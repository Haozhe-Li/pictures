import asyncio
import json
import random
from typing import Any, Dict, List

from core.config import settings
from core.embedding import EmbeddingClient
from core.generate_description import description_generator

# Redis queue of ready-to-serve random queries. Each item is a JSON object
# {"query": str, "image_id": str}; the image id only exists so a refill can avoid
# sampling the same source image twice. The query's embeddings are NOT stored here:
# they are warmed into the embedding cache (`embedding:{model}:{text}`), so the
# `/search` call that follows a click hits the cache instead of a cold service.
QUEUE_KEY = "random_query:queue"
REFILL_LOCK_KEY = "random_query:refill_lock"
IMAGE_POOL_KEY = "gallery:pool:active"  # ZSET holding every ingested image id

REFILL_THRESHOLD = 5  # refill once the queue is down to this many items
REFILL_BATCH = 10  # items added per refill
REFILL_LOCK_TTL_SECONDS = 300  # upper bound on one refill; guards against a crashed worker
REFILL_FAILURE_COOLDOWN_SECONDS = 30  # back off after a failed refill instead of retrying per click

# Queued queries can wait a long time on a quiet site, so keep their cached embeddings
# around much longer than the queue is expected to take to drain.
EMBEDDING_CACHE_TTL_SECONDS = 7 * 24 * 3600

QUERY_SYSTEM_PROMPT = """You turn photo titles and descriptions into short search queries, the way a person would type them into a photo library search box.

Requirements:
1) You get a numbered list of photos. Each has a title and a description (the description ends with a location).
2) For each photo write ONE query of 3-8 lowercase English words that captures its mood, subject and, when present, the place.
3) Do not copy the description verbatim. Reword and compress it; drop filler words and punctuation.
   Example: title "Where Daylight Lingers", description "Amber light spills across quiet streets and autumn canopies. Jacksonport, Lakeside Park Playground, Wisconsin" -> "daylight lingers on wisconsin autumn"
4) Output **JSON only**: {"queries": ["...", "..."]} with exactly one query per photo, in the same order.
"""


async def generate_random_query() -> str:
    """Static fallback, used whenever the queue is empty or unavailable."""
    choices = [
        "A serene landscape with mountains and a river",
        "A bustling city street at night with neon lights",
        "A close-up of a colorful butterfly on a flower",
        "A futuristic city skyline with flying cars",
        "A cozy cabin in the woods during autumn",
        "A majestic lion resting in the savannah",
        "A vibrant coral reef teeming with marine life",
        "A snowy mountain peak under a clear blue sky",
        "A group of friends enjoying a picnic in the park",
    ]
    return random.choice(choices)


class RandomQueryManager:
    """
    Serves random search queries that are guaranteed to be grounded in the gallery.

    A refill samples random images, has a small LLM rewrite each title + description
    into a search-style query, warms the query's embeddings, and appends it to a Redis
    queue. A click pops one query off the front; when the queue runs low a background
    refill tops it up. Anything going wrong falls back to the static list.
    """

    def __init__(self, redis_client, qdrant_wrapper, embedding_client: EmbeddingClient):
        self.redis = redis_client
        self.qdrant = qdrant_wrapper
        self.embedding_client = embedding_client
        self._tasks: set = set()

    # --- Serving ---

    async def next_query(self) -> str:
        """Pop the next query, or fall back to the static list. Never raises."""
        try:
            raw = await self.redis.lpop(QUEUE_KEY)
            remaining = await self.redis.llen(QUEUE_KEY)
            if remaining <= REFILL_THRESHOLD:
                self.schedule_refill()
            if raw:
                query = json.loads(raw).get("query")
                if query:
                    return query
            print("Random query queue empty, using static fallback")
        except Exception as e:
            print(f"Random query queue error, using static fallback: {e}")
        return await generate_random_query()

    def schedule_refill(self) -> None:
        """Start a background refill (no-op while another one holds the lock)."""
        task = asyncio.create_task(self._refill())
        self._tasks.add(task)  # keep a reference so the task isn't garbage collected
        task.add_done_callback(self._tasks.discard)

    async def warm_up(self) -> None:
        """Fill an empty queue at startup so the first click doesn't hit the fallback."""
        try:
            if await self.redis.llen(QUEUE_KEY) <= REFILL_THRESHOLD:
                self.schedule_refill()
        except Exception as e:
            print(f"Random query warm-up skipped: {e}")

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    # --- Refilling ---

    async def _refill(self) -> None:
        # Single-flight across requests and workers: only the lock holder refills.
        if not await self.redis.set(
            REFILL_LOCK_KEY, "1", nx=True, ex=REFILL_LOCK_TTL_SECONDS
        ):
            return

        try:
            added = await self._refill_batch()
        except asyncio.CancelledError:
            await self.redis.delete(REFILL_LOCK_KEY)
            raise
        except Exception as e:
            print(f"Random query refill failed: {e}")
            added = 0

        if added:
            await self.redis.delete(REFILL_LOCK_KEY)
        else:
            # Keep the lock briefly so a persistent failure isn't retried on every click.
            await self.redis.expire(REFILL_LOCK_KEY, REFILL_FAILURE_COOLDOWN_SECONDS)

    async def _refill_batch(self) -> int:
        queued_ids = await self._queued_image_ids()
        images = await self._sample_images(queued_ids, REFILL_BATCH)
        if not images:
            print("Random query refill: no images to sample")
            return 0

        queries = await self._rewrite_queries(images)

        added = 0
        for image, query in zip(images, queries):
            if not await self._warm_embeddings(query):
                # The client already retried for ~30s, so the embedding service is down:
                # trying the rest of the batch would only repeat that wait per query.
                print("Random query refill: embedding service unavailable, stopping batch")
                break
            item = json.dumps({"query": query, "image_id": image["id"]})
            await self.redis.rpush(QUEUE_KEY, item)
            added += 1
        print(f"Random query refill: queued {added}/{len(images)} queries")
        return added

    async def _queued_image_ids(self) -> set:
        ids = set()
        for raw in await self.redis.lrange(QUEUE_KEY, 0, -1):
            try:
                ids.add(json.loads(raw).get("image_id"))
            except Exception:
                continue
        return ids

    async def _sample_images(self, exclude: set, count: int) -> List[Dict[str, str]]:
        """Random images (id, title, description) from the whole gallery."""
        total = await self.redis.zcard(IMAGE_POOL_KEY)
        if total == 0:
            return []

        # Pick random ranks rather than ZRANDMEMBER so this works on any Redis version.
        # Oversample to absorb images already queued and ones that were since deleted.
        ranks = random.sample(range(total), min(total, count * 2 + len(exclude)))
        async with self.redis.pipeline(transaction=False) as pipe:
            for rank in ranks:
                pipe.zrange(IMAGE_POOL_KEY, rank, rank)
            found = await pipe.execute()
        ids = [i for hit in found for i in hit if i not in exclude]

        images: List[Dict[str, str]] = []
        for point in await self.qdrant.get_points(ids):
            payload = point.payload or {}
            title = str(payload.get("title") or "").strip()
            description = str(payload.get("description") or "").strip()
            if title or description:
                images.append(
                    {"id": str(point.id), "title": title, "description": description}
                )
        random.shuffle(images)
        return images[:count]

    async def _rewrite_queries(self, images: List[Dict[str, str]]) -> List[str]:
        """One LLM call for the whole batch; titles stand in for anything it fails to return."""
        titles = [(img["title"] or img["description"]).lower() for img in images]
        try:
            queries = await asyncio.to_thread(self._rewrite_queries_sync, images)
        except Exception as e:
            print(f"Random query LLM rewrite failed, using titles: {e}")
            return titles
        return [q or fallback for q, fallback in zip(queries, titles)]

    @staticmethod
    def _rewrite_queries_sync(images: List[Dict[str, str]]) -> List[str]:
        listing = "\n".join(
            f"{n}. title: {img['title']} | description: {img['description']}"
            for n, img in enumerate(images, 1)
        )
        completion = description_generator.client.chat.completions.create(
            model=description_generator.model,
            messages=[
                {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": listing},
            ],
            max_completion_tokens=512,
            stream=False,
            response_format={"type": "json_object"},
        )
        data: Any = json.loads(completion.choices[0].message.content or "{}")
        queries = data.get("queries")
        if not isinstance(queries, list) or len(queries) != len(images):
            raise ValueError(f"expected {len(images)} queries, got: {queries!r}")
        return [str(q).strip() for q in queries]

    async def _warm_embeddings(self, query: str) -> bool:
        """
        Embed the query with both models `/search` uses, so the results land in the
        embedding cache. The client retries cold-start failures itself; False means it
        gave up, and the query must not be queued (clicking it would hit a cold service).
        """
        for model in (settings.CLIP_MODEL, settings.TEXT_MODEL):
            try:
                await asyncio.to_thread(
                    self.embedding_client.get_embedding, text=query, model=model
                )
            except Exception as e:
                print(f"Random query embedding failed ({model}): '{query}': {e}")
                return False
            # Warmed entries get a TTL so one-off random queries don't pile up forever.
            await self.redis.expire(
                EmbeddingClient.text_cache_key(query, model), EMBEDDING_CACHE_TTL_SECONDS
            )
        return True
