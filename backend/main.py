import asyncio
import uuid
import os
import random
import time
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
    status,
    BackgroundTasks,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from core.random_query import generate_random_query
import secrets
import json
from enum import Enum
import redis.asyncio as redis

# Imports from Core
from core.config import settings
from core.embedding import EmbeddingClient, get_sparse_embedding
from core.storage import upload_file_to_r2
from core.db import QdrantClientWrapper
from core.utils import process_image_for_embedding, save_as_webp
from core.generate_description import description_generator
from core.autocomplete import autocomplete_manager

async def _none():
    return None


# --- Initialize Clients ---
embedding_client = EmbeddingClient()
qdrant_wrapper = QdrantClientWrapper()
redis_client = redis.Redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load autocomplete model if available
    autocomplete_manager.initialize()

    # Startup: Initialize Qdrant Collection
    await qdrant_wrapper.init_collection()
    yield
    await redis_client.close()
    await qdrant_wrapper.client.close()


app = FastAPI(title="Gallery RAG Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/autocomplete")
async def autocomplete(q: str):
    """
    Get autocomplete suggestions for the given query.
    """
    if not q:
        return {"suggestions": []}

    suggestions = autocomplete_manager.suggest(q)
    return {"suggestions": suggestions}


@app.get("/image/{image_id}")
async def get_image_by_id(image_id: str):
    """
    Get a single image detail by its ID.
    """
    try:
        point = await qdrant_wrapper.get_point(image_id)
        if not point:
            raise HTTPException(status_code=404, detail="Image not found")

        metadata = point.payload
        return SearchResult(
            preview_url=metadata.get("preview_url", ""),
            original_url=metadata.get("original_url", ""),
            score=1.0,  # Exact match by ID
            metadata=metadata,
            id=image_id,
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"Error fetching image {image_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/autocomplete/build")
async def build_autocomplete():
    """
    Trigger a rebuild of the autocomplete index from Qdrant data.
    """
    await autocomplete_manager.build_index(qdrant_wrapper)
    return {"status": "success", "message": "Autocomplete index rebuilt."}


# --- Security ---
security = HTTPBasic()


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    expected_username = os.getenv("ADMIN_USERNAME")
    expected_password = os.getenv("ADMIN_PASSWORD")

    # Without configured credentials there is nothing to authenticate against.
    # Refuse rather than crashing inside compare_digest on a None operand.
    if not expected_username or not expected_password:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin credentials are not configured on the server",
        )

    # Compare as bytes so non-ASCII credentials cannot raise, and evaluate both
    # halves before combining so the check stays constant-time.
    is_correct_username = secrets.compare_digest(
        credentials.username.encode("utf-8"), expected_username.encode("utf-8")
    )
    is_correct_password = secrets.compare_digest(
        credentials.password.encode("utf-8"), expected_password.encode("utf-8")
    )

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# --- Pydantic Models for Search ---
class SearchMode(str, Enum):
    HYBRID = "hybrid"
    TEXT_ONLY = "text-only"
    IMAGE_ONLY = "image-only"


class SearchRequest(BaseModel):
    query: str
    limit: int = 4
    similarity_threshold: Optional[float] = None
    search_mode: SearchMode = SearchMode.HYBRID


class SimilarToRequest(BaseModel):
    image_url: str
    limit: int = 4


class SearchResult(BaseModel):
    id: str
    preview_url: str
    original_url: str
    metadata: dict
    score: float


class GalleryResponse(BaseModel):
    items: List[SearchResult]
    next_cursor: Optional[str] = None


class FeedRequest(BaseModel):
    limit: int = 20
    seen_ids: List[str] = []


class GenerateDescriptionResponse(BaseModel):
    title: str
    description: str


# --- Endpoints ---


@app.post("/ingest")
async def ingest_image(
    username: str = Depends(verify_credentials),
    file: UploadFile = File(...),
    title: str = Form(...),
    taken_time: Optional[str] = Form(None),
    camera: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    """
    Ingest an image:
    1. Read and Convert to Base64
    2. Save Preview & Original Versions to R2
    3. Get Dense Embedding from the embedding service
    4. Generate Sparse Embedding from Metadata
    5. Save to Qdrant
    """
    try:
        # 1. Read file
        file_bytes = await file.read()
        file_uuid = uuid.uuid4()

        # Generate filenames for R2 (WebP)
        # Preview: Quality 10, Original: Quality 90
        storage_filename_preview = f"{file_uuid}_preview.webp"
        storage_filename_original = f"{file_uuid}_original.webp"

        # Save temp files for R2 upload
        temp_filename_preview = f"/tmp/{storage_filename_preview}"
        temp_filename_original = f"/tmp/{storage_filename_original}"

        # 1.1 Save Preview (Quality 10) with lower resolution
        await run_in_threadpool(
            save_as_webp, file_bytes, temp_filename_preview, 5, max_size=1200
        )

        # 1.2 Save Original (Quality 90)
        await run_in_threadpool(save_as_webp, file_bytes, temp_filename_original, 60)

        # Process image for embedding (Resize & Compress)
        # We generally deliver the compressed version to embedding model to save bandwidth and meet limits.
        # R2 gets the compressed WebP file via temp_filename logic.
        base64_str = await run_in_threadpool(process_image_for_embedding, file_bytes)

        print("Prepare to embed image via embedding service...")

        # 2. Dense Embedding (Image)
        try:
            dense_embedding = await run_in_threadpool(
                embedding_client.get_embedding, image_base64=base64_str
            )
        except Exception as e:
            print(e)
            raise HTTPException(
                status_code=500, detail=f"Image embedding failed: {str(e)}"
            )

        print(f"Dense embedding length: {len(dense_embedding)}")

        # 3. R2 Upload
        try:
            # Upload the Preview WebP file
            r2_url_preview = await run_in_threadpool(
                upload_file_to_r2, temp_filename_preview, storage_filename_preview
            )
            # Upload the Original WebP file
            r2_url_original = await run_in_threadpool(
                upload_file_to_r2, temp_filename_original, storage_filename_original
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"R2 Upload failed: {str(e)}")
        finally:
            if os.path.exists(temp_filename_preview):
                os.remove(temp_filename_preview)
            if os.path.exists(temp_filename_original):
                os.remove(temp_filename_original)

        print(f"Uploaded to R2. Preview: {r2_url_preview}, Original: {r2_url_original}")

        # 4. Sparse Embedding
        # Combine relevant metadata text for sparse search
        metadata_text = f"{title} {description or ''} {taken_time or ''} {camera or ''}"

        # FastEmbed is CPU bound, might want to offload if heavy, but for short text it's fast.
        sparse_vec = await run_in_threadpool(get_sparse_embedding, metadata_text)

        # 4.5 Dense Embedding (Metadata Text)
        try:
            text_dense_embedding = await run_in_threadpool(
                embedding_client.get_embedding,
                text=metadata_text,
                model=settings.TEXT_MODEL,
            )
        except Exception as e:
            print(f"Warning: Metadata Dense Embedding failed: {e}")
            # Fallback? Or just fail? Let's use zero vector or fail.
            # Ideally fail, but maybe we can just duplicate image embedding if we had to.
            # But let's fail to ensure quality.
            raise HTTPException(
                status_code=500, detail=f"Metadata Embedding failed: {str(e)}"
            )

        # 5. Qdrant Upsert
        point_id = str(file_uuid)
        payload = {
            "title": title,
            "taken_time": taken_time,
            "camera": camera,
            "description": description,
            "preview_url": r2_url_preview,
            "original_url": r2_url_original,
            "type": "image",
        }

        await qdrant_wrapper.upsert_point(
            point_id=point_id,
            image_dense_vector=dense_embedding,
            text_dense_vector=text_dense_embedding,
            sparse_vector=sparse_vec,
            payload=payload,
        )

        # 6. Add to Redis Pools for Waterfall Recommendation
        await redis_client.zadd("gallery:pool:active", {point_id: 0})
        await redis_client.zadd("gallery:pool:explore", {point_id: 0})

        return {
            "status": "success",
            "id": point_id,
            "preview_url": r2_url_preview,
            "original_url": r2_url_original,
            "metadata_used_for_sparse": metadata_text,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        # Cleanup?
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-description", response_model=GenerateDescriptionResponse)
async def generate_description_endpoint(file: UploadFile = File(...)):
    """
    Generate title and description for an image using LLM.
    """
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        # Compress/resize before sending to LLM
        image_base64 = await run_in_threadpool(process_image_for_embedding, file_bytes)

        result = await description_generator.generate(image_base64)

        if not result.get("title") or not result.get("description"):
            raise HTTPException(
                status_code=500, detail="LLM response missing title or description"
            )

        return GenerateDescriptionResponse(**result)

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=List[SearchResult])
async def search_images(request: SearchRequest):
    """
    Search for images using text query.
    1. Get the CLIP embedding of the query (matches `dense-image`)
    2. Get the MiniLM embedding of the query (matches `dense-text`)
    3. Get Sparse Embedding for Query (Text)
    4. Retrieve from Qdrant
    Only the embeddings the chosen search_mode actually uses are requested.
    """
    try:
        print("Getting embeddings for search query...")
        use_image = request.search_mode != SearchMode.TEXT_ONLY
        use_text = request.search_mode != SearchMode.IMAGE_ONLY

        async def dense(model: str):
            return await run_in_threadpool(
                embedding_client.get_embedding, text=request.query, is_query=True, model=model
            )

        # 1+2. Dense. Issued together; the service queues them on its single worker.
        image_vec, text_vec = await asyncio.gather(
            dense(settings.CLIP_MODEL) if use_image else _none(),
            dense(settings.TEXT_MODEL) if use_text else _none(),
        )
        # 3. Sparse (Text)
        sparse_vec = await run_in_threadpool(get_sparse_embedding, request.query)

        print("Searching Qdrant...")
        # 4. Search
        results = await qdrant_wrapper.search(
            image_vector=image_vec,
            text_vector=text_vec,
            sparse_vector=sparse_vec,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold,
            search_mode=request.search_mode,
        )

        print(results)

        # Format results
        output = []
        for hit in results:
            output.append(
                SearchResult(
                    id=str(hit.id),
                    preview_url=hit.payload.get("preview_url", ""),
                    original_url=hit.payload.get("original_url", ""),
                    metadata=hit.payload,
                    score=hit.score,
                )
            )

        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similar-to", response_model=List[SearchResult])
async def similar_to_image(request: SimilarToRequest):
    """
    Find similar images based on an existing image URL.
    1. Check if image URL exists in Qdrant
    2. Retrieve the image's stored vectors (dense-image + dense-text + sparse)
    3. Perform a hybrid search
    4. Return results in the same schema as /search
    """
    try:
        point = await qdrant_wrapper.find_point_by_image_url(request.image_url)
        if not point:
            raise HTTPException(status_code=404, detail="image_url not found")

        vectors = getattr(point, "vector", None) or getattr(point, "vectors", None)
        if not vectors:
            raise HTTPException(status_code=500, detail="Vectors not found for image")

        # Each stored vector is queried against its own named vector: the two
        # dense spaces (CLIP 768-d, MiniLM 384-d) are not interchangeable.
        image_vector = vectors.get("dense-image")
        text_vector = vectors.get("dense-text")
        sparse_vector = vectors.get("sparse")

        if image_vector is None or text_vector is None or sparse_vector is None:
            raise HTTPException(status_code=500, detail="Required vectors missing")

        sparse_vec = qdrant_wrapper.normalize_sparse_vector(sparse_vector)

        # Fetch one extra to allow removing the original image from results
        results = await qdrant_wrapper.search(
            image_vector=image_vector,
            text_vector=text_vector,
            sparse_vector=sparse_vec,
            limit=request.limit + 1,
        )

        output = []
        for hit in results:
            if (
                hit.payload.get("original_url") == request.image_url
                or hit.payload.get("preview_url") == request.image_url
            ):
                continue
            output.append(
                SearchResult(
                    id=str(hit.id),
                    preview_url=hit.payload.get("preview_url", ""),
                    original_url=hit.payload.get("original_url", ""),
                    metadata=hit.payload,
                    score=hit.score,
                )
            )

        return output[: request.limit]

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gallery", response_model=GalleryResponse)
async def get_gallery(request: FeedRequest):
    """
    Get dynamic waterfall images using Redis Waterfall logic.
    """
    try:
        limit = request.limit
        seen_set = set(request.seen_ids)

        # Calculate roughly 90% hot, 10% cold
        hot_limit = int(limit * 0.9)
        cold_limit = limit - hot_limit

        # Dynamically calculate how many items to fetch from Redis to guarantee enough unseen images
        fetch_count = len(seen_set) + limit + 20

        # 1. Fetch hot images (Exploit)
        # Fetch enough to account for seen_ids
        hot_pool = await redis_client.zrevrange("gallery:pool:active", 0, fetch_count)
        available_hot = [cid for cid in hot_pool if cid not in seen_set]
        hot_ids = available_hot[:hot_limit]

        # 2. Fetch cold images (Explore)
        # zrange returns ascending by score (impression count), so least-shown items come first
        cold_pool = await redis_client.zrange("gallery:pool:explore", 0, fetch_count)
        # Exclude IDs already in hot_ids and seen_ids; preserve ascending score order
        available_cold = [
            cid for cid in cold_pool if cid not in seen_set and cid not in hot_ids
        ]

        # Take least-shown items first (front of list has lowest impression counts).
        # Shuffle only within a small candidate window to add variety without causing repeats.
        candidate_window = available_cold[: cold_limit * 3]
        random.shuffle(candidate_window)
        cold_ids = candidate_window[:cold_limit]

        # 3. Combine and shuffle
        final_ids = hot_ids + cold_ids
        random.shuffle(final_ids)

        if not final_ids:
            return GalleryResponse(items=[], next_cursor=None)

        # 4. Fetch metadata from Qdrant
        points = await qdrant_wrapper.client.retrieve(
            collection_name=settings.COLLECTION_NAME, ids=final_ids
        )

        points_map = {str(p.id): p for p in points}

        items = []
        for pid in final_ids:
            if pid in points_map:
                point = points_map[pid]
                items.append(
                    SearchResult(
                        id=str(point.id),
                        preview_url=point.payload.get("preview_url", ""),
                        original_url=point.payload.get("original_url", ""),
                        metadata=point.payload,
                        score=1.0,  # Default score for browsing
                    )
                )

        # 5. Async trigger impressions for explore images
        # For simplicity and speed in Redis, we await them directly, it's very fast
        for cid in cold_ids:
            await redis_client.zincrby("gallery:pool:explore", 1, cid)
            await redis_client.zincrby("gallery:pool:active", 1, cid)

            # Check graduation: remove from explore pool after 30 impressions
            score = await redis_client.zscore("gallery:pool:explore", cid)
            if score and score >= 30:
                await redis_client.zrem("gallery:pool:explore", cid)

        return GalleryResponse(items=items, next_cursor=None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class InteractRequest(BaseModel):
    image_id: str
    action_type: str  # 'click' or 'share'


@app.post("/interact")
async def interact_image(request: InteractRequest):
    """
    Interaction loop for tracking clicks and shares.
    """
    try:
        image_id = request.image_id
        if request.action_type == "click":
            await redis_client.hincrby("gallery:counters:click", image_id, 1)
            await redis_client.zincrby("gallery:pool:active", 2, image_id)
        elif request.action_type == "share":
            await redis_client.hincrby("gallery:counters:share", image_id, 1)
            await redis_client.zincrby("gallery:pool:active", 4, image_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate-random-query")
async def generate_random_query_endpoint():
    """
    Generate a random photo description query using LLM.
    """
    try:
        query = await generate_random_query()
        return {"query": query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


DECAY_INTERVAL_SECONDS = 7200  # Decay at most once every 2 hours
DECAY_FACTOR = 0.98
DECAY_FLOOR = 0.1  # Scores below this collapse to 0
DECAY_BATCH = 500


async def _decay_active_pool():
    """
    Multiplicatively decay every score in the active pool.

    Runs out of the request path (via BackgroundTasks) and batches writes into
    pipelines, so a large pool cannot stall the caller.
    """
    try:
        photos = await redis_client.zrange(
            "gallery:pool:active", 0, -1, withscores=True
        )

        pipe = redis_client.pipeline(transaction=False)
        pending = 0
        for photo_id, score in photos:
            if score == 0:
                continue
            new_score = score * DECAY_FACTOR
            if new_score < DECAY_FLOOR:
                new_score = 0
            pipe.zadd("gallery:pool:active", {photo_id: new_score})
            pending += 1
            if pending % DECAY_BATCH == 0:
                await pipe.execute()

        if pending % DECAY_BATCH:
            await pipe.execute()

        print(f"Decay applied to {pending} photos.")
    except Exception as e:
        print(f"Decay Error: {e}")


@app.get("/health")
async def health_check(background_tasks: BackgroundTasks):
    """
    Health check endpoint, also used as a trigger for heat decay cron.
    """
    try:
        # SET NX EX is atomic, so exactly one worker wins each decay window even
        # when several health checks arrive at the same moment. The key expiring
        # is what re-opens the window.
        acquired = await redis_client.set(
            "gallery:decay:lock",
            time.time(),
            nx=True,
            ex=DECAY_INTERVAL_SECONDS,
        )
        if acquired:
            background_tasks.add_task(_decay_active_pool)
    except Exception as e:
        print(f"Decay scheduling error: {e}")

    return {"status": "ok"}
