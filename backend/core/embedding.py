import os
import requests
import json
import redis
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
from functools import lru_cache
from fastembed import SparseTextEmbedding
from core.config import settings


# --- Embedding Service Client (Dense) ---
class EmbeddingClient:
    def __init__(self):
        self.base_url = settings.EMBEDDING_SERVICE_URL.rstrip("/")
        self.session = requests.Session()
        # Initialize Redis connection
        try:
            if not settings.REDIS_URL:
                raise ValueError("REDIS_URL is not configured")
            self.redis_client = redis.Redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_timeout=2,  # Short timeout to not block app if Redis is down
            )
            self.redis_client.ping()  # Check connection
        except Exception as e:
            print(
                f"Warning: Redis connection failed ({e}). Running without Redis cache."
            )
            self.redis_client = None

    def get_embedding(
        self,
        text: Optional[str] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        is_query: bool = False,
        model: Optional[str] = None,
    ) -> List[float]:
        """
        Get a dense embedding for a single text or image from the embedding service.

        Images always use CLIP. Text uses `model`, which defaults to CLIP (for matching
        against image vectors); pass `settings.TEXT_MODEL` for text-to-text matching.
        Vectors from different models are not comparable.

        `is_query` is accepted for call-site compatibility; neither model has a
        query/document distinction.
        """
        # Use cache for text-only queries (e.g. Search)
        if text and not image_url and not image_base64:
            return self._get_cached_text_embedding(text, model or settings.CLIP_MODEL)

        return self._execute_embedding_request(
            text, image_url, image_base64, is_query=is_query, model=model
        )

    @lru_cache(maxsize=1024)
    def _get_cached_text_embedding(self, text: str, model: str) -> List[float]:
        """
        Layer 1: Memory Cache (LRU)
        Layer 2: Redis Cache (Persistent)
        Layer 3: API Call

        Keyed by model: vectors from different models must never be mixed up.
        """
        dim = settings.EMBEDDING_DIMS[model]
        redis_key = f"embedding:{model}:{text}"
        # Checks Redis before hitting API
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(redis_key)
                if cached_data:
                    cached = json.loads(cached_data)
                    if len(cached) == dim:
                        print(f"Hit Redis cache for query: '{text}' ({model})")
                        return cached
                    print(f"Ignoring stale {len(cached)}-d cache entry for query: '{text}'")
            except Exception as e:
                print(f"Redis get error: {e}")

        # If not in Redis or Redis failed, call the embedding service
        embedding = self._execute_embedding_request(text=text, model=model)

        # Save to Redis for future (no expiry: the key already pins the model)
        if self.redis_client:
            try:
                self.redis_client.set(redis_key, json.dumps(embedding))
            except Exception as e:
                print(f"Redis set error: {e}")

        return embedding

    def _execute_embedding_request(
        self,
        text: Optional[str] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        is_query: bool = False,
        model: Optional[str] = None,
    ) -> List[float]:
        if text:
            model = model or settings.CLIP_MODEL
            path, payload = "/embed/dense/text", {"texts": [text], "model": model}
        elif image_url or image_base64:
            model = settings.CLIP_MODEL
            path, payload = "/embed/dense/image", {"images": [image_url or image_base64]}
        else:
            raise ValueError("No input provided")

        response = self.session.post(
            f"{self.base_url}{path}", json=payload, timeout=(5, 60)
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_msg = response.text[:500]
            print("Embedding service error:", error_msg)
            raise ValueError(
                f"Embedding service failed: {response.status_code} - {error_msg}"
            ) from e

        body = response.json()
        embedding = body["embeddings"][0]
        dim = settings.EMBEDDING_DIMS[model]
        if body.get("model") != model or len(embedding) != dim:
            raise ValueError(
                f"Unexpected embedding from service: model={body.get('model')!r} "
                f"dim={len(embedding)}, expected model={model!r} dim={dim}"
            )
        return embedding


# --- Sparse Embedding (FastEmbed) ---

# Ensure model directory exists
bm25_model_path = Path(settings.MODELS_DIR) / "bm25"

# If the user wants to manage the model manually, we assume it's there or FastEmbed handles download to that path.
sparse_embedding_model = SparseTextEmbedding(
    model_name="Qdrant/bm25",
    cache_dir=str(settings.MODELS_DIR),  # fastembed uses cache_dir to store models
)


def get_sparse_embedding(text: str) -> Dict[str, Any]:
    """
    Generate sparse vector using FastEmbed (BM25).
    Returns dictionary format compatible with Qdrant: {'indices': [...], 'values': [...]}
    """
    # embed returns a generator of SparseEmbedding (which has .indices and .values)
    embedding_gen = sparse_embedding_model.embed([text])
    result = next(embedding_gen)

    # FastEmbed returns numpy arrays, convert to list for JSON serialization/Qdrant
    return {"indices": result.indices.tolist(), "values": result.values.tolist()}
