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
    ) -> List[float]:
        """
        Get a dense embedding for a single text or image from the embedding service.

        `is_query` is accepted for call-site compatibility; jina-clip-v1 has no
        query/document distinction.
        """
        # Use cache for text-only queries (e.g. Search)
        if text and not image_url and not image_base64:
            return self._get_cached_text_embedding(text)

        return self._execute_embedding_request(
            text, image_url, image_base64, is_query=is_query
        )

    @lru_cache(maxsize=1024)
    def _get_cached_text_embedding(self, text: str) -> List[float]:
        """
        Layer 1: Memory Cache (LRU)
        Layer 2: Redis Cache (Persistent)
        Layer 3: API Call
        """
        # Checks Redis before hitting API
        if self.redis_client:
            redis_key = f"embedding:{text}"
            try:
                cached_data = self.redis_client.get(redis_key)
                if cached_data:
                    print(f"Hit Redis cache for query: '{text}'")
                    return json.loads(cached_data)
            except Exception as e:
                print(f"Redis get error: {e}")

        # If not in Redis or Redis failed, call the embedding service
        embedding = self._execute_embedding_request(text=text)

        # Save to Redis for future
        if self.redis_client:
            try:
                # Cache for 1 week (604800 seconds) or indefinite?
                # Let's say 24h for now or indefinite. User said "persist", so maybe no expiry.
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
    ) -> List[float]:
        if text:
            path, payload = "/embed/dense/text", {"texts": [text]}
        elif image_url or image_base64:
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

        embedding = response.json()["embeddings"][0]
        if len(embedding) != settings.EMBEDDING_DIM:
            raise ValueError(
                f"Unexpected embedding dim {len(embedding)}, expected {settings.EMBEDDING_DIM}"
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
