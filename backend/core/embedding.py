import os
import requests
import json
import redis
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
from functools import lru_cache
from fastembed import SparseTextEmbedding
from core.config import settings


# --- Jina Client (Dense) ---
class JinaClient:
    def __init__(self):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.JINA_API_KEY}",
        }
        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                username=settings.REDIS_USERNAME,
                password=settings.REDIS_PASSWORD,
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
        Get embedding for a single text or image using Jina CLIP v2.
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

        # If not in Redis or Redis failed, get from API
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
        input_data = []
        if text:
            input_data.append({"text": text})

        if image_url:
            input_data.append({"image": image_url})

        if image_base64:
            input_data.append({"image": image_base64})

        if not input_data:
            raise ValueError("No input provided")

        data = {
            "model": "jina-clip-v2",
            "dimensions": 512,
            "input": input_data,
        }

        if is_query:
            data["task"] = "retrieval.query"

        response = requests.post(settings.JINA_URL, headers=self.headers, json=data)

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Print error detail, truncated to avoid logging huge base64 reflected in error (unlikely but safe)
            error_msg = response.text[:500]
            print("Jina API Error Detail:", error_msg)
            raise ValueError(
                f"Jina API Validation Failed: {response.status_code} - {error_msg}"
            ) from e

        result_data = response.json()["data"]
        return result_data[0]["embedding"]


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
