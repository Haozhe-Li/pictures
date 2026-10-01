import asyncio
from typing import List, Dict, Any, Optional
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from core.config import settings


class QdrantClientWrapper:
    def __init__(self):
        self.client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            port=None,  # honour the URL's own port (https://host => 443, not 6333)
            api_key=settings.QDRANT_API_KEY,
        )

    async def init_collection(self):
        """
        Initialize the collection with dense and sparse vector configuration.
        `dense-image` is CLIP-sized, `dense-text` is MiniLM-sized.
        """
        collections = await self.client.get_collections()
        exists = any(
            c.name == settings.COLLECTION_NAME for c in collections.collections
        )

        if not exists:
            await self.client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config={
                    "dense-image": VectorParams(
                        size=settings.CLIP_DIM, distance=Distance.COSINE
                    ),
                    "dense-text": VectorParams(
                        size=settings.TEXT_DIM, distance=Distance.COSINE
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        )
                    )
                },
            )
            print(f"Collection {settings.COLLECTION_NAME} created.")
        else:
            print(f"Collection {settings.COLLECTION_NAME} already exists.")
            cfg = (
                await self.client.get_collection(settings.COLLECTION_NAME)
            ).config.params.vectors
            for name, dim in (("dense-image", settings.CLIP_DIM), ("dense-text", settings.TEXT_DIM)):
                if name not in cfg or cfg[name].size != dim:
                    raise RuntimeError(
                        f"Collection {settings.COLLECTION_NAME}: '{name}' is not {dim}-d; "
                        "run scripts/migrate_text_model.py"
                    )

        # find_point_by_image_url filters on original_url; without an index that
        # is a full scan. Creating an existing index is a no-op on Qdrant's side,
        # so this is safe to run on every startup.
        try:
            await self.client.create_payload_index(
                collection_name=settings.COLLECTION_NAME,
                field_name="original_url",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            print(f"Warning: could not ensure payload index on original_url: {e}")

    async def upsert_point(
        self,
        point_id: str,
        image_dense_vector: List[float],
        text_dense_vector: List[float],
        sparse_vector: Dict[str, Any],
        payload: Dict[str, Any],
    ):
        await self.client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense-image": image_dense_vector,
                        "dense-text": text_dense_vector,
                        "sparse": models.SparseVector(
                            indices=sparse_vector["indices"],
                            values=sparse_vector["values"],
                        ),
                    },
                    payload=payload,
                )
            ],
        )

    async def search(
        self,
        image_vector: Optional[List[float]],
        text_vector: Optional[List[float]],
        sparse_vector: Dict[str, Any],
        limit: int = 10,
        similarity_threshold: Optional[float] = None,
        search_mode: str = "hybrid",
    ):
        """
        `image_vector` is a CLIP vector, queried against `dense-image`.
        `text_vector` is a MiniLM vector, queried against `dense-text`.
        Each mode only needs the vectors it uses.
        """
        sparse_prefetch = models.Prefetch(
            query=models.SparseVector(
                indices=sparse_vector["indices"],
                values=sparse_vector["values"],
            ),
            using="sparse",
            limit=limit * 2,
        )

        def dense_prefetch(vector, using):
            return models.Prefetch(
                query=vector,
                using=using,
                limit=limit * 2,
                score_threshold=similarity_threshold,
            )

        if search_mode == "hybrid":
            prefetch = [
                dense_prefetch(image_vector, "dense-image"),
                dense_prefetch(text_vector, "dense-text"),
                sparse_prefetch,
            ]
        elif search_mode == "text-only":
            prefetch = [dense_prefetch(text_vector, "dense-text"), sparse_prefetch]
        elif search_mode == "image-only":
            prefetch = [dense_prefetch(image_vector, "dense-image")]
        else:
            raise ValueError(f"Unsupported search_mode: {search_mode}")

        search_result = await self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            limit=limit,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            with_payload=True,
        )
        return search_result.points

    async def scroll(self, limit: int = 20, offset: str = None):
        """
        Scroll through points in the collection (pagination).
        """
        points, next_offset = await self.client.scroll(
            collection_name=settings.COLLECTION_NAME,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        return points, next_offset

    async def find_point_by_image_url(self, image_url: str):
        """
        Find a point by matching preview_url or original_url in payload.
        Returns the first matching point with vectors included.
        """
        flt = models.Filter(
            should=[
                # models.FieldCondition(
                #     key="preview_url",
                #     match=models.MatchValue(value=image_url),
                # ),
                models.FieldCondition(
                    key="original_url",
                    match=models.MatchValue(value=image_url),
                ),
            ]
        )

        res = await self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            limit=1,
            query_filter=flt,
            with_payload=True,
            with_vectors=True,
        )
        return res.points[0] if res.points else None

    async def get_point(self, point_id: str):
        """
        Fetch a single point by ID.
        """
        points = await self.client.retrieve(
            collection_name=settings.COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
            with_vectors=False
        )
        return points[0] if points else None

    @staticmethod
    def normalize_sparse_vector(sparse_vector: Any) -> Dict[str, Any]:
        """
        Normalize sparse vector to {'indices': [...], 'values': [...]}.
        """
        if isinstance(sparse_vector, models.SparseVector):
            return {
                "indices": list(sparse_vector.indices),
                "values": list(sparse_vector.values),
            }

        if (
            isinstance(sparse_vector, dict)
            and "indices" in sparse_vector
            and "values" in sparse_vector
        ):
            return {
                "indices": list(sparse_vector["indices"]),
                "values": list(sparse_vector["values"]),
            }

        raise ValueError("Unsupported sparse vector format")
