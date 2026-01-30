import asyncio
from typing import List, Dict, Any, Optional
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from core.config import settings


class QdrantClientWrapper:
    def __init__(self):
        self.client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )

    async def init_collection(self, vector_size: int = 512):
        """
        Initialize the collection with dense and sparse vector configuration.
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
                        size=vector_size, distance=Distance.COSINE
                    ),
                    "dense-text": VectorParams(
                        size=vector_size, distance=Distance.COSINE
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
        self, dense_vector: List[float], sparse_vector: Dict[str, Any], limit: int = 10, similarity_threshold: Optional[float] = None, search_mode: str = "hybrid"
    ):
        if search_mode == "hybrid":
            prefetch = [
                models.Prefetch(
                    query=dense_vector,
                    using="dense-image",
                    limit=limit * 2,
                    score_threshold=similarity_threshold,
                ),
                models.Prefetch(
                    query=dense_vector,
                    using="dense-text",
                    limit=limit * 2,
                    score_threshold=similarity_threshold,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector["indices"],
                        values=sparse_vector["values"],
                    ),
                    using="sparse",
                    limit=limit * 2,
                ),
            ]
        elif search_mode == "text-only":
            prefetch = [
                models.Prefetch(
                    query=dense_vector,
                    using="dense-text",
                    limit=limit * 2,
                    score_threshold=similarity_threshold,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector["indices"],
                        values=sparse_vector["values"],
                    ),
                    using="sparse",
                    limit=limit * 2,
                ),
            ]
        elif search_mode == "image-only":
            prefetch = [
                models.Prefetch(
                    query=dense_vector,
                    using="dense-image",
                    limit=limit * 2,
                    score_threshold=similarity_threshold,
                ),
            ]

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
