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

        # Collection membership lives on the image points; listing a collection's images
        # filters on it.
        try:
            await self.client.create_payload_index(
                collection_name=settings.COLLECTION_NAME,
                field_name="collection_ids",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            print(f"Warning: could not ensure payload index on collection_ids: {e}")

        await self._init_gallery_collections()

    async def _init_gallery_collections(self):
        """Create the store for collection anchors (text vectors only) if it is missing."""
        collections = await self.client.get_collections()
        if any(c.name == settings.GALLERY_COLLECTIONS_NAME for c in collections.collections):
            return
        await self.client.create_collection(
            collection_name=settings.GALLERY_COLLECTIONS_NAME,
            vectors_config={
                "dense-text": VectorParams(size=settings.TEXT_DIM, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            },
        )
        print(f"Collection {settings.GALLERY_COLLECTIONS_NAME} created.")

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

    async def scroll(
        self,
        limit: int = 20,
        offset: str = None,
        scroll_filter: Optional[models.Filter] = None,
    ):
        """
        Scroll through points in the collection (pagination), optionally filtered.
        """
        points, next_offset = await self.client.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        return points, next_offset

    async def count(self, count_filter: Optional[models.Filter] = None) -> int:
        """Number of points in the collection (matching `count_filter`, if given)."""
        result = await self.client.count(
            collection_name=settings.COLLECTION_NAME,
            count_filter=count_filter,
            exact=True,
        )
        return result.count

    async def update_metadata(
        self,
        point_id: str,
        text_dense_vector: List[float],
        sparse_vector: Dict[str, Any],
        payload: Dict[str, Any],
    ):
        """
        Replace the metadata-derived vectors and merge `payload` into the point.
        `dense-image` is left untouched. Both operations go out in one request so a
        failure cannot leave the payload and its vectors out of step.
        """
        await self.client.batch_update_points(
            collection_name=settings.COLLECTION_NAME,
            update_operations=[
                models.UpdateVectorsOperation(
                    update_vectors=models.UpdateVectors(
                        points=[
                            models.PointVectors(
                                id=point_id,
                                vector={
                                    "dense-text": text_dense_vector,
                                    "sparse": models.SparseVector(
                                        indices=sparse_vector["indices"],
                                        values=sparse_vector["values"],
                                    ),
                                },
                            )
                        ]
                    )
                ),
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(payload=payload, points=[point_id])
                ),
            ],
        )

    async def delete_point(self, point_id: str):
        await self.client.delete(
            collection_name=settings.COLLECTION_NAME,
            points_selector=models.PointIdsList(points=[point_id]),
        )

    # --- Collections (user-facing albums) ---

    @staticmethod
    def collection_filter(collection_id: str) -> models.Filter:
        """Matches the images that belong to `collection_id`."""
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="collection_ids", match=models.MatchValue(value=collection_id)
                )
            ]
        )

    async def set_membership(self, memberships: Dict[str, Dict[str, Any]]):
        """
        Write `collection_ids` / `collections_opt_out` onto image points. `memberships`
        maps point id -> the payload keys to set. Sent in batches, one operation per point.
        """
        items = list(memberships.items())
        for i in range(0, len(items), 200):
            await self.client.batch_update_points(
                collection_name=settings.COLLECTION_NAME,
                update_operations=[
                    models.SetPayloadOperation(
                        set_payload=models.SetPayload(payload=payload, points=[point_id])
                    )
                    for point_id, payload in items[i : i + 200]
                ],
            )

    async def search_images_by_text_vector(
        self, text_vector: List[float], limit: int, score_threshold: Optional[float] = None
    ):
        """Images ranked by cosine similarity of their `dense-text` vector to `text_vector`."""
        res = await self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=text_vector,
            using="dense-text",
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return res.points

    async def query_collection_images(
        self,
        collection_id: str,
        anchor_vector: List[float],
        limit: int,
        offset: int = 0,
    ):
        """
        The images of one collection, closest to its anchor first, so the most representative
        come first and the first one is the natural cover. Paged with a plain offset.
        """
        res = await self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=anchor_vector,
            using="dense-text",
            query_filter=self.collection_filter(collection_id),
            limit=limit,
            offset=offset,
            with_payload=True,
        )
        return res.points

    async def upsert_gallery_collection(
        self,
        collection_id: str,
        text_dense_vector: List[float],
        sparse_vector: Dict[str, Any],
        payload: Dict[str, Any],
    ):
        await self.client.upsert(
            collection_name=settings.GALLERY_COLLECTIONS_NAME,
            points=[
                models.PointStruct(
                    id=collection_id,
                    vector={
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

    async def update_gallery_collection(
        self,
        collection_id: str,
        payload: Dict[str, Any],
        text_dense_vector: Optional[List[float]] = None,
        sparse_vector: Optional[Dict[str, Any]] = None,
    ):
        """Merge `payload`; replace the anchor vectors too when they are given."""
        operations: List[Any] = []
        if text_dense_vector is not None and sparse_vector is not None:
            operations.append(
                models.UpdateVectorsOperation(
                    update_vectors=models.UpdateVectors(
                        points=[
                            models.PointVectors(
                                id=collection_id,
                                vector={
                                    "dense-text": text_dense_vector,
                                    "sparse": models.SparseVector(
                                        indices=sparse_vector["indices"],
                                        values=sparse_vector["values"],
                                    ),
                                },
                            )
                        ]
                    )
                )
            )
        operations.append(
            models.SetPayloadOperation(
                set_payload=models.SetPayload(payload=payload, points=[collection_id])
            )
        )
        await self.client.batch_update_points(
            collection_name=settings.GALLERY_COLLECTIONS_NAME,
            update_operations=operations,
        )

    async def get_gallery_collection(self, collection_id: str):
        """The collection point, with its `dense-text` anchor vector."""
        points = await self.client.retrieve(
            collection_name=settings.GALLERY_COLLECTIONS_NAME,
            ids=[collection_id],
            with_payload=True,
            with_vectors=["dense-text"],
        )
        return points[0] if points else None

    async def list_gallery_collections(self):
        """Every collection (there are only ever a handful), with its `dense-text` anchor vector."""
        points, offset = [], None
        while True:
            page, offset = await self.client.scroll(
                collection_name=settings.GALLERY_COLLECTIONS_NAME,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=["dense-text"],
            )
            points.extend(page)
            if offset is None:
                return points

    async def score_gallery_collections(self, text_vector: List[float]):
        """Every collection with its cosine score against `text_vector`, best first."""
        res = await self.client.query_points(
            collection_name=settings.GALLERY_COLLECTIONS_NAME,
            query=text_vector,
            using="dense-text",
            limit=200,
            with_payload=True,
        )
        return res.points

    async def delete_gallery_collection(self, collection_id: str):
        await self.client.delete(
            collection_name=settings.GALLERY_COLLECTIONS_NAME,
            points_selector=models.PointIdsList(points=[collection_id]),
        )

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

    async def get_points(self, point_ids: List[str]):
        """
        Fetch several points by ID (payload only). Unknown IDs are silently skipped.
        """
        if not point_ids:
            return []
        return await self.client.retrieve(
            collection_name=settings.COLLECTION_NAME,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )

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
