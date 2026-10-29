import asyncio
from typing import List, Dict, Any, Optional
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from core.config import settings
from core.taken_at import to_taken_at, window_bounds


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

        # Photo times (RFC 3339, see core/taken_at.py), for filtering by date
        try:
            await self.client.create_payload_index(
                collection_name=settings.COLLECTION_NAME,
                field_name="taken_at",
                field_schema=models.PayloadSchemaType.DATETIME,
            )
        except Exception as e:
            print(f"Warning: could not ensure payload index on taken_at: {e}")

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

        # The watermark check looks photos up by the id their watermark carries
        try:
            await self.client.create_payload_index(
                collection_name=settings.COLLECTION_NAME,
                field_name="watermark_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            print(f"Warning: could not ensure payload index on watermark_id: {e}")

        await self._init_gallery_collections()

    async def _init_gallery_collections(self):
        """
        Create the store for collections if it is missing, with the same three vectors as the
        images so a collection's text can be run as a search. The first time, the collections of
        the v1 store (MiniLM + BM25 only) are copied over; their CLIP vector is added the first
        time it is needed. The v1 store is left in place.
        """
        names = {c.name for c in (await self.client.get_collections()).collections}
        if settings.GALLERY_COLLECTIONS_NAME in names:
            return
        await self.client.create_collection(
            collection_name=settings.GALLERY_COLLECTIONS_NAME,
            vectors_config={
                "dense-image": VectorParams(size=settings.CLIP_DIM, distance=Distance.COSINE),
                "dense-text": VectorParams(size=settings.TEXT_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            },
        )
        print(f"Collection {settings.GALLERY_COLLECTIONS_NAME} created.")
        if settings.LEGACY_GALLERY_COLLECTIONS_NAME not in names:
            return
        try:
            copied, offset = 0, None
            while True:
                page, offset = await self.client.scroll(
                    collection_name=settings.LEGACY_GALLERY_COLLECTIONS_NAME,
                    limit=200,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
                if page:
                    await self.client.upsert(
                        collection_name=settings.GALLERY_COLLECTIONS_NAME,
                        points=[
                            models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                            for p in page
                        ],
                    )
                    copied += len(page)
                if offset is None:
                    break
        except Exception:
            # Start over on the next startup rather than run with half the collections
            await self.client.delete_collection(settings.GALLERY_COLLECTIONS_NAME)
            raise
        print(
            f"Copied {copied} collection(s) from {settings.LEGACY_GALLERY_COLLECTIONS_NAME} "
            f"to {settings.GALLERY_COLLECTIONS_NAME}."
        )

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
        query_filter: Optional[models.Filter] = None,
        offset: int = 0,
        prefetch_limit: Optional[int] = None,
        with_vectors: Any = False,
    ):
        """
        `image_vector` is a CLIP vector, queried against `dense-image`.
        `text_vector` is a MiniLM vector, queried against `dense-text`.
        Each mode only needs the vectors it uses. `query_filter` narrows every list before they
        are fused; `prefetch_limit` is how many each list contributes (default twice the page).
        """
        per_list = prefetch_limit or (offset + limit) * 2
        sparse_prefetch = models.Prefetch(
            query=models.SparseVector(
                indices=sparse_vector["indices"],
                values=sparse_vector["values"],
            ),
            using="sparse",
            filter=query_filter,
            limit=per_list,
        )

        def dense_prefetch(vector, using):
            return models.Prefetch(
                query=vector,
                using=using,
                filter=query_filter,
                limit=per_list,
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
            offset=offset,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            query_filter=query_filter,
            with_payload=True,
            with_vectors=with_vectors,
        )
        return search_result.points

    async def nearest_images(self, image_vector: List[float], limit: int, min_score: float):
        """The images whose CLIP vector is closest to `image_vector`, scoring at least `min_score`."""
        res = await self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=image_vector,
            using="dense-image",
            limit=limit,
            score_threshold=min_score,
            with_payload=True,
        )
        return res.points

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

    async def scroll_by_taken_at(
        self,
        limit: int,
        newest_first: bool,
        start_from: Optional[str] = None,
        skip: Optional[List[str]] = None,
    ):
        """
        Images that have a `taken_at`, in time order from `start_from` (inclusive) on, leaving out
        `skip`. Qdrant does not page an ordered scroll, so callers carry the position themselves.
        """
        must_not = [self._no_taken_at()]
        if skip:
            must_not.append(models.HasIdCondition(has_id=skip))
        points, _ = await self.client.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=models.Filter(must_not=must_not),
            order_by=models.OrderBy(
                key="taken_at",
                direction=models.Direction.DESC if newest_first else models.Direction.ASC,
                start_from=start_from,
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return points

    @staticmethod
    def _no_taken_at() -> models.IsEmptyCondition:
        return models.IsEmptyCondition(is_empty=models.PayloadField(key="taken_at"))

    @classmethod
    def undated(cls) -> models.Filter:
        """Images with no `taken_at` (no time, or one that could not be read)."""
        return models.Filter(must=[cls._no_taken_at()])

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

    async def backfill_taken_at(self) -> int:
        """
        Give every image the `taken_at` its `taken_time` reads as (images from before it existed,
        or edited by hand in Qdrant). Only points that differ are written; returns how many.
        """
        updates, offset = {}, None
        while True:
            page, offset = await self.scroll(limit=500, offset=offset)
            for p in page:
                payload = p.payload or {}
                wanted = to_taken_at(payload.get("taken_time"))
                if payload.get("taken_at") != wanted and (wanted or "taken_at" in payload):
                    updates[str(p.id)] = {"taken_at": wanted}
            if offset is None:
                break
        if updates:
            await self.set_payloads(updates)
        return len(updates)

    @staticmethod
    def taken_between(start: Optional[str], end: Optional[str]) -> Optional[models.Filter]:
        """Images taken within a window of whole days (YYYY-MM-DD, inclusive, either open)."""
        if not start and not end:
            return None
        lower, upper = window_bounds(start, end)
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="taken_at", range=models.DatetimeRange(gte=lower, lt=upper)
                )
            ]
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
        maps point id -> the payload keys to set.
        """
        await self.set_payloads(memberships)

    async def set_payloads(self, updates: Dict[str, Dict[str, Any]]):
        """Merge payload keys into many image points (point id -> keys), in batches, one
        operation per point."""
        items = list(updates.items())
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

    @staticmethod
    def _collection_vectors(vectors: Dict[str, Any]) -> Dict[str, Any]:
        """`vectors` as Qdrant takes them: the sparse one as a SparseVector."""
        return {
            name: (
                models.SparseVector(indices=v["indices"], values=v["values"])
                if name == "sparse"
                else v
            )
            for name, v in vectors.items()
        }

    async def upsert_gallery_collection(
        self,
        collection_id: str,
        vectors: Dict[str, Any],
        payload: Dict[str, Any],
    ):
        """`vectors` holds `dense-image`, `dense-text` and `sparse` (as {'indices', 'values'})."""
        await self.client.upsert(
            collection_name=settings.GALLERY_COLLECTIONS_NAME,
            points=[
                models.PointStruct(
                    id=collection_id,
                    vector=self._collection_vectors(vectors),
                    payload=payload,
                )
            ],
        )

    async def update_gallery_collection(
        self,
        collection_id: str,
        payload: Dict[str, Any],
        vectors: Optional[Dict[str, Any]] = None,
    ):
        """Merge `payload`; replace (some of) the collection's vectors too when given."""
        operations: List[Any] = []
        if vectors:
            operations.append(
                models.UpdateVectorsOperation(
                    update_vectors=models.UpdateVectors(
                        points=[
                            models.PointVectors(
                                id=collection_id, vector=self._collection_vectors(vectors)
                            )
                        ]
                    )
                )
            )
        if payload:
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
        """The collection point, with its vectors."""
        points = await self.client.retrieve(
            collection_name=settings.GALLERY_COLLECTIONS_NAME,
            ids=[collection_id],
            with_payload=True,
            with_vectors=True,
        )
        return points[0] if points else None

    async def find_gallery_collection_by_slug(self, slug: str):
        """The collection at this address, else the one that had it before a rename; with its vectors."""
        for key in ("slug", "former_slugs"):
            points, _ = await self.client.scroll(
                collection_name=settings.GALLERY_COLLECTIONS_NAME,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key=key, match=models.MatchValue(value=slug))]
                ),
                limit=1,
                with_payload=True,
                with_vectors=True,
            )
            if points:
                return points[0]
        return None

    async def list_gallery_collections(self):
        """Every collection (there are only ever a handful), with its vectors."""
        points, offset = [], None
        while True:
            page, offset = await self.client.scroll(
                collection_name=settings.GALLERY_COLLECTIONS_NAME,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=True,
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

    async def find_point_by_watermark(self, watermark_id: str):
        """The image whose watermark carries `watermark_id` (see core/watermark), payload only."""
        points, _ = await self.client.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="watermark_id", match=models.MatchValue(value=watermark_id))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return points[0] if points else None

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
