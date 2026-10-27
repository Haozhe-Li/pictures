import asyncio
import uuid
import os
import re
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
    Query,
    status,
)
from botocore.exceptions import ClientError
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from core.random_query import RandomQueryManager
import secrets
import json
from enum import Enum
import redis.asyncio as redis

# Imports from Core
from core.config import settings
from core.embedding import EmbeddingClient, get_sparse_embedding
from core.storage import (
    STAGING_PREFIX,
    delete_object,
    download_object,
    get_object_size,
    key_from_public_url,
    presign_staging_put,
    upload_file_to_r2,
)
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
random_query_manager = RandomQueryManager(redis_client, qdrant_wrapper, embedding_client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Qdrant Collection
    await qdrant_wrapper.init_collection()
    # Load the saved autocomplete model (if any) and refresh it from Qdrant in the background
    autocomplete_manager.start(qdrant_wrapper)
    decay_task = asyncio.create_task(_decay_loop())
    # Fill the random-query queue in the background so the first click is already served from it
    await random_query_manager.warm_up()
    yield
    decay_task.cancel()
    try:
        await decay_task
    except asyncio.CancelledError:
        pass
    await autocomplete_manager.close()
    await random_query_manager.close()
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


async def _embed_metadata(
    title: str,
    description: Optional[str],
    taken_time: Optional[str],
    camera: Optional[str],
):
    """
    Builds the metadata text and embeds it for the two text indexes (`sparse` BM25 and
    `dense-text`). Shared by ingest and metadata edits so both always index the same text.
    Returns (metadata_text, sparse_vector, text_dense_vector).
    """
    metadata_text = f"{title} {description or ''} {taken_time or ''} {camera or ''}"

    # FastEmbed is CPU bound, might want to offload if heavy, but for short text it's fast.
    sparse_vec = await run_in_threadpool(get_sparse_embedding, metadata_text)

    try:
        text_dense_embedding = await run_in_threadpool(
            embedding_client.get_embedding,
            text=metadata_text,
            model=settings.TEXT_MODEL,
        )
    except Exception as e:
        print(f"Warning: Metadata Dense Embedding failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Metadata Embedding failed: {str(e)}"
        )

    return metadata_text, sparse_vec, text_dense_embedding


async def _ingest_bytes(
    file_bytes: bytes,
    title: str,
    taken_time: Optional[str],
    camera: Optional[str],
    description: Optional[str],
):
    """
    Shared ingest pipeline (see /ingest for the steps). Takes the raw image bytes
    so it works for both multipart uploads and files pulled from the R2 staging area.
    """
    try:
        # 1. Prepare filenames
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

        # 4. Sparse + dense embedding of the metadata text
        metadata_text, sparse_vec, text_dense_embedding = await _embed_metadata(
            title, description, taken_time, camera
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

        # 7. New title/description text: refresh the autocomplete index after a quiet period
        autocomplete_manager.schedule_rebuild()

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
    Ingest an image sent directly as multipart form data:
    1. Read and Convert to Base64
    2. Save Preview & Original Versions to R2
    3. Get Dense Embedding from the embedding service
    4. Generate Sparse Embedding from Metadata
    5. Save to Qdrant

    Hosts with request-body limits (e.g. Vercel) cannot use this for large files;
    use /upload/presign + /ingest/staged instead.
    """
    file_bytes = await file.read()
    return await _ingest_bytes(file_bytes, title, taken_time, camera, description)


# --- Browser -> R2 -> backend upload flow ---

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_UPLOAD_TYPES = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}
STAGING_KEY_RE = re.compile(
    rf"^{re.escape(STAGING_PREFIX)}[0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}}\.(jpg|png|webp)$"
)


class PresignFile(BaseModel):
    content_type: str
    size: int


class PresignRequest(BaseModel):
    files: List[PresignFile]


class PresignedUpload(BaseModel):
    key: str
    upload_url: str
    content_type: str


class StagedIngestRequest(BaseModel):
    key: str
    title: str
    taken_time: Optional[str] = None
    camera: Optional[str] = None
    description: Optional[str] = None


@app.post("/upload/presign", response_model=List[PresignedUpload])
async def presign_uploads(
    req: PresignRequest, username: str = Depends(verify_credentials)
):
    """
    Issue one presigned R2 PUT URL per file. The browser uploads the raw file
    straight to R2, then calls /ingest/staged with the returned key.
    """
    if not req.files or len(req.files) > 50:
        raise HTTPException(status_code=400, detail="Provide between 1 and 50 files")

    results = []
    for f in req.files:
        ext = ALLOWED_UPLOAD_TYPES.get(f.content_type)
        if ext is None:
            raise HTTPException(
                status_code=400, detail=f"Unsupported content type: {f.content_type}"
            )
        if f.size <= 0 or f.size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File size must be between 1 byte and {MAX_UPLOAD_BYTES // 1024 // 1024} MB",
            )
        key = f"{STAGING_PREFIX}{uuid.uuid4()}.{ext}"
        url = await run_in_threadpool(presign_staging_put, key, f.content_type)
        results.append(PresignedUpload(key=key, upload_url=url, content_type=f.content_type))
    return results


async def _load_staged(key: str) -> bytes:
    """Validates a staging key and downloads the object (size-checked before reading)."""
    if not STAGING_KEY_RE.match(key):
        raise HTTPException(status_code=400, detail="Invalid staging key")
    try:
        size = await run_in_threadpool(get_object_size, key)
    except ClientError:
        raise HTTPException(status_code=404, detail="Staged upload not found")
    if size <= 0 or size > MAX_UPLOAD_BYTES:
        await run_in_threadpool(delete_object, key)
        raise HTTPException(status_code=413, detail="Staged file has invalid size")
    return await run_in_threadpool(download_object, key)


@app.post("/ingest/staged")
async def ingest_staged_image(
    req: StagedIngestRequest, username: str = Depends(verify_credentials)
):
    """
    Ingest an image the browser already uploaded to the R2 staging area: download
    it, run the normal ingest pipeline, then delete the staged object.
    """
    file_bytes = await _load_staged(req.key)
    result = await _ingest_bytes(
        file_bytes, req.title, req.taken_time, req.camera, req.description
    )
    try:
        await run_in_threadpool(delete_object, req.key)
    except Exception as e:
        # The image is already published; a leftover staged object is swept by the lifecycle rule.
        print(f"Warning: failed to delete staged object {req.key}: {e}")
    return result


async def _describe_bytes(file_bytes: bytes) -> GenerateDescriptionResponse:
    """Runs the LLM captioner on raw image bytes (downscaled first)."""
    try:
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        # Compress/resize before sending to LLM
        image_base64 = await run_in_threadpool(process_image_for_embedding, file_bytes)

        result = await description_generator.generate(image_base64)

        title, description = result.title.strip(), result.description.strip()
        if not title or not description:
            raise HTTPException(
                status_code=500, detail="LLM response missing title or description"
            )

        return GenerateDescriptionResponse(title=title, description=description)

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-description", response_model=GenerateDescriptionResponse)
async def generate_description_endpoint(
    file: UploadFile = File(...), username: str = Depends(verify_credentials)
):
    """
    Generate title and description for an image using LLM.
    """
    return await _describe_bytes(await file.read())


# --- Admin: manage published images ---


class AdminImageList(BaseModel):
    items: List[SearchResult]
    next_cursor: Optional[str] = None
    total: int


class ImageUpdateRequest(BaseModel):
    """
    Partial update: omitted/null fields are left alone, an empty string clears an
    optional field. `title` can be changed but never blanked.
    """

    title: Optional[str] = None
    description: Optional[str] = None
    taken_time: Optional[str] = None
    camera: Optional[str] = None


EDITABLE_FIELDS = ("title", "description", "taken_time", "camera")


def _to_search_result(point_id: str, payload: dict) -> SearchResult:
    return SearchResult(
        id=point_id,
        preview_url=payload.get("preview_url", ""),
        original_url=payload.get("original_url", ""),
        metadata=payload,
        score=1.0,
    )


async def _require_point(image_id: str):
    try:
        point = await qdrant_wrapper.get_point(image_id)
    except Exception:
        # Qdrant rejects ids that are neither integers nor UUIDs
        raise HTTPException(status_code=404, detail="Image not found")
    if not point:
        raise HTTPException(status_code=404, detail="Image not found")
    return point


@app.get("/admin/autocomplete/status")
async def admin_autocomplete_status(username: str = Depends(verify_credentials)):
    """
    State of the autocomplete index: whether a rebuild is running or scheduled (and in how
    many seconds), and when it was last built.
    """
    return autocomplete_manager.status()


@app.post("/autocomplete/build")
async def build_autocomplete(username: str = Depends(verify_credentials)):
    """
    Rebuild the autocomplete index from Qdrant right now, dropping any scheduled rebuild.
    Returns once the new index is live. Normally unnecessary: ingest, edits and deletes
    schedule a rebuild by themselves.
    """
    try:
        await autocomplete_manager.rebuild_now()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Autocomplete rebuild failed: {e}")
    return {"status": "success", "message": "Autocomplete index rebuilt."}


@app.get("/admin/images", response_model=AdminImageList)
async def admin_list_images(
    limit: int = Query(40, ge=1, le=100),
    cursor: Optional[str] = None,
    username: str = Depends(verify_credentials),
):
    """
    Stable, paginated listing of every image (unlike /gallery, which is a random feed).
    Pass the returned `next_cursor` to fetch the following page.
    """
    try:
        points, next_cursor = await qdrant_wrapper.scroll(limit=limit, offset=cursor)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not list images: {e}")

    total = await qdrant_wrapper.count()
    return AdminImageList(
        items=[_to_search_result(str(p.id), p.payload or {}) for p in points],
        next_cursor=str(next_cursor) if next_cursor is not None else None,
        total=total,
    )


@app.patch("/admin/images/{image_id}", response_model=SearchResult)
async def admin_update_image(
    image_id: str,
    req: ImageUpdateRequest,
    username: str = Depends(verify_credentials),
):
    """
    Edit an image's metadata and re-embed it. The title, description, time and camera
    are all part of the text that gets indexed, so any change re-computes the sparse and
    `dense-text` vectors; the image vector is untouched.
    """
    point = await _require_point(image_id)
    payload = point.payload or {}

    updates = {}
    for field in EDITABLE_FIELDS:
        value = getattr(req, field)
        if value is None:
            continue
        value = value.strip()
        if field == "title":
            if not value:
                raise HTTPException(status_code=400, detail="Title cannot be empty")
            updates[field] = value
        else:
            updates[field] = value or None

    changed = {k: v for k, v in updates.items() if payload.get(k) != v}
    if not changed:
        return _to_search_result(image_id, payload)

    merged = {**payload, **changed}
    # Embed first: if the embedding service fails nothing has been written yet.
    _, sparse_vec, text_dense_embedding = await _embed_metadata(
        merged.get("title", ""),
        merged.get("description"),
        merged.get("taken_time"),
        merged.get("camera"),
    )
    await qdrant_wrapper.update_metadata(
        point_id=image_id,
        text_dense_vector=text_dense_embedding,
        sparse_vector=sparse_vec,
        payload=changed,
    )
    # Autocomplete is built from titles and descriptions only
    if "title" in changed or "description" in changed:
        autocomplete_manager.schedule_rebuild()
    return _to_search_result(image_id, merged)


@app.delete("/admin/images/{image_id}")
async def admin_delete_image(
    image_id: str, username: str = Depends(verify_credentials)
):
    """
    Delete an image everywhere: the Qdrant point, its Redis pool/counter entries and the
    preview + original files in R2. Order matters: the point goes first so the gallery
    stops serving it immediately; a failure afterwards leaves only orphaned files or
    stale Redis entries (the gallery skips ids Qdrant no longer has), never a record
    pointing at missing files. Those leftovers are reported in `warnings`.
    """
    point = await _require_point(image_id)
    payload = point.payload or {}

    warnings: List[str] = []
    keys = []
    for field in ("preview_url", "original_url"):
        key = key_from_public_url(payload.get(field))
        if key:
            keys.append(key)
        elif payload.get(field):
            warnings.append(f"Not deleting {field}: unrecognised file {payload[field]}")

    await qdrant_wrapper.delete_point(image_id)
    autocomplete_manager.schedule_rebuild()

    try:
        pipe = redis_client.pipeline(transaction=False)
        pipe.zrem("gallery:pool:active", image_id)
        pipe.zrem("gallery:pool:explore", image_id)
        pipe.hdel("gallery:counters:click", image_id)
        pipe.hdel("gallery:counters:share", image_id)
        await pipe.execute()
    except Exception as e:
        print(f"Warning: failed to clean Redis entries for {image_id}: {e}")
        warnings.append(f"Redis cleanup failed: {e}")

    for key in keys:
        try:
            await run_in_threadpool(delete_object, key)
        except Exception as e:
            print(f"Warning: failed to delete {key} from R2: {e}")
            warnings.append(f"Storage delete failed for {key}: {e}")

    return {"status": "deleted", "id": image_id, "warnings": warnings}


@app.post(
    "/admin/images/{image_id}/describe", response_model=GenerateDescriptionResponse
)
async def admin_describe_image(
    image_id: str, username: str = Depends(verify_credentials)
):
    """
    Suggest a title and description for an already published image, read from the stored
    original (the preview is too heavily compressed to caption well).
    """
    point = await _require_point(image_id)
    key = key_from_public_url((point.payload or {}).get("original_url"))
    if not key:
        raise HTTPException(status_code=404, detail="Original file not found")
    try:
        file_bytes = await run_in_threadpool(download_object, key)
    except ClientError:
        raise HTTPException(status_code=404, detail="Original file not found")
    return await _describe_bytes(file_bytes)


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


# Impressions an image gets in the explore pool before it graduates to the hot pool only.
EXPLORE_GRADUATION_IMPRESSIONS = 10


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

            # Check graduation: remove from explore pool after enough impressions
            score = await redis_client.zscore("gallery:pool:explore", cid)
            if score and score >= EXPLORE_GRADUATION_IMPRESSIONS:
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
    Pop a random search query from the Redis queue. Queries are LLM-rewritten from the
    title + description of random gallery images, with their embeddings pre-warmed.
    Falls back to a static list if the queue is empty or unavailable.
    """
    try:
        query = await random_query_manager.next_query()
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

    Runs from a background loop (not in a request) and batches writes into
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


DECAY_CHECK_INTERVAL_SECONDS = 600  # How often the background loop tries to open a window


async def _acquire_decay_window() -> bool:
    """
    SET NX EX is atomic, so exactly one caller wins each decay window, even across
    processes, replicas and concurrent /health hits. The key expiring re-opens it.
    """
    acquired = await redis_client.set(
        "gallery:decay:lock",
        time.time(),
        nx=True,
        ex=DECAY_INTERVAL_SECONDS,
    )
    return bool(acquired)


async def _decay_loop():
    """
    Background scheduler: wake up periodically and decay the active pool whenever the
    window is open. Because the lock outlives restarts, redeploying does not cause
    extra decays, and a restart after a long outage catches up immediately.
    """
    while True:
        try:
            if await _acquire_decay_window():
                await _decay_active_pool()
        except Exception as e:
            print(f"Decay loop error: {e}")
        await asyncio.sleep(DECAY_CHECK_INTERVAL_SECONDS)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
