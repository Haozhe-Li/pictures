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
from pydantic import BaseModel, Field
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
from core import gallery_collections as gc

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


def _public_metadata(payload: dict) -> dict:
    """An image's payload minus the bookkeeping the site has no use for."""
    return {k: v for k, v in (payload or {}).items() if k != "collections_opt_out"}


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
            metadata=_public_metadata(metadata),
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
    collection_ids: Optional[List[str]] = None,
):
    """
    Shared ingest pipeline (see /ingest for the steps). Takes the raw image bytes
    so it works for both multipart uploads and files pulled from the R2 staging area.

    `collection_ids` is the admin's explicit choice of collections ([] = none, and it sticks);
    None lets the server match the new image against the collections' anchors.
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

        # 5. Collections: the admin's explicit choice, or whatever matches the anchors
        if collection_ids is None:
            member_ids = await _auto_collections(text_dense_embedding)
            membership = gc.membership_payload(member_ids, by_admin=False)
        else:
            membership = gc.membership_payload(collection_ids, by_admin=True)

        # 6. Qdrant Upsert
        point_id = str(file_uuid)
        payload = {
            "title": title,
            "taken_time": taken_time,
            "camera": camera,
            "description": description,
            "preview_url": r2_url_preview,
            "original_url": r2_url_original,
            "type": "image",
            **membership,
        }

        await qdrant_wrapper.upsert_point(
            point_id=point_id,
            image_dense_vector=dense_embedding,
            text_dense_vector=text_dense_embedding,
            sparse_vector=sparse_vec,
            payload=payload,
        )

        # 7. Add to Redis Pools for Waterfall Recommendation
        await redis_client.zadd("gallery:pool:active", {point_id: 0})
        await redis_client.zadd("gallery:pool:explore", {point_id: 0})

        # 8. New title/description text: refresh the autocomplete index after a quiet period
        autocomplete_manager.schedule_rebuild()
        if membership["collection_ids"]:
            _invalidate_collections_cache()

        return {
            "status": "success",
            "id": point_id,
            "preview_url": r2_url_preview,
            "original_url": r2_url_original,
            "metadata_used_for_sparse": metadata_text,
            "collection_ids": membership["collection_ids"],
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
    # Explicit collections ([] = none); omit to let the server match the anchors
    collection_ids: Optional[List[str]] = None


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
    collection_ids = None
    if req.collection_ids is not None:
        # Reject unknown collections before downloading and publishing anything
        collection_ids = gc.dedupe(req.collection_ids)
        await _require_collections(collection_ids)
    file_bytes = await _load_staged(req.key)
    result = await _ingest_bytes(
        file_bytes,
        req.title,
        req.taken_time,
        req.camera,
        req.description,
        collection_ids,
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
    # Explicit collections ([] = none, and it sticks); omit to leave them as they are
    # (an image with none is matched against the anchors when its text changes)
    collection_ids: Optional[List[str]] = None


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

    # Explicit membership from the admin; validated before anything is written
    membership: dict = {}
    if req.collection_ids is not None:
        ids = gc.dedupe(req.collection_ids)
        await _require_collections(ids)
        wanted = gc.membership_payload(ids, by_admin=True)
        if wanted != {
            "collection_ids": gc.current_membership(payload),
            "collections_opt_out": bool(payload.get("collections_opt_out")),
        }:
            membership = wanted

    if not changed and not membership:
        return _to_search_result(image_id, payload)

    merged = {**payload, **changed}
    if changed:
        # Embed first: if the embedding service fails nothing has been written yet.
        _, sparse_vec, text_dense_embedding = await _embed_metadata(
            merged.get("title", ""),
            merged.get("description"),
            merged.get("taken_time"),
            merged.get("camera"),
        )
        # New text for an image that is in no collection (and has not opted out): match it
        if req.collection_ids is None and gc.is_auto_eligible(payload):
            matched = await _auto_collections(text_dense_embedding)
            if matched:
                membership = gc.membership_payload(matched, by_admin=False)
        merged.update(membership)
        await qdrant_wrapper.update_metadata(
            point_id=image_id,
            text_dense_vector=text_dense_embedding,
            sparse_vector=sparse_vec,
            payload={**changed, **membership},
        )
    else:
        merged.update(membership)
        await qdrant_wrapper.set_membership({image_id: membership})

    # Autocomplete is built from titles and descriptions only
    if "title" in changed or "description" in changed:
        autocomplete_manager.schedule_rebuild()
    if membership:
        _invalidate_collections_cache()
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
    if gc.current_membership(payload):
        _invalidate_collections_cache()

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


# --- Admin: collections ---
#
# A collection is a user-facing group of images. Its title+description is embedded like an
# image's text (`dense-text` MiniLM + `sparse` BM25) and kept as the collection's anchor in a
# separate Qdrant collection; the images carry their membership in `collection_ids`.
# Only the dense score decides automatic assignment (BM25 scores have no fixed scale).


class CollectionInfo(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    threshold: float
    image_count: int
    # Admin-chosen cover; when unset the cover is the image closest to the collection's anchor
    cover_image_id: Optional[str] = None
    cover_preview_url: Optional[str] = None


class CollectionMatch(BaseModel):
    id: str
    title: str
    score: float
    threshold: float
    matched: bool


class CollectionCreateRequest(BaseModel):
    title: str
    description: Optional[str] = None
    threshold: Optional[float] = Field(None, ge=0, le=1)
    # Images to put in right away (what the admin ticked on the review screen)
    image_ids: List[str] = []


class CollectionUpdateRequest(BaseModel):
    """Partial update; an empty `description` or `cover_image_id` clears it."""

    title: Optional[str] = None
    description: Optional[str] = None
    threshold: Optional[float] = Field(None, ge=0, le=1)
    cover_image_id: Optional[str] = None


class CollectionPreviewRequest(BaseModel):
    title: str
    description: Optional[str] = None
    limit: int = Field(200, ge=1, le=500)
    min_score: float = Field(0.2, ge=-1, le=1)


class CollectionPreviewResponse(BaseModel):
    # Existing images ranked by similarity to the draft collection. `metadata` carries each
    # image's current `collection_ids`, so the UI can tell which ones are already placed.
    items: List[SearchResult]
    default_threshold: float


class SuggestRequest(BaseModel):
    title: str
    description: Optional[str] = None
    taken_time: Optional[str] = None
    camera: Optional[str] = None


class MembershipRequest(BaseModel):
    image_ids: List[str]
    add: List[str] = []
    remove: List[str] = []


class MembershipItem(BaseModel):
    id: str
    collection_ids: List[str]
    collections_opt_out: bool


class MembershipResponse(BaseModel):
    updated: List[MembershipItem]
    missing: List[str]


MAX_MEMBERSHIP_IMAGES = 1000


def _collection_threshold(payload: dict) -> float:
    value = payload.get("threshold")
    return settings.DEFAULT_COLLECTION_THRESHOLD if value is None else float(value)


async def _require_collection(collection_id: str):
    try:
        point = await qdrant_wrapper.get_gallery_collection(collection_id)
    except Exception:
        # Qdrant rejects ids that are neither integers nor UUIDs
        raise HTTPException(status_code=404, detail="Collection not found")
    if not point:
        raise HTTPException(status_code=404, detail="Collection not found")
    return point


async def _require_collections(ids: List[str]):
    """400 unless every id is an existing collection."""
    if not ids:
        return
    known = {str(p.id) for p in await qdrant_wrapper.list_gallery_collections()}
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise HTTPException(
            status_code=400, detail=f"Unknown collection(s): {', '.join(unknown)}"
        )


async def _score_collections(text_vector: List[float]) -> List[CollectionMatch]:
    """Every collection scored against an image's text vector, best first."""
    points = await qdrant_wrapper.score_gallery_collections(text_vector)
    matches = []
    for p in points:
        payload = p.payload or {}
        threshold = _collection_threshold(payload)
        matches.append(
            CollectionMatch(
                id=str(p.id),
                title=payload.get("title", ""),
                score=p.score,
                threshold=threshold,
                matched=p.score >= threshold,
            )
        )
    return matches


async def _auto_collections(text_vector: List[float]) -> List[str]:
    """
    Ids of the collections an image matches. Matching is a convenience: if it fails the
    image simply lands in none rather than the upload or edit failing.
    """
    try:
        return [m.id for m in await _score_collections(text_vector) if m.matched]
    except Exception as e:
        print(f"Warning: collection matching failed: {e}")
        return []


def _clean_image_ids(ids: List[str]) -> List[str]:
    ids = gc.dedupe(ids)
    if len(ids) > MAX_MEMBERSHIP_IMAGES:
        raise HTTPException(
            status_code=400, detail=f"At most {MAX_MEMBERSHIP_IMAGES} images at a time"
        )
    for i in ids:
        try:
            uuid.UUID(i)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid image id: {i}")
    return ids


async def _fetch_images(ids: List[str]) -> dict:
    """id -> payload for the images that exist (unknown ids are left out)."""
    found = {}
    for i in range(0, len(ids), 200):
        for p in await qdrant_wrapper.get_points(ids[i : i + 200]):
            found[str(p.id)] = p.payload or {}
    return found


async def _change_membership(
    image_ids: List[str], add: List[str], remove: List[str], by_admin: bool = True
) -> MembershipResponse:
    images = await _fetch_images(image_ids)
    memberships = {}
    for image_id, payload in images.items():
        new_ids = gc.apply_membership(gc.current_membership(payload), add, remove)
        memberships[image_id] = gc.membership_payload(new_ids, by_admin)
    if memberships:
        await qdrant_wrapper.set_membership(memberships)
        _invalidate_collections_cache()
    return MembershipResponse(
        updated=[
            MembershipItem(
                id=i,
                collection_ids=m["collection_ids"],
                collections_opt_out=m["collections_opt_out"],
            )
            for i, m in memberships.items()
        ],
        missing=[i for i in image_ids if i not in images],
    )


def _anchor_vector(point) -> Optional[List[float]]:
    vector = getattr(point, "vector", None)
    return vector.get("dense-text") if isinstance(vector, dict) else vector


async def _collection_cover(point, image_count: int) -> Optional[dict]:
    """
    Payload of the collection's cover image: the admin's pick while it is still a member,
    otherwise the member closest to the collection's anchor (the most representative one).
    """
    collection_id = str(point.id)
    cover_id = (point.payload or {}).get("cover_image_id")
    if cover_id:
        try:
            candidate = await qdrant_wrapper.get_point(cover_id)
            if candidate and collection_id in gc.current_membership(candidate.payload or {}):
                return candidate.payload or {}
        except Exception:
            pass  # a deleted or invalid cover falls back to the anchor's nearest image
    anchor = _anchor_vector(point)
    if image_count and anchor:
        nearest = await qdrant_wrapper.query_collection_images(collection_id, anchor, limit=1)
        if nearest:
            return nearest[0].payload or {}
    return None


async def _collection_info(point) -> CollectionInfo:
    collection_id = str(point.id)
    payload = point.payload or {}
    image_count = await qdrant_wrapper.count(qdrant_wrapper.collection_filter(collection_id))
    cover = await _collection_cover(point, image_count)

    return CollectionInfo(
        id=collection_id,
        title=payload.get("title", ""),
        description=payload.get("description"),
        threshold=_collection_threshold(payload),
        image_count=image_count,
        cover_image_id=payload.get("cover_image_id"),
        cover_preview_url=cover.get("preview_url") if cover else None,
    )


@app.get("/admin/collections", response_model=List[CollectionInfo])
async def admin_list_collections(username: str = Depends(verify_credentials)):
    """All collections, oldest first, with their image counts and covers."""
    points = await qdrant_wrapper.list_gallery_collections()
    points.sort(key=lambda p: (p.payload or {}).get("created_at", 0))
    return list(await asyncio.gather(*(_collection_info(p) for p in points)))


@app.post("/admin/collections", response_model=CollectionInfo)
async def admin_create_collection(
    req: CollectionCreateRequest, username: str = Depends(verify_credentials)
):
    """
    Create a collection and put `image_ids` in it. The images come from the review step
    (`/admin/collections/preview`), where the admin picked which matches to keep.
    """
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    description = (req.description or "").strip() or None
    image_ids = _clean_image_ids(req.image_ids)
    if image_ids:
        existing = await _fetch_images(image_ids)
        missing = [i for i in image_ids if i not in existing]
        if missing:
            raise HTTPException(
                status_code=400, detail=f"Unknown image(s): {', '.join(missing)}"
            )

    _, sparse_vec, text_dense_embedding = await _embed_metadata(
        title, description, None, None
    )
    collection_id = str(uuid.uuid4())
    await qdrant_wrapper.upsert_gallery_collection(
        collection_id=collection_id,
        text_dense_vector=text_dense_embedding,
        sparse_vector=sparse_vec,
        payload={
            "title": title,
            "description": description,
            "threshold": (
                settings.DEFAULT_COLLECTION_THRESHOLD
                if req.threshold is None
                else req.threshold
            ),
            "cover_image_id": None,
            "created_at": time.time(),
        },
    )
    if image_ids:
        try:
            await _change_membership(image_ids, [collection_id], [])
        except Exception:
            # Do not leave a half-filled collection behind; the admin can simply retry
            try:
                await qdrant_wrapper.delete_gallery_collection(collection_id)
            except Exception as e:
                print(f"Warning: could not roll back collection {collection_id}: {e}")
            raise
    _invalidate_collections_cache()
    return await _collection_info(await _require_collection(collection_id))


@app.post("/admin/collections/preview", response_model=CollectionPreviewResponse)
async def admin_preview_collection(
    req: CollectionPreviewRequest, username: str = Depends(verify_credentials)
):
    """
    Review step for a new (or extended) collection: embeds the draft title+description and
    returns the existing images closest to it, best first, with their cosine scores.
    Nothing is written. The UI filters by threshold locally and then sends the kept ids to
    `POST /admin/collections`.
    """
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    _, _, vector = await _embed_metadata(title, (req.description or "").strip() or None, None, None)
    points = await qdrant_wrapper.search_images_by_text_vector(
        vector, limit=req.limit, score_threshold=req.min_score
    )
    return CollectionPreviewResponse(
        items=[
            SearchResult(
                id=str(p.id),
                preview_url=(p.payload or {}).get("preview_url", ""),
                original_url=(p.payload or {}).get("original_url", ""),
                metadata=p.payload or {},
                score=p.score,
            )
            for p in points
        ],
        default_threshold=settings.DEFAULT_COLLECTION_THRESHOLD,
    )


@app.post("/admin/collections/suggest", response_model=List[CollectionMatch])
async def admin_suggest_collections(
    req: SuggestRequest, username: str = Depends(verify_credentials)
):
    """
    Which collections an image with this metadata would join, for the upload and edit forms.
    Every collection comes back with its score and threshold (best first); `matched` marks the
    ones to pre-select, the rest are near misses the admin may still want.
    """
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    # Same text as is stored for the image, so the score equals what ingest would compute
    _, _, vector = await _embed_metadata(
        title,
        (req.description or "").strip() or None,
        (req.taken_time or "").strip() or None,
        (req.camera or "").strip() or None,
    )
    return await _score_collections(vector)


@app.get("/admin/collections/{collection_id}", response_model=CollectionInfo)
async def admin_get_collection(
    collection_id: str, username: str = Depends(verify_credentials)
):
    return await _collection_info(await _require_collection(collection_id))


@app.patch("/admin/collections/{collection_id}", response_model=CollectionInfo)
async def admin_update_collection(
    collection_id: str,
    req: CollectionUpdateRequest,
    username: str = Depends(verify_credentials),
):
    """
    Edit a collection. Changing its title or description re-embeds the anchor; the members
    are left as they are (use the review step again to find more images).
    """
    point = await _require_collection(collection_id)
    payload = point.payload or {}

    updates = {}
    if req.title is not None:
        title = req.title.strip()
        if not title:
            raise HTTPException(status_code=400, detail="Title cannot be empty")
        updates["title"] = title
    if req.description is not None:
        updates["description"] = req.description.strip() or None
    if req.threshold is not None:
        updates["threshold"] = req.threshold
    if req.cover_image_id is not None:
        cover_id = req.cover_image_id.strip()
        if cover_id:
            cover_id = _clean_image_ids([cover_id])[0]
            cover = (await _fetch_images([cover_id])).get(cover_id)
            if cover is None or collection_id not in gc.current_membership(cover):
                raise HTTPException(
                    status_code=400, detail="Cover must be an image in this collection"
                )
        updates["cover_image_id"] = cover_id or None

    changed = {k: v for k, v in updates.items() if payload.get(k) != v}
    if changed:
        vectors = {}
        if "title" in changed or "description" in changed:
            merged = {**payload, **changed}
            _, sparse_vec, text_dense_embedding = await _embed_metadata(
                merged.get("title", ""), merged.get("description"), None, None
            )
            vectors = {"text_dense_vector": text_dense_embedding, "sparse_vector": sparse_vec}
        await qdrant_wrapper.update_gallery_collection(collection_id, changed, **vectors)
        _invalidate_collections_cache()
    return await _collection_info(await _require_collection(collection_id))


@app.delete("/admin/collections/{collection_id}")
async def admin_delete_collection(
    collection_id: str, username: str = Depends(verify_credentials)
):
    """
    Delete a collection. Its images are not touched, they just leave the collection (and may
    be matched into others again). The members are cleaned first, so a failure midway leaves
    a collection that can simply be deleted again, never images pointing at a missing one.
    """
    await _require_collection(collection_id)
    members = qdrant_wrapper.collection_filter(collection_id)
    member_ids, offset = [], None
    while True:
        page, offset = await qdrant_wrapper.scroll(
            limit=200, offset=offset, scroll_filter=members
        )
        member_ids.extend(str(p.id) for p in page)
        if offset is None:
            break
    for i in range(0, len(member_ids), 200):
        await _change_membership(
            member_ids[i : i + 200], [], [collection_id], by_admin=False
        )
    await qdrant_wrapper.delete_gallery_collection(collection_id)
    _invalidate_collections_cache()
    return {"status": "deleted", "id": collection_id, "removed_images": len(member_ids)}


@app.get("/admin/collections/{collection_id}/images", response_model=AdminImageList)
async def admin_collection_images(
    collection_id: str,
    limit: int = Query(40, ge=1, le=100),
    cursor: Optional[str] = None,
    username: str = Depends(verify_credentials),
):
    """Paginated images of one collection, shaped like `GET /admin/images`."""
    await _require_collection(collection_id)
    members = qdrant_wrapper.collection_filter(collection_id)
    try:
        points, next_cursor = await qdrant_wrapper.scroll(
            limit=limit, offset=cursor, scroll_filter=members
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not list images: {e}")
    return AdminImageList(
        items=[_to_search_result(str(p.id), p.payload or {}) for p in points],
        next_cursor=str(next_cursor) if next_cursor is not None else None,
        total=await qdrant_wrapper.count(members),
    )


@app.post("/admin/images/membership", response_model=MembershipResponse)
async def admin_change_membership(
    req: MembershipRequest, username: str = Depends(verify_credentials)
):
    """
    Add images to and/or remove them from collections in one go (multi-select in the
    admin portal). Taking an image out of its last collection is remembered, so automatic
    assignment does not put it back. Images that no longer exist come back in `missing`.
    """
    image_ids = _clean_image_ids(req.image_ids)
    add, remove = gc.dedupe(req.add), gc.dedupe(req.remove)
    if not image_ids or not (add or remove):
        raise HTTPException(
            status_code=400, detail="Provide image_ids and something to add or remove"
        )
    if set(add) & set(remove):
        raise HTTPException(status_code=400, detail="Cannot add and remove the same collection")
    await _require_collections(add)
    return await _change_membership(image_ids, add, remove)


# --- Collections: public ---


class PublicCollection(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    image_count: int
    cover_preview_url: Optional[str] = None


class CollectionImages(BaseModel):
    items: List[SearchResult]
    next_cursor: Optional[str] = None
    total: int


# The menu of every page lists collections, so the (comparatively expensive) listing is kept
# in memory. Any change made through this process drops it at once; the TTL only bounds how
# stale it can get when another instance made the change.
PUBLIC_COLLECTIONS_TTL_SECONDS = 300
_public_collections_cache: dict = {"at": 0.0, "items": None, "generation": 0}


def _invalidate_collections_cache():
    _public_collections_cache["items"] = None
    _public_collections_cache["generation"] += 1


def _to_public_collection(info: CollectionInfo) -> PublicCollection:
    return PublicCollection(
        id=info.id,
        title=info.title,
        description=info.description,
        image_count=info.image_count,
        cover_preview_url=info.cover_preview_url,
    )


@app.get("/collections", response_model=List[PublicCollection])
async def list_collections():
    """
    Every collection that has images, largest first (ties by title). The cover is the image
    closest to the collection's anchor unless the admin picked one.
    """
    cache = _public_collections_cache
    if cache["items"] is not None and time.time() - cache["at"] < PUBLIC_COLLECTIONS_TTL_SECONDS:
        return cache["items"]

    generation = cache["generation"]
    points = await qdrant_wrapper.list_gallery_collections()
    infos = await asyncio.gather(*(_collection_info(p) for p in points))
    items = [_to_public_collection(i) for i in infos if i.image_count > 0]
    items.sort(key=lambda c: (-c.image_count, c.title.lower()))
    if generation == cache["generation"]:  # nothing changed while this was being built
        cache.update(items=items, at=time.time())
    return items


@app.get("/collections/{collection_id}", response_model=PublicCollection)
async def get_collection(collection_id: str):
    return _to_public_collection(await _collection_info(await _require_collection(collection_id)))


@app.get("/collections/{collection_id}/images", response_model=CollectionImages)
async def get_collection_images(
    collection_id: str,
    limit: int = Query(40, ge=1, le=100),
    cursor: Optional[str] = None,
):
    """
    The images of a collection, most representative (closest to its anchor) first. Pass the
    returned `next_cursor` to get the following page.
    """
    point = await _require_collection(collection_id)
    try:
        offset = int(cursor) if cursor else 0
        if offset < 0:
            raise ValueError
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid cursor")

    total = await qdrant_wrapper.count(qdrant_wrapper.collection_filter(collection_id))
    anchor = _anchor_vector(point)
    points = (
        await qdrant_wrapper.query_collection_images(collection_id, anchor, limit, offset)
        if anchor and offset < total
        else []
    )
    return CollectionImages(
        items=[
            SearchResult(
                id=str(p.id),
                preview_url=(p.payload or {}).get("preview_url", ""),
                original_url=(p.payload or {}).get("original_url", ""),
                metadata=_public_metadata(p.payload or {}),
                score=p.score,
            )
            for p in points
        ],
        next_cursor=str(offset + limit) if offset + limit < total else None,
        total=total,
    )


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
                    metadata=_public_metadata(hit.payload),
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
                    metadata=_public_metadata(hit.payload),
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
                        metadata=_public_metadata(point.payload),
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
