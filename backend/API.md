# Gallery RAG Backend API Specification

Base URL: http://localhost:8000 (or deployed domain)

## 1. Get Gallery (Browse)

Get all uploaded images with pagination support.

- **URL**: `GET /gallery`
- **Query Parameters**:
  - `limit` (int, optional): Number of items per page. Default: 20.
  - `cursor` (string, optional): Token for fetching the next page.
- **Response Example**:
  ```json
  {
    "items": [
      {
        "preview_url": "https://cdn.haozheli.com/uuid_preview.webp",
        "original_url": "https://cdn.haozheli.com/uuid_original.webp",
        "score": 1.0,
        "metadata": {
          "title": "Summer Beach",
          "description": "Fun times",
          "taken_time": "2023-08-01",
          "camera": "Sony A7M4",
          "preview_url": "https://cdn.haozheli.com/uuid_preview.webp",
          "original_url": "https://cdn.haozheli.com/uuid_original.webp",
          "type": "image"
        }
      }
    ],
    "next_cursor": "offset_token_for_next_page"
  }
  ```

## 2. Search Images (Semantic Search)

Search for images using natural language text (Hybrid Search: Semantic + Keywords).

- **URL**: `POST /search`
- **Request Body (JSON)**:
  ```json
  {
    "query": "A dog running on grass",
    "limit": 4
  }
  ```
- **Response Example**:
  ```json
  [
    {
      "preview_url": "https://cdn.haozheli.com/dog_preview.webp",
      "original_url": "https://cdn.haozheli.com/dog_original.webp",
      "score": 0.89,
      "metadata": {
        "title": "My Dog",
        "description": "Playing in the park",
        "taken_time": "2024-01-15",
        "camera": "iPhone 15",
        "preview_url": "https://cdn.haozheli.com/dog_preview.webp",
        "original_url": "https://cdn.haozheli.com/dog_original.webp",
        "type": "image"
      }
    }
  ]
  ```

## 3. Similar Images (By Existing Image URL)

Find similar images based on an existing image URL.

- **URL**: `POST /similar-to`
- **Request Body (JSON)**:
  ```json
  {
    "image_url": "https://cdn.haozheli.com/dog_original.webp",
    "limit": 4
  }
  ```
- **Response Example**:
  ```json
  [
    {
      "preview_url": "https://cdn.haozheli.com/other_preview.webp",
      "original_url": "https://cdn.haozheli.com/other_original.webp",
      "score": 0.82,
      "metadata": {
        "title": "Another Dog",
        "description": "Running on the beach",
        "taken_time": "2024-02-10",
        "camera": "Sony A7M4",
        "preview_url": "https://cdn.haozheli.com/other_preview.webp",
        "original_url": "https://cdn.haozheli.com/other_original.webp",
        "type": "image"
      }
    }
  ]
  ```

## 4. Ingest Image (Upload)

Upload an image, generate embeddings (Dense + Sparse), and store metadata.
**Requires Authentication (Basic Auth).**

- **URL**: `POST /ingest`
- **Authentication**: HTTP Basic Auth
  - Username: value of environment variable `ADMIN_USERNAME`
  - Password: value of environment variable `ADMIN_PASSWORD`
- **Content-Type**: `multipart/form-data`
- **Form Data**:
  - `file`: (File, Required) The image file (jpg/png).
  - `title`: (String, Required) Image title.
  - `description`: (String, Optional) Detailed description.
  - `taken_time`: (String, Optional) Date/Time taken.
  - `camera`: (String, Optional) Camera model.
- **Response Example**:
  ```json
  {
    "status": "success",
    "id": "uuid-string",
    "preview_url": "https://cdn.haozheli.com/uuid_preview.webp",
    "original_url": "https://cdn.haozheli.com/uuid_original.webp",
    "metadata_used_for_sparse": "Title Description ..."
  }
  ```

### 4.1 Browser-direct upload (presign + ingest staged)

For hosts with request body limits (e.g. Vercel), the file goes browser -> R2 directly,
then the backend pulls it from R2, processes it and deletes the staged object.
Both endpoints use the same Basic Auth as `/ingest`.

**Step 1: `POST /upload/presign`** (JSON)

```json
{ "files": [{ "content_type": "image/jpeg", "size": 5242880 }] }
```

Allowed types: `image/jpeg`, `image/png`, `image/webp`. Max 50 MB per file, 50 files per call.
Returns one entry per file; `PUT` the raw file to `upload_url` with the same `Content-Type`
(URL valid for 15 minutes). The R2 bucket needs a CORS rule allowing `PUT` from the site origin.

```json
[{ "key": "staging/<uuid>.jpg", "upload_url": "https://...", "content_type": "image/jpeg" }]
```

**Step 2: `POST /ingest/staged`** (JSON)

```json
{ "key": "staging/<uuid>.jpg", "title": "...", "description": null, "taken_time": null, "camera": null,
  "collection_ids": ["<collection uuid>"] }
```

`collection_ids` is optional: a list puts the image in exactly those collections (`[]` = none, and
that is remembered so it is never matched automatically later); leave it out to let the server match
the new image against the collections (see 12). Unknown collection ids are rejected (400) before
anything is downloaded or published.

Runs the same pipeline as `/ingest` and returns the same response, plus the image's `collection_ids`. Add an R2 lifecycle
rule expiring the `staging/` prefix (e.g. after 1 day) to clean up uploads that were never ingested.

## 5. Generate Random Query

Get a random search query that is guaranteed to be grounded in the gallery.

Queries come from a Redis queue (`random_query:queue`). A refill samples random images,
has a small LLM rewrite each title + description into a short search query, and warms the
query's embeddings into the embedding cache so the follow-up `/search` is fast. Each call
pops one query; when 5 or fewer remain, 10 more are added in the background. If the queue is
empty or Redis/LLM fails, a query from a small static list is returned instead.

- **URL**: `GET /generate-random-query`
- **Response Example**:
  ```json
  {
    "query": "daylight lingers on wisconsin autumn"
  }
  ```

## 6. Generate Description (Auto Title & Description)

**Requires Authentication (Basic Auth)**, same credentials as `/ingest`.

Generate a short title and detailed description for an uploaded image.

- **URL**: `POST /generate-description`
- **Content-Type**: `multipart/form-data`
- **Form Data**:
  - `file`: (File, Required) The image file (jpg/png/webp).
- **Response Example**:
  ```json
  {
    "title": "海边日落",
    "description": "画面展示了海边的日落景象，天空被橙红色渲染，远处有低矮的海岸线与零散的人影。照片看起来像是在海滩或海岸步道拍摄，可能位于临海城市或度假区。前景有沙滩或礁石纹理，整体氛围温暖宁静。"
  }
  ```

## 7. Health Check

- **URL**: `GET /health`
- **Response Example**:
  ```json
  {
    "status": "ok"
  }
  ```

## 8. Autocomplete Suggestions

Get search suggestions based on the current input prefix. Supports fuzzy matching and phrase completion.

- **URL**: `GET /autocomplete`
- **Query Parameters**:
  - `q` (string, required): The prefix or partial query to autocomplete.
- **Response Example**:
  ```json
  {
    "suggestions": [
      "Osaka Castle",
      "Osaka streets",
      "Osaka night view"
    ]
  }
  ```

The suggestions come from an in-memory trie built from every image's title and description.
It is not stored in the repository:

- On startup the backend loads the model saved on its own disk (`data/trie_model.pkl`, or
  `AUTOCOMPLETE_MODEL_PATH`) if there is one, and always rebuilds from Qdrant in the background.
  Until the first build finishes, suggestions are empty.
- Ingesting an image, deleting one, or changing a title/description schedules a rebuild. It runs
  after 5 minutes without further changes (each new change restarts the wait), and at the latest
  30 minutes after the first pending change. A failed rebuild is retried the same way.
- A rebuild builds a new trie off the request path and swaps it in when finished, so queries
  always see a complete index. State is per process: with several backend instances, each one
  only rebuilds for changes it handled itself.

## 9. Rebuild Autocomplete Index

**Requires Authentication (Basic Auth).**

Rebuild the index from Qdrant right now, discarding any scheduled rebuild, and save it to disk.
Returns once the new index is live. If a rebuild is already running, it is waited for and one more
is run afterwards.

- **URL**: `POST /autocomplete/build`
- **Response Example**:
  ```json
  {
    "status": "success",
    "message": "Autocomplete index rebuilt."
  }
  ```
- **Errors**: `500` if the rebuild fails (the previous index keeps serving).

### 9.1 Autocomplete Status

**Requires Authentication (Basic Auth).**

- **URL**: `GET /admin/autocomplete/status`
- **Response Example**:
  ```json
  {
    "ready": true,
    "building": false,
    "pending": true,
    "seconds_until_rebuild": 241.3,
    "last_built_at": 1761580000.0
  }
  ```
  `pending` means a rebuild is scheduled, `seconds_until_rebuild` is the time left until it starts
  (`null` when nothing is pending), `last_built_at` is epoch seconds (`null` before the first build).

## 10. Get Image by ID

Get details for a specific image by its UUID.

- **URL**: `GET /image/{image_id}`
- **Path Parameters**:
  - `image_id` (string, required): The UUID of the image to retrieve.
- **Response Example**:
  ```json
  {
    "preview_url": "https://cdn.haozheli.com/uuid_preview.webp",
    "original_url": "https://cdn.haozheli.com/uuid_original.webp",
    "score": 1.0,
    "metadata": {
      "title": "Summer Beach",
      "description": "Fun times",
      "taken_time": "2023-08-01",
      "camera": "Sony A7M4",
      "preview_url": "https://cdn.haozheli.com/uuid_preview.webp",
      "original_url": "https://cdn.haozheli.com/uuid_original.webp",
      "type": "image"
    }
  }
  ```

## 11. Admin: Manage Images

All endpoints below **require Authentication (Basic Auth)**, same credentials as `/ingest`.

### 11.1 List Images

Stable, paginated listing of every image (`/gallery` is a random feed and cannot be paged reliably).

- **URL**: `GET /admin/images`
- **Query Parameters**:
  - `limit` (int, optional): 1-100. Default: 40.
  - `cursor` (string, optional): `next_cursor` from the previous page.
- **Response Example**:
  ```json
  {
    "items": [
      {
        "id": "uuid-string",
        "preview_url": "https://cdn.haozheli.com/uuid_preview.webp",
        "original_url": "https://cdn.haozheli.com/uuid_original.webp",
        "score": 1.0,
        "metadata": { "title": "Summer Beach", "description": "Fun times", "taken_time": "2023-08-01", "camera": "Sony A7M4", "type": "image" }
      }
    ],
    "next_cursor": "uuid-of-next-page-or-null",
    "total": 128
  }
  ```

### 11.2 Update Metadata

- **URL**: `PATCH /admin/images/{image_id}`
- **Request Body (JSON)**: any of `title`, `description`, `taken_time`, `camera`, `collection_ids`.
  Omitted or `null` fields are left unchanged; an empty string clears an optional field;
  `title` can never be empty (400).
- **Collections**: `collection_ids` sets the image's collections to exactly that list (`[]` = none,
  remembered as an opt-out, see 12). Unknown ids are a 400 and nothing is written. Left out, the
  membership is kept, except that an image in **no** collection that has **not** opted out is matched
  against the collections when its text changes, and joins those at or above their thresholds.
  Membership is written in the same request as the text and vectors.
- **Re-embedding**: the title, description, taken time and camera are all part of the indexed
  text, so any change recomputes the `sparse` and `dense-text` vectors (the `dense-image`
  vector is untouched). The vectors and payload are written in a single Qdrant request, and
  embeddings are computed first, so a failed embedding changes nothing. A request that changes
  nothing returns immediately without re-embedding.
- **Response**: the updated image, in the same shape as an item of `/admin/images`.
- **Errors**: `404` unknown id, `400` blank title.

### 11.3 Delete Image

Removes the Qdrant point, its entries in the Redis recommendation pools and click/share
counters, and the preview and original files in R2. Not reversible.

- **URL**: `DELETE /admin/images/{image_id}`
- **Response Example**:
  ```json
  { "status": "deleted", "id": "uuid-string", "warnings": [] }
  ```
  The point is deleted first, so the image disappears from the gallery immediately. If the
  Redis or R2 cleanup fails afterwards the request still succeeds and `warnings` lists what
  was left behind. Only files named `<uuid>_preview.webp` / `<uuid>_original.webp` under the
  public CDN base URL are ever deleted from R2.
- **Errors**: `404` unknown id.
- **Not reflected automatically**: the autocomplete index (see 9) and copies already cached at
  the CDN edge.

### 11.4 Describe a Published Image

Like `/generate-description`, but for an image that is already published (reads the stored
original from R2).

- **URL**: `POST /admin/images/{image_id}/describe`
- **Response**: `{ "title": "...", "description": "..." }`
- **Errors**: `404` unknown id or missing original.

## 12. Admin: Collections

All endpoints below **require Authentication (Basic Auth)**.

A collection is a user-facing group of images (not a Qdrant collection). Its title and description are
embedded the way `/search` embeds a query (CLIP for `dense-image`, MiniLM for `dense-text`, BM25 for
`sparse`) and stored as the collection's *anchor* in the Qdrant collection `gallery_collections_v2`.
Images carry their membership in the payload:

- `collection_ids`: the collections the image is in. An image can be in several.
- `collections_opt_out`: `true` once an admin deliberately left the image in none (removed it from its
  last collection, or chose none on upload/edit), so automatic assignment never puts it back.

The two directions use the anchor differently:

- **Collection → images** (the review in 12.2, the order of a collection's images in 12.6 and 13.3, the
  default cover): the collection's text is run through the site's own hybrid search (the three vectors,
  fused with RRF), so keyword and picture matches count, not only similar wording.
- **Image → collections** (automatic assignment): needs a yes or no, which a fused search score cannot
  give (RRF only reflects rank; every query has a first result). So it compares the image's `dense-text`
  vector with each anchor's by cosine and joins every collection whose score reaches its own `threshold`
  (default `COLLECTION_DEFAULT_THRESHOLD`, 0.5). It only ever applies to an image with no collection that
  has not opted out; an image that already has collections is never changed automatically. If matching
  fails the image simply lands in none. A new collection reaches the existing images through the review
  step (12.2), not silently.

**Upgrading from `gallery_collections`** (MiniLM + BM25 only): on first startup the collections are
copied into `gallery_collections_v2`; the old store is left in place and can be deleted afterwards. Each
copied collection gets its CLIP vector the first time it is needed (if the embedding service is down then,
it is searched on its text alone and the CLIP vector is added on a later request).

### 12.1 List / Get Collections

- **URL**: `GET /admin/collections` (oldest first), `GET /admin/collections/{id}`
- **Response Example** (a list for the first, a single object for the second):
  ```json
  [{
    "id": "uuid-string",
    "title": "Desert Road Trip",
    "description": "Cacti, boneyards and long straight roads.",
    "threshold": 0.5,
    "image_count": 12,
    "cover_image_id": null,
    "cover_preview_url": "https://cdn.haozheli.com/uuid_preview.webp"
  }]
  ```
  `cover_image_id` is the admin's explicit choice. While it is `null` (or that image has been deleted or
  left the collection) the cover is the image the search ranks first for the collection's text, i.e. the
  most representative one.
- **Errors**: `404` unknown id.

### 12.2 Review Matches for a New Collection

Runs a draft title and description through the site's hybrid search and returns the results. **Nothing
is written.** The UI pre-selects by threshold locally, lets the admin tick which to keep, then calls 12.3.

- **URL**: `POST /admin/collections/preview`
- **Request Body (JSON)**: `title` (required), `description`, `limit` (1-500, default 200).
- **Response Example**:
  ```json
  {
    "items": [{ "id": "uuid", "preview_url": "...", "original_url": "...", "score": 0.83,
                "metadata": { "title": "...", "collection_ids": ["other-uuid"], "collections_opt_out": false } }],
    "default_threshold": 0.5
  }
  ```
  In search order. Each `score` is the image's MiniLM cosine to the text (what thresholds compare), so an
  image the search ranks high on keywords or looks alone can sit near the top with a low score: shown,
  but not pre-selected. `metadata.collection_ids` shows which images are already in other collections.
- **Errors**: `400` blank title.

### 12.3 Create a Collection

- **URL**: `POST /admin/collections`
- **Request Body (JSON)**: `title` (required), `description`, `threshold` (0-1, default
  `COLLECTION_DEFAULT_THRESHOLD`), `image_ids` (the images to put in it, at most 1000).
- **Response**: the collection (see 12.1). If putting the images in fails, the new collection is rolled
  back, so a retry starts clean.
- **Errors**: `400` blank title, invalid or unknown image id; `422` threshold out of range.

### 12.4 Update a Collection

- **URL**: `PATCH /admin/collections/{id}`
- **Request Body (JSON)**: any of `title`, `description` (empty clears), `threshold`, `cover_image_id`
  (empty clears; must be an image in the collection).
- Changing the title or description re-embeds the anchor (all three vectors). **Members are left as they are**; run 12.2 again
  to find more images. A new threshold only affects future matching.
- **Response**: the collection. **Errors**: `404` unknown id, `400` blank title / bad cover.

### 12.5 Delete a Collection

- **URL**: `DELETE /admin/collections/{id}`
- **Response Example**: `{ "status": "deleted", "id": "uuid", "removed_images": 12 }`
- The images are untouched and stay in the library; they just leave the collection (this does **not**
  count as an opt-out, so they may be matched into others later). Members are cleaned before the
  collection is removed, so a failure midway leaves a collection that can be deleted again.
  Deleting an *image* needs no collection cleanup.
- **Errors**: `404` unknown id.

### 12.6 List a Collection's Images

- **URL**: `GET /admin/collections/{id}/images?limit=40&cursor=...`
- **Response**: same shape as `GET /admin/images` (`items`, `next_cursor`, `total`), in the same order as
  13.3. The cursor is a position; pass back `next_cursor`.
- **Errors**: `404` unknown id.

### 12.7 Add / Remove Images (multi-select)

- **URL**: `POST /admin/images/membership`
- **Request Body (JSON)**: `image_ids` (at most 1000), `add` and `remove` (collection ids; at least one
  non-empty, and no id in both).
- **Response Example**:
  ```json
  { "updated": [{ "id": "uuid", "collection_ids": ["a"], "collections_opt_out": false }],
    "missing": ["uuid-of-an-image-that-no-longer-exists"] }
  ```
  Taking an image out of its last collection sets `collections_opt_out`; adding it to one clears it.
  Removing a collection id that no longer exists is allowed (cleans dangling ids).
- **Errors**: `400` invalid image id, unknown collection in `add`, add/remove overlap, nothing to do.

### 12.8 Suggest Collections for an Image

For the upload and edit forms: which collections an image with this metadata would join.

- **URL**: `POST /admin/collections/suggest`
- **Request Body (JSON)**: `title` (required), `description`, `taken_time`, `camera`. Embedded the same way
  as at ingest, so the score equals what ingest would compute.
- **Response Example**:
  ```json
  [{ "id": "uuid", "title": "Desert Road Trip", "score": 0.72, "threshold": 0.5, "matched": true },
   { "id": "uuid", "title": "Winter", "score": 0.08, "threshold": 0.5, "matched": false }]
  ```
  Every collection, best first. Pre-select the `matched` ones; the rest are near misses.
- **Errors**: `400` blank title.

### 12.9 Rewrite a Collection's Title and Description

Suggests a more poetic title and description for a draft collection, from the same LLM and structured
output as photo captions (`/generate-description`). **Nothing is saved**; the admin keeps, edits or
discards the text. The prompt keeps every place, date and event the admin wrote, since those words are
what the collection's photos are found by (12.2), and adds no facts.

- **URL**: `POST /admin/collections/rewrite`
- **Request Body (JSON)**: `title` (required), `description` (optional; a title alone is fine).
- **Response Example**:
  ```json
  { "title": "Japan, Summer 2024", "description": "Osaka, Kyoto and Tokyo under a long summer light." }
  ```
- **Errors**: `400` blank title; `500` the LLM call failed or returned no title.

## 13. Collections (public)

No authentication. Only collections that have at least one image are listed. `collections_opt_out` is
internal and never appears in public image metadata; `collection_ids` does.

The listing is kept in memory for up to 5 minutes and dropped immediately by any change made through the
same instance (membership, edits, deletes, ingest into a collection).

### 13.1 List Collections

- **URL**: `GET /collections`
- **Response Example**: largest collection first, ties by title.
  ```json
  [{
    "id": "uuid-string",
    "title": "Racing",
    "description": "Grandstands, pit lanes and long straights",
    "image_count": 48,
    "cover_preview_url": "https://cdn.haozheli.com/uuid_preview.webp"
  }]
  ```
  The cover is the image the search ranks first for the collection's text unless the admin picked one
  (see 12.1).

### 13.2 Get a Collection

- **URL**: `GET /collections/{collection_id}`
- **Response**: one collection in the shape above (an empty collection is returned too, with `image_count` 0).
- **Errors**: `404` unknown id.

### 13.3 List a Collection's Images

- **URL**: `GET /collections/{collection_id}/images?limit=40&cursor=...`
- **Query Parameters**: `limit` (1-100, default 40), `cursor` (the previous response's `next_cursor`).
- **Response Example**:
  ```json
  {
    "items": [{ "id": "uuid", "preview_url": "...", "original_url": "...", "score": 0.91, "metadata": { } }],
    "next_cursor": "40",
    "total": 48
  }
  ```
  In the order the site's search ranks the members for the collection's text, so the most representative
  images come first (`score` is the fused search score, only meaningful for ordering). `next_cursor` is
  `null` on the last page.
- **Errors**: `404` unknown id, `400` invalid cursor.
