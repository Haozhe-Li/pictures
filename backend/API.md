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
{ "key": "staging/<uuid>.jpg", "title": "...", "description": null, "taken_time": null, "camera": null }
```

Runs the same pipeline as `/ingest` and returns the same response. Add an R2 lifecycle
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

- **URL**: `GET /api/autocomplete`
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

## 9. Build Autocomplete Index

Trigger a rebuild of the autocomplete Trie index from the Qdrant database. This scans all image titles and descriptions and updates the in-memory and on-disk model.

- **URL**: `POST /autocomplete/build`
- **Response Example**:
  ```json
  {
    "status": "success",
    "message": "Autocomplete index rebuilt."
  }
  ```

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
- **Request Body (JSON)**: any of `title`, `description`, `taken_time`, `camera`.
  Omitted or `null` fields are left unchanged; an empty string clears an optional field;
  `title` can never be empty (400).
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
