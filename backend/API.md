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

## 5. Generate Random Query

Generate a random photo description query using LLM.

- **URL**: `GET /generate-random-query`
- **Response Example**:
  ```json
  {
    "query": "A golden retriever running on the beach at sunset"
  }
  ```

## 6. Generate Description (Auto Title & Description)

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
