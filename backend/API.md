# Gallery RAG Backend API Specification

Base URL: `http://localhost:8000` (or deployed domain)

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
        "image_url": "https://cdn.haozheli.com/uuid.jpg",
        "score": 1.0,
        "metadata": {
          "title": "Summer Beach",
          "description": "Fun times",
          "taken_time": "2023-08-01",
          "camera": "Sony A7M4"
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
    "limit": 20
  }
  ```
- **Response Example**:
  ```json
  [
    {
      "image_url": "https://cdn.haozheli.com/dog.jpg",
      "score": 0.89,
      "metadata": {
        "title": "My Dog",
        "description": "Playing in the park",
        "taken_time": "2024-01-15",
        "camera": "iPhone 15"
      }
    }
  ]
  ```

## 3. Ingest Image (Upload)

Upload an image, generate embeddings (Dense + Sparse), and store metadata.
**Requires Authentication (Basic Auth).**

- **URL**: `POST /ingest`
- **Authentication**: HTTP Basic Auth
  - Username: `admin`
  - Password: `admin`
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
    "url": "https://cdn.haozheli.com/uuid.jpg",
    "metadata_used_for_sparse": "Title Description ..."
  }
  ```
