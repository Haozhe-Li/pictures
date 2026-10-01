<div align="center">

# 📸 Pictures

### AI-Powered Multimodal Gallery for Photographers

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-haozheli.pictures-blue?style=for-the-badge)](https://www.haozheli.pictures/)

[English](README.md) | [简体中文](README_ZH.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=flat-square&logo=next.js&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-1E1E1E?style=flat-square&logo=qdrant&logoColor=white)
![Cloudflare R2](https://img.shields.io/badge/Cloudflare_R2-F38020?style=flat-square&logo=cloudflare&logoColor=white)

**Pictures** is a production-ready multimodal, multilingual image gallery that combines hybrid retrieval (sparse + dense) with intelligent caching to deliver fast, accurate search results from natural language or images.

![Homepage Demo](assets/pictures-home.gif)

[Features](#-features) • [Architecture](#️-architecture) • [Tech Stack](#️-tech-stack) • [Quick Start](#-quick-start) • [Configuration](#️-configuration) • [Deployment](#-deployment)

</div>

---

## ✨ Features

### 🔍 Natural Language Search

Powered by hybrid sparse+dense retrieval for accurate, context-aware results.

![Natural language search](assets/pictures-search.gif)

### 🌐 Multilingual Support

Search in any language — no barriers, powered by CLIP embeddings.

![Multilingual search](assets/pictures-multilingual.gif)

### 🖼️ Image-to-Image Search

Find visually similar photos using advanced visual embeddings.

![Image-to-image search](assets/pictures-similar.gif)

### 🎲 Smart Recommendations

Discover new photos with intelligent random recommendations.

![Random recommendations](assets/pictures-random-search.gif)

### ⚡ High Performance

Multi-layer caching architecture minimizes latency and reduces costs at scale.

---

## 🏗️ Architecture

### Hybrid Retrieval

Pictures uses a sophisticated hybrid retrieval system that combines:

![Hybrid Retrieval Flow](https://cdn.haozheli.com/pictures-flow-chart.webp)

1. **Sparse Retrieval (BM25)** — Keyword-based metadata matching
2. **Dense Retrieval (CLIP)** — Semantic text and image embeddings
3. **Vector Storage (Qdrant)** — Efficient similarity search
4. **Rank Fusion (RRF)** — Reciprocal Rank Fusion for optimal results

### Multi-Layer Caching

Intelligent caching reduces latency and API costs:

![Multi-Layer Cache](https://cdn.haozheli.com/pictures-cache-layer.webp)

- **Layer 1:** Frontend cache (Next.js + Upstash Redis)
- **Layer 2:** Backend service cache
- **Layer 3:** Embedding cache (query vectors)

---

## 🛠️ Tech Stack

### Backend

- **Framework:** FastAPI (Python 3.10+)
- **Vector Database:** Qdrant
- **Embeddings:** CLIP (`jinaai/jina-clip-v1`) via a self-hosted FastEmbed service, BM25 (FastEmbed)
- **Retrieval:** BM25 + Dense Vector Search
- **Cache:** Redis
- **Storage:** Cloudflare R2

### Frontend

- **Framework:** Next.js 14 (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **UI Components:** Radix UI + shadcn/ui
- **Cache:** Upstash Redis
- **Deployment:** Vercel

### Infrastructure

- **CDN:** Cloudflare
- **Container:** Docker
- **Monitoring:** Built-in logging

---

## 📂 Project Structure

```
gallery_RAG/
├── backend/                 # FastAPI backend service
│   ├── main.py             # Application entry point
│   ├── requirements.txt    # Python dependencies
│   ├── Dockerfile          # Container configuration
│   ├── core/               # Core modules
│   │   ├── config.py       # Configuration management
│   │   ├── db.py           # Database connections
│   │   ├── embedding.py    # Embedding generation
│   │   ├── storage.py      # R2 storage integration
│   │   └── utils.py        # Utility functions
│   └── models/             # Model artifacts
├── frontend/               # Next.js gallery app
│   ├── app/                # App router pages
│   │   ├── page.tsx        # Homepage
│   │   └── api/            # API routes
│   ├── components/         # React components
│   │   ├── gallery.tsx
│   │   ├── search-bar.tsx
│   │   └── ui/             # UI primitives
│   └── lib/                # Utilities
│       ├── api.ts          # API client
│       └── types.ts        # TypeScript types
├── upload_GUI/             # Admin upload interface
│   ├── app/                # Upload app pages
│   └── components/         # Upload components
└── assets/                 # Documentation assets
```

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.10 or higher
- **Node.js** 18 or higher
- **pnpm** (recommended) or npm
- **Docker** (optional, for containerized deployment)
- **Redis** instance (local or cloud)
- **Qdrant** instance (local or cloud)

### Backend Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/gallery_RAG.git
   cd gallery_RAG/backend
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Run the backend**

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   Or with Docker:

   ```bash
   docker build -t pictures-backend .
   docker run -p 8000:8000 --env-file .env pictures-backend
   ```

### Frontend Setup

1. **Navigate to frontend directory**

   ```bash
   cd frontend
   ```

2. **Install dependencies**

   ```bash
   pnpm install
   ```

3. **Configure environment variables**

   ```bash
   cp .env.example .env.local
   # Edit .env.local with your configuration
   ```

4. **Run the development server**

   ```bash
   pnpm dev
   ```

   Open [http://localhost:3000](http://localhost:3000) in your browser.

### Upload GUI Setup

1. **Navigate to upload_GUI directory**

   ```bash
   cd upload_GUI
   ```

2. **Install dependencies**

   ```bash
   pnpm install
   ```

3. **Configure environment variables**

   ```bash
   cp .env.example .env.local
   # Set BACKEND_URL to your backend URL
   ```

4. **Run the upload interface**
   ```bash
   pnpm dev
   ```

---

## ⚙️ Configuration

### Backend Environment Variables

Create a `.env` file in the `backend/` directory:

| Variable            | Description                                        | Required |
| ------------------- | -------------------------------------------------- | -------- |
| `CF_API_URL`        | Cloudflare R2 S3 endpoint URL                      | ✅       |
| `CF_API_KEY_ID`     | R2 access key ID                                   | ✅       |
| `CF_API_KEY_SECRET` | R2 access key secret                               | ✅       |
| `QDRANT_URL`        | Qdrant endpoint (default: `http://localhost:6333`) | ✅       |
| `QDRANT_API_KEY`    | Qdrant API key (if using cloud)                    | ⚠️       |
| `EMBEDDING_SERVICE_URL` | Base URL of the embedding service (dense vectors) | ✅       |
| `REDIS_HOST`        | Redis host address                                 | ✅       |
| `REDIS_PORT`        | Redis port (default: `16666`)                      | ✅       |
| `REDIS_USERNAME`    | Redis username (default: `default`)                | ✅       |
| `REDIS_PASSWORD`    | Redis password                                     | ✅       |
| `ADMIN_USERNAME`    | Basic auth username for `/ingest` endpoint         | ✅       |
| `ADMIN_PASSWORD`    | Basic auth password for `/ingest` endpoint         | ✅       |

### Frontend Environment Variables

Create a `.env.local` file in the `frontend/` directory:

| Variable               | Description                                      | Required |
| ---------------------- | ------------------------------------------------ | -------- |
| `GALLERY_API_BASE_URL` | Backend base URL (e.g., `http://localhost:8000`) | ✅       |
| `KV_REST_API_URL`      | Upstash Redis REST API URL                       | ✅       |
| `KV_REST_API_TOKEN`    | Upstash Redis REST API token                     | ✅       |

### Upload GUI Environment Variables

Create a `.env.local` file in the `upload_GUI/` directory:

| Variable      | Description                          | Required |
| ------------- | ------------------------------------ | -------- |
| `BACKEND_URL` | Backend base URL for image ingestion | ✅       |

---

## 📚 API Documentation

Comprehensive API documentation is available at [backend/API.md](backend/API.md).

### Key Endpoints

- `GET /gallery` — Retrieve all gallery images
- `POST /search` — Search images by text query
- `POST /similar` — Find similar images by image URL
- `GET /random-query` — Get random search suggestions
- `POST /ingest` — Upload and process new images (admin only)

For detailed request/response schemas and examples, see the [API documentation](backend/API.md).

---

## 🚢 Deployment

### Production Architecture

- **Frontend:** Deployed on Vercel with edge caching
- **Backend:** Containerized deployment (Docker) on any cloud provider
- **Storage:** Cloudflare R2 with CDN
- **Vector DB:** Qdrant Cloud or self-hosted
- **Cache:** Upstash Redis (frontend) + Redis (backend)

### Backend Deployment (Docker)

```bash
cd backend

# Build the Docker image
docker build -t pictures-backend .

# Run the container
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  --name pictures-backend \
  pictures-backend
```

### Frontend Deployment (Vercel)

1. **Push to GitHub**

   ```bash
   git push origin main
   ```

2. **Import to Vercel**
   - Visit [vercel.com](https://vercel.com)
   - Import your repository
   - Set root directory to `frontend/`
   - Add environment variables from `.env.local`

3. **Deploy**
   - Vercel will automatically deploy on every push to `main`

### Upload GUI Deployment

The upload GUI can be deployed similarly to the frontend, or hosted locally for admin use only. Ensure HTTP Basic Auth credentials are properly configured.

---

## 📄 License

This project is currently unlicensed. If you plan to open-source this project, please add a LICENSE file (MIT, Apache 2.0, etc.).

---

<div align="center">
**Built with ❤️ by Haozhe Li**

[⬆ Back to Top](#-pictures)

</div>
