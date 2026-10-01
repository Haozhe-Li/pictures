import os
import dotenv

dotenv.load_dotenv()


class Settings:
    # R2 Config
    CF_API_URL = os.getenv("CF_API_URL")
    CF_API_KEY_ID = os.getenv("CF_API_KEY_ID")
    CF_API_KEY_SECRET = os.getenv("CF_API_KEY_SECRET")
    CF_BUCKET = "haozheli-pictures"
    CLOUDFLARE_FREE_URL = "https://img-cdn.haozheli.com/"
    # Qdrant Config
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
    COLLECTION_NAME = "gallery_rag_hybrid"

    # Self-hosted embedding service (jinaai/jina-clip-v1, 768-d)
    EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8000")
    EMBEDDING_DIM = 768

    # Redis Config
    REDIS_URL = os.getenv("REDIS_URL")

    # Project Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


settings = Settings()
