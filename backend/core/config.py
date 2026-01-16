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

    # Jina Config
    JINA_API_KEY = os.getenv(
        "JINA_API_KEY",
    )
    JINA_URL = "https://api.jina.ai/v1/embeddings"

    # Redis Config
    REDIS_HOST = os.getenv("REDIS_HOST")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 16666))
    REDIS_USERNAME = os.getenv("REDIS_USERNAME", "default")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

    # Project Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


settings = Settings()
