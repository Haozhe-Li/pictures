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
    # v2: `dense-text` is 384-d (multilingual MiniLM); v1 had it 768-d (CLIP) and is
    # incompatible, so it lives in a different collection (see scripts/migrate_text_model.py).
    COLLECTION_NAME = "gallery_rag_hybrid_v2"
    LEGACY_COLLECTION_NAME = "gallery_rag_hybrid"

    # Self-hosted embedding service. Two dense models, each with its own job:
    #  - CLIP: text query <-> image (`dense-image`) and image <-> image. Shared text/image space.
    #  - MiniLM: text query <-> title+description (`dense-text`). Multilingual, text only.
    EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8000")
    CLIP_MODEL = "jinaai/jina-clip-v1"
    CLIP_DIM = 768
    TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    TEXT_DIM = 384
    EMBEDDING_DIMS = {CLIP_MODEL: CLIP_DIM, TEXT_MODEL: TEXT_DIM}

    # Redis Config
    REDIS_URL = os.getenv("REDIS_URL")

    # Project Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


settings = Settings()
