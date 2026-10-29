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
    # User-facing "collections" (albums). Each one is a point in its own Qdrant collection holding
    # its title+description embedded the way /search embeds a query (CLIP, MiniLM, BM25), so its
    # photos can be found and ranked with the site's own search. New photos are matched against
    # the MiniLM vector. v1 had no CLIP vector; its points are copied over on first startup.
    GALLERY_COLLECTIONS_NAME = "gallery_collections_v2"
    LEGACY_GALLERY_COLLECTIONS_NAME = "gallery_collections"
    # Cosine score on `dense-text` above which an image joins a collection. Each collection can
    # override it; this is the default a new one starts with.
    DEFAULT_COLLECTION_THRESHOLD = float(os.getenv("COLLECTION_DEFAULT_THRESHOLD", "0.5"))
    # Cosine score on `dense-image` (CLIP) at or above which an upload is held back as a likely
    # duplicate of a photo already in the library. The same file scores about 1.0, a re-export
    # (resized, re-encoded) a little less; neighbouring burst shots can come close too.
    DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", "0.95"))

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

    # TrustMark invisible watermarks (core/watermark). Variant Q, fixed: a mark can only be read with the
    # variant that wrote it. Its ONNX models are fetched into WATERMARK_MODEL_DIR when the Docker image is
    # built, and loaded only while a photo is watermarked or checked.
    WATERMARK_MODEL_TYPE = "Q"
    WATERMARK_MODEL_DIR = os.getenv("WATERMARK_MODEL_DIR", os.path.join(MODELS_DIR, "trustmark"))
    WATERMARK_THREADS = int(os.getenv("WATERMARK_THREADS", "2"))


settings = Settings()
