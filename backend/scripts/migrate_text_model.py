"""Move the gallery from collection v1 to v2: `dense-text` goes from CLIP (768-d) to
the multilingual MiniLM model (384-d).

Qdrant cannot change a named vector's size in place, so this builds a new collection
(settings.COLLECTION_NAME) next to the old one (settings.LEGACY_COLLECTION_NAME) in the
same Qdrant instance. For every point:
  - `dense-image` (CLIP) and `sparse` (BM25) are copied unchanged, so no image is
    re-processed;
  - `dense-text` is recomputed from title/description/time/camera with the new model;
  - id and payload are preserved.

The old collection is only read. Re-running is safe: ids already in the new
collection are skipped. With --delete-old the old collection is dropped, but only
after the new one is verified (same ids, same image vectors on a sample, correct
dimensions).

Env: QDRANT_URL / QDRANT_API_KEY / EMBEDDING_SERVICE_URL (as for the backend).

Run from backend/:
  python -m scripts.migrate_text_model --limit 5       # rehearsal on a few points
  python -m scripts.migrate_text_model                 # copy everything
  python -m scripts.migrate_text_model --delete-old    # verify, then drop the old collection
"""

import argparse
import random
import sys
from typing import Any, Dict, List

import requests
from qdrant_client import QdrantClient, models

from core.config import settings

OLD = settings.LEGACY_COLLECTION_NAME
NEW = settings.COLLECTION_NAME
EMBED_URL = settings.EMBEDDING_SERVICE_URL.rstrip("/")


def metadata_text(payload: Dict[str, Any]) -> str:
    # Must match the text built in main.py's ingest endpoint.
    return (
        f"{payload.get('title', '')} {payload.get('description') or ''} "
        f"{payload.get('taken_time') or ''} {payload.get('camera') or ''}"
    )


def embed_texts(texts: List[str]) -> List[List[float]]:
    r = requests.post(
        f"{EMBED_URL}/embed/dense/text",
        json={"texts": texts, "model": settings.TEXT_MODEL},
        timeout=(5, 120),
    )
    r.raise_for_status()
    body = r.json()
    vecs = body["embeddings"]
    assert body["model"] == settings.TEXT_MODEL, f"service answered with {body['model']}"
    assert len(vecs) == len(texts) and all(len(v) == settings.TEXT_DIM for v in vecs), "bad shape"
    return vecs


def ensure_target(client: QdrantClient) -> None:
    if client.collection_exists(NEW):
        cfg = client.get_collection(NEW).config.params.vectors
        for name, dim in (("dense-image", settings.CLIP_DIM), ("dense-text", settings.TEXT_DIM)):
            if name not in cfg or cfg[name].size != dim:
                sys.exit(f"{NEW} exists but '{name}' is not {dim}-d")
        print(f"{NEW} exists, resuming")
    else:
        client.create_collection(
            collection_name=NEW,
            vectors_config={
                "dense-image": models.VectorParams(size=settings.CLIP_DIM, distance=models.Distance.COSINE),
                "dense-text": models.VectorParams(size=settings.TEXT_DIM, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            },
        )
        print(f"created {NEW}")
    client.create_payload_index(NEW, field_name="original_url", field_schema=models.PayloadSchemaType.KEYWORD)


def scroll_all(client: QdrantClient, collection: str, with_vectors: bool) -> List[models.Record]:
    out, offset = [], None
    while True:
        points, offset = client.scroll(
            collection, limit=100, offset=offset, with_payload=True, with_vectors=with_vectors
        )
        out += points
        if offset is None:
            return out


def verify(client: QdrantClient) -> None:
    old = {p.id: p for p in scroll_all(client, OLD, with_vectors=True)}
    new = {p.id: p for p in scroll_all(client, NEW, with_vectors=True)}
    if set(old) != set(new):
        sys.exit(f"verify failed: {len(old)} old ids vs {len(new)} new ids, sets differ")
    for p in new.values():
        v = p.vector
        if len(v["dense-image"]) != settings.CLIP_DIM or len(v["dense-text"]) != settings.TEXT_DIM or "sparse" not in v:
            sys.exit(f"verify failed: bad vectors on point {p.id}")
    for pid in random.sample(sorted(old), min(20, len(old))):
        if old[pid].vector["dense-image"] != new[pid].vector["dense-image"]:
            sys.exit(f"verify failed: dense-image changed for {pid}")
        if old[pid].payload != new[pid].payload:
            sys.exit(f"verify failed: payload changed for {pid}")
    print(f"verified: {len(new)} points, ids match, vector sizes correct, sample of image vectors/payloads identical")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only migrate the first N points")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--delete-old", action="store_true", help="verify, then drop the old collection")
    args = ap.parse_args()

    client = QdrantClient(url=settings.QDRANT_URL, port=None, api_key=settings.QDRANT_API_KEY, timeout=60)
    if not client.collection_exists(OLD):
        if args.delete_old:
            sys.exit(f"{OLD} does not exist; nothing to delete")
        sys.exit(f"{OLD} does not exist; nothing to migrate")

    h = requests.get(f"{EMBED_URL}/health", timeout=15)
    h.raise_for_status()
    if settings.TEXT_MODEL not in h.json().get("text_models", {}):
        sys.exit(f"embedding service does not serve {settings.TEXT_MODEL}: {h.json()}")

    src = scroll_all(client, OLD, with_vectors=True)
    if args.limit:
        src = src[: args.limit]
    print(f"{OLD}: {len(src)} points")

    if not args.delete_old:
        ensure_target(client)
        done = {p.id for p in scroll_all(client, NEW, with_vectors=False)}
        todo = [p for p in src if p.id not in done]
        print(f"already in {NEW}: {len(src) - len(todo)}, to do: {len(todo)}")
        for i in range(0, len(todo), args.batch):
            batch = todo[i : i + args.batch]
            vecs = embed_texts([metadata_text(p.payload) for p in batch])
            client.upsert(
                NEW,
                points=[
                    models.PointStruct(
                        id=p.id,
                        vector={
                            "dense-image": p.vector["dense-image"],
                            "dense-text": tv,
                            "sparse": p.vector["sparse"],
                        },
                        payload=p.payload,
                    )
                    for p, tv in zip(batch, vecs)
                ],
            )
            print(f"  {min(i + args.batch, len(todo))}/{len(todo)}", end="\r", flush=True)
        print()
        print(f"{NEW} now has {client.count(NEW, exact=True).count} points")
        if args.limit:
            return

    if args.delete_old:
        verify(client)
        client.delete_collection(OLD)
        print(f"deleted {OLD}")
    else:
        print("Next: deploy the backend, then run again with --delete-old to drop the old collection.")


if __name__ == "__main__":
    main()
