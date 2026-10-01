"""Re-embed every point of the existing collection with the self-hosted embedding
service and write the result into a new Qdrant instance.

The source collection is only read. Point ids and payloads are preserved. Safe to
re-run: ids already present in the target are skipped.

Env:
  QDRANT_URL / QDRANT_API_KEY          source (the existing .env values)
  NEW_QDRANT_URL / NEW_QDRANT_API_KEY  target (use ":memory:" for a dry rehearsal)
  EMBEDDING_SERVICE_URL                embedding service base URL

Run from backend/:
  python -m scripts.migrate_embeddings --limit 5      # rehearsal on a few points
  python -m scripts.migrate_embeddings                # everything
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests
from qdrant_client import QdrantClient, models

from core.config import settings
from core.embedding import get_sparse_embedding
from core.utils import process_image_for_embedding

COLLECTION = settings.COLLECTION_NAME
DIM = settings.EMBEDDING_DIM
EMBED_URL = settings.EMBEDDING_SERVICE_URL.rstrip("/")


def metadata_text(payload: Dict[str, Any]) -> str:
    # Must match the text built in main.py's ingest endpoint.
    return (
        f"{payload.get('title', '')} {payload.get('description') or ''} "
        f"{payload.get('taken_time') or ''} {payload.get('camera') or ''}"
    )


def ensure_target(target: QdrantClient) -> None:
    if target.collection_exists(COLLECTION):
        cfg = target.get_collection(COLLECTION).config.params.vectors
        for name in ("dense-image", "dense-text"):
            if name not in cfg or cfg[name].size != DIM:
                sys.exit(f"target collection exists but '{name}' is not {DIM}-d")
        print(f"target collection {COLLECTION} exists, resuming")
    else:
        target.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense-image": models.VectorParams(size=DIM, distance=models.Distance.COSINE),
                "dense-text": models.VectorParams(size=DIM, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            },
        )
        print(f"created target collection {COLLECTION} ({DIM}-d)")
    target.create_payload_index(
        COLLECTION, field_name="original_url", field_schema=models.PayloadSchemaType.KEYWORD
    )


def scroll_all(client: QdrantClient, with_payload: bool) -> List[models.Record]:
    out, offset = [], None
    while True:
        points, offset = client.scroll(
            COLLECTION, limit=100, offset=offset, with_payload=with_payload, with_vectors=False
        )
        out += points
        if offset is None:
            return out


def embed(path: str, items: List[str]) -> List[List[float]]:
    key = "images" if "image" in path else "texts"
    r = requests.post(f"{EMBED_URL}{path}", json={key: items}, timeout=(5, 180))
    r.raise_for_status()
    vecs = r.json()["embeddings"]
    assert len(vecs) == len(items) and all(len(v) == DIM for v in vecs), "bad embedding shape"
    return vecs


def download(payload: Dict[str, Any]) -> bytes:
    last: Optional[Exception] = None
    for field in ("original_url", "preview_url"):  # prefer the higher-quality file
        url = payload.get(field)
        if not url:
            continue
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.content
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"could not download image: {last}")


def process_batch(target: QdrantClient, batch: List[models.Record]) -> int:
    # Same preprocessing as the upload path (1024px JPEG) so vectors are comparable.
    images = [process_image_for_embedding(download(p.payload)) for p in batch]
    texts = [metadata_text(p.payload) for p in batch]
    image_vecs = embed("/embed/dense/image", images)
    text_vecs = embed("/embed/dense/text", texts)
    points = []
    for p, iv, tv, text in zip(batch, image_vecs, text_vecs, texts):
        sp = get_sparse_embedding(text)
        points.append(
            models.PointStruct(
                id=p.id,
                vector={
                    "dense-image": iv,
                    "dense-text": tv,
                    "sparse": models.SparseVector(indices=sp["indices"], values=sp["values"]),
                },
                payload=p.payload,
            )
        )
    target.upsert(COLLECTION, points=points)
    return len(points)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only migrate the first N points")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="list what would be migrated, write nothing")
    args = ap.parse_args()

    new_url = os.getenv("NEW_QDRANT_URL")
    if not new_url:
        sys.exit("NEW_QDRANT_URL is not set")
    if new_url != ":memory:" and new_url.rstrip("/") == (settings.QDRANT_URL or "").rstrip("/"):
        sys.exit("NEW_QDRANT_URL equals the source QDRANT_URL; refusing to run")

    source = QdrantClient(url=settings.QDRANT_URL, port=None, api_key=settings.QDRANT_API_KEY, timeout=60)
    target = (
        QdrantClient(":memory:")
        if new_url == ":memory:"
        else QdrantClient(url=new_url, port=None, api_key=os.getenv("NEW_QDRANT_API_KEY"), timeout=60)
    )

    health = requests.get(f"{EMBED_URL}/health", timeout=15)
    health.raise_for_status()
    print("embedding service:", health.json())

    src_points = scroll_all(source, with_payload=True)
    if args.limit:
        src_points = src_points[: args.limit]
    print(f"source points: {len(src_points)}")

    if not args.dry_run:
        ensure_target(target)
        done = {p.id for p in scroll_all(target, with_payload=False)}
    else:
        done = set()
    todo = [p for p in src_points if p.id not in done]
    print(f"already migrated: {len(src_points) - len(todo)}, to do: {len(todo)}")
    if args.dry_run:
        for p in todo[:5]:
            print(" ", p.id, p.payload.get("title"))
        return

    batches = [todo[i : i + args.batch] for i in range(0, len(todo), args.batch)]
    migrated, failed = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_batch, target, b): b for b in batches}
        for f in as_completed(futs):
            try:
                migrated += f.result()
            except Exception as e:  # noqa: BLE001
                failed += [p.id for p in futs[f]]
                print(f"batch failed ({[p.payload.get('title') for p in futs[f]][:2]}...): {e}")
            print(f"  progress: {migrated}/{len(todo)}", end="\r", flush=True)
    print()

    total = target.count(COLLECTION, exact=True).count
    print(f"migrated {migrated}, failed {len(failed)}, target now has {total} points")
    if failed:
        print("failed ids (re-run the script to retry):", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
