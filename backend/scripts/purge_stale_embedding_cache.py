"""Delete cached query embeddings whose dimension does not match EMBEDDING_DIM.

Only `embedding:*` keys are considered, and only those with a wrong-sized vector;
everything else in Redis (recommendation pools, counters, ...) is left alone.

  python -m scripts.purge_stale_embedding_cache            # list only (dry run)
  python -m scripts.purge_stale_embedding_cache --apply    # delete them

Uses REDIS_URL from .env (including its database number).
"""

import argparse
import json

import redis

from core.config import settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    args = ap.parse_args()

    r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True, socket_timeout=10)
    stale, ok = [], 0
    for key in r.scan_iter("embedding:*", count=1000):
        try:
            n = len(json.loads(r.get(key) or "[]"))
        except (ValueError, TypeError):
            n = -1
        if n == settings.EMBEDDING_DIM:
            ok += 1
        else:
            stale.append((key, n))

    print(f"{ok} cache entries are {settings.EMBEDDING_DIM}-d (kept), {len(stale)} are stale:")
    for key, n in stale:
        print(f"  {n}-d  {key[:80]!r}")
    if not stale:
        return
    if not args.apply:
        print("dry run: nothing deleted. Re-run with --apply to delete the entries above.")
        return
    r.delete(*[k for k, _ in stale])
    print(f"deleted {len(stale)} keys")


if __name__ == "__main__":
    main()
