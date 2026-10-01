"""Delete cached query embeddings that the backend can no longer use.

Cache keys are `embedding:{model}:{text}`. This removes every `embedding:*` key that
is either an old unprefixed entry (from before keys carried a model name) or holds a
vector of the wrong size for its model. Anything else in Redis (recommendation pools,
counters, ...) is left alone.

  python -m scripts.purge_stale_embedding_cache            # list only (dry run)
  python -m scripts.purge_stale_embedding_cache --apply    # delete them

Uses REDIS_URL from .env (including its database number).
"""

import argparse
import json

import redis

from core.config import settings


def check(key: str, value) -> bool:
    """True if this cache entry is valid under the current models."""
    body = key[len("embedding:") :]
    for model, dim in settings.EMBEDDING_DIMS.items():
        if body.startswith(model + ":"):
            try:
                return len(json.loads(value or "[]")) == dim
            except (ValueError, TypeError):
                return False
    return False  # no known model prefix: legacy key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    args = ap.parse_args()

    r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True, socket_timeout=10)
    stale, ok = [], 0
    for key in r.scan_iter("embedding:*", count=1000):
        if check(key, r.get(key)):
            ok += 1
        else:
            stale.append(key)

    print(f"{ok} cache entries are valid (kept), {len(stale)} are unusable:")
    for key in stale[:20]:
        print(f"  {key[:80]!r}")
    if len(stale) > 20:
        print(f"  ... and {len(stale) - 20} more")
    if not stale:
        return
    if not args.apply:
        print("dry run: nothing deleted. Re-run with --apply to delete the entries above.")
        return
    for i in range(0, len(stale), 500):
        r.delete(*stale[i : i + 500])
    print(f"deleted {len(stale)} keys")


if __name__ == "__main__":
    main()
