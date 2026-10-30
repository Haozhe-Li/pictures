"""Remove malformed ids from the recommendation pools.

/interact used to add whatever id it was given to `gallery:pool:active`. This removes every
member of the active and explore pools that is not a canonical UUID, along with its counters.
Scores of valid ids are left alone, which is why this is used rather than init_redis_pools.py.

  python -m scripts.purge_invalid_pool_ids            # list only (dry run)
  python -m scripts.purge_invalid_pool_ids --apply    # delete them

Uses REDIS_URL from .env (including its database number).
"""

import argparse

import redis

from core.config import settings
from core.gallery_collections import is_canonical_uuid

POOLS = ("gallery:pool:active", "gallery:pool:explore")
COUNTERS = ("gallery:counters:click", "gallery:counters:share")
BATCH = 500


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    args = ap.parse_args()

    r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True, socket_timeout=10)

    invalid = {}
    for pool in POOLS:
        invalid[pool] = [m for m, _ in r.zscan_iter(pool) if not is_canonical_uuid(m)]
    for counters in COUNTERS:
        invalid[counters] = [f for f, _ in r.hscan_iter(counters) if not is_canonical_uuid(f)]

    for key, members in invalid.items():
        print(f"{key}: {len(members)} invalid")
        for member in members[:20]:
            print(f"  {member!r}")
        if len(members) > 20:
            print(f"  ... and {len(members) - 20} more")

    if not args.apply:
        print("Dry run; pass --apply to delete them.")
        return

    for key, members in invalid.items():
        remove = r.zrem if key in POOLS else r.hdel
        for start in range(0, len(members), BATCH):
            remove(key, *members[start : start + BATCH])
    print("Done.")


if __name__ == "__main__":
    main()
