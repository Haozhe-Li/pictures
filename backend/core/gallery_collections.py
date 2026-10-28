"""
Pure helpers for collections: their addresses on the site, and membership.

A collection's address is its `slug`, made from its title ("Formula 1" -> /collections/formula-1).
A rename gives it a new one; the old ones are kept in `former_slugs` and still lead to it.

An image's membership is two payload keys:

- `collection_ids`: the collections it belongs to (an image can be in several).
- `collections_opt_out`: true once an admin has deliberately left it in none, so automatic
  assignment must not put it back. Automatic assignment only ever applies to an image that
  has no collection and has not opted out.
"""

import random
import re
import unicodedata
import uuid
from typing import Any, Dict, Iterable, List, Optional, Set

SLUG_MAX_LENGTH = 60
# Lower case letters and digits that cannot be mistaken for one another
_SUFFIX_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"


def dedupe(ids: Iterable[str]) -> List[str]:
    """Drop repeats and blanks, keeping the first occurrence's order."""
    seen, out = set(), []
    for i in ids:
        i = (i or "").strip()
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def current_membership(payload: Dict[str, Any]) -> List[str]:
    return list(payload.get("collection_ids") or [])


def is_auto_eligible(payload: Dict[str, Any]) -> bool:
    """Automatic assignment may only touch an image with no collection that has not opted out."""
    return not current_membership(payload) and not payload.get("collections_opt_out")


def membership_payload(ids: List[str], by_admin: bool) -> Dict[str, Any]:
    """
    Payload keys for a new membership list. An admin leaving an image in no collection is
    remembered as an opt-out; a list that came out empty by itself is not.
    """
    return {"collection_ids": ids, "collections_opt_out": bool(by_admin and not ids)}


def apply_membership(current: List[str], add: List[str], remove: List[str]) -> List[str]:
    drop = set(remove)
    return [i for i in dedupe([*current, *add]) if i not in drop]


def _fold(ch: str) -> str:
    """é -> e, ﬁ -> fi, Ａ -> A; a letter with no plain Latin form (が, ß) stays as it is."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
    return plain if plain.isascii() else ch


def slugify(title: str) -> str:
    """
    The readable part of a collection's address: its title in lower case with the words joined
    by hyphens ("Formula 1" -> "formula-1"). Letters of other scripts are kept ("四川之夏").
    """
    text = "".join(_fold(ch) for ch in unicodedata.normalize("NFC", title)).lower()
    text = re.sub(r"['’]", "", text)  # "it's" -> "its", not "it-s"
    slug = re.sub(r"[\W_]+", "-", text).strip("-")
    if len(slug) > SLUG_MAX_LENGTH:
        cut = slug[:SLUG_MAX_LENGTH]
        # End on a whole word when there is one to end on
        if slug[SLUG_MAX_LENGTH] != "-" and "-" in cut:
            cut = cut.rsplit("-", 1)[0]
        slug = cut.strip("-")
    return slug or "collection"


def is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


def unique_slug(base: str, taken: Set[str]) -> str:
    """`base`, or `base` with a short random suffix when it is taken. Never a UUID (ids are)."""
    slug = base
    while slug in taken or is_uuid(slug):
        slug = f"{base}-{''.join(random.choices(_SUFFIX_ALPHABET, k=4))}"
    return slug


def taken_slugs(points: Iterable[Any], except_id: Optional[str] = None) -> Set[str]:
    """Every address the collections answer to (current or from before a rename), but `except_id`'s."""
    taken: Set[str] = set()
    for p in points:
        if str(p.id) == except_id:
            continue
        payload = p.payload or {}
        if payload.get("slug"):
            taken.add(payload["slug"])
        taken.update(payload.get("former_slugs") or [])
    return taken
