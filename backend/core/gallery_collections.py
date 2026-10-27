"""
Pure helpers for collection membership. An image's membership is two payload keys:

- `collection_ids`: the collections it belongs to (an image can be in several).
- `collections_opt_out`: true once an admin has deliberately left it in none, so automatic
  assignment must not put it back. Automatic assignment only ever applies to an image that
  has no collection and has not opted out.
"""

from typing import Any, Dict, Iterable, List


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
