"""
Photo times as something Qdrant can filter on.

`taken_time` is the text that is shown and embedded ("2024-07-15 14:32:10", the camera's own
clock). `taken_at` is the same moment in RFC 3339 ("2024-07-15T14:32:10Z"), kept beside it under
a datetime index. EXIF times carry no time zone, so they are stored as if they were UTC, the way
the site already shows them; whole-day windows are unaffected.

A collection can have a window of whole days (`window_start`, `window_end`, both inclusive and
either one open). Only photos taken inside it are offered for it or matched into it.
"""

from datetime import date, datetime, timedelta, timezone
from typing import Optional, Tuple

_FORMATS = (
    "%Y-%m-%d %H:%M:%S",  # what the upload form fills in from EXIF
    "%Y:%m:%d %H:%M:%S",  # raw EXIF
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
)
_RFC3339 = "%Y-%m-%dT%H:%M:%SZ"


def to_taken_at(taken_time: Optional[str]) -> Optional[str]:
    """`taken_time` as RFC 3339 in UTC, or None when it is empty or not a time."""
    text = (taken_time or "").strip()
    if not text:
        return None
    for fmt in _FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime(_RFC3339)
        except ValueError:
            pass
    try:
        # Python 3.10 (the deployed one) does not read a trailing "Z" itself
        moment = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError:
        return None
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)
    return moment.strftime(_RFC3339)


def parse_day(value: Optional[str]) -> Optional[str]:
    """A window bound as "YYYY-MM-DD", None when blank. Raises ValueError for anything else."""
    text = (value or "").strip()
    if not text:
        return None
    return date.fromisoformat(text).isoformat()


def window_bounds(start: Optional[str], end: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """RFC 3339 bounds of a window of whole days: from the start of `start` up to (not
    including) the day after `end`. A missing side stays None (open)."""
    lower = f"{start}T00:00:00Z" if start else None
    upper = (date.fromisoformat(end) + timedelta(days=1)).strftime(_RFC3339) if end else None
    return lower, upper


def in_window(taken_at: Optional[str], start: Optional[str], end: Optional[str]) -> bool:
    """Whether a photo falls in a window. Everything does when there is no window; a photo
    without a time falls in none."""
    if not start and not end:
        return True
    if not taken_at:
        return False
    lower, upper = window_bounds(start, end)
    # Same fixed-width format on both sides, so plain string comparison orders them
    return (lower is None or taken_at >= lower) and (upper is None or taken_at < upper)
