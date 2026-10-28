"""Find the photos in the gallery that look most alike: to choose DUPLICATE_THRESHOLD, and to
spot photos that were uploaded twice before uploads were checked.

For every image, the closest other image by CLIP (`dense-image`, cosine) is looked up, the same
comparison the upload check makes. Prints how those closest scores are spread, then the closest
pairs with titles, times and preview links to compare by eye. Only reads.

The same file uploaded twice scores about 1.0 and a re-export (resized, re-encoded) a little
less, but neighbouring burst shots can score nearly as high: look at the pairs around the
threshold before changing it.

Env: QDRANT_URL / QDRANT_API_KEY (as for the backend).

Run from backend/:
  python -m scripts.find_duplicates                    # pairs scoring 0.9 or more
  python -m scripts.find_duplicates --min 0.95 --top 50
"""

import argparse
from typing import Dict, List, Tuple

from qdrant_client import QdrantClient, models

from core.config import settings

BUCKETS = [0.99, 0.97, 0.95, 0.93, 0.9]


def load_images(client: QdrantClient) -> list:
    """Every image with its CLIP vector and the payload fields the report shows."""
    points, offset = [], None
    while True:
        page, offset = client.scroll(
            collection_name=settings.COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=["title", "taken_time", "preview_url"],
            with_vectors=["dense-image"],
        )
        points.extend(page)
        if offset is None:
            return points


def nearest_neighbours(client: QdrantClient, points: list) -> Dict[str, Tuple[float, str]]:
    """For each image id, the score of the closest other image and that image's id."""
    nearest: Dict[str, Tuple[float, str]] = {}
    for i in range(0, len(points), 64):
        batch = points[i : i + 64]
        responses = client.query_batch_points(
            collection_name=settings.COLLECTION_NAME,
            requests=[
                models.QueryRequest(
                    query=p.vector["dense-image"],
                    using="dense-image",
                    filter=models.Filter(must_not=[models.HasIdCondition(has_id=[p.id])]),
                    limit=1,
                )
                for p in batch
            ],
        )
        for p, response in zip(batch, responses):
            if response.points:
                best = response.points[0]
                nearest[str(p.id)] = (best.score, str(best.id))
    return nearest


def report(points: list, nearest: Dict[str, Tuple[float, str]], min_score: float, top: int) -> List[str]:
    payloads = {str(p.id): p.payload or {} for p in points}
    scores = [score for score, _ in nearest.values()]
    lines = [f"{len(points)} images. Score of each image's closest other image (CLIP cosine):"]
    upper = None
    for bound in BUCKETS:
        count = sum(1 for s in scores if s >= bound and (upper is None or s < upper))
        span = f">= {bound:.2f}" if upper is None else f"{bound:.2f}-{upper:.2f}"
        lines.append(f"  {span:>11}  {count}")
        upper = bound
    lines.append(f"  {'< ' + format(BUCKETS[-1], '.2f'):>11}  {sum(1 for s in scores if s < BUCKETS[-1])}")
    flagged = sum(1 for s in scores if s >= settings.DUPLICATE_THRESHOLD)
    lines.append(
        f"At DUPLICATE_THRESHOLD={settings.DUPLICATE_THRESHOLD}, {flagged} of these images would be "
        "held back if uploaded again now."
    )

    # Each pair once, closest first
    pairs: Dict[Tuple[str, str], float] = {}
    for a, (score, b) in nearest.items():
        if score >= min_score:
            pairs[(min(a, b), max(a, b))] = score
    ranked = sorted(pairs.items(), key=lambda item: -item[1])
    lines.append("")
    lines.append(f"{len(ranked)} pair(s) at {min_score} or more" + (f", closest {top}:" if len(ranked) > top else ":"))

    def describe(image_id: str) -> str:
        payload = payloads.get(image_id, {})
        when = f" ({payload['taken_time']})" if payload.get("taken_time") else ""
        return f"{payload.get('title') or 'Untitled'}{when}  {image_id}\n          {payload.get('preview_url', '')}"

    for (a, b), score in ranked[:top]:
        lines.append(f"  {score:.4f}  {describe(a)}")
        lines.append(f"          {describe(b)}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min", type=float, default=0.9, help="list pairs scoring at least this (default 0.9)")
    parser.add_argument("--top", type=int, default=30, help="list at most this many pairs (default 30)")
    args = parser.parse_args()

    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    points = load_images(client)
    print("\n".join(report(points, nearest_neighbours(client, points), args.min, args.top)))


if __name__ == "__main__":
    main()
