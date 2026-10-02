"""Concurrency benchmark for the embedding service (dense text endpoint, jina-clip-v1 only).

Sends one text per request, like the backend's search path does, at increasing
concurrency, and reports throughput and latency percentiles per level. Meant to be
run from inside the private network, so the numbers exclude public-internet latency.

  python -m scripts.bench_embedding_service
  python -m scripts.bench_embedding_service --levels 1,8,32 --requests 200
  python -m scripts.bench_embedding_service --url http://embedding.railway.internal:8080

Uses EMBEDDING_SERVICE_URL (the same setting the backend uses) unless --url is given.
Only needs `requests`. It adds real load to the service; do not run it during busy hours.
"""

import argparse
import math
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

from core.config import settings

MODEL = settings.CLIP_MODEL  # jina-clip-v1 only; MiniLM is deliberately not benchmarked

TEXTS = [
    "A golden retriever running through a field of sunflowers",
    "The quarterly earnings report exceeded analyst expectations",
    "How to bake sourdough bread at home with a cast iron pot",
    "A red sports car parked in front of a modern glass building",
    "Quantum computers use qubits that can exist in superposition",
    "Two children building a sandcastle on the beach at sunset",
    "The Federal Reserve raised interest rates by a quarter point",
    "A bowl of ramen with soft boiled egg, pork belly and scallions",
    "Snow-capped mountains reflected in a calm alpine lake",
    "Best practices for securing a PostgreSQL database in production",
    "A cat sleeping on a windowsill in the morning sun",
    "The history of the Roman Empire from Augustus to Constantine",
    "A crowded night market in Taipei with neon signs and street food",
    "Gradient descent minimizes a loss function by following its slope",
    "An astronaut floating outside the International Space Station",
    "A cozy wooden cabin in the woods surrounded by autumn leaves",
    "How do vaccines train the immune system to recognize pathogens?",
    "A barista pouring latte art in a small independent coffee shop",
    "The transformer architecture replaced recurrent networks in NLP",
    "A vintage bicycle leaning against a brick wall covered in ivy",
    "monkeys and apes",
    "Golden Gate Bridge",
    "red torii gate in Japan",
    "fighter jets at an airshow",
    "Hong Kong skyline at night",
    "seagulls on the beach at sunset",
    "a man flying a drone over a cliff",
    "neon lights in Tokyo",
]


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def health(base: str) -> dict:
    r = requests.get(f"{base}/health", timeout=15)
    r.raise_for_status()
    return r.json()


def run_level(base: str, conc: int, n: int, counter: list):
    local = threading.local()
    results = []  # (status, seconds)
    lock = threading.Lock()

    def one(i: int):
        if not hasattr(local, "s"):  # one keep-alive connection per worker thread
            local.s = requests.Session()
        with lock:
            counter[0] += 1
            text = f"{TEXTS[counter[0] % len(TEXTS)]} {counter[0]}"  # unique => no cache effects
        t = time.perf_counter()
        try:
            code = local.s.post(
                f"{base}/embed/dense/text", json={"texts": [text], "model": MODEL}, timeout=120
            ).status_code
        except requests.RequestException as e:
            code = type(e).__name__
        dt = time.perf_counter() - t
        with lock:
            results.append((code, dt))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        list(ex.map(one, range(n)))
    return results, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=settings.EMBEDDING_SERVICE_URL)
    ap.add_argument("--levels", default="1,2,4,8,16,32,64,128", help="comma-separated concurrency levels")
    ap.add_argument("--requests", type=int, default=0, help="requests per level (default: max(60, 4*concurrency))")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds to rest between levels")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    h = health(base)
    print(f"target: {base}")
    print(f"model={h.get('dense_model')} queue_max={h.get('queue_max')} rss={h.get('rss_mb')}MB queue={h.get('queue_depth')}")
    counter = [0]
    for _ in range(5):  # warm up connections and ONNX
        requests.post(f"{base}/embed/dense/text", json={"texts": ["warm up"], "model": MODEL}, timeout=60)

    print(f"\n{'conc':>4} {'n':>4} {'ok':>4} {'req/s':>7} | {'mean':>6} {'p50':>6} {'p95':>6} {'p99':>6} {'max':>6} (ms) | not-200 | rss  q-depth")
    for conc in [int(x) for x in args.levels.split(",")]:
        n = args.requests or max(60, conc * 4)
        results, wall = run_level(base, conc, n, counter)
        ok = [dt * 1000 for code, dt in results if code == 200]
        bad = dict(Counter(code for code, _ in results if code != 200))
        h = health(base)
        stats = (
            f"{statistics.mean(ok):6.0f} {pct(ok, .5):6.0f} {pct(ok, .95):6.0f} {pct(ok, .99):6.0f} {max(ok):6.0f}"
            if ok
            else "     -      -      -      -      -"
        )
        print(
            f"{conc:>4} {len(results):>4} {len(ok):>4} {len(ok) / wall:7.1f} | {stats}      | "
            f"{bad or '-'} | {h.get('rss_mb')}MB q={h.get('queue_depth')}",
            flush=True,
        )
        time.sleep(args.pause)

    print("\n503 = the service's queue was full (it is protecting itself); latency at high concurrency is mostly queue wait.")


if __name__ == "__main__":
    main()
