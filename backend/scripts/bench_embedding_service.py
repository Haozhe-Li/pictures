"""Concurrency benchmark for the embedding service (dense text endpoint).

Sends one text per request, like the backend's search path does, at increasing
concurrency, and reports throughput and latency percentiles per level. Meant to be
run from inside the private network, so the numbers exclude public-internet latency.

Three phases run in order, each over every concurrency level:
  1. minilm : only paraphrase-multilingual-MiniLM-L12-v2
  2. jina   : only jina-clip-v1
  3. mixed  : both at once; concurrency N = N/2 workers per model, sending the SAME texts
              (e.g. conc 32 -> 16 jina + 16 MiniLM in flight). Each model is reported on its
              own row, plus a combined row. Needs conc >= 2.

  python -m scripts.bench_embedding_service
  python -m scripts.bench_embedding_service --levels 2,8,32 --requests 200
  python -m scripts.bench_embedding_service --phases mixed
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

MINILM = settings.TEXT_MODEL
JINA = settings.CLIP_MODEL
PHASES = {"minilm": [MINILM], "jina": [JINA], "mixed": [JINA, MINILM]}

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


def make_texts(counter: list, n: int) -> list:
    """n unique texts (=> no cache effects); the same list is reused for every model in a level."""
    out = []
    for _ in range(n):
        counter[0] += 1
        out.append(f"{TEXTS[counter[0] % len(TEXTS)]} {counter[0]}")
    return out


def run_model(base: str, model: str, conc: int, texts: list):
    """Send every text to `model` with `conc` workers. Returns ([(status, seconds)], wall seconds)."""
    local = threading.local()
    results = []
    lock = threading.Lock()

    def one(text: str):
        if not hasattr(local, "s"):  # one keep-alive connection per worker thread
            local.s = requests.Session()
        t = time.perf_counter()
        try:
            code = local.s.post(
                f"{base}/embed/dense/text", json={"texts": [text], "model": model}, timeout=120
            ).status_code
        except requests.RequestException as e:
            code = type(e).__name__
        dt = time.perf_counter() - t
        with lock:
            results.append((code, dt))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        list(ex.map(one, texts))
    return results, time.perf_counter() - t0


def run_level(base: str, models: list, conc: int, n: int, counter: list) -> list:
    """One concurrency level. One model: all `conc` workers on it, n requests. Several models:
    conc split evenly across them, n split evenly, same texts to each, all running at once.
    Returns [(model, workers, results, wall)]."""
    per = [conc // len(models) + (i < conc % len(models)) for i in range(len(models))]
    texts = make_texts(counter, max(1, n // len(models)))
    out = [None] * len(models)

    def work(i: int):
        out[i] = (models[i], per[i], *run_model(base, models[i], per[i], texts))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(models))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def fmt(results: list, wall: float, h: dict) -> str:
    ok = [dt * 1000 for code, dt in results if code == 200]
    bad = dict(Counter(code for code, _ in results if code != 200))
    stats = (
        f"{statistics.mean(ok):6.0f} {pct(ok, .5):6.0f} {pct(ok, .95):6.0f} {pct(ok, .99):6.0f} {max(ok):6.0f}"
        if ok
        else "     -      -      -      -      -"
    )
    return (
        f"{len(results):>4} {len(ok):>4} {len(ok) / wall:7.1f} | {stats}      | "
        f"{bad or '-'} | {h.get('rss_mb')}MB q={h.get('queue_depth')}"
    )


def short(model: str) -> str:
    return "jina" if model == JINA else "minilm" if model == MINILM else model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=settings.EMBEDDING_SERVICE_URL)
    ap.add_argument("--phases", default="minilm,jina,mixed", help=f"comma-separated, from {list(PHASES)}")
    ap.add_argument("--levels", default="1,2,4,8,16,32,64,128", help="comma-separated total concurrency levels")
    ap.add_argument("--requests", type=int, default=0, help="total requests per level (default: max(60, 4*concurrency)); split evenly in mixed")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds to rest between levels")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    levels = [int(x) for x in args.levels.split(",")]
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    for p in phases:
        if p not in PHASES:
            ap.error(f"unknown phase {p!r}; choose from {list(PHASES)}")

    h = health(base)
    print(f"target: {base}")
    print(f"service dense_model={h.get('dense_model')} queue_max={h.get('queue_max')} rss={h.get('rss_mb')}MB queue={h.get('queue_depth')}")
    counter = [0]
    header = f"{'conc':>4} {'model':>8} {'work':>4} {'n':>4} {'ok':>4} {'req/s':>7} | {'mean':>6} {'p50':>6} {'p95':>6} {'p99':>6} {'max':>6} (ms) | not-200 | rss  q-depth"

    for phase in phases:
        models = PHASES[phase]
        print(f"\n=== phase: {phase} ({' + '.join(short(m) for m in models)}) ===")
        for m in models:  # warm up connections and ONNX
            for _ in range(5):
                requests.post(f"{base}/embed/dense/text", json={"texts": ["warm up"], "model": m}, timeout=60)
        print(header)
        for conc in levels:
            if conc < len(models):
                print(f"{conc:>4} skipped (needs concurrency >= {len(models)})")
                continue
            n = args.requests or max(60, conc * 4)
            rows = run_level(base, models, conc, n, counter)
            h = health(base)
            for model, workers, results, wall in rows:
                print(f"{conc:>4} {short(model):>8} {workers:>4} " + fmt(results, wall, h), flush=True)
            if len(rows) > 1:  # combined: all requests over the longer of the two walls
                allres = [r for _, _, res, _ in rows for r in res]
                print(f"{conc:>4} {'combined':>8} {conc:>4} " + fmt(allres, max(w for *_, w in rows), h), flush=True)
            time.sleep(args.pause)

    print("\n503 = the service's queue was full (it is protecting itself); latency at high concurrency is mostly queue wait.")


if __name__ == "__main__":
    main()
