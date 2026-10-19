"""Local load test for the API: latency and throughput at several concurrency levels.

Sends real sample DICOMs to a running server (e.g. the Docker container) from N concurrent
clients, and reports client-side p50 / p95 / p99 latency, requests per second, errors, and the
batch sizes the server reports. Run once with batching on and once with it off to compare.

    uv run python -m brainct.serving.loadtest --url http://127.0.0.1:8000 --label batching_on

Needs `httpx` (in the dev dependency group), so it is not part of the API image.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SAMPLES = Path("samples/dicom")


def summarize(latencies_ms: list[float], seconds: float, errors: int) -> dict[str, Any]:
    values = np.array(latencies_ms) if latencies_ms else np.array([np.nan])
    return {
        "requests": len(latencies_ms) + errors,
        "errors": errors,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "requests_per_s": len(latencies_ms) / seconds if seconds > 0 else 0.0,
    }


async def _level(client, url: str, files: list[bytes], concurrency: int, total: int):
    """`total` requests from `concurrency` clients, each sending one request at a time."""
    latencies: list[float] = []
    batch_sizes: Counter[int] = Counter()
    errors = 0
    counter = iter(range(total))

    async def worker() -> None:
        nonlocal errors
        for k in counter:
            data = files[k % len(files)]
            start = time.perf_counter()
            response = await client.post(f"{url}/predict", files={"file": ("slice.dcm", data)})
            elapsed = (time.perf_counter() - start) * 1000
            if response.status_code == 200:
                latencies.append(elapsed)
                batch_sizes[response.json()["timing"]["batch_size"]] += 1
            else:
                errors += 1

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    result = summarize(latencies, time.perf_counter() - start, errors)
    result["concurrency"] = concurrency
    result["batch_sizes"] = {str(k): v for k, v in sorted(batch_sizes.items())}
    return result


async def run(
    url: str, sample_dir: Path, levels: list[int], requests: int, warmup: int
) -> dict[str, Any]:
    import httpx  # dev dependency; imported here so the API image never needs it

    files = [p.read_bytes() for p in sorted(sample_dir.glob("*.dcm"))]
    if not files:
        raise SystemExit(f"no .dcm files in {sample_dir}")
    async with httpx.AsyncClient(timeout=60) as client:
        ready = (await client.get(f"{url}/ready")).json()
        await _level(client, url, files, 1, warmup)  # warm connections and caches, not reported
        results = []
        for concurrency in levels:
            result = await _level(client, url, files, concurrency, requests)
            print(
                f"concurrency {concurrency:3d}: p50 {result['p50_ms']:7.1f} ms  "
                f"p95 {result['p95_ms']:7.1f} ms  {result['requests_per_s']:6.1f} req/s  "
                f"errors {result['errors']}  batches {result['batch_sizes']}",
                flush=True,
            )
            results.append(result)
    return {"url": url, "server": ready, "requests_per_level": requests, "levels": results}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--requests", type=int, default=300, help="requests per level")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--label", default="run", help="name for the output file")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/loadtest"))
    args = parser.parse_args(argv)

    report = asyncio.run(run(args.url, args.samples, args.concurrency, args.requests, args.warmup))
    report["label"] = args.label
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.label}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
