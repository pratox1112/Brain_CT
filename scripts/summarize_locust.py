"""Per-step load-test numbers from the Locust request log (load_tests/locustfile.py).

Reads PREFIX_requests.csv (one row per request: time_s, latency_ms, users, ok) and prints a
Markdown table for docs/results.md: req/s, p50/p95/p99 and error % for each user step.

- The first --warmup seconds of each step are dropped (users still spawning, and requests
  started under the previous step still finishing).
- req/s counts successful requests only. Percentiles are exact, from every successful
  request in the step (numpy's default linear method).
- "Max sustained" = the highest step whose p95 is at most --factor x the lowest step's p95
  and whose error rate is at most --max-errors %. Decided before measuring (Phase 9 plan).

    uv run python scripts/summarize_locust.py load_tests/results/local-k8s/run_requests.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Step:
    users: int
    seconds: float  # length of the measured window, after warm-up
    requests: int
    req_per_s: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    error_pct: float


def load_requests(path: Path) -> np.ndarray:
    """Columns: time_s, latency_ms, users, ok."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))[1:]  # skip the header
    return np.array(rows, dtype=float).reshape(-1, 4)


def summarize(requests: np.ndarray, warmup_s: float = 15.0) -> list[Step]:
    steps = []
    for users in sorted(set(requests[:, 2].astype(int)) - {0}):  # 0 = stopping
        step = requests[requests[:, 2] == users]
        start, end = step[:, 0].min(), step[:, 0].max()
        window = step[step[:, 0] >= start + warmup_s]
        seconds = end - (start + warmup_s)
        if len(window) == 0 or seconds <= 0:
            continue  # step shorter than the warm-up: nothing to measure
        ok = window[:, 3] == 1
        # Latency percentiles over successful requests; failures are counted in error %.
        latency = window[ok, 1] if ok.any() else window[:, 1]
        p50, p95, p99 = np.percentile(latency, [50, 95, 99])
        steps.append(
            Step(
                users=users,
                seconds=float(seconds),
                requests=len(window),
                req_per_s=float(ok.sum() / seconds),
                p50_ms=float(p50),
                p95_ms=float(p95),
                p99_ms=float(p99),
                error_pct=float(100 * (~ok).mean()),
            )
        )
    return steps


def max_sustained(steps: list[Step], factor: float = 2.0, max_errors: float = 1.0) -> Step | None:
    if not steps:
        return None
    limit = factor * steps[0].p95_ms
    ok = [s for s in steps if s.p95_ms <= limit and s.error_pct <= max_errors]
    return max(ok, key=lambda s: s.users) if ok else None


def to_markdown(steps: list[Step], best: Step | None, factor: float, max_errors: float) -> str:
    lines = [
        "| Users | Window (s) | Requests | req/s | p50 (ms) | p95 (ms) | p99 (ms) | Errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in steps:
        mark = " **(max sustained)**" if s == best else ""
        lines.append(
            f"| {s.users}{mark} | {s.seconds:.0f} | {s.requests} | {s.req_per_s:.1f} "
            f"| {s.p50_ms:.0f} | {s.p95_ms:.0f} | {s.p99_ms:.0f} | {s.error_pct:.2f}% |"
        )
    if steps:
        lines.append("")
        lines.append(
            f"Degraded = p95 above {factor:g} x {steps[0].p95_ms:.0f} ms "
            f"(the {steps[0].users}-user p95) or errors above {max_errors:g}%."
        )
        if best is None:
            lines.append("No step met the limit.")
        else:
            lines.append(
                f"Max sustained: {best.req_per_s:.1f} req/s at {best.users} "
                f"user{'s' if best.users != 1 else ''} "
                f"(p95 {best.p95_ms:.0f} ms)."
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("requests_csv", type=Path)
    parser.add_argument("--warmup", type=float, default=15.0, help="seconds dropped per step")
    parser.add_argument("--factor", type=float, default=2.0, help="p95 limit vs the first step")
    parser.add_argument("--max-errors", type=float, default=1.0, help="error-rate limit, %%")
    args = parser.parse_args()

    steps = summarize(load_requests(args.requests_csv), args.warmup)
    best = max_sustained(steps, args.factor, args.max_errors)
    print(to_markdown(steps, best, args.factor, args.max_errors))


if __name__ == "__main__":
    main()
