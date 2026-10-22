"""Per-step load-test summary (scripts/summarize_locust.py) on a made-up request log."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts/summarize_locust.py"
_spec = importlib.util.spec_from_file_location("summarize_locust", _PATH)
summarize_locust = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = summarize_locust  # @dataclass looks the module up here
_spec.loader.exec_module(summarize_locust)


def _step(users: int, start: float, seconds: int, latency_ms: float, failures: int = 0):
    """10 requests per second for `seconds`, all with the same latency; first `failures` fail."""
    n = 10 * seconds
    t = start + np.arange(n) / 10
    ok = np.ones(n)
    ok[:failures] = 0
    return np.column_stack([t, np.full(n, latency_ms), np.full(n, users), ok])


def test_steps_drop_warmup_and_measure_each_step():
    log = np.vstack(
        [
            # warm-up requests are slow and must not count
            _step(1, 0, 5, latency_ms=999),
            _step(1, 5, 30, latency_ms=20),
            _step(2, 35, 5, latency_ms=999),
            _step(2, 40, 30, latency_ms=30),
        ]
    )

    steps = summarize_locust.summarize(log, warmup_s=5)

    assert [s.users for s in steps] == [1, 2]
    assert steps[0].p95_ms == pytest.approx(20) and steps[1].p95_ms == pytest.approx(30)
    assert steps[0].req_per_s == pytest.approx(10, rel=0.01)
    assert all(s.error_pct == 0 for s in steps)


def test_max_sustained_uses_p95_factor_and_error_limit(tmp_path):
    log = np.vstack(
        [
            _step(1, 0, 20, latency_ms=20),
            _step(2, 20, 20, latency_ms=39),  # within 2x
            _step(4, 40, 20, latency_ms=35, failures=150),  # fast but too many errors
            _step(8, 60, 20, latency_ms=41),  # above 2x 20 ms
            _step(0, 80, 2, latency_ms=10),  # stopping: ignored
        ]
    )
    path = tmp_path / "run_requests.csv"
    np.savetxt(path, log, delimiter=",", header="time_s,latency_ms,users,ok", comments="")

    steps = summarize_locust.summarize(summarize_locust.load_requests(path), warmup_s=0)
    best = summarize_locust.max_sustained(steps)

    assert [s.users for s in steps] == [1, 2, 4, 8]
    assert steps[2].error_pct == pytest.approx(75)
    assert best is not None and best.users == 2
    table = summarize_locust.to_markdown(steps, best, 2.0, 1.0)
    assert "| 2 **(max sustained)** |" in table
    assert "Max sustained: " in table and "at 2 users" in table
