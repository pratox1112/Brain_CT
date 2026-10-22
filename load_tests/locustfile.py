"""Step load test for POST /predict. Research use only.

Uploads synthetic 512x512 CT DICOMs (built by tests/dicom_factory.py, no RSNA data), the same
size as real RSNA slices, so upload and decode cost match real traffic.

Each user sends its next request as soon as the previous one returns (no wait time), so the
number of users is the number of requests in flight. Users step up through BRAINCT_STEP_USERS,
holding each step for BRAINCT_STEP_SECONDS, then the test stops.

Run (laptop, against the local API or Kubernetes Service):
  uv run --group loadtest locust -f load_tests/locustfile.py --headless \
      --host http://127.0.0.1:8080 --csv load_tests/results/local --csv-full-history
Use 127.0.0.1, not localhost: on Windows, localhost tries IPv6 (::1) first, and Docker
Desktop accepts that connection but never answers it.
Quick check:  set BRAINCT_STEP_USERS=1,2 and BRAINCT_STEP_SECONDS=15 first.

With --csv PREFIX, every request is also written to PREFIX_requests.csv (time, latency, users,
ok) when the test stops; scripts/summarize_locust.py turns it into exact per-step numbers.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from locust import FastHttpUser, LoadTestShape, constant, events, task

# dicom_factory lives in tests/ in the repo; the load-test image copies it next to this file.
_TESTS = Path(__file__).resolve().parent.parent / "tests"
if _TESTS.is_dir():
    sys.path.insert(0, str(_TESTS))
from dicom_factory import make_dicom_bytes  # noqa: E402

STEP_USERS = [int(n) for n in os.environ.get("BRAINCT_STEP_USERS", "1,2,4,8,16,32").split(",")]
STEP_SECONDS = int(os.environ.get("BRAINCT_STEP_SECONDS", "120"))
N_IMAGES = 8
BOUNDARY = "brainct-load-test-boundary"


def _multipart(name: str, data: bytes) -> bytes:
    """One-file multipart/form-data body, the same request as the API docs' "Try it out"."""
    head = (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
        "Content-Type: application/dicom\r\n\r\n"
    ).encode()
    return head + data + f"\r\n--{BOUNDARY}--\r\n".encode()


def _bodies() -> list[bytes]:
    # Stored values 0..2000 with intercept -1024: roughly -1024..976 HU, a plausible CT range.
    rng = np.random.default_rng(2026)
    return [
        _multipart(
            f"synthetic_{i}.dcm",
            make_dicom_bytes(f"ID_load{i}", pixels=rng.integers(0, 2000, (512, 512))),
        )
        for i in range(N_IMAGES)
    ]


BODIES = _bodies()  # built once at startup, so the load generator only sends bytes

# One row per request, kept in memory and written once at the end so logging doesn't add
# disk writes during the test (a 12-minute run is ~30k rows, about 1 MB).
REQUEST_LOG: list[tuple[float, float, int, int]] = []
_STATE: dict = {"env": None}


@events.init.add_listener
def _keep_environment(environment, **_kwargs) -> None:
    _STATE["env"] = environment


@events.request.add_listener
def _log_request(response_time: float, exception: BaseException | None, **_kwargs) -> None:
    # users = the step this request finished in; ok = 0 for any failure (incl. non-2xx).
    runner = _STATE["env"].runner if _STATE["env"] is not None else None
    users = runner.user_count if runner is not None else 0
    ok = int(exception is None)
    REQUEST_LOG.append((round(time.time(), 3), round(response_time, 2), users, ok))


@events.test_stop.add_listener
def _write_request_log(environment, **_kwargs) -> None:
    prefix = getattr(environment.parsed_options, "csv_prefix", None)
    if not prefix:
        return
    with open(f"{prefix}_requests.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "latency_ms", "users", "ok"])
        writer.writerows(REQUEST_LOG)


class PredictUser(FastHttpUser):
    wait_time = constant(0)

    def on_start(self) -> None:
        self.i = 0

    @task
    def predict(self) -> None:
        body = BODIES[self.i % N_IMAGES]
        self.i += 1
        # Any status other than 2xx counts as a failure (Locust's default).
        self.client.post(
            "/predict",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
            name="/predict",
        )


class StepLoad(LoadTestShape):
    """1 -> 2 -> 4 -> ... users, each step held for STEP_SECONDS, then stop."""

    def tick(self) -> tuple[int, float] | None:
        step = int(self.get_run_time() // STEP_SECONDS)
        if step >= len(STEP_USERS):
            return None
        users = STEP_USERS[step]
        return users, users  # spawn the whole step at once
