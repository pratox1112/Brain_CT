"""Micro-batcher tests, plus the API with batching on under concurrent requests."""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn

from brainct.constants import LABELS
from brainct.optimization.export_onnx import export_onnx
from brainct.serving.app import create_app
from brainct.serving.batching import MicroBatcher
from brainct.serving.loadtest import summarize
from brainct.serving.settings import Settings


def _row_sums(batch: np.ndarray) -> np.ndarray:
    """A fake model: 6 outputs per image, each equal to the image's pixel sum."""
    return np.repeat(batch.reshape(len(batch), -1).sum(axis=1, keepdims=True), 6, axis=1)


def test_concurrent_requests_share_a_batch_and_get_their_own_results():
    sizes = []

    async def scenario():
        batcher = MicroBatcher(_row_sums, max_batch=8, max_wait_ms=50, on_batch=sizes.append)
        await batcher.start()
        inputs = [np.full((1, 2, 2), float(k)) for k in range(5)]
        results = await asyncio.gather(*(batcher.predict(x) for x in inputs))
        await batcher.stop()
        return results

    results = asyncio.run(scenario())

    for k, (probs, batch_size) in enumerate(results):
        assert np.allclose(probs, 4.0 * k)  # each request gets *its* image's output
        assert batch_size == 5
    assert sizes == [5]


def test_batches_are_capped_at_max_batch():
    sizes = []

    async def scenario():
        batcher = MicroBatcher(_row_sums, max_batch=3, max_wait_ms=50, on_batch=sizes.append)
        await batcher.start()
        await asyncio.gather(*(batcher.predict(np.zeros((1, 1, 1))) for _ in range(7)))
        await batcher.stop()

    asyncio.run(scenario())

    assert sum(sizes) == 7 and max(sizes) <= 3


def test_a_lone_request_waits_at_most_max_wait():
    async def scenario():
        batcher = MicroBatcher(_row_sums, max_batch=8, max_wait_ms=20)
        await batcher.start()
        start = time.perf_counter()
        _, batch_size = await batcher.predict(np.zeros((1, 1, 1)))
        elapsed = time.perf_counter() - start
        await batcher.stop()
        return batch_size, elapsed

    batch_size, elapsed = asyncio.run(scenario())

    assert batch_size == 1
    assert elapsed < 1.0  # 20 ms wait plus overhead; generous for slow CI machines


def test_model_errors_reach_every_waiting_request():
    def broken(batch):
        raise ValueError("model exploded")

    async def scenario():
        batcher = MicroBatcher(broken, max_batch=4, max_wait_ms=20)
        await batcher.start()
        outcomes = await asyncio.gather(
            *(batcher.predict(np.zeros((1, 1, 1))) for _ in range(3)), return_exceptions=True
        )
        await batcher.stop()
        return outcomes

    outcomes = asyncio.run(scenario())

    assert all(isinstance(o, ValueError) for o in outcomes)


def test_invalid_max_batch_is_rejected():
    with pytest.raises(ValueError, match="max_batch"):
        MicroBatcher(_row_sums, max_batch=0)


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(4, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).mean(dim=(2, 3)))


def test_api_with_batching_matches_unbatched_results(tmp_path, dicom_bytes):
    torch.manual_seed(0)
    model_path = tmp_path / "tiny.onnx"
    export_onnx(TinyNet().eval(), model_path, image_size=32)
    thresholds_path = tmp_path / "metrics_val.json"
    thresholds_path.write_text(json.dumps({"thresholds": dict.fromkeys(LABELS, 0.5)}))
    rng = np.random.default_rng(0)
    uploads = [dicom_bytes(f"ID_{k}", pixels=rng.integers(0, 2000, (16, 16))) for k in range(8)]

    def responses(batching: bool) -> list[dict]:
        settings = Settings(
            model_path=model_path,
            thresholds_path=thresholds_path,
            threads=1,
            batching=batching,
            max_wait_ms=100,
        )
        with TestClient(create_app(settings)) as client, ThreadPoolExecutor(8) as pool:

            def post(data):
                return client.post("/predict", files={"file": ("x.dcm", data)}).json()

            return list(pool.map(post, uploads))

    batched, single = responses(True), responses(False)

    for a, b in zip(batched, single, strict=True):
        for label in LABELS:
            assert a["probabilities"][label] == pytest.approx(b["probabilities"][label], abs=1e-5)
    assert all(r["timing"]["batch_size"] == 1 for r in single)
    assert max(r["timing"]["batch_size"] for r in batched) > 1  # requests were really grouped


def test_ready_reports_batching_mode(tmp_path):
    torch.manual_seed(0)
    model_path = tmp_path / "tiny.onnx"
    export_onnx(TinyNet().eval(), model_path, image_size=32)
    thresholds_path = tmp_path / "metrics_val.json"
    thresholds_path.write_text(json.dumps({"thresholds": dict.fromkeys(LABELS, 0.5)}))
    settings = Settings(model_path=model_path, thresholds_path=thresholds_path, batching=False)

    with TestClient(create_app(settings)) as client:
        assert client.get("/ready").json()["batching"] == "off"


def test_settings_parse_batching_from_environment(monkeypatch):
    monkeypatch.setenv("BRAINCT_BATCHING", "false")
    monkeypatch.setenv("BRAINCT_MAX_BATCH", "16")
    monkeypatch.setenv("BRAINCT_MAX_WAIT_MS", "2.5")

    settings = Settings.from_env()

    assert (settings.batching, settings.max_batch, settings.max_wait_ms) == (False, 16, 2.5)


def test_loadtest_summary():
    summary = summarize([10.0, 20.0, 30.0, 40.0], seconds=2.0, errors=1)

    assert summary["requests"] == 5 and summary["errors"] == 1
    assert summary["p50_ms"] == 25.0
    assert summary["requests_per_s"] == 2.0
