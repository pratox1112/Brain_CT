"""Phase 5b benchmark harness: timing logic, sample agreement, and a small end-to-end run."""

from __future__ import annotations

import json

import numpy as np
import torch

from brainct.data.labels import LABELS
from brainct.optimization import benchmark as bench
from brainct.optimization.export_onnx import export_onnx
from brainct.training.model import build_model


def test_time_batches_excludes_warmup_and_counts_iterations():
    calls = []

    def predict(batch):
        calls.append(len(batch))
        return np.zeros((len(batch), 6), dtype=np.float32)

    times = bench.time_batches(predict, np.zeros((4, 3, 8, 8), np.float32), warmup=3, iterations=5)

    assert len(calls) == 8  # 3 warmup + 5 timed
    assert times.shape == (5,) and (times >= 0).all()


def test_benchmark_reports_percentiles_and_throughput_per_batch_size():
    def predict(batch):
        return np.zeros((len(batch), 6), dtype=np.float32)

    rows = bench.benchmark(predict, 8, [1, 4], {1: 6, 4: 3}, warmup=1, repeats=2)

    assert [r["batch_size"] for r in rows] == [1, 4]
    for row in rows:
        assert len(row["repeats"]) == 2
        assert row["p50_ms"] <= row["p95_ms"] or np.isclose(row["p50_ms"], row["p95_ms"])
        assert row["images_per_s"] > 0


def test_cpu_info_has_core_fields():
    info = bench.cpu_info()

    assert info["logical_cpus"] and "vnni" in info and "machine" in info


def test_run_end_to_end_with_pytorch_and_onnx(tmp_path, dicom_bytes):
    size = 32
    artifacts = tmp_path / "artifacts"
    (artifacts / "onnx").mkdir(parents=True)
    torch.manual_seed(0)
    model = build_model(pretrained=False).eval()
    torch.save({"model": model.state_dict()}, artifacts / "best.pt")
    export_onnx(model, artifacts / "onnx" / "model_fp32.onnx", image_size=size)
    thresholds = dict.fromkeys(LABELS, 0.5)
    (artifacts / "metrics_val.json").write_text(json.dumps({"thresholds": thresholds}))
    samples = tmp_path / "samples"
    samples.mkdir()
    for k in range(3):
        pixels = np.full((16, 16), 1024 + 40 * k)
        (samples / f"ID_s{k}.dcm").write_bytes(dicom_bytes(f"ID_s{k}", pixels=pixels))

    report = bench.run(
        artifacts,
        variants=["pytorch_fp32", "onnx_fp32"],
        threads=2,
        batch_sizes=(1, 2),
        iterations=(3, 2),
        warmup=1,
        repeats=1,
        image_size=size,
        sample_dirs=[samples],
        out=tmp_path / "bench.json",
    )

    assert set(report["timings"]) == {"pytorch_fp32", "onnx_fp32"}
    agreement = report["sample_agreement_vs_pytorch_fp32"]
    assert agreement["images"] == 3
    assert agreement["onnx_fp32"]["max_abs_prob_diff"] < 1e-4  # FP32 export matches PyTorch
    assert sum(agreement["onnx_fp32"]["decision_flips"].values()) == 0
    assert json.loads((tmp_path / "bench.json").read_text())["threads"] == 2
