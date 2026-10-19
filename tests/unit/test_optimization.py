"""ONNX export, INT8 quantization and the Phase 5a pipeline on a tiny synthetic cache (CPU)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from brainct.data.labels import LABELS
from brainct.optimization.evaluate_onnx import agreement, run
from brainct.optimization.export_onnx import (
    compare_with_torch,
    export_onnx,
    make_session,
    run_session,
)
from brainct.optimization.quantize import CacheCalibrationReader, quantize_int8
from brainct.training.config import TrainConfig
from brainct.training.dataset import CacheDataset
from brainct.training.evaluate import evaluate
from brainct.training.train import train


class TinyNet(nn.Module):
    """Same interface as the real model ((N, 3, H, W) -> (N, 6) logits), but fast."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        self.fc = nn.Linear(8, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


def test_export_matches_pytorch_and_has_dynamic_batch(tmp_path):
    torch.manual_seed(0)
    model = TinyNet().eval()
    info = export_onnx(model, tmp_path / "tiny.onnx", image_size=32)
    session = make_session(info["path"])

    batches = [
        np.random.default_rng(k).random((n, 3, 32, 32), dtype=np.float32)
        for k, n in enumerate([1, 8, 5])
    ]
    parity = compare_with_torch(model, session, batches)

    assert info["exporter"] in ("torchscript", "dynamo")
    assert parity["images"] == 14
    assert parity["max_abs_logit_diff"] < 1e-4
    assert run_session(session, batches[1]).shape == (8, 6)  # batch size is not fixed at 1


def test_quantize_int8_runs_and_stays_close(tmp_path, tiny_cache):
    torch.manual_seed(0)
    model = TinyNet().eval()
    fp32 = export_onnx(model, tmp_path / "tiny.onnx", image_size=32)["path"]
    train_set = CacheDataset(tiny_cache["cache_dir"], tiny_cache["labels_path"], "train")
    reader = CacheCalibrationReader(train_set, n_images=8, batch_size=4, seed=0)

    info = quantize_int8(fp32, tmp_path / "tiny_int8.onnx", reader, "minmax")

    batch = np.stack([train_set[i][0].numpy() for i in range(4)])
    int8_logits = run_session(make_session(info["path"]), batch)
    fp32_logits = run_session(make_session(fp32), batch)
    assert info["calibration_images"] == 8
    assert np.abs(int8_logits - fp32_logits).max() < 0.5  # quantized, so close but not equal
    with pytest.raises(ValueError, match="method must be"):
        quantize_int8(fp32, tmp_path / "x.onnx", reader, "best-guess")


def test_agreement_counts_decision_flips():
    ids = ["a", "b", "c", "d"]
    reference = pd.DataFrame({"image_id": ids})
    other = pd.DataFrame({"image_id": ids})
    for label in LABELS:
        reference[f"prob_{label}"] = [0.1, 0.4, 0.6, 0.9]
        other[f"prob_{label}"] = [0.1, 0.6, 0.6, 0.9]  # "b" crosses a 0.5 threshold
    thresholds = dict.fromkeys(LABELS, 0.5)

    result = agreement(reference, other, thresholds)

    assert result["decision_flips"]["epidural"] == {"count": 1, "rate": 0.25}
    assert result["mean_abs_prob_diff"]["epidural"] == pytest.approx(0.05)


def test_full_pipeline_on_a_tiny_trained_run(tmp_path, tiny_cache):
    cfg = TrainConfig(
        run_id="tiny",
        runs_dir=str(tmp_path / "runs"),
        cache_dir=str(tiny_cache["cache_dir"]),
        labels_path=str(tiny_cache["labels_path"]),
        device="cpu",
        pretrained=False,
        epochs=1,
        batch_size=4,
        warmup_steps=1,
        amp=False,
        channels_last=False,
        num_workers=0,
        val_batch_size=8,
        wandb_mode="disabled",
    )
    train(cfg)
    evaluate(cfg.run_dir, "val", n_boot=0)
    evaluate(cfg.run_dir, "test", n_boot=0)

    result = run(
        cfg.run_dir,
        tmp_path / "onnx",
        methods=("minmax", "percentile"),
        n_calibration=8,
        n_verify=8,
        batch_size=4,
        workers=0,
    )

    assert result["parity_onnx_fp32_vs_pytorch_fp32"]["max_abs_logit_diff"] < 1e-3
    assert result["quantization"]["chosen_method"] in ("minmax", "percentile")
    formats = result["test_split"]["formats"]
    assert set(formats) == {"pytorch_fp32", "onnx_fp32", "onnx_int8"}
    assert formats["onnx_fp32"]["vs_pytorch_fp32"]["images"] == 16
    # FP32 export loses nothing (compared on probabilities: AUC on 16 images is too coarse).
    assert max(formats["onnx_fp32"]["vs_pytorch_fp32"]["mean_abs_prob_diff"].values()) < 1e-4
    assert result["file_bytes"]["onnx_int8"] > 0
    saved = json.loads((tmp_path / "onnx" / "onnx_report.json").read_text())
    assert saved["test_split"]["thresholds_from"] == "val"
