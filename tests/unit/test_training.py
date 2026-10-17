"""Training, evaluation and metrics tests on a tiny synthetic cache (CPU only, no W&B)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from brainct.data.build_cache import CACHE_FILE, INDEX_FILE
from brainct.data.labels import LABELS, SUBTYPES
from brainct.preprocessing.window import model_input
from brainct.training import bench_loader
from brainct.training.config import TrainConfig, load_config
from brainct.training.dataset import Augment, CacheDataset, EpochSampler
from brainct.training.evaluate import TestAlreadyEvaluatedError, evaluate
from brainct.training.metrics import (
    auc_per_label,
    best_f1_thresholds,
    bootstrap_auc,
    confusion,
    macro,
)
from brainct.training.model import build_model
from brainct.training.train import lr_factor, train, weighted_bce

SIZE = 32  # tiny images keep CPU training fast; the real cache is 256 x 256


@pytest.fixture
def tiny_cache(tmp_path):
    """48 images (16 per split, 2 per patient); every label has positives and negatives in
    every split; one train image is marked ok=False like the real corrupt file."""
    n = 48
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = np.lib.format.open_memmap(
        cache_dir / CACHE_FILE, mode="w+", dtype=np.int16, shape=(n, SIZE, SIZE)
    )
    cache[:] = np.random.default_rng(0).integers(-1024, 1500, size=(n, SIZE, SIZE))
    del cache

    splits = ["train", "val", "test"]
    ids = [f"ID_{k:03d}" for k in range(n)]
    index = pd.DataFrame(
        {
            "row": range(n),
            "image_id": ids,
            "split": [splits[k // 16] for k in range(n)],
            "ok": [k != 3 for k in range(n)],
        }
    )
    index.to_parquet(cache_dir / INDEX_FILE, index=False)

    labels = pd.DataFrame({"image_id": ids, "patient_id": [f"P_{k // 2}" for k in range(n)]})
    for j, subtype in enumerate(SUBTYPES):
        labels[subtype] = [int(k % 6 == j) for k in range(n)]  # k % 6 == 5 -> no bleed
    labels["any"] = labels[SUBTYPES].max(axis=1)
    labels_path = tmp_path / "labels.parquet"
    labels[["image_id", "patient_id", *LABELS]].to_parquet(labels_path, index=False)
    return {"cache_dir": cache_dir, "labels_path": labels_path, "tmp": tmp_path}


def tiny_config(tiny_cache, **overrides) -> TrainConfig:
    values = {
        "run_id": "tiny",
        "runs_dir": str(tiny_cache["tmp"] / "runs"),
        "cache_dir": str(tiny_cache["cache_dir"]),
        "labels_path": str(tiny_cache["labels_path"]),
        "device": "cpu",
        "pretrained": False,
        "epochs": 2,
        "batch_size": 4,
        "warmup_steps": 2,
        "amp": False,
        "channels_last": False,
        "num_workers": 0,
        "val_batch_size": 8,
        "log_every": 1,
        "ckpt_every": 2,
        "wandb_mode": "disabled",
    }
    values.update(overrides)
    return TrainConfig(**values)


# --- Dataset -------------------------------------------------------------------------------


def test_dataset_matches_serving_preprocessing_and_drops_bad_rows(tiny_cache):
    train_set = CacheDataset(tiny_cache["cache_dir"], tiny_cache["labels_path"], "train")
    cache = np.load(tiny_cache["cache_dir"] / CACHE_FILE, mmap_mode="r")

    assert len(train_set) == 15  # 16 train rows minus the ok=False one
    assert "ID_003" not in set(train_set.image_ids)
    x, y, i = train_set[0]
    expected = torch.from_numpy(model_input(np.asarray(cache[train_set.rows[0]])))
    torch.testing.assert_close(x, expected)
    assert y.shape == (6,) and x.shape == (3, SIZE, SIZE)
    del cache


def test_augment_keeps_shape_and_changes_pixels(tiny_cache):
    torch.manual_seed(0)
    plain = CacheDataset(tiny_cache["cache_dir"], tiny_cache["labels_path"], "train")
    augmented = CacheDataset(
        tiny_cache["cache_dir"], tiny_cache["labels_path"], "train", augment=Augment()
    )

    x_plain, _, _ = plain[0]
    x_aug, _, _ = augmented[0]

    assert x_aug.shape == x_plain.shape
    assert torch.isfinite(x_aug).all()
    assert not torch.equal(x_aug, x_plain)


def test_dataset_hu_output_is_the_raw_cache_row(tiny_cache):
    hu_set = CacheDataset(tiny_cache["cache_dir"], tiny_cache["labels_path"], "val", output="hu")
    cache = np.load(tiny_cache["cache_dir"] / CACHE_FILE, mmap_mode="r")

    x, _, _ = hu_set[0]

    assert x.dtype == torch.int16 and x.shape == (SIZE, SIZE)
    assert torch.equal(x, torch.from_numpy(np.array(cache[hu_set.rows[0]])))
    with pytest.raises(ValueError, match="augmentation"):
        CacheDataset(
            tiny_cache["cache_dir"], tiny_cache["labels_path"], "val", Augment(), output="hu"
        )
    del cache


def test_direct_and_mmap_reads_are_identical(tiny_cache):
    args = (tiny_cache["cache_dir"], tiny_cache["labels_path"], "train")
    direct = CacheDataset(*args, output="hu", read="direct")
    mmap = CacheDataset(*args, output="hu", read="mmap")

    for i in range(len(direct)):
        assert torch.equal(direct[i][0], mmap[i][0])
    with pytest.raises(ValueError, match="read must be"):
        CacheDataset(*args, read="fast")


def test_bench_loader_measures_every_stage(tiny_cache):
    report = bench_loader.run(
        tiny_cache["cache_dir"],
        tiny_cache["labels_path"],
        workers=[0],
        batch_size=4,
        batches=2,
        warmup=1,
    )

    assert [r["stage"] for r in report["results"]] == ["read_direct", "read_mmap", "augment"]
    assert all(r["images"] == 8 and r["images_per_s"] > 0 for r in report["results"])


def test_epoch_sampler_resumes_in_the_same_order():
    full = list(EpochSampler(20, seed=1, epoch=3))
    resumed = list(EpochSampler(20, seed=1, epoch=3, skip=8))

    assert sorted(full) == list(range(20))
    assert resumed == full[8:]
    assert full != list(EpochSampler(20, seed=1, epoch=4))


# --- Model, loss, schedule, config ----------------------------------------------------------


def test_model_has_six_outputs():
    model = build_model(pretrained=False).eval()

    with torch.no_grad():
        out = model(torch.zeros(2, 3, 64, 64))

    assert out.shape == (2, 6)


def test_weighted_bce_matches_manual_formula():
    logits = torch.tensor([[0.0, 1.0, -1.0, 2.0, 0.5, -0.5]])
    targets = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    weights = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    per_class = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    expected = (per_class * weights).sum() / weights.sum()

    torch.testing.assert_close(weighted_bce(logits, targets, weights), expected)


def test_lr_schedule_warms_up_then_decays_to_zero():
    assert lr_factor(0, 10, 100) == pytest.approx(0.1)
    assert lr_factor(9, 10, 100) == pytest.approx(1.0)
    assert lr_factor(10, 10, 100) == pytest.approx(1.0)
    assert lr_factor(100, 10, 100) == pytest.approx(0.0, abs=1e-12)


def test_load_config_overrides_and_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINCT_TEST_ROOT", str(tmp_path))
    path = tmp_path / "c.yaml"
    path.write_text("runs_dir: $BRAINCT_TEST_ROOT/runs\nepochs: 3\n")

    cfg = load_config(path, ["epochs=7", "amp=false", "max_steps=500", "lr=1e-4"])

    assert Path(cfg.runs_dir) == tmp_path / "runs"  # compare as paths: separators differ
    assert (cfg.epochs, cfg.amp, cfg.max_steps, cfg.lr) == (7, False, 500, 1e-4)
    assert isinstance(cfg.lr, float)  # YAML 1.1 reads "1e-4" as a string; it must be coerced
    assert load_config(None, ["class_weights=[2, 1, 1, 1, 1, 1]"]).class_weights[0] == 2.0
    assert load_config(None, ["max_steps=null"]).max_steps is None
    with pytest.raises(ValueError, match="unknown"):
        load_config(None, ["not_a_key=1"])
    with pytest.raises(ValueError, match="epochs expects"):
        load_config(None, ["epochs=five"])


# --- Training: full run, resume, evaluation ------------------------------------------------


def test_train_completes_and_is_idempotent(tiny_cache):
    cfg = tiny_config(tiny_cache)

    summary = train(cfg)

    assert summary["complete"] and summary["stop_reason"] == "epochs_done"
    assert summary["step"] == 6  # 2 epochs x (15 // 4) steps
    assert summary["best_epoch"] is not None
    assert (cfg.run_dir / "checkpoints" / "best.pt").exists()
    assert (cfg.run_dir / "DONE").exists()
    logged = [json.loads(line) for line in (cfg.run_dir / "metrics.jsonl").read_text().splitlines()]
    assert sum("val/macro_auc" in row for row in logged) == 2

    again = train(cfg)  # a chained job after completion must do nothing
    assert again["step"] == 6


def test_train_resumes_after_stopping_on_time_budget(tiny_cache):
    first = train(tiny_config(tiny_cache, time_budget_hours=0.0))  # stops after one step
    assert not first["complete"] and first["stop_reason"] == "time_budget"
    assert first["step"] == 1

    second = train(tiny_config(tiny_cache))  # same run_id: resumes from last.pt

    assert second["complete"] and second["step"] == 6
    run_dir = tiny_config(tiny_cache).run_dir
    steps = [
        json.loads(line)["step"]
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if "train/loss" in line
    ]
    assert steps == list(range(1, 7))  # no gap and no repeat across the resume


def test_max_steps_marks_a_speed_test_complete(tiny_cache):
    summary = train(tiny_config(tiny_cache, run_id="speed", max_steps=2, validate=False))

    assert summary["complete"] and summary["stop_reason"] == "max_steps"
    assert summary["step"] == 2
    assert summary["best_epoch"] is None


def test_evaluate_val_then_test_once(tiny_cache):
    cfg = tiny_config(tiny_cache)
    train(cfg)

    val = evaluate(cfg.run_dir, "val", n_boot=20)
    test = evaluate(cfg.run_dir, "test", n_boot=20)

    assert val["images"] == 16 and test["images"] == 16
    assert test["thresholds"] == val["thresholds"]  # chosen on val, applied to test
    assert set(test["auc_ci95"]) == {*LABELS, "macro"}
    assert (cfg.run_dir / "eval" / "predictions_test.parquet").exists()
    with pytest.raises(TestAlreadyEvaluatedError):
        evaluate(cfg.run_dir, "test", n_boot=20)
    evaluate(cfg.run_dir, "test", n_boot=20, force=True)


# --- Metrics ------------------------------------------------------------------------------


def _toy_predictions():
    y = np.zeros((8, 6), dtype=np.float32)
    y[4:, :] = 1
    p = np.tile(np.linspace(0.1, 0.8, 8, dtype=np.float32)[:, None], (1, 6))
    return y, p


def test_auc_and_macro_on_separable_data():
    y, p = _toy_predictions()
    p[:, 0] = p[::-1, 0]  # make `any` perfectly wrong; macro must ignore it

    aucs = auc_per_label(y, p)

    assert aucs["any"] == 0.0
    assert all(aucs[s] == 1.0 for s in SUBTYPES)
    assert macro(aucs) == 1.0


def test_best_f1_threshold_and_confusion():
    y, p = _toy_predictions()

    thresholds = best_f1_thresholds(y, p)
    matrices = confusion(y, p, thresholds)

    assert thresholds["epidural"] == pytest.approx(p[4, 1])  # lowest positive score
    assert matrices["epidural"] == {"tp": 4, "fp": 0, "fn": 0, "tn": 4, "f1": 1.0}


def test_bootstrap_ci_is_deterministic_and_brackets_the_estimate():
    rng = np.random.default_rng(0)
    y = (rng.random((200, 6)) < 0.3).astype(np.float32)
    p = np.clip(y * 0.4 + rng.random((200, 6)) * 0.6, 0, 1).astype(np.float32)
    groups = np.repeat(np.arange(50), 4)

    a = bootstrap_auc(y, p, groups, n_boot=100, seed=1)
    b = bootstrap_auc(y, p, groups, n_boot=100, seed=1)
    point = auc_per_label(y, p)

    assert a == b
    for label in SUBTYPES:
        low, high = a[label]
        assert low <= point[label] <= high
