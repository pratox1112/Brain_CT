"""Evaluate a trained run on val or test: per-subtype AUC with patient-bootstrap CIs, macro AUC,
macro F1 at thresholds chosen on val, and confusion matrices.

Usage (on the cluster, inside a Slurm GPU job - see jobs/evaluate.sh):

    python -m brainct.training.evaluate --run-dir /scratch/$USER/brainct/runs/resnet50_baseline \
        --split val
    python -m brainct.training.evaluate --run-dir ... --split test

The test split is evaluated **once** per run: a second test evaluation is refused unless
--force is given, so the reported test numbers cannot be tuned against.

Outputs in <run_dir>/eval/:
    predictions_<split>.parquet   image_id, patient_id, labels and probabilities
    metrics_<split>.json          the full report for the split
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from brainct.data.labels import LABELS
from brainct.training.config import TrainConfig, load_config
from brainct.training.dataset import CacheDataset
from brainct.training.metrics import best_f1_thresholds, report
from brainct.training.model import build_model
from brainct.training.train import predict


class TestAlreadyEvaluatedError(RuntimeError):
    """The test split was already evaluated for this run."""

    __test__ = False  # not a pytest test class, despite the name


def _load_run(run_dir: Path, overrides: list[str]) -> tuple[TrainConfig, torch.nn.Module, Path]:
    saved = yaml.safe_load((run_dir / "config.yaml").read_text())
    extra = [f"{k}={json.dumps(v)}" for k, v in saved.items()]
    cfg = load_config(None, extra + overrides)
    weights = run_dir / "checkpoints" / "best.pt"
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return cfg, model, weights


def _predictions(
    run_dir: Path, cfg: TrainConfig, model: torch.nn.Module, split: str, device: torch.device
) -> pd.DataFrame:
    """Predict a split, reusing saved predictions if they exist."""
    path = run_dir / "eval" / f"predictions_{split}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    dataset = CacheDataset(cfg.cache_dir, cfg.labels_path, split)
    y, p = predict(model, dataset, cfg, device)
    table = pd.DataFrame({"image_id": dataset.image_ids, "patient_id": dataset.patient_ids})
    for j, label in enumerate(LABELS):
        table[label] = y[:, j].astype(np.int8)
        table[f"prob_{label}"] = p[:, j]
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return table


def _arrays(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = table[LABELS].to_numpy(dtype=np.float32)
    p = table[[f"prob_{label}" for label in LABELS]].to_numpy(dtype=np.float32)
    return y, p


def evaluate(
    run_dir: str | Path,
    split: str,
    n_boot: int = 1000,
    force: bool = False,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    metrics_path = run_dir / "eval" / f"metrics_{split}.json"
    if split == "test" and metrics_path.exists() and not force:
        raise TestAlreadyEvaluatedError(
            f"{metrics_path} exists: the test split is evaluated once per run (use --force "
            "only if you understand that re-running invalidates it as an unbiased estimate)"
        )

    cfg, model, weights = _load_run(run_dir, overrides or [])
    use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # Thresholds always come from val, so test is never used to choose anything.
    val_table = _predictions(run_dir, cfg, model, "val", device)
    thresholds = best_f1_thresholds(*_arrays(val_table))
    table = val_table if split == "val" else _predictions(run_dir, cfg, model, split, device)
    y, p = _arrays(table)

    result = report(y, p, thresholds, groups=table["patient_id"].to_numpy(), n_boot=n_boot)
    result.update(
        {
            "split": split,
            "run_id": cfg.run_id,
            "weights": str(weights),
            "patients": int(table["patient_id"].nunique()),
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "thresholds_from": "val",
        }
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2))
    _log_to_wandb(cfg, run_dir, split, result)
    return result


def _log_to_wandb(cfg: TrainConfig, run_dir: Path, split: str, result: dict[str, Any]) -> None:
    summary_path = run_dir / "run_summary.json"
    if cfg.wandb_mode == "disabled" or not summary_path.exists():
        return
    wandb_id = json.loads(summary_path.read_text()).get("wandb_id")
    if wandb_id is None:
        return
    import wandb

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        id=wandb_id,
        resume="allow",
        mode=cfg.wandb_mode,
        dir=str(run_dir),
    )
    flat = {f"{split}/auc_{k}": v for k, v in result["auc"].items()}
    flat.update({f"{split}/f1_{k}": v for k, v in result["f1"].items()})
    flat[f"{split}/macro_auc"] = result["macro_auc"]
    flat[f"{split}/macro_f1"] = result["macro_f1"]
    for label, (low, high) in result.get("auc_ci95", {}).items():
        flat[f"{split}/auc_{label}_ci95_low"] = low
        flat[f"{split}/auc_{label}_ci95_high"] = high
    run.summary.update(flat)
    run.finish()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--force", action="store_true", help="re-evaluate test (see docstring)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser()
    result = evaluate(run_dir, args.split, args.n_boot, args.force, args.overrides)
    print(json.dumps({k: v for k, v in result.items() if k != "confusion"}, indent=2))


if __name__ == "__main__":
    main()
