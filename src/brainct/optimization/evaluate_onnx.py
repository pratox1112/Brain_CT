"""Phase 5a: export the final model to ONNX, quantize it to INT8, and measure the accuracy cost.

Steps (all on CPU, the target for serving):
  1. Export best.pt to ONNX FP32 and check it against PyTorch FP32 on val images.
  2. Quantize to INT8 with each calibration method (calibrated on train images).
  3. Choose the calibration method by **val** macro AUC.
  4. On the **same test set and val thresholds** as the PyTorch evaluation, compare PyTorch FP32
     (saved predictions from evaluate.py), ONNX FP32 and ONNX INT8: AUC / AP / F1, the change
     against PyTorch, and how often a yes/no decision flips.

The model itself is frozen: nothing here is tuned on test. Test measures what each format loses.

Usage (on the cluster - see jobs/optimize.sh):

    python -m brainct.optimization.evaluate_onnx \
        --run-dir /scratch/$USER/brainct/runs/resnet50_baseline \
        --out ~/brainct_outputs/final/resnet50_baseline/onnx

Outputs in --out: model_fp32.onnx, model_int8.onnx (best calibration), int8_<method>.onnx,
predictions_<split>_<format>.parquet, onnx_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from brainct.data.labels import LABELS
from brainct.optimization.export_onnx import (
    compare_with_torch,
    export_onnx,
    make_session,
    run_session,
)
from brainct.optimization.quantize import (
    CALIBRATION_METHODS,
    DEFAULT_METHODS,
    CacheCalibrationReader,
    quantize_int8,
)
from brainct.training.dataset import CacheDataset
from brainct.training.evaluate import load_run
from brainct.training.metrics import auc_per_label, macro, report

DEFAULT_SEED = 2026
PROB_COLUMNS = [f"prob_{label}" for label in LABELS]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def predict_onnx(session, dataset: CacheDataset, batch_size: int, workers: int) -> pd.DataFrame:
    """Probabilities for every image of `dataset`, in dataset order."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    probs = [_sigmoid(run_session(session, x.numpy())) for x, _, _ in loader]
    table = pd.DataFrame({"image_id": dataset.image_ids, "patient_id": dataset.patient_ids})
    table[LABELS] = dataset.labels.astype(np.int8)
    table[PROB_COLUMNS] = np.concatenate(probs).astype(np.float32)
    return table


def _arrays(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return table[LABELS].to_numpy(np.float32), table[PROB_COLUMNS].to_numpy(np.float32)


def agreement(
    reference: pd.DataFrame, other: pd.DataFrame, thresholds: dict[str, float]
) -> dict[str, Any]:
    """How far `other`'s probabilities are from `reference`'s, matched by image_id."""
    merged = reference.merge(other, on="image_id", suffixes=("_ref", "_new"), validate="1:1")
    out: dict[str, Any] = {"images": len(merged), "mean_abs_prob_diff": {}, "decision_flips": {}}
    for label in LABELS:
        ref, new = merged[f"prob_{label}_ref"], merged[f"prob_{label}_new"]
        out["mean_abs_prob_diff"][label] = float((ref - new).abs().mean())
        flips = (ref >= thresholds[label]) != (new >= thresholds[label])
        out["decision_flips"][label] = {"count": int(flips.sum()), "rate": float(flips.mean())}
    return out


def _summary(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = ["auc", "macro_auc", "average_precision", "macro_average_precision", "f1", "macro_f1"]
    return {k: metrics[k] for k in keys}


def run(
    run_dir: str | Path,
    out_dir: str | Path,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    n_calibration: int = 512,
    n_verify: int = 512,
    batch_size: int = 64,
    workers: int = 8,
    threads: int | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    start = time.time()
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, model, weights = load_run(run_dir, [])
    model = model.eval().to("cpu")
    pytorch_test = pd.read_parquet(run_dir / "eval" / "predictions_test_fp32.parquet")
    thresholds = json.loads((run_dir / "eval" / "metrics_val.json").read_text())["thresholds"]

    def dataset(split: str) -> CacheDataset:
        return CacheDataset(cfg.cache_dir, cfg.labels_path, split, read=cfg.cache_read)

    val_set, test_set = dataset("val"), dataset("test")
    image_size = int(val_set[0][0].shape[-1])

    # 1. Export and check against PyTorch on a fixed sample of val images.
    fp32_path = out_dir / "model_fp32.onnx"
    export_info = export_onnx(model, fp32_path, image_size)
    fp32_session = make_session(fp32_path, threads)
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(len(val_set), size=min(n_verify, len(val_set)), replace=False))
    stacked = np.stack([val_set[int(i)][0].numpy() for i in picked])
    batches = [stacked[k : k + batch_size] for k in range(0, len(stacked), batch_size)]
    parity = compare_with_torch(model, fp32_session, batches)
    print(f"export: {export_info['exporter']}, parity {parity}", flush=True)

    # 2-3. Quantize with each calibration method; choose by val macro AUC.
    reader = CacheCalibrationReader(dataset("train"), n_calibration, batch_size=16, seed=seed)
    calibration: dict[str, Any] = {}
    for method in methods:
        info = quantize_int8(fp32_path, out_dir / f"int8_{method}.onnx", reader, method)
        session = make_session(info["path"], threads)
        val_table = predict_onnx(session, val_set, batch_size, workers)
        val_table.to_parquet(out_dir / f"predictions_val_int8_{method}.parquet", index=False)
        info["val_auc"] = auc_per_label(*_arrays(val_table))
        info["val_macro_auc"] = macro(info["val_auc"])
        calibration[method] = info
        print(f"int8 {method}: val macro AUC {info['val_macro_auc']:.5f}", flush=True)
    best = max(calibration, key=lambda m: calibration[m]["val_macro_auc"])
    int8_path = out_dir / "model_int8.onnx"
    shutil.copyfile(calibration[best]["path"], int8_path)

    # 4. Same test set, same val thresholds: PyTorch FP32 vs ONNX FP32 vs ONNX INT8.
    tables = {"pytorch_fp32": pytorch_test}
    tables["onnx_fp32"] = predict_onnx(fp32_session, test_set, batch_size, workers)
    int8_session = make_session(int8_path, threads)
    tables["onnx_int8"] = predict_onnx(int8_session, test_set, batch_size, workers)
    formats: dict[str, Any] = {}
    for name, table in tables.items():
        table.to_parquet(out_dir / f"predictions_test_{name}.parquet", index=False)
        metrics = report(*_arrays(table), thresholds, n_boot=0)
        formats[name] = _summary(metrics)
        if name != "pytorch_fp32":
            formats[name]["vs_pytorch_fp32"] = agreement(pytorch_test, table, thresholds)
            reference = formats["pytorch_fp32"]
            formats[name]["delta_macro_auc"] = metrics["macro_auc"] - reference["macro_auc"]
            formats[name]["delta_auc"] = {
                label: metrics["auc"][label] - reference["auc"][label] for label in LABELS
            }

    result = {
        "run_id": cfg.run_id,
        "weights": str(weights),
        "export": export_info,
        "parity_onnx_fp32_vs_pytorch_fp32": parity,
        "quantization": {
            "format": "QDQ",
            "per_channel": True,
            "activations": "uint8",
            "weights": "int8",
            "calibration": calibration,
            "chosen_method": best,
            "chosen_by": "val macro AUC",
        },
        "file_bytes": {
            "onnx_fp32": fp32_path.stat().st_size,
            "onnx_int8": int8_path.stat().st_size,
        },
        "test_split": {"images": len(test_set), "thresholds_from": "val", "formats": formats},
        "threads": threads,
        "cpus": os.cpu_count(),
        "elapsed_seconds": round(time.time() - start, 1),
    }
    (out_dir / "onnx_report.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(CALIBRATION_METHODS),
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument("--n-calibration", type=int, default=512)
    parser.add_argument("--n-verify", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=None, help="ONNX Runtime intra-op threads")
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads or torch.get_num_threads())
    result = run(
        args.run_dir.expanduser(),
        args.out.expanduser(),
        tuple(args.methods),
        args.n_calibration,
        args.n_verify,
        args.batch_size,
        args.workers,
        args.threads,
    )
    print(json.dumps(result["test_split"], indent=2))
    print(json.dumps({k: result[k] for k in ["file_bytes", "elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
