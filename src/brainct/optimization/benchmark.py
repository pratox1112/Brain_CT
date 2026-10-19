"""Phase 5b: latency and throughput of PyTorch FP32 vs ONNX Runtime FP32 / INT8 on this CPU.

Measures the model alone (inputs are ready-made tensors; preprocessing is excluded and is
measured end to end by the API in Phase 6). For each variant and batch size: warmup runs (not
timed), then timed runs, repeated; reports p50 / p95 latency per batch and throughput.

Also checks, on the local sample DICOMs, how far each ONNX variant's probabilities are from
PyTorch FP32 and how many yes/no decisions flip at the val thresholds. The samples are test-split
images, so this is a sanity check on this CPU, never used to choose anything.

Usage (on the laptop, plugged in, same power mode for every run, heavy apps closed):

    uv run python -m brainct.optimization.benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from brainct.data.labels import LABELS
from brainct.optimization.export_onnx import make_session, run_session
from brainct.preprocessing.dicom import IMAGE_SIZE, preprocess_dicom
from brainct.preprocessing.window import model_input
from brainct.training.model import build_model

DEFAULT_ARTIFACTS = Path("artifacts/resnet50_baseline")
DEFAULT_SAMPLES = [Path("samples/dicom"), Path("samples_edge/dicom")]
DEFAULT_OUT = Path("artifacts/benchmark/benchmark.json")

# name -> file inside the artifacts folder (best.pt is the PyTorch checkpoint).
VARIANTS = {
    "pytorch_fp32": "best.pt",
    "onnx_fp32": "onnx/model_fp32.onnx",
    "onnx_int8": "onnx/model_int8.onnx",  # final: percentile calibration
    "onnx_int8_minmax": "onnx/int8_minmax.onnx",
}

Predict = Callable[[np.ndarray], np.ndarray]  # float32 (N, 3, H, W) -> logits (N, 6)


def cpu_info() -> dict[str, Any]:
    """CPU model and whether it has VNNI (INT8 accuracy depends on it, see Phase 5a)."""
    info: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "vnni": None,
    }
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        text = cpuinfo.read_text()
        names = [ln.split(":", 1)[1].strip() for ln in text.splitlines() if "model name" in ln]
        info["model_name"] = names[0] if names else None
        info["vnni"] = "avx512_vnni" in text or "avx_vnni" in text
    elif platform.system() == "Windows":
        try:
            query = "(Get-CimInstance Win32_Processor).Name"
            command = ["powershell", "-NoProfile", "-Command", query]
            info["model_name"] = subprocess.run(
                command, capture_output=True, text=True, timeout=20, check=True
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            info["model_name"] = None
    return info


def load_pytorch(checkpoint: Path, threads: int) -> Predict:
    torch.set_num_threads(threads)
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    model.eval()

    @torch.inference_mode()
    def predict(batch: np.ndarray) -> np.ndarray:
        return model(torch.from_numpy(batch)).numpy()

    return predict


def load_onnx(path: Path, threads: int) -> Predict:
    session = make_session(path, threads)
    return lambda batch: run_session(session, batch)


def time_batches(predict: Predict, batch: np.ndarray, warmup: int, iterations: int) -> np.ndarray:
    """Wall-clock milliseconds for each of `iterations` calls, after `warmup` untimed calls."""
    for _ in range(warmup):
        predict(batch)
    times = np.empty(iterations)
    for i in range(iterations):
        start = time.perf_counter()
        predict(batch)
        times[i] = (time.perf_counter() - start) * 1000
    return times


def benchmark(
    predict: Predict,
    image_size: int,
    batch_sizes: list[int],
    iterations: dict[int, int],
    warmup: int,
    repeats: int,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """p50 / p95 latency and throughput per batch size; each repeat is reported, plus medians."""
    rng = np.random.default_rng(seed)
    results = []
    for batch_size in batch_sizes:
        batch = rng.standard_normal((batch_size, 3, image_size, image_size), dtype=np.float32)
        runs = []
        for _ in range(repeats):
            times = time_batches(predict, batch, warmup, iterations[batch_size])
            runs.append(
                {
                    "p50_ms": float(np.percentile(times, 50)),
                    "p95_ms": float(np.percentile(times, 95)),
                    "images_per_s": batch_size * len(times) / (times.sum() / 1000),
                }
            )
        results.append(
            {
                "batch_size": batch_size,
                "iterations": iterations[batch_size],
                "repeats": runs,
                "p50_ms": float(np.median([r["p50_ms"] for r in runs])),
                "p95_ms": float(np.median([r["p95_ms"] for r in runs])),
                "images_per_s": float(np.median([r["images_per_s"] for r in runs])),
                "p95_ms_spread": [min(r["p95_ms"] for r in runs), max(r["p95_ms"] for r in runs)],
            }
        )
    return results


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sample_agreement(
    predictors: dict[str, Predict],
    sample_files: list[Path],
    thresholds: dict[str, float],
    image_size: int = IMAGE_SIZE,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Each non-PyTorch variant's probabilities vs PyTorch FP32 on the same sample DICOMs."""
    inputs = np.stack(
        [model_input(preprocess_dicom(p.read_bytes(), size=image_size)[0]) for p in sample_files]
    ).astype(np.float32)
    probs = {}
    for name, predict in predictors.items():
        chunks = [predict(inputs[k : k + batch_size]) for k in range(0, len(inputs), batch_size)]
        probs[name] = _sigmoid(np.concatenate(chunks))
    reference = probs["pytorch_fp32"]
    cut = np.array([thresholds[label] for label in LABELS])
    out: dict[str, Any] = {"images": len(sample_files)}
    for name, p in probs.items():
        if name == "pytorch_fp32":
            continue
        diff = np.abs(p - reference)
        flips = (p >= cut) != (reference >= cut)
        out[name] = {
            "mean_abs_prob_diff": {lab: float(diff[:, j].mean()) for j, lab in enumerate(LABELS)},
            "max_abs_prob_diff": float(diff.max()),
            "decision_flips": {lab: int(flips[:, j].sum()) for j, lab in enumerate(LABELS)},
        }
    return out


def run(
    artifacts: Path = DEFAULT_ARTIFACTS,
    variants: list[str] | None = None,
    threads: int = 8,
    batch_sizes: tuple[int, ...] = (1, 8, 32),
    iterations: tuple[int, ...] = (100, 50, 25),
    warmup: int = 10,
    repeats: int = 3,
    image_size: int = IMAGE_SIZE,
    sample_dirs: list[Path] | None = None,
    out: Path | None = DEFAULT_OUT,
) -> dict[str, Any]:
    variants = variants or list(VARIANTS)
    iters = dict(zip(batch_sizes, iterations, strict=True))
    predictors: dict[str, Predict] = {}
    for name in variants:
        path = artifacts / VARIANTS[name]
        loader = load_pytorch if name == "pytorch_fp32" else load_onnx
        predictors[name] = loader(path, threads)

    timings = {}
    for name, predict in predictors.items():
        print(f"benchmarking {name} ...", flush=True)
        timings[name] = benchmark(predict, image_size, list(batch_sizes), iters, warmup, repeats)
        for row in timings[name]:
            print(
                f"  batch {row['batch_size']:3d}: p50 {row['p50_ms']:8.1f} ms   "
                f"p95 {row['p95_ms']:8.1f} ms   {row['images_per_s']:7.1f} img/s",
                flush=True,
            )

    agreement = None
    files = sorted(p for d in (sample_dirs or []) for p in Path(d).glob("*.dcm"))
    if files and "pytorch_fp32" in predictors:
        thresholds = json.loads((artifacts / "metrics_val.json").read_text())["thresholds"]
        agreement = sample_agreement(predictors, files, thresholds, image_size)

    report = {
        "cpu": cpu_info(),
        "threads": threads,
        "image_size": image_size,
        "warmup": warmup,
        "repeats": repeats,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnxruntime": version("onnxruntime"),
        },
        "file_bytes": {name: (artifacts / VARIANTS[name]).stat().st_size for name in variants},
        "timings": timings,
        "sample_agreement_vs_pytorch_fp32": agreement,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=[100, 50, 25],
        help="timed runs per batch size, in the same order as --batch-sizes",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-dir", type=Path, action="append", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if len(args.batch_sizes) != len(args.iterations):
        parser.error("--iterations needs one value per --batch-sizes entry")

    report = run(
        args.artifacts,
        args.variants,
        args.threads,
        tuple(args.batch_sizes),
        tuple(args.iterations),
        args.warmup,
        args.repeats,
        sample_dirs=args.sample_dir if args.sample_dir is not None else DEFAULT_SAMPLES,
        out=args.out,
    )
    summary = {"cpu": report["cpu"], "agreement": report["sample_agreement_vs_pytorch_fp32"]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
