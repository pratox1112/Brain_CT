"""Export a PyTorch model to ONNX with a dynamic batch size, and run it with ONNX Runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

INPUT_NAME = "input"
OUTPUT_NAME = "logits"
OPSET = 17


def export_onnx(model: torch.nn.Module, path: str | Path, image_size: int) -> dict[str, Any]:
    """Export `model` (input (N, 3, H, W) float32 -> logits (N, 6)) and validate the file.

    Tries the TorchScript-based exporter first (mature for CNNs like ResNet), then the newer
    dynamo-based exporter if the installed torch no longer provides the old one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().to("cpu", memory_format=torch.contiguous_format)
    dummy = torch.zeros(1, 3, image_size, image_size)
    options = {
        "input_names": [INPUT_NAME],
        "output_names": [OUTPUT_NAME],
        "opset_version": OPSET,
        "dynamic_axes": {INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
    }
    try:
        torch.onnx.export(model, (dummy,), str(path), dynamo=False, **options)
        exporter = "torchscript"
    except (TypeError, RuntimeError, NotImplementedError) as legacy_error:
        try:
            torch.onnx.export(model, (dummy,), str(path), dynamo=True, **options)
        except Exception as exc:
            raise RuntimeError(f"ONNX export failed (legacy: {legacy_error})") from exc
        exporter = "dynamo"
    onnx.checker.check_model(str(path))
    return {
        "path": str(path),
        "exporter": exporter,
        "opset": OPSET,
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "bytes": path.stat().st_size,
    }


def make_session(path: str | Path, threads: int | None = None) -> ort.InferenceSession:
    """A CPU ONNX Runtime session (intra-op threads = `threads`, or ORT's default)."""
    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def run_session(session: ort.InferenceSession, batch: np.ndarray) -> np.ndarray:
    """Logits for one float32 batch of shape (N, 3, H, W)."""
    return session.run([OUTPUT_NAME], {INPUT_NAME: np.ascontiguousarray(batch, np.float32)})[0]


@torch.no_grad()
def compare_with_torch(
    model: torch.nn.Module, session: ort.InferenceSession, batches: list[np.ndarray]
) -> dict[str, float]:
    """Max / mean absolute logit difference between PyTorch (fp32, CPU) and ONNX Runtime."""
    model = model.eval().to("cpu")
    diffs = []
    for batch in batches:
        reference = model(torch.from_numpy(batch)).numpy()
        diffs.append(np.abs(run_session(session, batch) - reference).ravel())
    diff = np.concatenate(diffs)
    return {
        "images": int(sum(len(b) for b in batches)),
        "max_abs_logit_diff": float(diff.max()),
        "mean_abs_logit_diff": float(diff.mean()),
    }
