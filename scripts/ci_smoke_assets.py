"""Create stand-in assets so CI can build and smoke-test the Docker image without real data.

The real INT8 model and RSNA images are not in git, so CI builds the image with:
  - a tiny ONNX model with the real model's interface ((N, 3, 256, 256) -> (N, 6) logits),
  - a thresholds file in the evaluate.py format,
  - one synthetic CT DICOM to POST to /predict.
Needs only numpy, onnx and pydicom (no PyTorch).

    python scripts/ci_smoke_assets.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

# This import needs the sys.path entry above.
from dicom_factory import make_dicom_bytes  # noqa: E402

LABELS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
OUT = REPO / "artifacts/resnet50_baseline"


def tiny_model(path: Path, size: int = 256) -> None:
    """Global average pool over the 3 channels, then a 3 -> 6 linear layer."""
    rng = np.random.default_rng(0)
    weight = numpy_helper.from_array(rng.normal(size=(6, 3)).astype(np.float32), "weight")
    bias = numpy_helper.from_array(np.zeros(6, dtype=np.float32), "bias")
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["flat"]),
            helper.make_node("Gemm", ["flat", "weight", "bias"], ["logits"], transB=1),
        ],
        "ci_smoke_model",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, size, size])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 6])],
        initializer=[weight, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8  # readable by any recent onnxruntime
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def main() -> None:
    target = OUT / "onnx" / "model_int8.onnx"
    # These paths are where the REAL model lives on the laptop: never overwrite it there.
    if target.exists() and os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit(f"refusing to overwrite {target} outside CI (it may be the real model)")
    tiny_model(target)
    (OUT / "metrics_val.json").write_text(json.dumps({"thresholds": dict.fromkeys(LABELS, 0.5)}))
    pixels = np.random.default_rng(1).integers(0, 2000, (512, 512))
    (REPO / "smoke.dcm").write_bytes(make_dicom_bytes("ID_smoke", pixels=pixels))
    print("wrote stand-in model, thresholds and smoke.dcm")


if __name__ == "__main__":
    main()
