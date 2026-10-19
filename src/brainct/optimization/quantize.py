"""Static INT8 quantization with ONNX Runtime, calibrated on training images from the cache.

QDQ format, per-channel weights, uint8 activations / int8 weights (ORT's usual CPU choice for
CNNs). Calibration images are model inputs exactly as training/serving produce them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from brainct.optimization.export_onnx import INPUT_NAME
from brainct.training.dataset import CacheDataset

CALIBRATION_METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "percentile": CalibrationMethod.Percentile,
    "entropy": CalibrationMethod.Entropy,
}
# Compared by default. Entropy is left out: with our settings and this ONNX Runtime version it
# produced a model bit-identical to minmax (same md5, 2026-10-19), so it added no comparison.
DEFAULT_METHODS = ("minmax", "percentile")


class CacheCalibrationReader(CalibrationDataReader):
    """Feeds fixed, randomly chosen dataset images to the calibrator in small batches."""

    def __init__(self, dataset: CacheDataset, n_images: int, batch_size: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        count = min(n_images, len(dataset))
        self.indices = np.sort(rng.choice(len(dataset), size=count, replace=False))
        self.dataset = dataset
        self.batch_size = batch_size
        self._position = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._position >= len(self.indices):
            return None
        chunk = self.indices[self._position : self._position + self.batch_size]
        self._position += self.batch_size
        batch = np.stack([self.dataset[int(i)][0].numpy() for i in chunk]).astype(np.float32)
        return {INPUT_NAME: batch}

    def rewind(self) -> None:
        self._position = 0


def quantize_int8(
    fp32_path: str | Path,
    out_path: str | Path,
    reader: CacheCalibrationReader,
    method: str,
) -> dict[str, Any]:
    """Pre-process (shape inference + graph optimizations), then quantize statically."""
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"method must be one of {sorted(CALIBRATION_METHODS)}, got {method!r}")
    out_path = Path(out_path)
    prepared = out_path.with_name(out_path.stem + "_prep.onnx")
    quant_pre_process(str(fp32_path), str(prepared))
    reader.rewind()
    quantize_static(
        str(prepared),
        str(out_path),
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CALIBRATION_METHODS[method],
    )
    prepared.unlink(missing_ok=True)
    return {
        "method": method,
        "path": str(out_path),
        "bytes": out_path.stat().st_size,
        "calibration_images": len(reader.indices),
    }
