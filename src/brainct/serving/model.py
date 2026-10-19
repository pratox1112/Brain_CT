"""Loads the ONNX model and val thresholds, and turns DICOM bytes into predictions.

Uses exactly the training preprocessing (preprocess_dicom -> model_input), so the API sees the
same tensors as training and the offline ONNX evaluation.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from brainct.constants import LABELS
from brainct.preprocessing.dicom import PreprocessInfo, preprocess_dicom
from brainct.preprocessing.window import model_input

INPUT_NAME = "input"
OUTPUT_NAME = "logits"


@dataclass(frozen=True)
class Prepared:
    """One request's preprocessed image, ready to be batched."""

    x: np.ndarray  # float32 (3, H, W)
    info: PreprocessInfo
    preprocess_ms: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class InferenceModel:
    def __init__(self, model_path: Path, thresholds_path: Path, name: str, threads: int) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        shape = self.session.get_inputs()[0].shape  # [batch, 3, H, W]
        self.image_size = int(shape[-1])
        thresholds = json.loads(Path(thresholds_path).read_text())["thresholds"]
        self.thresholds = {label: float(thresholds[label]) for label in LABELS}
        self.name = name
        self.version = _sha256(Path(model_path))[:12]  # identifies the exact model file served
        self.format = "onnx-int8" if "int8" in Path(model_path).name else "onnx"

    def prepare(self, data: bytes) -> Prepared:
        """DICOM bytes -> model input. Raises DicomError for unusable files."""
        start = time.perf_counter()
        hu, info = preprocess_dicom(data, size=self.image_size)
        x = model_input(hu).astype(np.float32)
        return Prepared(x, info, (time.perf_counter() - start) * 1000)

    def predict_batch(self, batch: np.ndarray) -> np.ndarray:
        """float32 (N, 3, H, W) -> sigmoid probabilities (N, 6)."""
        logits = self.session.run([OUTPUT_NAME], {INPUT_NAME: np.ascontiguousarray(batch)})[0]
        return 1.0 / (1.0 + np.exp(-logits))
