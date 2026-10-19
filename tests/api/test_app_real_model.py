"""The real INT8 model behind the API equals an offline ONNX run, on the local sample DICOMs.

Needs artifacts/ (model, thresholds) and samples/, which exist only on the owner's laptop
(RSNA-derived data is never committed), so it is skipped in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from brainct.constants import LABELS
from brainct.optimization.export_onnx import make_session, run_session
from brainct.preprocessing.dicom import preprocess_dicom
from brainct.preprocessing.window import model_input
from brainct.serving.app import create_app
from brainct.serving.settings import Settings

REPO = Path(__file__).resolve().parents[2]
SETTINGS = Settings(
    model_path=REPO / "artifacts/resnet50_baseline/onnx/model_int8.onnx",
    thresholds_path=REPO / "artifacts/resnet50_baseline/metrics_val.json",
    threads=4,
)
SAMPLES = sorted((REPO / "samples" / "dicom").glob("*.dcm"))[:10]

pytestmark = pytest.mark.skipif(
    not (SETTINGS.model_path.exists() and SETTINGS.thresholds_path.exists() and SAMPLES),
    reason="needs local artifacts/ and samples/",
)


def test_api_equals_offline_int8_on_real_samples():
    session = make_session(SETTINGS.model_path, 4)
    with TestClient(create_app(SETTINGS)) as client:
        for path in SAMPLES:
            data = path.read_bytes()
            hu, _ = preprocess_dicom(data)
            offline = 1 / (1 + np.exp(-run_session(session, model_input(hu)[None])[0]))

            body = client.post("/predict", files={"file": (path.name, data)}).json()

            served = [body["probabilities"][label] for label in LABELS]
            np.testing.assert_allclose(served, offline, rtol=0, atol=1e-6, err_msg=path.name)
            assert body["model"]["format"] == "onnx-int8"
