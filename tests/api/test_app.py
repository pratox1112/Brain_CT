"""API tests with FastAPI's TestClient, using a tiny ONNX model and synthetic DICOMs."""

from __future__ import annotations

import io
import json

import numpy as np
import pydicom
import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn

from brainct.constants import DISCLAIMER, LABELS
from brainct.optimization.export_onnx import export_onnx, make_session, run_session
from brainct.preprocessing.dicom import preprocess_dicom
from brainct.preprocessing.window import model_input
from brainct.serving.app import create_app
from brainct.serving.settings import Settings

SIZE = 32  # tiny model input; the real model uses 256


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(4, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).mean(dim=(2, 3)))


@pytest.fixture
def model_files(tmp_path):
    torch.manual_seed(0)
    model_path = tmp_path / "tiny.onnx"
    export_onnx(TinyNet().eval(), model_path, image_size=SIZE)
    thresholds_path = tmp_path / "metrics_val.json"
    thresholds_path.write_text(json.dumps({"thresholds": dict.fromkeys(LABELS, 0.5)}))
    return model_path, thresholds_path


@pytest.fixture
def client(model_files):
    model_path, thresholds_path = model_files
    settings = Settings(model_path=model_path, thresholds_path=thresholds_path, threads=1)
    with TestClient(create_app(settings)) as test_client:  # runs startup (model load)
        yield test_client


def _upload(client, data: bytes, name: str = "slice.dcm"):
    return client.post("/predict", files={"file": (name, data, "application/dicom")})


def _ct(dicom_bytes, **kwargs) -> bytes:
    return dicom_bytes("ID_api", pixels=np.full((16, 16), 1064), **kwargs)


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["status"] == "ready"


def test_predict_returns_probabilities_and_disclaimer(client, dicom_bytes):
    response = _upload(client, _ct(dicom_bytes))

    assert response.status_code == 200
    body = response.json()
    assert set(body["probabilities"]) == set(LABELS)
    assert all(0 <= p <= 1 for p in body["probabilities"].values())
    for label, p in body["probabilities"].items():
        assert body["predictions"][label] == (p >= body["thresholds"][label])
    assert body["disclaimer"] == DISCLAIMER
    assert body["model"]["input_size"] == SIZE and len(body["model"]["version"]) == 12
    assert body["timing"]["total_ms"] >= body["timing"]["inference_ms"] >= 0


def test_predict_matches_offline_onnx(client, model_files, dicom_bytes):
    # Serving must equal an offline run of the same model on the same preprocessing.
    data = dicom_bytes("ID_x", pixels=np.random.default_rng(0).integers(0, 2000, (20, 20)))
    hu, _ = preprocess_dicom(data, size=SIZE)
    logits = run_session(make_session(model_files[0], 1), model_input(hu)[None])
    offline = 1 / (1 + np.exp(-logits[0]))

    served = _upload(client, data).json()["probabilities"]

    np.testing.assert_allclose([served[label] for label in LABELS], offline, rtol=0, atol=1e-6)


def test_predict_reports_preprocessing_branches(client, dicom_bytes):
    data = dicom_bytes("ID_fix", intercept=0, pixels=np.full((8, 16), 3096))  # mislabelled, wide

    body = _upload(client, data).json()

    assert body["preprocessing"] == {"original_shape": [8, 16], "sign_fixed": True, "padded": True}


def test_empty_upload_is_400(client):
    response = _upload(client, b"")

    assert response.status_code == 400
    assert response.json()["disclaimer"] == DISCLAIMER


def test_oversized_upload_is_413(model_files, dicom_bytes):
    model_path, thresholds_path = model_files
    settings = Settings(
        model_path=model_path, thresholds_path=thresholds_path, max_upload_mb=0.0001
    )
    with TestClient(create_app(settings)) as small_client:
        response = _upload(small_client, _ct(dicom_bytes))

    assert response.status_code == 413


def test_non_dicom_is_422(client):
    response = _upload(client, b"this is a text file, not a DICOM", name="notes.txt")

    assert response.status_code == 422
    assert "DICOM" in response.json()["detail"]


def test_dicom_missing_tags_is_422(client, dicom_bytes):
    ds = pydicom.dcmread(io.BytesIO(_ct(dicom_bytes)))
    del ds.RescaleIntercept
    buffer = io.BytesIO()
    ds.save_as(buffer)

    response = _upload(client, buffer.getvalue())

    assert response.status_code == 422
    assert "RescaleIntercept" in response.json()["detail"]


def test_non_ct_modality_is_422(client, dicom_bytes):
    ds = pydicom.dcmread(io.BytesIO(_ct(dicom_bytes)))
    ds.Modality = "MR"
    buffer = io.BytesIO()
    ds.save_as(buffer)

    response = _upload(client, buffer.getvalue())

    assert response.status_code == 422
    assert "CT" in response.json()["detail"]


def test_missing_model_reports_not_ready(tmp_path, dicom_bytes):
    settings = Settings(model_path=tmp_path / "nope.onnx", thresholds_path=tmp_path / "nope.json")
    with TestClient(create_app(settings)) as broken:
        assert broken.get("/health").status_code == 200  # alive...
        assert broken.get("/ready").status_code == 503  # ...but not ready
        assert _upload(broken, _ct(dicom_bytes)).status_code == 503


def test_metrics_count_requests(client, dicom_bytes):
    _upload(client, _ct(dicom_bytes))
    _upload(client, b"")

    text = client.get("/metrics").text

    assert 'brainct_requests_total{endpoint="/predict",status="200"} 1.0' in text
    assert 'brainct_requests_total{endpoint="/predict",status="400"} 1.0' in text
    assert "brainct_stage_seconds_count" in text
    assert "brainct_model_loaded 1.0" in text


def test_settings_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAINCT_MODEL_PATH", str(tmp_path / "m.onnx"))
    monkeypatch.setenv("BRAINCT_THREADS", "3")

    settings = Settings.from_env()

    assert settings.model_path == tmp_path / "m.onnx"
    assert settings.threads == 3
    assert settings.max_upload_bytes == 10 * 1024 * 1024
