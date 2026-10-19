"""Fails if the served model's outputs on the local sample DICOMs drift. Laptop only.

The real INT8 model and the RSNA samples can't be published, so this needs the git-ignored
artifacts/ and samples/ folders and is skipped elsewhere (including CI). It uses the serving
code path (InferenceModel.prepare + predict_batch), i.e. exactly what /predict computes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from brainct.constants import LABELS
from brainct.serving.model import InferenceModel

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "artifacts/resnet50_baseline/onnx/model_int8.onnx"
THRESHOLDS = REPO / "artifacts/resnet50_baseline/metrics_val.json"
EXPECTED = REPO / "artifacts/regression/expected_int8_samples.json"
TOLERANCE = 1e-5

pytestmark = pytest.mark.skipif(
    not (MODEL.exists() and THRESHOLDS.exists() and EXPECTED.exists()),
    reason="needs local artifacts/ (run scripts/make_regression_expected.py --model)",
)


def test_int8_outputs_match_stored_expectations():
    expected = json.loads(EXPECTED.read_text())
    model = InferenceModel(MODEL, THRESHOLDS, "regression", threads=4)
    assert model.version == expected["model_version"], "a different model file is in artifacts/"

    for name, reference in expected["probabilities"].items():
        path = REPO / expected["sample_dir"] / name
        probs = model.predict_batch(model.prepare(path.read_bytes()).x[None])[0]
        np.testing.assert_allclose(
            probs, [reference[label] for label in LABELS], rtol=0, atol=TOLERANCE, err_msg=name
        )
