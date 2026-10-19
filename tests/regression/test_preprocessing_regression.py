"""Fails if preprocessing behaviour changes (see synthetic_cases.py). Runs in CI."""

from __future__ import annotations

import json

import pytest

from dicom_factory import make_dicom_bytes
from synthetic_cases import EXPECTED_PATH, fingerprints

pytestmark = pytest.mark.skipif(
    not EXPECTED_PATH.exists(), reason="run scripts/make_regression_expected.py --synthetic"
)


def test_preprocessing_matches_stored_fingerprints():
    expected = json.loads(EXPECTED_PATH.read_text())["cases"]
    actual = fingerprints(make_dicom_bytes)

    assert set(actual) == set(expected)
    for name in expected:
        for field in ("sign_fixed", "padded", "original_shape"):
            assert actual[name][field] == expected[name][field], f"{name}: {field}"
        for stage in ("hu", "model_input"):
            assert actual[name][stage] == expected[name][stage], (
                f"{name}: {stage} changed - preprocessing behaviour differs from the stored "
                "fingerprint. If intended, regenerate with scripts/make_regression_expected.py"
            )
