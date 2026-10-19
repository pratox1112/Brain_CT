"""Deterministic synthetic DICOMs covering every preprocessing branch, and their fingerprints.

The regression test compares today's preprocessing of these cases with the fingerprints stored
in expected_synthetic.json. Any change in behaviour - HU conversion, the sign fix, clipping,
padding, resizing, windowing, normalization - changes a fingerprint and fails the test.

Synthetic, so safe to commit and to run in CI (no RSNA data). To accept an *intended* change:
    uv run python scripts/make_regression_expected.py --synthetic --reason "why it changed"
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np

from brainct.preprocessing.dicom import preprocess_dicom
from brainct.preprocessing.window import model_input

EXPECTED_PATH = Path(__file__).with_name("expected_synthetic.json")


def _cases(make: Callable[..., bytes]) -> dict[str, bytes]:
    rng = np.random.default_rng(2026)
    head = rng.integers(0, 1800, (512, 512))  # unsigned stored values, roughly -1024..776 HU
    wrapped = np.where(rng.random((512, 512)) < 0.5, 3096, rng.integers(0, 80, (512, 512)))
    padded = rng.integers(-1000, 1500, (490, 638))
    padded[:40, :] = -3024  # scanner padding outside the field of view
    return {
        "unsigned12_intercept_m1024_512": make("ID_case1", pixels=head),
        # mislabelled signed 12-bit: 3096 must become -1000 HU after the sign fix
        "unsigned12_intercept_0_signfix": make("ID_case2", intercept=0, pixels=wrapped),
        "signed16_intercept_0_nonsquare_padding": make(
            "ID_case3", intercept=0, pixel_representation=1, bits_stored=16, pixels=padded
        ),
        "signed12_intercept_0_already_hu": make(
            "ID_case4",
            intercept=0,
            pixel_representation=1,
            bits_stored=12,
            pixels=rng.integers(-1000, 1500, (512, 512)),
        ),
    }


def _fingerprint(array: np.ndarray) -> dict:
    data = np.ascontiguousarray(array)
    return {
        "dtype": str(data.dtype),
        "shape": list(data.shape),
        "sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": round(float(data.mean()), 6),
    }


def fingerprints(make: Callable[..., bytes]) -> dict[str, dict]:
    """For each case: fingerprints of the cached HU image and of the model input, plus flags."""
    out = {}
    for name, data in _cases(make).items():
        hu, info = preprocess_dicom(data)
        out[name] = {
            "hu": _fingerprint(hu),
            "model_input": _fingerprint(model_input(hu)),
            "sign_fixed": info.sign_fixed,
            "padded": info.padded,
            "original_shape": list(info.original_shape),
        }
    return out
