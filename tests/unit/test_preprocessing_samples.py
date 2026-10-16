"""Preprocessing checks on real RSNA DICOMs from samples/ and samples_edge/.

These folders exist only on the owner's laptop (RSNA data is never committed), so the tests are
skipped in CI and anywhere the folders are missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brainct.preprocessing.dicom import IMAGE_SIZE, preprocess_dicom
from brainct.preprocessing.hu import HU_MAX, HU_MIN
from brainct.preprocessing.window import model_input

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIRS = [REPO_ROOT / "samples", REPO_ROOT / "samples_edge"]


def _sample_files() -> list[Path]:
    return sorted(p for d in SAMPLE_DIRS for p in (d / "dicom").glob("*.dcm"))


def _edge_reasons() -> dict[str, str]:
    csv = REPO_ROOT / "samples_edge" / "samples_edge.csv"
    if not csv.exists():
        return {}
    table = pd.read_csv(csv)
    return dict(zip(table["image_id"], table["reasons"], strict=True))


FILES = _sample_files()
EDGE_REASONS = _edge_reasons()

pytestmark = pytest.mark.skipif(not FILES, reason="no local RSNA sample DICOMs")


@pytest.mark.parametrize("path", FILES, ids=[p.stem for p in FILES])
def test_real_dicom_preprocesses_to_plausible_ct(path):
    image, info = preprocess_dicom(path.read_bytes())

    assert image.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert image.dtype == np.int16
    assert HU_MIN <= image.min() and image.max() <= HU_MAX
    # Every head CT slice has air around the head; the broken group had none before the fix.
    assert (image < -900).mean() > 0.05
    # Implausible values: almost nothing should sit at the top of the HU range.
    assert (image > 3000).mean() < 0.01

    x = model_input(image)
    assert x.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert np.isfinite(x).all()


@pytest.mark.skipif(not EDGE_REASONS, reason="no samples_edge/samples_edge.csv")
def test_edge_cases_take_the_expected_branch():
    for image_id, reasons in EDGE_REASONS.items():
        data = (REPO_ROOT / "samples_edge" / "dicom" / f"{image_id}.dcm").read_bytes()
        _, info = preprocess_dicom(data)
        broken = "unsigned12_intercept_0" in reasons or "intercept_1" in reasons
        assert info.sign_fixed == broken, image_id
        if "not_512x512" in reasons:
            assert info.original_shape != (512, 512), image_id
            assert info.padded == (info.original_shape[0] != info.original_shape[1]), image_id
