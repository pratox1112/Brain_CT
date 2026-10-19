"""Shared test fixtures: synthetic DICOMs and a small zip laid out like the RSNA download.

No real RSNA data is used anywhere in the tests (licence terms; CI has no access to it).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brainct.data.build_cache import CACHE_FILE, INDEX_FILE
from brainct.data.labels import LABELS, SUBTYPES
from brainct.data.zip_reader import TRAIN_PREFIX
from dicom_factory import make_dicom_bytes  # tests/ is on sys.path (rootdir conftest)

TEST_PREFIX = "rsna-intracranial-hemorrhage-detection/stage_2_test/"


@pytest.fixture
def dicom_bytes():
    """Factory fixture: dicom_bytes("ID_x", patient_id=...) -> bytes."""
    return make_dicom_bytes


@pytest.fixture
def rsna_zip(tmp_path: Path) -> Path:
    """RSNA-style zip: 5 valid train DICOMs (2 patients), 1 corrupt one, a test DICOM, a CSV."""
    path = tmp_path / "rsna.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in range(3):
            zf.writestr(
                f"{TRAIN_PREFIX}ID_a{i}.dcm",
                make_dicom_bytes(f"ID_a{i}", patient_id="ID_p1", z=5.0 * i),
            )
        for i in range(2):
            zf.writestr(
                f"{TRAIN_PREFIX}ID_b{i}.dcm",
                make_dicom_bytes(
                    f"ID_b{i}",
                    patient_id="ID_p2",
                    study_uid="ID_st2",
                    series_uid="ID_se2",
                    window_center=[40, 40],
                ),
            )
        zf.writestr(f"{TRAIN_PREFIX}ID_bad.dcm", b"this is not a dicom file")
        zf.writestr(f"{TEST_PREFIX}ID_t0.dcm", make_dicom_bytes("ID_t0"))
        zf.writestr("rsna-intracranial-hemorrhage-detection/stage_2_train.csv", "ID,Label\n")
    return path


# Tiny HU cache for training / evaluation / ONNX tests (images are TINY_SIZE x TINY_SIZE).
TINY_SIZE = 32


@pytest.fixture
def tiny_cache(tmp_path):
    """48 images (16 per split, 2 per patient); every label has positives and negatives in
    every split; one train image is marked ok=False like the real corrupt file."""
    n = 48
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = np.lib.format.open_memmap(
        cache_dir / CACHE_FILE, mode="w+", dtype=np.int16, shape=(n, TINY_SIZE, TINY_SIZE)
    )
    cache[:] = np.random.default_rng(0).integers(-1024, 1500, size=(n, TINY_SIZE, TINY_SIZE))
    del cache

    splits = ["train", "val", "test"]
    ids = [f"ID_{k:03d}" for k in range(n)]
    index = pd.DataFrame(
        {
            "row": range(n),
            "image_id": ids,
            "split": [splits[k // 16] for k in range(n)],
            "ok": [k != 3 for k in range(n)],
        }
    )
    index.to_parquet(cache_dir / INDEX_FILE, index=False)

    labels = pd.DataFrame({"image_id": ids, "patient_id": [f"P_{k // 2}" for k in range(n)]})
    for j, subtype in enumerate(SUBTYPES):
        labels[subtype] = [int(k % 6 == j) for k in range(n)]  # k % 6 == 5 -> no bleed
    labels["any"] = labels[SUBTYPES].max(axis=1)
    labels_path = tmp_path / "labels.parquet"
    labels[["image_id", "patient_id", *LABELS]].to_parquet(labels_path, index=False)
    return {"cache_dir": cache_dir, "labels_path": labels_path, "tmp": tmp_path}
