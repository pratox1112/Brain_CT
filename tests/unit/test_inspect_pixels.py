from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from brainct.data.zip_reader import TRAIN_PREFIX
from brainct.preprocessing.inspect_pixels import GROUPS, pixel_stats, run


def half_air_half_brain(offset: int) -> np.ndarray:
    """4x4 stored pixels: top half -> -1000 HU (air), bottom half -> 40 HU, given the offset."""
    pixels = np.empty((4, 4), dtype=np.int32)
    pixels[:2] = -1000 - offset
    pixels[2:] = 40 - offset
    return pixels


@pytest.mark.parametrize(
    ("pixel_representation", "bits_stored", "intercept"),
    [(0, 12, -1024), (1, 16, -1024), (1, 12, 0)],
)
def test_pixel_stats_hu(dicom_bytes, pixel_representation, bits_stored, intercept):
    data = dicom_bytes(
        "ID_x",
        intercept=intercept,
        pixel_representation=pixel_representation,
        bits_stored=bits_stored,
        pixels=half_air_half_brain(intercept),
    )

    stats = pixel_stats(data)

    assert stats["hu_min"] == -1000
    assert stats["hu_max"] == 40
    assert stats["frac_hu_air"] == 0.5
    assert stats["frac_hu_brain"] == 0.5
    assert stats["frac_hu_below_-1100"] == 0


def test_pixel_stats_flags_unsigned_with_zero_intercept(dicom_bytes):
    # Unsigned pixels with intercept 0 cannot go below 0 HU, so no air is found.
    data = dicom_bytes("ID_y", intercept=0, pixels=np.full((4, 4), 1024))

    stats = pixel_stats(data)

    assert stats["hu_min"] == 1024
    assert stats["frac_hu_air"] == 0


def test_run_end_to_end(tmp_path, dicom_bytes):
    # One image per pixel-format group; all in train except two. The shape is what the header
    # table reports (selection uses the table); the stored pixels are always a tiny 8x8 array.
    specs = [
        ("ID_u1024", 0, 12, -1024, (512, 512), "train"),
        ("ID_u1000", 0, 12, -1000, (512, 512), "val"),
        ("ID_u0", 0, 12, 0, (512, 512), "train"),
        ("ID_u1", 0, 12, 1, (512, 512), "train"),
        ("ID_s12_0", 1, 12, 0, (512, 512), "train"),
        ("ID_s16", 1, 16, -1024, (436, 436), "train"),
        ("ID_s16_0", 1, 16, 0, (512, 512), "test"),
    ]
    zip_path = tmp_path / "rsna.zip"
    rows = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for image_id, pr, bits, intercept, shape, split in specs:
            name = f"{TRAIN_PREFIX}{image_id}.dcm"
            pixels = np.zeros((8, 8), dtype=np.int32)
            zf.writestr(
                name,
                dicom_bytes(
                    image_id,
                    intercept=intercept,
                    pixel_representation=pr,
                    bits_stored=bits,
                    pixels=pixels,
                ),
            )
            rows.append(
                {
                    "image_id": image_id,
                    "patient_id": f"P_{image_id}",
                    "zip_path": name,
                    "rescale_intercept": float(intercept),
                    "pixel_representation": pr,
                    "bits_stored": bits,
                    "n_rows": shape[0],
                    "n_cols": shape[1],
                    "split": split,
                }
            )
    table = pd.DataFrame(rows)
    headers, splits = tmp_path / "headers.parquet", tmp_path / "splits.parquet"
    table.drop(columns="split").to_parquet(headers, index=False)
    table[["image_id", "split"]].to_parquet(splits, index=False)

    summary = run(headers, splits, zip_path, tmp_path / "stats", tmp_path / "edge")

    assert list(summary.index) == list(GROUPS)
    assert (summary["images"] == 1).all()
    assert (summary["errors"] == 0).all()
    edge = pd.read_csv(tmp_path / "edge" / "samples_edge.csv")
    # ID_s16_0 is an edge case but in test, so it must not be picked.
    assert sorted(edge["image_id"]) == ["ID_s12_0", "ID_s16", "ID_u0", "ID_u1"]
    assert (tmp_path / "edge" / "dicom" / "ID_s16.dcm").exists()
    assert (tmp_path / "stats" / "pixel_stats.csv").exists()
