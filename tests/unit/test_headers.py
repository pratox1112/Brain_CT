from __future__ import annotations

import json
import zipfile

import pandas as pd
import pytest

from brainct.data.headers import extract_headers, parse_header
from brainct.data.zip_reader import PIXEL_DATA_TAG, TRAIN_PREFIX, list_dicoms, read_header_bytes

VALID_IDS = ["ID_a0", "ID_a1", "ID_a2", "ID_b0", "ID_b1"]


def test_list_dicoms_returns_only_train_dicoms(rsna_zip):
    names = list_dicoms(rsna_zip, TRAIN_PREFIX)

    assert len(names) == 6  # 5 valid + 1 corrupt; the test folder and CSV are excluded
    assert all(n.startswith(TRAIN_PREFIX) and n.endswith(".dcm") for n in names)


def test_read_header_bytes_reads_only_a_prefix(tmp_path, dicom_bytes):
    data = dicom_bytes("ID_big", size=128)  # 32 KB of pixels, larger than max_bytes
    path = tmp_path / "big.zip"
    name = f"{TRAIN_PREFIX}ID_big.dcm"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, data)

    with zipfile.ZipFile(path) as zf:
        head, full_read = read_header_bytes(zf, name, max_bytes=16384)
        tiny, tiny_full_read = read_header_bytes(zf, name, max_bytes=64)

    assert len(head) == 16384
    assert PIXEL_DATA_TAG in head
    assert not full_read
    # 64 bytes cannot contain the whole header, so it falls back to reading the whole file.
    assert tiny == data
    assert tiny_full_read


def test_parse_header_extracts_fields(dicom_bytes):
    data = dicom_bytes("ID_x1", patient_id="ID_p9", z=12.5, window_center=[40, 40])

    row = parse_header(data, f"{TRAIN_PREFIX}ID_x1.dcm")

    assert row["image_id"] == "ID_x1"
    assert row["sop_instance_uid"] == "ID_x1"
    assert row["patient_id"] == "ID_p9"
    assert row["ipp_z"] == 12.5
    assert row["iop"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert row["window_center"] == 40.0  # multi-valued tag -> first value
    assert row["rescale_slope"] == 1.0
    assert row["rescale_intercept"] == -1024.0
    assert (row["n_rows"], row["n_cols"]) == (8, 8)
    assert (row["bits_stored"], row["pixel_representation"]) == (12, 0)


def test_parse_header_rejects_non_dicom():
    # Any parse error is acceptable here; the extractor records it as a failure.
    with pytest.raises(Exception):  # noqa: B017
        parse_header(b"this is not a dicom file", "ID_bad.dcm")


@pytest.mark.parametrize("workers", [1, 2])
def test_extract_headers_end_to_end(rsna_zip, tmp_path, workers):
    out = tmp_path / f"out_{workers}"

    summary = extract_headers(rsna_zip, out, workers=workers)

    df = pd.read_parquet(out / "dicom_headers.parquet")
    failures = pd.read_csv(out / "header_failures.csv")
    saved = json.loads((out / "header_summary.json").read_text())

    assert df["image_id"].tolist() == VALID_IDS
    assert failures["image_id"].tolist() == ["ID_bad"]
    assert summary["files_listed"] == 6
    assert summary["parsed"] == 5
    assert summary["failed"] == 1
    assert summary["unique_patients"] == 2
    assert summary["unique_studies"] == 2
    assert summary["unique_series"] == 2
    assert summary["duplicate_image_ids"] == 0
    assert summary["image_id_ne_sop_uid"] == 0
    assert summary["value_counts"]["rescale_intercept"] == {"-1024.0": 5}
    assert saved["parsed"] == 5


def test_extract_headers_falls_back_to_full_reads(rsna_zip, tmp_path):
    summary = extract_headers(rsna_zip, tmp_path / "out", max_bytes=64)

    assert summary["parsed"] == 5
    assert summary["full_file_reads"] >= 5


def test_extract_headers_limit(rsna_zip, tmp_path):
    summary = extract_headers(rsna_zip, tmp_path / "out", limit=2)

    assert summary["files_listed"] == 2
    assert summary["parsed"] + summary["failed"] == 2
