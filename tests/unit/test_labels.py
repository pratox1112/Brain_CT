from __future__ import annotations

import pandas as pd
import pytest

from brainct.data.labels import LABELS, build_label_table, read_long_labels, to_wide


def write_long_csv(path, labels_by_image: dict[str, list[int]], extra_rows=()):
    rows = [
        (f"{image_id}_{label}", value)
        for image_id, values in labels_by_image.items()
        for label, value in zip(LABELS, values, strict=True)
    ]
    rows.extend(extra_rows)
    pd.DataFrame(rows, columns=["ID", "Label"]).to_csv(path, index=False)


def write_headers(path, image_ids, patient_of=lambda i: f"P_{i}"):
    pd.DataFrame(
        {
            "image_id": image_ids,
            "patient_id": [patient_of(i) for i in image_ids],
            "study_instance_uid": [f"ST_{i}" for i in image_ids],
            "series_instance_uid": [f"SE_{i}" for i in image_ids],
        }
    ).to_parquet(path, index=False)


#   order:            any epi ipa ivh sah sdh
LABELS_A = {
    "ID_a": [0, 0, 0, 0, 0, 0],
    "ID_b": [1, 1, 0, 0, 0, 1],
    "ID_c": [1, 0, 0, 1, 0, 0],
}


def test_read_long_labels_splits_image_and_label(tmp_path):
    csv = tmp_path / "labels.csv"
    write_long_csv(csv, {"ID_12cadc6af": [0, 1, 0, 0, 0, 0]})

    long = read_long_labels(csv)

    assert set(long["image_id"]) == {"ID_12cadc6af"}
    assert list(long["label"]) == LABELS
    assert list(long["value"]) == [0, 1, 0, 0, 0, 0]


def test_to_wide_pivots_and_removes_exact_duplicates(tmp_path):
    csv = tmp_path / "labels.csv"
    write_long_csv(csv, LABELS_A, extra_rows=[("ID_b_epidural", 1)])  # exact duplicate row

    wide, stats = to_wide(read_long_labels(csv))

    assert list(wide.columns) == ["image_id", *LABELS]
    assert wide.set_index("image_id").loc["ID_b"].tolist() == [1, 1, 0, 0, 0, 1]
    assert stats["exact_duplicate_rows"] == 1
    assert stats["images"] == 3
    assert stats["any_inconsistent_with_subtypes"] == 0


def test_to_wide_counts_any_inconsistency(tmp_path):
    csv = tmp_path / "labels.csv"
    write_long_csv(csv, {"ID_x": [1, 0, 0, 0, 0, 0]})  # any=1 but no subtype

    _, stats = to_wide(read_long_labels(csv))

    assert stats["any_inconsistent_with_subtypes"] == 1


def test_to_wide_rejects_conflicting_duplicates(tmp_path):
    csv = tmp_path / "labels.csv"
    write_long_csv(csv, LABELS_A, extra_rows=[("ID_b_epidural", 0)])  # contradicts the 1

    with pytest.raises(ValueError, match="conflicting"):
        to_wide(read_long_labels(csv))


def test_to_wide_rejects_missing_labels(tmp_path):
    csv = tmp_path / "labels.csv"
    write_long_csv(csv, LABELS_A)
    long = read_long_labels(csv)
    long = long[~((long["image_id"] == "ID_c") & (long["label"] == "subdural"))]

    with pytest.raises(ValueError, match="missing"):
        to_wide(long)


def test_build_label_table_joins_headers(tmp_path):
    csv, headers = tmp_path / "labels.csv", tmp_path / "headers.parquet"
    write_long_csv(csv, LABELS_A)
    write_headers(headers, list(LABELS_A))

    table, stats = build_label_table(csv, headers)

    assert len(table) == 3
    assert table.set_index("image_id").loc["ID_c", "patient_id"] == "P_ID_c"
    assert stats["image_positives"]["any"] == 2


def test_build_label_table_rejects_missing_headers(tmp_path):
    csv, headers = tmp_path / "labels.csv", tmp_path / "headers.parquet"
    write_long_csv(csv, LABELS_A)
    write_headers(headers, ["ID_a", "ID_b"])  # ID_c has labels but no header

    with pytest.raises(ValueError, match="mismatch"):
        build_label_table(csv, headers)
