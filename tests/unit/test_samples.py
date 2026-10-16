from __future__ import annotations

import json
import zipfile

import pandas as pd

from brainct.data.labels import LABELS, SUBTYPES
from brainct.data.samples import make_samples, select_samples
from brainct.data.zip_reader import TRAIN_PREFIX


def make_images(n_per_split: int = 40) -> pd.DataFrame:
    """Synthetic joined table: IDs, split, labels and header fields."""
    rows = []
    for split in ("train", "val", "test"):
        for i in range(n_per_split):
            image_id = f"ID_{split}_{i:03d}"
            subtype = SUBTYPES[i % 6] if i % 6 < len(SUBTYPES) else None  # every 6th is normal
            rows.append(
                {
                    "image_id": image_id,
                    "patient_id": f"P_{split}_{i // 2}",  # 2 images per patient
                    "study_instance_uid": f"ST_{split}_{i // 2}",
                    "series_instance_uid": f"SE_{split}_{i // 2}",
                    "split": split,
                    "any": int(subtype is not None),
                    **{s: int(s == subtype) for s in SUBTYPES},
                    "zip_path": f"{TRAIN_PREFIX}{image_id}.dcm",
                    "rescale_intercept": 0.0 if i == 7 else -1024.0,
                    "pixel_representation": i % 2,
                    "bits_stored": 12 if i % 4 < 2 else 16,
                    "n_rows": 436 if i == 9 else 512,
                    "n_cols": 436 if i == 9 else 512,
                }
            )
    return pd.DataFrame(rows)


def test_select_samples_uses_only_test_split_and_covers_categories():
    selected = select_samples(make_images(), n_normal=3, n_per_subtype=2, n_per_edge_case=1)

    assert (selected["split"] == "test").all()
    reasons = ";".join(selected["reasons"]).split(";")
    assert reasons.count("normal") == 3
    for subtype in SUBTYPES:
        assert reasons.count(subtype) == 2
    assert reasons.count("intercept_0") == 1
    assert reasons.count("not_512x512") == 1
    assert "intercept_1" not in reasons  # no such image exists; skipped, not an error
    assert selected["image_id"].is_unique


def test_select_samples_prefers_distinct_patients():
    selected = select_samples(make_images(), n_normal=3, n_per_subtype=0, n_per_edge_case=0)

    assert selected["patient_id"].nunique() == 3


def test_select_samples_is_deterministic():
    a = select_samples(make_images(), seed=1)
    b = select_samples(make_images().sample(frac=1, random_state=0), seed=1)

    assert a["image_id"].tolist() == b["image_id"].tolist()


def test_make_samples_extracts_exact_bytes(tmp_path, dicom_bytes):
    images = make_images(n_per_split=12)
    zip_path = tmp_path / "rsna.zip"
    originals = {}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for image_id, zip_name in zip(images["image_id"], images["zip_path"], strict=True):
            originals[image_id] = dicom_bytes(image_id)
            zf.writestr(zip_name, originals[image_id])

    paths = {name: tmp_path / f"{name}.parquet" for name in ["split_images", "labels", "headers"]}
    id_cols = ["image_id", "patient_id", "study_instance_uid", "series_instance_uid"]
    images[[*id_cols, "split"]].to_parquet(paths["split_images"], index=False)
    images[[*id_cols, *LABELS]].to_parquet(paths["labels"], index=False)
    images.drop(columns=["split", *LABELS]).to_parquet(paths["headers"], index=False)
    out = tmp_path / "samples"

    summary = make_samples(
        paths["split_images"], paths["labels"], paths["headers"], zip_path, out, n_normal=2
    )

    table = pd.read_csv(out / "samples.csv")
    assert summary["selected_images"] == len(table) > 0
    assert table["image_id"].str.startswith("ID_test_").all()
    for image_id in table["image_id"]:
        assert (out / "dicom" / f"{image_id}.dcm").read_bytes() == originals[image_id]
    saved = json.loads((out / "samples_summary.json").read_text())
    assert saved["all_from_test_split"] is True
    assert saved["bytes_written"] == sum(len(originals[i]) for i in table["image_id"])
