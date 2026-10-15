from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from brainct.data.labels import LABELS, SUBTYPES
from brainct.data.split import (
    SplitLeakError,
    assign_splits,
    check_no_overlap,
    make_split,
    patient_strata,
)


def make_table(n_patients: int = 300, n_epidural: int = 20, seed: int = 0) -> pd.DataFrame:
    """Synthetic label table: 4 images per patient, some patients with two studies."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        for i in range(4):
            study = f"ST_{p}_{i // 2}" if p % 10 == 0 else f"ST_{p}_0"
            subtypes = (rng.random(len(SUBTYPES)) < 0.1).astype(int)
            subtypes[0] = int(p < n_epidural and i == 0)  # epidural is rare
            rows.append(
                {
                    "image_id": f"ID_{p}_{i}",
                    "patient_id": f"P_{p:04d}",
                    "study_instance_uid": study,
                    "series_instance_uid": f"SE_{study}",
                    "any": int(subtypes.max()),
                    **dict(zip(SUBTYPES, subtypes.tolist(), strict=True)),
                }
            )
    return pd.DataFrame(rows)


def split_images(table: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    splits = assign_splits(patient_strata(table), seed=seed)
    return table.assign(split=table["patient_id"].map(splits))


def test_patient_strata_uses_rarest_subtype():
    table = make_table()
    strata = patient_strata(table)

    epidural_patients = table.loc[table["epidural"] == 1, "patient_id"].unique()
    assert (strata[epidural_patients] == "epidural").all()
    no_bleed = table.groupby("patient_id")["any"].max() == 0
    assert (strata[no_bleed[no_bleed].index] == "none").all()


def test_split_has_no_overlap_and_rough_80_10_10():
    images = split_images(make_table())

    check_no_overlap(images)  # must not raise
    fractions = images.groupby("split")["patient_id"].nunique() / images["patient_id"].nunique()
    assert fractions["train"] == pytest.approx(0.8, abs=0.03)
    assert fractions["val"] == pytest.approx(0.1, abs=0.03)
    assert fractions["test"] == pytest.approx(0.1, abs=0.03)


def test_rare_subtype_reaches_val_and_test():
    images = split_images(make_table(n_epidural=20))

    epidural_splits = set(images.loc[images["epidural"] == 1, "split"])
    assert epidural_splits == {"train", "val", "test"}


def test_split_is_deterministic_and_order_independent():
    table = make_table()
    shuffled = table.sample(frac=1, random_state=3)

    a = assign_splits(patient_strata(table), seed=7).sort_index()
    b = assign_splits(patient_strata(shuffled), seed=7).sort_index()
    c = assign_splits(patient_strata(table), seed=8).sort_index()

    assert a.equals(b)
    assert not a.equals(c)


def test_overlap_check_catches_patient_leak():
    images = split_images(make_table())
    leaked = images.copy()
    first = leaked.index[0]
    leaked.loc[first, "split"] = "test" if leaked.loc[first, "split"] != "test" else "train"

    with pytest.raises(SplitLeakError, match="patient_id"):
        check_no_overlap(leaked)


def test_overlap_check_catches_study_shared_across_patients():
    images = split_images(make_table())
    train_row = images[images["split"] == "train"].index[0]
    test_row = images[images["split"] == "test"].index[0]
    images.loc[test_row, "study_instance_uid"] = images.loc[train_row, "study_instance_uid"]

    with pytest.raises(SplitLeakError, match="study_instance_uid"):
        check_no_overlap(images)


def test_make_split_end_to_end(tmp_path):
    table = make_table(n_patients=100, n_epidural=10)
    csv, headers, out = tmp_path / "labels.csv", tmp_path / "headers.parquet", tmp_path / "out"
    long = table.melt(id_vars=["image_id"], value_vars=LABELS, var_name="label", value_name="Label")
    long.assign(ID=long["image_id"] + "_" + long["label"])[["ID", "Label"]].to_csv(csv, index=False)
    table.drop(columns=LABELS).to_parquet(headers, index=False)

    report = make_split(csv, headers, out)

    assert report["overlap_check"] == "passed"
    assert report["total"]["images"] == len(table)
    assert report["total"]["patients"] == 100
    for name in ["labels.parquet", "split_images.parquet", "split_patients.csv"]:
        assert (out / name).exists()
    saved = json.loads((out / "split_report.json").read_text())
    assert saved["train"]["patients"] + saved["val"]["patients"] + saved["test"]["patients"] == 100
