"""Build the per-image label table from RSNA's stage_2_train.csv and join it with the headers.

RSNA stores labels in long format, one row per image *and* label type:

    ID,Label
    ID_12cadc6af_epidural,0
    ID_12cadc6af_intraparenchymal,0
    ...

This module turns that into one row per image with 6 binary columns, checks it for duplicates,
conflicts and gaps, and attaches patient / study / series IDs from the header table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from brainct.constants import LABELS, SUBTYPES

HEADER_KEYS = ["image_id", "patient_id", "study_instance_uid", "series_instance_uid"]


def read_long_labels(csv_path: str | Path) -> pd.DataFrame:
    """Read RSNA's long-format CSV into columns image_id, label, value."""
    raw = pd.read_csv(csv_path, dtype={"ID": str, "Label": "int8"})
    parts = raw["ID"].str.rsplit("_", n=1, expand=True)
    return pd.DataFrame({"image_id": parts[0], "label": parts[1], "value": raw["Label"]})


def to_wide(long: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Pivot long labels to one row per image. Raises on unknown, conflicting or missing labels."""
    unknown = sorted(set(long["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"unknown label types: {unknown}")

    exact_duplicates = int(long.duplicated().sum())
    deduped = long.drop_duplicates()
    conflicting = deduped.duplicated(["image_id", "label"], keep=False)
    if conflicting.any():
        n_images = deduped.loc[conflicting, "image_id"].nunique()
        raise ValueError(f"{n_images} images have conflicting values for the same label")

    wide = deduped.pivot(index="image_id", columns="label", values="value")
    incomplete = wide.isna().any(axis=1)
    if incomplete.any():
        raise ValueError(f"{int(incomplete.sum())} images are missing one or more of {LABELS}")

    wide = wide[LABELS].astype("int8").reset_index()
    wide.columns.name = None
    stats = {
        "label_rows": len(long),
        "exact_duplicate_rows": exact_duplicates,
        "images": len(wide),
        # RSNA defines `any` as "at least one subtype"; count rows where that does not hold.
        "any_inconsistent_with_subtypes": int((wide["any"] != wide[SUBTYPES].max(axis=1)).sum()),
    }
    return wide, stats


def build_label_table(
    labels_csv: str | Path, headers_path: str | Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return one row per image (IDs + 6 labels) and stats. Every image must have both."""
    wide, stats = to_wide(read_long_labels(labels_csv))
    headers = pd.read_parquet(headers_path, columns=HEADER_KEYS)

    merged = wide.merge(headers, on="image_id", how="outer", indicator=True)
    only_labels = int((merged["_merge"] == "left_only").sum())
    only_headers = int((merged["_merge"] == "right_only").sum())
    if only_labels or only_headers:
        raise ValueError(
            f"label/header mismatch: {only_labels} images have labels but no header, "
            f"{only_headers} have a header but no labels"
        )

    table = merged[HEADER_KEYS + LABELS].sort_values("image_id").reset_index(drop=True)
    stats["image_positives"] = {label: int(table[label].sum()) for label in LABELS}
    return table, stats
