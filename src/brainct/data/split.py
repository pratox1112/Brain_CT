"""Patient-level train/val/test split, with an automated check that nothing leaks across splits.

Usage (on the cluster, inside a Slurm job - see jobs/make_split.sh):

    python -m brainct.data.split \
        --labels-csv /scratch/$USER/rsna/stage_2_train.csv \
        --headers ~/brainct_outputs/metadata/dicom_headers.parquet \
        --out ~/brainct_outputs/splits

How patients are assigned:
    Each patient gets a stratum: the rarest hemorrhage subtype found anywhere in their scans,
    or "none". Within each stratum, patients are shuffled with a fixed seed and divided
    80/10/10. This spreads rare subtypes (epidural above all) across all three splits.
    All images of a patient go to the same split.

Outputs written to --out:
    labels.parquet          one row per image: IDs + 6 labels
    split_images.parquet    image -> patient / study / series -> split
    split_patients.csv      patient -> stratum -> split
    split_report.json       counts per split, for docs/results.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from brainct.data.labels import LABELS, SUBTYPES, build_label_table

SPLITS = ("train", "val", "test")
GROUP_KEYS = ("patient_id", "study_instance_uid", "series_instance_uid")
DEFAULT_SEED = 2026


class SplitLeakError(ValueError):
    """A patient, study or series appears in more than one split."""


def patient_strata(table: pd.DataFrame) -> pd.Series:
    """Return each patient's stratum: the rarest subtype they have, or "none"."""
    flags = table.groupby("patient_id")[SUBTYPES].max()
    # Rarest first, by number of patients with the subtype; ties keep column order.
    rarest_first = flags.sum().sort_values(kind="stable").index.tolist()
    strata = pd.Series("none", index=flags.index, dtype=object)
    # Assign from most common to rarest, so a patient ends up with their rarest subtype.
    for subtype in reversed(rarest_first):
        strata[flags[subtype] == 1] = subtype
    return strata


def assign_splits(
    strata: pd.Series, val_frac: float = 0.1, test_frac: float = 0.1, seed: int = DEFAULT_SEED
) -> pd.Series:
    """Assign each patient (index of `strata`) to train / val / test, stratum by stratum."""
    rng = np.random.default_rng(seed)
    splits = pd.Series(index=strata.index, dtype=object)
    for stratum in sorted(strata.unique()):
        # Sorting first makes the result independent of input row order.
        members = strata.index[(strata == stratum).to_numpy()].to_numpy()
        patients = rng.permutation(np.sort(members))
        n_test = round(len(patients) * test_frac)
        n_val = round(len(patients) * val_frac)
        splits.loc[patients[:n_test]] = "test"
        splits.loc[patients[n_test : n_test + n_val]] = "val"
        splits.loc[patients[n_test + n_val :]] = "train"
    return splits


def check_no_overlap(images: pd.DataFrame, split_col: str = "split") -> None:
    """Raise SplitLeakError if any patient, study or series is in more than one split."""
    for key in GROUP_KEYS:
        splits_per_group = images.groupby(key)[split_col].nunique()
        leaked = splits_per_group[splits_per_group > 1]
        if len(leaked):
            examples = ", ".join(map(str, leaked.index[:5]))
            raise SplitLeakError(f"{len(leaked)} {key} values span splits (e.g. {examples})")


def _summarize(part: pd.DataFrame) -> dict[str, Any]:
    patient_flags = part.groupby("patient_id")[LABELS].max()
    return {
        "patients": int(part["patient_id"].nunique()),
        "studies": int(part["study_instance_uid"].nunique()),
        "series": int(part["series_instance_uid"].nunique()),
        "images": len(part),
        "image_positives": {label: int(part[label].sum()) for label in LABELS},
        "image_positive_rate": {label: round(float(part[label].mean()), 5) for label in LABELS},
        "patient_positives": {label: int(patient_flags[label].sum()) for label in LABELS},
    }


def split_report(images: pd.DataFrame) -> dict[str, Any]:
    report = {split: _summarize(images[images["split"] == split]) for split in SPLITS}
    report["total"] = _summarize(images)
    total_patients = report["total"]["patients"]
    for split in SPLITS:
        report[split]["patient_fraction"] = round(report[split]["patients"] / total_patients, 5)
    return report


def make_split(
    labels_csv: str | Path,
    headers_path: str | Path,
    out_dir: str | Path,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Build the label table, split it by patient, verify no leakage and write the outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table, label_stats = build_label_table(labels_csv, headers_path)
    strata = patient_strata(table)
    patient_splits = assign_splits(strata, val_frac=val_frac, test_frac=test_frac, seed=seed)
    images = table.assign(split=table["patient_id"].map(patient_splits))
    check_no_overlap(images)

    table.to_parquet(out_dir / "labels.parquet", index=False)
    images[["image_id", *GROUP_KEYS, "split"]].to_parquet(
        out_dir / "split_images.parquet", index=False
    )
    patients = pd.DataFrame({"stratum": strata, "split": patient_splits})
    patients.rename_axis("patient_id").reset_index().to_csv(
        out_dir / "split_patients.csv", index=False
    )

    report = {
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "overlap_check": "passed",
        "label_stats": label_stats,
        "strata": {k: int(v) for k, v in strata.value_counts().items()},
        **split_report(images),
    }
    (out_dir / "split_report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels-csv", required=True, type=Path, help="RSNA stage_2_train.csv")
    parser.add_argument("--headers", required=True, type=Path, help="dicom_headers.parquet")
    parser.add_argument("--out", required=True, type=Path, help="Output directory.")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    report = make_split(
        args.labels_csv.expanduser(),
        args.headers.expanduser(),
        args.out.expanduser(),
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
