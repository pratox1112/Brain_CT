"""Pick a small, varied set of test-split DICOMs and extract only those files from the zip.

The samples go to the laptop for preprocessing tests, API/Docker checks and regression tests.
They come from **test-split patients only**, so they never overlap with training data.

Usage (on the cluster, inside a Slurm job - see jobs/extract_samples.sh):

    python -m brainct.data.samples \
        --split-images ~/brainct_outputs/splits/split_images.parquet \
        --labels ~/brainct_outputs/splits/labels.parquet \
        --headers ~/brainct_outputs/metadata/dicom_headers.parquet \
        --zip /scratch/$USER/rsna/rsna-intracranial-hemorrhage-detection.zip \
        --out ~/brainct_outputs/samples

Outputs written to --out:
    dicom/<image_id>.dcm    the selected DICOM files, byte-for-byte as stored in the zip
    samples.csv             one row per sample: IDs, why it was picked, labels, key header fields
    samples_summary.json    counts per reason and per label
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from brainct.data.labels import LABELS, SUBTYPES

DEFAULT_SEED = 2026

HEADER_FIELDS = [
    "zip_path",
    "rescale_intercept",
    "pixel_representation",
    "bits_stored",
    "n_rows",
    "n_cols",
]

# Unusual header values seen in the full header run; each gets a few samples for Phase 2.
EDGE_CASES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "intercept_0": lambda d: d["rescale_intercept"] == 0,
    "intercept_1": lambda d: d["rescale_intercept"] == 1,
    "not_512x512": lambda d: (d["n_rows"] != 512) | (d["n_cols"] != 512),
    "unsigned_12bit": lambda d: (d["pixel_representation"] == 0) & (d["bits_stored"] == 12),
    "unsigned_16bit": lambda d: (d["pixel_representation"] == 0) & (d["bits_stored"] == 16),
    "signed_12bit": lambda d: (d["pixel_representation"] == 1) & (d["bits_stored"] == 12),
    "signed_16bit": lambda d: (d["pixel_representation"] == 1) & (d["bits_stored"] == 16),
}


def pick_spread(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Pick up to n rows at random, one image per patient first, then top up if needed."""
    shuffled = pool.iloc[rng.permutation(len(pool))]
    one_per_patient = shuffled.drop_duplicates("patient_id")
    rest = shuffled.drop(one_per_patient.index)
    return pd.concat([one_per_patient, rest]).head(n)


def select_samples(
    images: pd.DataFrame,
    n_normal: int = 60,
    n_per_subtype: int = 40,
    n_per_edge_case: int = 5,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Select test-split samples. `images` needs IDs, split, the 6 labels and HEADER_FIELDS.

    Returns one row per selected image with a `reasons` column ("normal;subdural;...").
    """
    rng = np.random.default_rng(seed)
    pool = images[images["split"] == "test"].sort_values("image_id").set_index("image_id")

    categories: list[tuple[str, pd.Series, int]] = [("normal", pool["any"] == 0, n_normal)]
    categories += [(subtype, pool[subtype] == 1, n_per_subtype) for subtype in SUBTYPES]
    categories += [(name, rule(pool), n_per_edge_case) for name, rule in EDGE_CASES.items()]

    reasons: dict[str, list[str]] = {}
    for name, mask, n in categories:
        for image_id in pick_spread(pool[mask], n, rng).index:
            reasons.setdefault(image_id, []).append(name)

    selected = pool.loc[sorted(reasons)].reset_index()
    selected.insert(1, "reasons", [";".join(reasons[i]) for i in selected["image_id"]])
    return selected


def extract_files(zip_path: str | Path, selected: pd.DataFrame, out_dir: str | Path) -> int:
    """Copy the selected DICOMs out of the zip into out_dir. Returns total bytes written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(zip_path) as zf:
        for image_id, zip_name in zip(selected["image_id"], selected["zip_path"], strict=True):
            data = zf.read(zip_name)
            (out_dir / f"{image_id}.dcm").write_bytes(data)
            total += len(data)
    return total


def make_samples(
    split_images_path: str | Path,
    labels_path: str | Path,
    headers_path: str | Path,
    zip_path: str | Path,
    out_dir: str | Path,
    n_normal: int = 60,
    n_per_subtype: int = 40,
    n_per_edge_case: int = 5,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Join split, labels and headers, select samples, extract them and write the outputs."""
    out_dir = Path(out_dir)
    splits = pd.read_parquet(split_images_path, columns=["image_id", "split"])
    labels = pd.read_parquet(labels_path)
    headers = pd.read_parquet(headers_path, columns=["image_id", *HEADER_FIELDS])
    images = labels.merge(splits, on="image_id", validate="1:1").merge(
        headers, on="image_id", validate="1:1"
    )

    selected = select_samples(images, n_normal, n_per_subtype, n_per_edge_case, seed)
    if not (selected["split"] == "test").all():
        raise AssertionError("a sample outside the test split was selected")

    bytes_written = extract_files(zip_path, selected, out_dir / "dicom")
    selected.drop(columns="split").to_csv(out_dir / "samples.csv", index=False)

    reason_counts: dict[str, int] = {}
    for reasons in selected["reasons"]:
        for reason in reasons.split(";"):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary = {
        "seed": seed,
        "requested": {
            "normal": n_normal,
            "per_subtype": n_per_subtype,
            "per_edge_case": n_per_edge_case,
        },
        "selected_images": len(selected),
        "unique_patients": int(selected["patient_id"].nunique()),
        "all_from_test_split": True,
        "bytes_written": bytes_written,
        "images_per_reason": reason_counts,
        "label_positives": {label: int(selected[label].sum()) for label in LABELS},
    }
    (out_dir / "samples_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split-images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-normal", type=int, default=60)
    parser.add_argument("--n-per-subtype", type=int, default=40)
    parser.add_argument("--n-per-edge-case", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    summary = make_samples(
        args.split_images.expanduser(),
        args.labels.expanduser(),
        args.headers.expanduser(),
        args.zip.expanduser(),
        args.out.expanduser(),
        n_normal=args.n_normal,
        n_per_subtype=args.n_per_subtype,
        n_per_edge_case=args.n_per_edge_case,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
