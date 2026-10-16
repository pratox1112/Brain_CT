"""Inspect real pixel values per pixel-format group, and extract edge-case samples from train.

Phase 1 showed that pixel format and rescale intercept combine in unusual ways (for example
unsigned 12-bit pixels with intercept 0, which can never produce air at about -1000 HU). This
script measures what the pixel values actually look like in each group, so any correction in
the preprocessing code is based on evidence. It also extracts a few edge-case DICOMs from the
train split for preprocessing tests only (never for evaluation).

Usage (on the cluster, inside a Slurm job - see jobs/inspect_pixels.sh):

    python -m brainct.preprocessing.inspect_pixels \
        --headers ~/brainct_outputs/metadata/dicom_headers.parquet \
        --split-images ~/brainct_outputs/splits/split_images.parquet \
        --zip /scratch/$USER/rsna/rsna-intracranial-hemorrhage-detection.zip \
        --stats-out ~/brainct_outputs/analysis \
        --edge-out ~/brainct_outputs/samples_edge

Outputs:
    <stats-out>/pixel_stats.csv           one row per inspected image
    <stats-out>/pixel_stats_summary.csv   median of each statistic per group
    <edge-out>/dicom/<image_id>.dcm       edge-case DICOMs from the train split
    <edge-out>/samples_edge.csv           why each edge sample was picked
"""

from __future__ import annotations

import argparse
import io
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
from pydicom import config as pydicom_config

from brainct.data.samples import extract_files, pick_spread
from brainct.preprocessing.dicom import dicom_to_hu

DEFAULT_SEED = 2026

HEADER_COLUMNS = [
    "image_id",
    "patient_id",
    "zip_path",
    "rescale_intercept",
    "pixel_representation",
    "bits_stored",
    "n_rows",
    "n_cols",
]

Rule = Callable[[pd.DataFrame], pd.Series]


def _format(pixel_representation: int, bits_stored: int, intercept: float) -> Rule:
    return lambda d: (
        (d["pixel_representation"] == pixel_representation)
        & (d["bits_stored"] == bits_stored)
        & (d["rescale_intercept"] == intercept)
    )


# Every pixel-format x intercept combination that occurs in the RSNA training set.
GROUPS: dict[str, Rule] = {
    "unsigned12_-1024": _format(0, 12, -1024.0),
    "unsigned12_-1000": _format(0, 12, -1000.0),
    "unsigned12_0": _format(0, 12, 0.0),
    "unsigned12_1": _format(0, 12, 1.0),
    "signed12_0": _format(1, 12, 0.0),
    "signed16_-1024": _format(1, 16, -1024.0),
    "signed16_0": _format(1, 16, 0.0),
}

# Edge cases for preprocessing tests. All of these occur only in the train split.
EDGE_CASES: dict[str, Rule] = {
    "not_512x512": lambda d: (d["n_rows"] != 512) | (d["n_cols"] != 512),
    "intercept_1": lambda d: d["rescale_intercept"] == 1,
    "unsigned12_intercept_0": _format(0, 12, 0.0),
    "signed12_intercept_0": _format(1, 12, 0.0),
    "signed16_intercept_0": _format(1, 16, 0.0),
}

PERCENTILES = {"min": 0, "p01": 1, "p50": 50, "p99": 99, "max": 100}


def _percentiles(prefix: str, values: np.ndarray) -> dict[str, float]:
    points = np.percentile(values, list(PERCENTILES.values()))
    return {f"{prefix}_{name}": float(v) for name, v in zip(PERCENTILES, points, strict=True)}


def _fractions(prefix: str, hu: np.ndarray) -> dict[str, float]:
    return {
        f"frac_{prefix}_air": float((hu < -900).mean()),  # air is about -1000 HU
        f"frac_{prefix}_brain": float(((hu >= 0) & (hu <= 80)).mean()),  # brain / blood range
        f"frac_{prefix}_bone": float((hu > 300).mean()),
        f"frac_{prefix}_below_-1100": float((hu < -1100).mean()),  # implausible for CT
        f"frac_{prefix}_above_3000": float((hu > 3000).mean()),  # implausible except metal
    }


def pixel_stats(data: bytes) -> dict[str, Any]:
    """Statistics of one DICOM's pixels: as stored, as decoded by pydicom, in naive HU
    (decoded x slope + intercept) and in fixed HU (the preprocessing code: sign fix + clip)."""
    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    stored = np.frombuffer(ds.PixelData, dtype=np.uint16)  # raw 16-bit words, no interpretation
    decoded = ds.pixel_array.astype(np.float64)  # pydicom applies PixelRepresentation
    hu = decoded * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    fixed, sign_fixed = dicom_to_hu(ds)
    return {
        "transfer_syntax": str(ds.file_meta.TransferSyntaxUID),
        "sign_fixed": sign_fixed,
        **_percentiles("stored", stored),
        **_percentiles("decoded", decoded),
        **_percentiles("hu", hu),
        **_fractions("hu", hu),
        **_percentiles("fixed_hu", fixed),
        **_fractions("fixed_hu", fixed),
    }


def inspect_groups(
    headers: pd.DataFrame, zip_path: str | Path, n_per_group: int, seed: int
) -> pd.DataFrame:
    """Return per-image pixel statistics for up to n_per_group images from each group."""
    rng = np.random.default_rng(seed)
    pool = headers.sort_values("image_id").set_index("image_id")
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for group, rule in GROUPS.items():
            for image_id, info in pick_spread(pool[rule(pool)], n_per_group, rng).iterrows():
                try:
                    stats = pixel_stats(zf.read(info["zip_path"]))
                    error = None
                except Exception as exc:  # record and continue; this is a diagnostic
                    stats, error = {}, f"{type(exc).__name__}: {exc}"
                rows.append({"group": group, "image_id": image_id, "error": error, **stats})
    return pd.DataFrame(rows)


def summarize(stats: pd.DataFrame) -> pd.DataFrame:
    numeric = stats.select_dtypes("number").columns
    summary = stats.groupby("group", sort=False)[list(numeric)].median()
    summary.insert(0, "images", stats.groupby("group", sort=False).size())
    summary.insert(1, "errors", stats.groupby("group", sort=False)["error"].count())
    if "sign_fixed" in stats:
        sign_fixed = stats["sign_fixed"].astype(float).groupby(stats["group"], sort=False).mean()
        summary.insert(2, "sign_fixed_frac", sign_fixed)
    return summary


def select_edge_samples(images: pd.DataFrame, n_per_case: int, seed: int) -> pd.DataFrame:
    """Pick up to n_per_case train-split images per edge case, spread across patients."""
    rng = np.random.default_rng(seed)
    pool = images[images["split"] == "train"].sort_values("image_id").set_index("image_id")
    reasons: dict[str, list[str]] = {}
    for name, rule in EDGE_CASES.items():
        for image_id in pick_spread(pool[rule(pool)], n_per_case, rng).index:
            reasons.setdefault(image_id, []).append(name)
    selected = pool.loc[sorted(reasons)].reset_index()
    selected.insert(1, "reasons", [";".join(reasons[i]) for i in selected["image_id"]])
    return selected


def run(
    headers_path: str | Path,
    split_images_path: str | Path,
    zip_path: str | Path,
    stats_out: str | Path,
    edge_out: str | Path,
    n_per_group: int = 20,
    n_edge_per_case: int = 5,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    pydicom_config.settings.reading_validation_mode = pydicom_config.IGNORE
    stats_out, edge_out = Path(stats_out), Path(edge_out)
    stats_out.mkdir(parents=True, exist_ok=True)
    edge_out.mkdir(parents=True, exist_ok=True)

    headers = pd.read_parquet(headers_path, columns=HEADER_COLUMNS)
    splits = pd.read_parquet(split_images_path, columns=["image_id", "split"])

    stats = inspect_groups(headers, zip_path, n_per_group, seed)
    summary = summarize(stats)
    stats.to_csv(stats_out / "pixel_stats.csv", index=False)
    summary.to_csv(stats_out / "pixel_stats_summary.csv")

    images = headers.merge(splits, on="image_id", validate="1:1")
    edge = select_edge_samples(images, n_edge_per_case, seed)
    extract_files(zip_path, edge, edge_out / "dicom")
    edge.to_csv(edge_out / "samples_edge.csv", index=False)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--split-images", required=True, type=Path)
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--stats-out", required=True, type=Path)
    parser.add_argument("--edge-out", required=True, type=Path)
    parser.add_argument("--n-per-group", type=int, default=20)
    parser.add_argument("--n-edge-per-case", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    summary = run(
        args.headers.expanduser(),
        args.split_images.expanduser(),
        args.zip.expanduser(),
        args.stats_out.expanduser(),
        args.edge_out.expanduser(),
        n_per_group=args.n_per_group,
        n_edge_per_case=args.n_edge_per_case,
        seed=args.seed,
    )
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(summary.round(4).T.to_string())


if __name__ == "__main__":
    main()
