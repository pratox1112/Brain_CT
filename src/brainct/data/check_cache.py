"""Equivalence checks: the cache matches preprocess_dicom, and the laptop matches the cluster.

Two subcommands:

    verify   (cluster) For N random cache rows, re-run preprocess_dicom on the zip bytes and
             require an exact match. Also preprocess the local sample DICOMs and save the
             results as expected_hu256.npz, for the cross-machine check.

        python -m brainct.data.check_cache verify \
            --cache-dir /scratch/$USER/brainct/preprocessed \
            --zip /scratch/$USER/rsna/rsna-intracranial-hemorrhage-detection.zip \
            --sample-dir ~/brainct_outputs/samples/dicom \
            --sample-dir ~/brainct_outputs/samples_edge/dicom \
            --expected-out ~/brainct_outputs/samples/expected_hu256.npz

    compare  (laptop) Re-run preprocess_dicom on the same sample DICOMs and require bit-identical
             results to the cluster's expected_hu256.npz.

        python -m brainct.data.check_cache compare \
            --expected samples/expected_hu256.npz \
            --sample-dir samples/dicom --sample-dir samples_edge/dicom

Both exit with status 1 if anything differs.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import zipfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from brainct.data.build_cache import CACHE_FILE, INDEX_FILE
from brainct.preprocessing.dicom import preprocess_dicom

DEFAULT_SEED = 2026


def _versions() -> dict[str, str]:
    packages = {name: version(name) for name in ("numpy", "pillow", "pydicom")}
    return {"python": platform.python_version(), **packages}


def _sample_files(sample_dirs: list[Path]) -> dict[str, Path]:
    return {p.stem: p for d in sample_dirs for p in sorted(Path(d).glob("*.dcm"))}


def verify(
    cache_dir: str | Path,
    zip_path: str | Path,
    n_random: int = 1000,
    sample_dirs: list[Path] | None = None,
    expected_out: str | Path | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    index = pd.read_parquet(cache_dir / INDEX_FILE)
    cache = np.load(cache_dir / CACHE_FILE, mmap_mode="r")
    size = cache.shape[1]

    ok_rows = index[index["ok"]]
    picked = ok_rows.sample(n=min(n_random, len(ok_rows)), random_state=seed)
    mismatched: list[str] = []
    columns = picked[["row", "image_id", "zip_path"]]
    with zipfile.ZipFile(zip_path) as zf:
        for row, image_id, zip_name in columns.itertuples(index=False):
            fresh, _ = preprocess_dicom(zf.read(zip_name), size=size)
            if not np.array_equal(cache[row], fresh):
                mismatched.append(image_id)

    expected: dict[str, np.ndarray] = {}
    sample_cache_mismatched: list[str] = []
    row_of = dict(zip(index["image_id"], index["row"], strict=True))
    for image_id, path in _sample_files(sample_dirs or []).items():
        expected[image_id], _ = preprocess_dicom(path.read_bytes(), size=size)
        if image_id in row_of and not np.array_equal(cache[row_of[image_id]], expected[image_id]):
            sample_cache_mismatched.append(image_id)
    if expected_out is not None and expected:
        expected_out = Path(expected_out)
        expected_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(expected_out, **expected)

    summary = {
        "random_rows_checked": len(picked),
        "random_rows_mismatched": len(mismatched),
        "mismatched_examples": mismatched[:10],
        "samples_preprocessed": len(expected),
        "samples_in_cache": sum(image_id in row_of for image_id in expected),
        "samples_cache_mismatched": len(sample_cache_mismatched),
        "expected_out": None if expected_out is None else str(expected_out),
        "versions": _versions(),
    }
    (cache_dir / "check_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def compare(expected_path: str | Path, sample_dirs: list[Path]) -> dict[str, Any]:
    with np.load(expected_path) as expected_file:
        expected = {key: expected_file[key] for key in expected_file.files}
    files = _sample_files(sample_dirs)
    missing = sorted(set(expected) - set(files))
    mismatched = []
    for image_id, reference in expected.items():
        if image_id in files:
            local, _ = preprocess_dicom(files[image_id].read_bytes(), size=reference.shape[0])
            if not np.array_equal(local, reference):
                mismatched.append(image_id)
    return {
        "expected_images": len(expected),
        "compared": len(expected) - len(missing),
        "missing_locally": len(missing),
        "mismatched": len(mismatched),
        "mismatched_examples": mismatched[:10],
        "versions": _versions(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_verify = sub.add_parser("verify", help="cluster: cache vs fresh preprocessing")
    p_verify.add_argument("--cache-dir", required=True, type=Path)
    p_verify.add_argument("--zip", required=True, type=Path)
    p_verify.add_argument("--n", type=int, default=1000, help="random cache rows to check")
    p_verify.add_argument("--sample-dir", action="append", type=Path, default=[])
    p_verify.add_argument("--expected-out", type=Path, default=None)
    p_verify.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p_compare = sub.add_parser("compare", help="laptop: local preprocessing vs cluster output")
    p_compare.add_argument("--expected", required=True, type=Path)
    p_compare.add_argument("--sample-dir", action="append", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify(
            args.cache_dir.expanduser(),
            args.zip.expanduser(),
            n_random=args.n,
            sample_dirs=[d.expanduser() for d in args.sample_dir],
            expected_out=None if args.expected_out is None else args.expected_out.expanduser(),
            seed=args.seed,
        )
        failed = result["random_rows_mismatched"] or result["samples_cache_mismatched"]
    else:
        result = compare(args.expected.expanduser(), [d.expanduser() for d in args.sample_dir])
        failed = result["mismatched"] or result["missing_locally"]
    print(json.dumps(result, indent=2))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
