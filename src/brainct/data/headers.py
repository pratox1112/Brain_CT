"""Extract DICOM header metadata for every RSNA training image, read straight from the zip.

Usage (on the cluster, inside a Slurm job - see jobs/extract_headers.sh):

    python -m brainct.data.headers \
        --zip /scratch/$USER/rsna/rsna-intracranial-hemorrhage-detection.zip \
        --out ~/brainct_outputs/metadata --workers 16

Outputs written to --out:
    dicom_headers.parquet   one row per parsed image
    header_failures.csv     images that could not be parsed, with the reason
    header_summary.json     counts and value distributions for docs/results.md
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import socket
import time
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import pydicom
from pydicom import config as pydicom_config
from pydicom.multival import MultiValue

from brainct.data.zip_reader import TRAIN_PREFIX, list_dicoms, read_header_bytes, read_member

# Without these an image cannot be placed in a patient-level split or converted to HU.
REQUIRED_TAGS = (
    "SOPInstanceUID",
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "RescaleSlope",
    "RescaleIntercept",
)

COLUMNS = [
    "image_id",
    "zip_path",
    "sop_instance_uid",
    "patient_id",
    "study_instance_uid",
    "series_instance_uid",
    "modality",
    "ipp_x",
    "ipp_y",
    "ipp_z",
    "iop",
    "pixel_spacing_row",
    "pixel_spacing_col",
    "slice_thickness",
    "rescale_slope",
    "rescale_intercept",
    "window_center",
    "window_width",
    "bits_allocated",
    "bits_stored",
    "high_bit",
    "pixel_representation",
    "n_rows",
    "n_cols",
    "photometric_interpretation",
]

FAILURE_COLUMNS = ["image_id", "zip_path", "error"]

# Value distributions reported in the summary; used to spot RSNA quirks before preprocessing.
SUMMARY_VALUE_COUNTS = [
    "modality",
    "rescale_slope",
    "rescale_intercept",
    "bits_allocated",
    "bits_stored",
    "pixel_representation",
    "photometric_interpretation",
]

CHUNK_SIZE = 1000
PROGRESS_EVERY_CHUNKS = 50

# Per-process state, set by _init_worker (each worker opens its own handle on the zip).
_zf: zipfile.ZipFile | None = None
_max_bytes = 16384


def _quiet_pydicom() -> None:
    # RSNA uses non-standard UIDs such as "ID_000012eaf"; don't warn on every file.
    pydicom_config.settings.reading_validation_mode = pydicom_config.IGNORE


def _init_worker(zip_path: str, max_bytes: int) -> None:
    global _zf, _max_bytes
    _quiet_pydicom()
    _zf = zipfile.ZipFile(zip_path)
    _max_bytes = max_bytes


def _close_worker() -> None:
    global _zf
    if _zf is not None:
        _zf.close()
        _zf = None


def _first(value: Any) -> Any:
    if isinstance(value, MultiValue | list | tuple):
        return value[0] if len(value) else None
    return value


def _to_float(value: Any) -> float | None:
    value = _first(value)
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return None if number is None else int(number)


def _to_floats(value: Any, n: int) -> list[float | None]:
    """Return exactly n floats, or n Nones if the tag is missing or malformed."""
    try:
        numbers = [float(v) for v in value]
    except (TypeError, ValueError):
        return [None] * n
    return numbers if len(numbers) == n else [None] * n


def _to_str(value: Any) -> str | None:
    return None if value is None else str(value)


def parse_header(data: bytes, zip_name: str) -> dict[str, Any]:
    """Parse one DICOM header into a flat row. Raises if the bytes are not a usable DICOM."""
    ds = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True, force=True)
    missing = [tag for tag in REQUIRED_TAGS if ds.get(tag) in (None, "")]
    if missing:
        raise ValueError(f"missing required tags: {', '.join(missing)}")

    ipp_x, ipp_y, ipp_z = _to_floats(ds.get("ImagePositionPatient"), 3)
    spacing_row, spacing_col = _to_floats(ds.get("PixelSpacing"), 2)
    iop = _to_floats(ds.get("ImageOrientationPatient"), 6)
    return {
        "image_id": PurePosixPath(zip_name).stem,
        "zip_path": zip_name,
        "sop_instance_uid": str(ds.SOPInstanceUID),
        "patient_id": str(ds.PatientID),
        "study_instance_uid": str(ds.StudyInstanceUID),
        "series_instance_uid": str(ds.SeriesInstanceUID),
        "modality": _to_str(ds.get("Modality")),
        "ipp_x": ipp_x,
        "ipp_y": ipp_y,
        "ipp_z": ipp_z,
        "iop": None if iop[0] is None else iop,
        "pixel_spacing_row": spacing_row,
        "pixel_spacing_col": spacing_col,
        "slice_thickness": _to_float(ds.get("SliceThickness")),
        "rescale_slope": _to_float(ds.RescaleSlope),
        "rescale_intercept": _to_float(ds.RescaleIntercept),
        "window_center": _to_float(ds.get("WindowCenter")),
        "window_width": _to_float(ds.get("WindowWidth")),
        "bits_allocated": _to_int(ds.get("BitsAllocated")),
        "bits_stored": _to_int(ds.get("BitsStored")),
        "high_bit": _to_int(ds.get("HighBit")),
        "pixel_representation": _to_int(ds.get("PixelRepresentation")),
        "n_rows": _to_int(ds.get("Rows")),
        "n_cols": _to_int(ds.get("Columns")),
        "photometric_interpretation": _to_str(ds.get("PhotometricInterpretation")),
    }


def _parse_member(zf: zipfile.ZipFile, name: str, max_bytes: int):
    """Return (row, failure, full_read). Exactly one of row / failure is set."""
    full_read = False
    try:
        data, full_read = read_header_bytes(zf, name, max_bytes)
        try:
            return parse_header(data, name), None, full_read
        except Exception:
            if full_read:
                raise
            # The partial read may have cut the header short; retry with the whole file.
            full_read = True
            return parse_header(read_member(zf, name), name), None, full_read
    except Exception as exc:  # corrupt files must be logged, not crash a 752K-file run
        failure = {
            "image_id": PurePosixPath(name).stem,
            "zip_path": name,
            "error": f"{type(exc).__name__}: {exc}",
        }
        return None, failure, full_read


def _run_chunk(names: list[str]) -> tuple[list[dict], list[dict], int]:
    rows, failures, full_reads = [], [], 0
    for name in names:
        row, failure, full_read = _parse_member(_zf, name, _max_bytes)
        full_reads += full_read
        if row is not None:
            rows.append(row)
        else:
            failures.append(failure)
    return rows, failures, full_reads


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def build_summary(
    df: pd.DataFrame, failures: pd.DataFrame, files_listed: int, full_reads: int
) -> dict[str, Any]:
    return {
        "files_listed": files_listed,
        "parsed": len(df),
        "failed": len(failures),
        "full_file_reads": full_reads,
        "unique_patients": int(df["patient_id"].nunique()),
        "unique_studies": int(df["study_instance_uid"].nunique()),
        "unique_series": int(df["series_instance_uid"].nunique()),
        "duplicate_image_ids": int(df["image_id"].duplicated().sum()),
        "image_id_ne_sop_uid": int((df["image_id"] != df["sop_instance_uid"]).sum()),
        "value_counts": {col: _value_counts(df[col]) for col in SUMMARY_VALUE_COUNTS},
        "image_shapes": _value_counts(df["n_rows"].astype(str) + "x" + df["n_cols"].astype(str)),
    }


def extract_headers(
    zip_path: str | Path,
    out_dir: str | Path,
    workers: int = 1,
    max_bytes: int = 16384,
    limit: int | None = None,
    prefix: str = TRAIN_PREFIX,
) -> dict[str, Any]:
    """Read every DICOM header under `prefix` in the zip and write the outputs to `out_dir`."""
    _quiet_pydicom()
    start = time.time()
    zip_path = str(zip_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list_dicoms(zip_path, prefix)
    if limit is not None:
        names = names[:limit]
    chunks = [names[i : i + CHUNK_SIZE] for i in range(0, len(names), CHUNK_SIZE)]
    print(f"Listed {len(names)} DICOMs under {prefix!r}: {len(chunks)} chunks, {workers} workers")
    print(f"Listing took {time.time() - start:.0f}s", flush=True)

    rows: list[dict] = []
    failures: list[dict] = []
    full_reads = 0

    def consume(results: Iterable[tuple[list[dict], list[dict], int]]) -> None:
        nonlocal full_reads
        for i, (chunk_rows, chunk_failures, chunk_full) in enumerate(results, 1):
            rows.extend(chunk_rows)
            failures.extend(chunk_failures)
            full_reads += chunk_full
            if i % PROGRESS_EVERY_CHUNKS == 0 or i == len(chunks):
                done = len(rows) + len(failures)
                print(
                    f"  {done}/{len(names)} files, {len(failures)} failures, "
                    f"{time.time() - start:.0f}s",
                    flush=True,
                )

    if workers <= 1:
        _init_worker(zip_path, max_bytes)
        try:
            consume(map(_run_chunk, chunks))
        finally:
            _close_worker()
    else:
        with mp.Pool(workers, initializer=_init_worker, initargs=(zip_path, max_bytes)) as pool:
            consume(pool.imap_unordered(_run_chunk, chunks))

    df = pd.DataFrame(rows, columns=COLUMNS).sort_values("image_id").reset_index(drop=True)
    failures_df = pd.DataFrame(failures, columns=FAILURE_COLUMNS).sort_values("image_id")
    df.to_parquet(out_dir / "dicom_headers.parquet", index=False)
    failures_df.to_csv(out_dir / "header_failures.csv", index=False)

    summary = {
        "zip": zip_path,
        "prefix": prefix,
        "limit": limit,
        "host": socket.gethostname(),
        "workers": workers,
        "max_header_bytes": max_bytes,
        "elapsed_seconds": round(time.time() - start, 1),
        **build_summary(df, failures_df, len(names), full_reads),
    }
    (out_dir / "header_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", required=True, type=Path, help="Path to the RSNA zip.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes.")
    parser.add_argument(
        "--max-header-bytes",
        type=int,
        default=16384,
        help="Bytes to decompress per file before falling back to a full read.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N files (test runs)."
    )
    parser.add_argument("--prefix", default=TRAIN_PREFIX, help="Folder inside the zip to read.")
    args = parser.parse_args(argv)

    summary = extract_headers(
        args.zip,
        args.out.expanduser(),
        workers=args.workers,
        max_bytes=args.max_header_bytes,
        limit=args.limit,
        prefix=args.prefix,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
