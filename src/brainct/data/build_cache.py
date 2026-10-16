"""Build the preprocessed HU cache: every training DICOM -> int16 (256, 256) in one .npy memmap.

Each image goes through brainct.preprocessing.dicom.preprocess_dicom - the same function the API
uses - so training and serving see identical inputs. The zip is read directly, never unzipped.

Usage (on the cluster, inside a Slurm job - see jobs/build_cache.sh):

    python -m brainct.data.build_cache \
        --zip /scratch/$USER/rsna/rsna-intracranial-hemorrhage-detection.zip \
        --split-images ~/brainct_outputs/splits/split_images.parquet \
        --out /scratch/$USER/brainct/preprocessed --workers 32

Outputs written to --out:
    hu256.npy               int16 array [n_images, 256, 256] (open with np.load(mmap_mode="r"))
    hu256_index.parquet     row -> image_id, zip_path, split, ok, sign_fixed, padded, original shape
    cache_failures.csv      images whose pixels could not be preprocessed (row filled with air)
    cache_summary.json      counts and timing for docs/results.md
    cache_meta.json         layout used, checked when resuming
    chunks/                 one finished-chunk file each; a rerun skips chunks already done
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import socket
import time
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from brainct.data.zip_reader import TRAIN_PREFIX, list_dicoms
from brainct.preprocessing.dicom import IMAGE_SIZE, preprocess_dicom
from brainct.preprocessing.hu import HU_MIN

CACHE_FILE = "hu256.npy"
INDEX_FILE = "hu256_index.parquet"
FAILURES_FILE = "cache_failures.csv"
SUMMARY_FILE = "cache_summary.json"
META_FILE = "cache_meta.json"
CHUNK_DIR = "chunks"
CHUNK_SIZE = 1000
PROGRESS_EVERY_CHUNKS = 25

INDEX_COLUMNS = [
    "row",
    "image_id",
    "zip_path",
    "split",
    "ok",
    "sign_fixed",
    "padded",
    "orig_rows",
    "orig_cols",
    "error",
]

# Per-process state, set by _init_worker (each worker opens its own zip and memmap handles).
_zf: zipfile.ZipFile | None = None
_cache: np.memmap | None = None


def _init_worker(zip_path: str, cache_path: str) -> None:
    global _zf, _cache
    _zf = zipfile.ZipFile(zip_path)
    _cache = np.load(cache_path, mmap_mode="r+")


def _close_worker() -> None:
    global _zf, _cache
    if _zf is not None:
        _zf.close()
    if _cache is not None:
        _cache.flush()
    _zf, _cache = None, None


def _process_chunk(task: tuple[list[int], list[str], list[str], int, str]) -> int:
    """Preprocess one chunk into the memmap, then write its index rows as the 'done' marker."""
    rows, names, splits, size, chunk_path = task
    records = []
    for row, name, split in zip(rows, names, splits, strict=True):
        record = {
            "row": row,
            "image_id": PurePosixPath(name).stem,
            "zip_path": name,
            "split": split,
            "ok": True,
            "sign_fixed": False,
            "padded": False,
            "orig_rows": 0,
            "orig_cols": 0,
            "error": None,
        }
        try:
            image, info = preprocess_dicom(_zf.read(name), size=size)
            _cache[row] = image
            record.update(
                sign_fixed=info.sign_fixed,
                padded=info.padded,
                orig_rows=info.original_shape[0],
                orig_cols=info.original_shape[1],
            )
        except Exception as exc:  # one bad file must not stop a 752K-image build
            _cache[row] = HU_MIN
            record.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        records.append(record)
    _cache.flush()

    # Write to a temporary name first, so a crash never leaves a half-written marker.
    path = Path(chunk_path)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(records, columns=INDEX_COLUMNS).to_parquet(tmp, index=False)
    tmp.replace(path)
    return len(records)


def _prepare_cache(out_dir: Path, n_rows: int, size: int, chunk_size: int) -> bool:
    """Create the memmap and meta file, or check an existing one matches. Returns True if new."""
    cache_path, meta_path = out_dir / CACHE_FILE, out_dir / META_FILE
    meta = {"n_rows": n_rows, "size": size, "chunk_size": chunk_size, "dtype": "int16"}
    if cache_path.exists():
        existing = json.loads(meta_path.read_text()) if meta_path.exists() else None
        if existing != meta:
            raise ValueError(
                f"{cache_path} exists with a different layout ({existing} vs {meta}); "
                "delete the output folder or choose another --out"
            )
        return False
    cache = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.int16, shape=(n_rows, size, size)
    )
    del cache  # close the handle; workers open their own
    meta_path.write_text(json.dumps(meta, indent=2))
    return True


def build_cache(
    zip_path: str | Path,
    split_images_path: str | Path,
    out_dir: str | Path,
    workers: int = 1,
    size: int = IMAGE_SIZE,
    limit: int | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, Any]:
    start = time.time()
    zip_path, out_dir = str(zip_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = out_dir / CHUNK_DIR
    chunk_dir.mkdir(exist_ok=True)

    names = list_dicoms(zip_path, TRAIN_PREFIX)
    if limit is not None:
        names = names[:limit]
    image_ids = [PurePosixPath(name).stem for name in names]
    splits = pd.read_parquet(split_images_path, columns=["image_id", "split"])
    split_of = splits.set_index("image_id")["split"].reindex(image_ids)
    if split_of.isna().any():
        raise ValueError(f"{int(split_of.isna().sum())} images in the zip have no split")

    if _prepare_cache(out_dir, len(names), size, chunk_size):
        for stale in chunk_dir.glob("chunk_*"):  # markers from an older, deleted cache
            stale.unlink()

    tasks, skipped = [], 0
    for chunk_id, first in enumerate(range(0, len(names), chunk_size)):
        chunk_path = chunk_dir / f"chunk_{chunk_id:05d}.parquet"
        if chunk_path.exists():
            skipped += 1
            continue
        last = min(first + chunk_size, len(names))
        rows, chunk_names = list(range(first, last)), names[first:last]
        chunk_splits = split_of.iloc[first:last].tolist()
        tasks.append((rows, chunk_names, chunk_splits, size, str(chunk_path)))
    n_chunks = len(tasks) + skipped
    print(f"{len(names)} images, {n_chunks} chunks ({skipped} already done), {workers} workers")
    print(f"Setup took {time.time() - start:.0f}s", flush=True)

    def consume(results: Iterable[int]) -> None:
        done = 0
        for i, n in enumerate(results, 1):
            done += n
            if i % PROGRESS_EVERY_CHUNKS == 0 or i == len(tasks):
                rate = done / max(time.time() - start, 1e-9)
                print(f"  {done}/{len(names)} images this run, {rate:.0f} img/s", flush=True)

    cache_path = str(out_dir / CACHE_FILE)
    if workers <= 1:
        _init_worker(zip_path, cache_path)
        try:
            consume(map(_process_chunk, tasks))
        finally:
            _close_worker()
    else:
        with mp.Pool(workers, initializer=_init_worker, initargs=(zip_path, cache_path)) as pool:
            consume(pool.imap_unordered(_process_chunk, tasks))

    index = pd.concat(
        [pd.read_parquet(p) for p in sorted(chunk_dir.glob("chunk_*.parquet"))],
        ignore_index=True,
    )
    index = index.sort_values("row").reset_index(drop=True)
    if not index["row"].equals(pd.Series(range(len(names)), name="row")):
        raise RuntimeError("cache index is incomplete or has duplicate rows")
    index.to_parquet(out_dir / INDEX_FILE, index=False)
    failures = index.loc[~index["ok"], ["row", "image_id", "zip_path", "error"]]
    failures.to_csv(out_dir / FAILURES_FILE, index=False)

    ok = index[index["ok"]]
    summary = {
        "zip": zip_path,
        "out": str(out_dir),
        "host": socket.gethostname(),
        "workers": workers,
        "size": size,
        "limit": limit,
        "images": len(index),
        "ok": int(index["ok"].sum()),
        "failed": len(failures),
        "sign_fixed": int(ok["sign_fixed"].sum()),
        "padded": int(ok["padded"].sum()),
        "not_512x512": int(((ok["orig_rows"] != 512) | (ok["orig_cols"] != 512)).sum()),
        "ok_by_split": {k: int(v) for k, v in ok["split"].value_counts().items()},
        "chunks": n_chunks,
        "chunks_skipped_resume": skipped,
        "cache_bytes": (out_dir / CACHE_FILE).stat().st_size,
        "elapsed_seconds_this_run": round(time.time() - start, 1),
    }
    (out_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--split-images", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N images (tests).")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args(argv)

    summary = build_cache(
        args.zip.expanduser(),
        args.split_images.expanduser(),
        args.out.expanduser(),
        workers=args.workers,
        size=args.size,
        limit=args.limit,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
