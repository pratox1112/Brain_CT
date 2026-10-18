"""Benchmark the training input pipeline without a model, to find what limits throughput.

The Phase 3 speed test ran at about 350 img/s with and without AMP, so the GPU was waiting for
data. This measures the DataLoader alone, stage by stage and for several worker counts:

    read_direct  cache row only, one positioned read of the row's bytes (the default)
    read_mmap    cache row only, through a numpy memmap (the original method)
    augment      the full training path (direct read + windows + flip/affine + normalize)

The first benchmark (memmap reads only) showed storage latency as the limit: throughput grew
linearly with workers at only about 64 MB/s. A memmap row read on network storage can become many
small page-fault reads; a direct read is one request per row. read_direct runs first, while the
page cache is coldest, so any advantage it shows isn't from caching. Each measurement uses a
different random order.

Usage (on the cluster - see jobs/bench_loader.sh):

    python -m brainct.training.bench_loader \
        --cache-dir /scratch/$USER/brainct/preprocessed \
        --labels ~/brainct_outputs/splits/labels.parquet \
        --workers 4 8 16 32 --out ~/brainct_outputs/analysis/loader_bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from brainct.training.dataset import Augment, CacheDataset, EpochSampler

DEFAULT_SEED = 2026
STAGES = ("read_direct", "read_mmap", "augment")


def _dataset(cache_dir: Path, labels: Path, stage: str) -> CacheDataset:
    if stage == "read_direct":
        return CacheDataset(cache_dir, labels, "train", output="hu", read="direct")
    if stage == "read_mmap":
        return CacheDataset(cache_dir, labels, "train", output="hu", read="mmap")
    return CacheDataset(cache_dir, labels, "train", augment=Augment(), read="direct")


def measure(
    dataset: CacheDataset,
    workers: int,
    batch_size: int,
    batches: int,
    warmup: int,
    seed: int,
    epoch: int,
) -> dict[str, Any]:
    """Images per second over `batches` batches, after `warmup` batches (worker start-up)."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=EpochSampler(len(dataset), seed, epoch),
        num_workers=workers,
        drop_last=True,
    )
    batches_iter = iter(loader)
    for _ in range(warmup):
        next(batches_iter)
    images, start = 0, time.perf_counter()
    for _ in range(batches):
        x, _, _ = next(batches_iter)
        images += len(x)
    seconds = time.perf_counter() - start
    del batches_iter  # shut the workers down before the next measurement
    return {"images": images, "seconds": round(seconds, 3), "images_per_s": images / seconds}


def run(
    cache_dir: str | Path,
    labels: str | Path,
    workers: list[int],
    stages: tuple[str, ...] = STAGES,
    batch_size: int = 64,
    batches: int = 100,
    warmup: int = 5,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    results = []
    epoch = 1000  # an epoch number training never uses, so the order is unrelated to training
    for stage in stages:
        dataset = _dataset(Path(cache_dir), Path(labels), stage)
        for n_workers in workers:
            epoch += 1
            result = measure(dataset, n_workers, batch_size, batches, warmup, seed, epoch)
            result.update({"stage": stage, "workers": n_workers})
            results.append(result)
            rate = result["images_per_s"]
            print(f"{stage:8s} workers={n_workers:3d}  {rate:8.1f} img/s", flush=True)
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return {
        "host": socket.gethostname(),
        "cpus_available": cpus,
        "batch_size": batch_size,
        "batches": batches,
        "warmup_batches": warmup,
        "results": results,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run(
        args.cache_dir.expanduser(),
        args.labels.expanduser(),
        args.workers,
        tuple(args.stages),
        args.batch_size,
        args.batches,
        args.warmup,
    )
    if args.out is not None:
        out = args.out.expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
