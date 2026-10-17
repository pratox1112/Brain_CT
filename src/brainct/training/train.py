"""Train the ResNet-50 baseline on the HU cache, with checkpoint/resume and W&B tracking.

Usage (on the cluster, inside a Slurm GPU job - see jobs/train.sh and jobs/train_chain.sh):

    python -m brainct.training.train --config configs/resnet50_baseline.yaml \
        --set run_id=resnet50_baseline

Overrides use --set key=value, e.g. a speed test:
    --set run_id=speed_amp_off --set amp=false --set max_steps=500 --set validate=false

Everything for a run lives in <runs_dir>/<run_id>/:
    config.yaml          resolved config (written on first start)
    checkpoints/last.pt  full state for resuming (model, optimizer, schedule, RNG, position)
    checkpoints/best.pt  model weights with the best val macro AUC
    metrics.jsonl        one line per log step / validation
    run_summary.json     final numbers (throughput, peak memory, best val AUC, ...)
    DONE                 written when the run is complete; later chained jobs exit at once
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import socket
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from brainct.training.config import TrainConfig, load_config
from brainct.training.dataset import Augment, CacheDataset, EpochSampler
from brainct.training.metrics import auc_per_label, macro
from brainct.training.model import build_model

SPEED_WARMUP_STEPS = 20  # steps excluded from the throughput measurement (cudnn autotuning etc.)


def weighted_bce(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Binary cross-entropy per output, weighted per class, normalized so weights sum to 1."""
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * weights).sum(dim=1).mean() / weights.sum()


def lr_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup to 1, then cosine decay to 0 at total_steps."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
    return 0.5 * (1 + math.cos(math.pi * progress))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_atomic(obj: Any, path: Path) -> None:
    """Write to a temporary file first, so a job killed mid-save never corrupts a checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _environment(device: torch.device) -> dict[str, Any]:
    env = {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": version("torchvision"),
        "cuda": torch.version.cuda,
    }
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
    return env


def _init_wandb(cfg: TrainConfig, wandb_id: str | None):
    if cfg.wandb_mode == "disabled":
        return None
    import wandb

    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_id,
        id=wandb_id,
        resume="allow",
        mode=cfg.wandb_mode,
        dir=str(cfg.run_dir),
        config=cfg.to_dict(),
    )


def _uses_amp(cfg: TrainConfig, device: torch.device) -> bool:
    return cfg.amp and device.type == "cuda"


@torch.no_grad()
def predict(
    model: torch.nn.Module, dataset: CacheDataset, cfg: TrainConfig, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Labels and sigmoid probabilities for a whole dataset, in dataset order."""
    loader = DataLoader(
        dataset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    was_training = model.training
    model.eval()
    labels, probs = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=_uses_amp(cfg, device)):
            logits = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(y.numpy())
    model.train(was_training)
    return np.concatenate(labels), np.concatenate(probs)


def train(cfg: TrainConfig) -> dict[str, Any]:
    run_dir = cfg.run_dir
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "DONE").exists():
        print(f"{run_dir} is already complete; nothing to do.")
        return json.loads((run_dir / "run_summary.json").read_text())
    if not (run_dir / "config.yaml").exists():
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config asks for CUDA but no GPU is visible")
    device = torch.device(cfg.device)
    use_amp = _uses_amp(cfg, device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    seed_everything(cfg.seed)

    augment = None
    if cfg.augment:
        augment = Augment(
            cfg.hflip_p, cfg.rotate_degrees, cfg.scale_min, cfg.scale_max, cfg.translate
        )
    train_set = CacheDataset(
        cfg.cache_dir, cfg.labels_path, "train", augment=augment, read=cfg.cache_read
    )
    val_set = None
    if cfg.validate:
        val_set = CacheDataset(cfg.cache_dir, cfg.labels_path, "val", read=cfg.cache_read)
    steps_per_epoch = len(train_set) // cfg.batch_size
    total_steps = cfg.epochs * steps_per_epoch

    model = build_model(cfg.pretrained).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, cfg.warmup_steps, total_steps)
    )
    weights = torch.tensor(cfg.class_weights, device=device)

    state: dict[str, Any] = {
        "epoch": 0,
        "batch": 0,
        "step": 0,
        "best_val_macro_auc": None,
        "best_epoch": None,
        "train_seconds": 0.0,
        "wandb_id": None,
    }
    last_ckpt = ckpt_dir / "last.pt"
    if last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        state.update(ckpt["state"])
        _set_rng_state(ckpt["rng"])
        print(f"Resumed at step {state['step']} (epoch {state['epoch']}, batch {state['batch']})")

    run = _init_wandb(cfg, state["wandb_id"])
    if run is not None:
        state["wandb_id"] = run.id
        extra = {"steps_per_epoch": steps_per_epoch, "train_images": len(train_set)}
        run.config.update(extra, allow_val_change=True)

    def save_checkpoint() -> None:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "state": state,
            "rng": _rng_state(),
        }
        _save_atomic(checkpoint, last_ckpt)

    def log(values: dict[str, Any]) -> None:
        values = {"step": state["step"], "epoch": state["epoch"], **values}
        with (run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(values) + "\n")
        if run is not None:
            run.log(values, step=state["step"])

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    env = _environment(device)
    print(f"{len(train_set)} train images, {steps_per_epoch} steps/epoch, {total_steps} total")
    print(f"environment: {env}", flush=True)

    job_start = time.time()
    budget_seconds = cfg.time_budget_hours * 3600
    job_steps = 0
    speed_images, speed_start = 0, None
    window_start, window_images = time.time(), 0
    stop_reason = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    while state["epoch"] < cfg.epochs and stop_reason is None:
        skip = state["batch"] * cfg.batch_size
        sampler = EpochSampler(len(train_set), cfg.seed, state["epoch"], skip)
        worker_seed = cfg.seed * 1_000_000 + state["epoch"] * 10_000 + state["batch"]
        loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
            generator=torch.Generator().manual_seed(worker_seed),
        )
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if cfg.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
            loss = weighted_bce(logits.float(), y, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            state["step"] += 1
            state["batch"] += 1
            job_steps += 1
            window_images += len(x)

            # Throughput: count images after the first SPEED_WARMUP_STEPS steps of this job.
            if job_steps == SPEED_WARMUP_STEPS:
                sync()
                speed_start = time.time()
            elif speed_start is not None:
                speed_images += len(x)

            if state["step"] % cfg.log_every == 0:
                sync()
                now = time.time()
                values = {
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/images_per_s": window_images / max(now - window_start, 1e-9),
                }
                if device.type == "cuda":
                    values["gpu/peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                log(values)
                window_start, window_images = now, 0
            if state["step"] % cfg.ckpt_every == 0:
                save_checkpoint()
            if cfg.max_steps is not None and state["step"] >= cfg.max_steps:
                stop_reason = "max_steps"
                break
            if time.time() - job_start > budget_seconds:
                stop_reason = "time_budget"
                break
        else:
            # The epoch finished: validate, keep the best model, move on to the next epoch.
            if val_set is not None:
                y_val, p_val = predict(model, val_set, cfg, device)
                aucs = auc_per_label(y_val, p_val)
                val_macro = macro(aucs)
                log({**{f"val/auc_{k}": v for k, v in aucs.items()}, "val/macro_auc": val_macro})
                best = state["best_val_macro_auc"]
                if best is None or val_macro > best:
                    state["best_val_macro_auc"] = val_macro
                    state["best_epoch"] = state["epoch"]
                    best_ckpt = {
                        "model": model.state_dict(),
                        "epoch": state["epoch"],
                        "val_macro_auc": val_macro,
                        "val_auc": aucs,
                    }
                    _save_atomic(best_ckpt, ckpt_dir / "best.pt")
            state["epoch"] += 1
            state["batch"] = 0
            save_checkpoint()

    sync()
    speed_seconds = time.time() - speed_start if speed_start is not None else 0.0
    state["train_seconds"] += time.time() - job_start
    save_checkpoint()

    complete = stop_reason == "max_steps" or state["epoch"] >= cfg.epochs
    summary = {
        "run_id": cfg.run_id,
        "complete": complete,
        "stop_reason": stop_reason or "epochs_done",
        "step": state["step"],
        "epoch": state["epoch"],
        "steps_per_epoch": steps_per_epoch,
        "train_images": len(train_set),
        "batch_size": cfg.batch_size,
        "amp": use_amp,
        "speed_steps_this_job": max(job_steps - SPEED_WARMUP_STEPS, 0),
        "images_per_s_this_job": speed_images / speed_seconds if speed_seconds > 0 else None,
        "peak_mem_gb": (
            torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
        ),
        "best_val_macro_auc": state["best_val_macro_auc"],
        "best_epoch": state["best_epoch"],
        "train_hours_total": state["train_seconds"] / 3600,
        "environment": env,
        "wandb_id": state["wandb_id"],
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    if run is not None:
        run.summary.update({k: v for k, v in summary.items() if not isinstance(v, dict)})
        if complete:
            _log_model_artifact(run, cfg, ckpt_dir, summary)
    if complete:
        (run_dir / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    if run is not None:
        run.finish()
    return summary


def _log_model_artifact(run, cfg: TrainConfig, ckpt_dir: Path, summary: dict[str, Any]) -> None:
    import wandb

    weights = ckpt_dir / "best.pt"
    if not weights.exists():
        return  # speed tests run without validation, so there is no "best" model to keep
    metadata = {k: v for k, v in summary.items() if not isinstance(v, dict)}
    artifact = wandb.Artifact(f"{cfg.run_id}-model", type="model", metadata=metadata)
    artifact.add_file(str(weights))
    run.log_artifact(artifact, aliases=["best"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value (repeatable)",
    )
    args = parser.parse_args(argv)
    summary = train(load_config(args.config, args.overrides))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
