"""Training configuration: a YAML file plus `--set key=value` overrides on the command line."""

from __future__ import annotations

import dataclasses
import os
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PATH_FIELDS = ("runs_dir", "cache_dir", "labels_path")


@dataclass
class TrainConfig:
    run_id: str = "resnet50_baseline"
    runs_dir: str = "/scratch/$USER/brainct/runs"
    cache_dir: str = "/scratch/$USER/brainct/preprocessed"
    labels_path: str = "~/brainct_outputs/splits/labels.parquet"
    device: str = "cuda"
    seed: int = 2026

    # Model and optimisation
    pretrained: bool = True
    epochs: int = 5
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    # Loss weight per output, in LABELS order: any, then the 5 subtypes (RSNA metric weighting).
    class_weights: list[float] = field(default_factory=lambda: [2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    amp: bool = True  # bf16 autocast on CUDA
    channels_last: bool = True

    # Augmentation (train split only), applied to windowed channels before normalization
    augment: bool = True
    hflip_p: float = 0.5
    rotate_degrees: float = 10.0
    scale_min: float = 0.9
    scale_max: float = 1.1
    translate: float = 0.05

    # Data loading
    cache_read: str = "direct"  # direct (one read per row) | mmap; see CacheDataset
    num_workers: int = 12
    val_batch_size: int = 256

    # Run control
    validate: bool = True
    max_steps: int | None = None  # stop early (speed tests, smoke runs)
    log_every: int = 50
    ckpt_every: int = 2000
    time_budget_hours: float = 7.5  # checkpoint and exit before the 8 h Slurm limit

    # Weights & Biases
    wandb_mode: str = "online"  # online | offline | disabled
    wandb_project: str = "brainct"
    wandb_entity: str | None = None

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_dir) / self.run_id

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _expand(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))


def _coerce(name: str, value: Any, hint: Any) -> Any:
    """Convert a YAML value to the field's declared type.

    PyYAML follows YAML 1.1, which reads "1e-4" (no decimal point) as a string, not a float.
    """
    if value is None:
        return None
    if typing.get_origin(hint) is list:
        return [float(v) for v in value]
    allowed = typing.get_args(hint) or (hint,)
    if isinstance(value, bool):
        if bool not in allowed:
            raise ValueError(f"{name} expects {hint}, got a boolean")
        return value
    try:
        if float in allowed:
            return float(value)
        if int in allowed:
            return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} expects {hint}, got {value!r}") from exc
    return value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> TrainConfig:
    """Load a YAML config and apply `key=value` overrides (values parsed as YAML)."""
    values: dict[str, Any] = {}
    if path is not None:
        values.update(yaml.safe_load(Path(path).read_text()) or {})
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        values[key.strip()] = yaml.safe_load(raw)

    known = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    hints = typing.get_type_hints(TrainConfig)
    config = TrainConfig(**{k: _coerce(k, v, hints[k]) for k, v in values.items()})
    for name in PATH_FIELDS:
        setattr(config, name, _expand(getattr(config, name)))
    if len(config.class_weights) != 6:
        raise ValueError("class_weights needs 6 values (any + 5 subtypes)")
    return config
