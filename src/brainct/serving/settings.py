"""Service settings, read from environment variables (so Docker / Kubernetes can set them)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PREFIX = "BRAINCT_"
TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path("artifacts/resnet50_baseline/onnx/model_int8.onnx")
    # metrics_val.json from evaluate.py: its "thresholds" are the val-chosen F1 thresholds.
    thresholds_path: Path = Path("artifacts/resnet50_baseline/metrics_val.json")
    model_name: str = "resnet50_baseline-int8"
    threads: int = 8  # ONNX Runtime intra-op threads (8 = the Phase 5b benchmark setting)
    max_upload_mb: float = 10.0
    # Micro-batching: concurrent requests are run as one batch of up to max_batch images,
    # waiting at most max_wait_ms for others to arrive after the first one.
    # Off by default: on the laptop (2026-10-19) it gave no throughput gain (about 68 req/s
    # either way) and added about 5 ms at low load (results.md). Re-measure on GKE in Phase 9.
    batching: bool = False
    max_batch: int = 8
    max_wait_ms: float = 5.0
    warmup: bool = True  # one dummy prediction at startup, before /ready reports ready

    @classmethod
    def from_env(cls) -> Settings:
        """Defaults, overridden by BRAINCT_MODEL_PATH, BRAINCT_THREADS, ... if set."""
        env = os.environ
        d = cls()

        def get(name: str, default):
            return env.get(f"{PREFIX}{name}", default)

        return cls(
            model_path=Path(get("MODEL_PATH", d.model_path)),
            thresholds_path=Path(get("THRESHOLDS_PATH", d.thresholds_path)),
            model_name=get("MODEL_NAME", d.model_name),
            threads=int(get("THREADS", d.threads)),
            max_upload_mb=float(get("MAX_UPLOAD_MB", d.max_upload_mb)),
            batching=str(get("BATCHING", d.batching)).lower() in TRUE,
            max_batch=int(get("MAX_BATCH", d.max_batch)),
            max_wait_ms=float(get("MAX_WAIT_MS", d.max_wait_ms)),
            warmup=str(get("WARMUP", d.warmup)).lower() in TRUE,
        )

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)
