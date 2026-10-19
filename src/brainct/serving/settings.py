"""Service settings, read from environment variables (so Docker / Kubernetes can set them)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PREFIX = "BRAINCT_"


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path("artifacts/resnet50_baseline/onnx/model_int8.onnx")
    # metrics_val.json from evaluate.py: its "thresholds" are the val-chosen F1 thresholds.
    thresholds_path: Path = Path("artifacts/resnet50_baseline/metrics_val.json")
    model_name: str = "resnet50_baseline-int8"
    threads: int = 8  # ONNX Runtime intra-op threads (8 = the Phase 5b benchmark setting)
    max_upload_mb: float = 10.0

    @classmethod
    def from_env(cls) -> Settings:
        """Defaults, overridden by BRAINCT_MODEL_PATH, BRAINCT_THREADS, ... if set."""
        env = os.environ
        defaults = cls()
        return cls(
            model_path=Path(env.get(f"{PREFIX}MODEL_PATH", defaults.model_path)),
            thresholds_path=Path(env.get(f"{PREFIX}THRESHOLDS_PATH", defaults.thresholds_path)),
            model_name=env.get(f"{PREFIX}MODEL_NAME", defaults.model_name),
            threads=int(env.get(f"{PREFIX}THREADS", defaults.threads)),
            max_upload_mb=float(env.get(f"{PREFIX}MAX_UPLOAD_MB", defaults.max_upload_mb)),
        )

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)
