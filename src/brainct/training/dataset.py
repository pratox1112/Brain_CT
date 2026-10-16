"""PyTorch dataset over the HU cache, plus training augmentation and a resumable epoch sampler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tvf

from brainct.data.build_cache import CACHE_FILE, INDEX_FILE
from brainct.data.labels import LABELS
from brainct.preprocessing.window import IMAGENET_MEAN, IMAGENET_STD, to_channels

_MEAN = torch.from_numpy(IMAGENET_MEAN)[:, None, None]
_STD = torch.from_numpy(IMAGENET_STD)[:, None, None]


class Augment:
    """Random flip + small affine, on windowed channels in [0, 1].

    Air is 0 in every window, so filling uncovered borders with 0 is the same as adding air.
    """

    def __init__(
        self,
        hflip_p: float = 0.5,
        rotate_degrees: float = 10.0,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
        translate: float = 0.05,
    ) -> None:
        self.hflip_p = hflip_p
        self.rotate_degrees = rotate_degrees
        self.scale_range = (scale_min, scale_max)
        self.translate = translate

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < self.hflip_p:
            x = x.flip(-1)
        angle = float((torch.rand(()) * 2 - 1) * self.rotate_degrees)
        low, high = self.scale_range
        scale = float(low + torch.rand(()) * (high - low))
        height, width = x.shape[-2:]
        shift_x = int(round(float((torch.rand(()) * 2 - 1) * self.translate * width)))
        shift_y = int(round(float((torch.rand(()) * 2 - 1) * self.translate * height)))
        return tvf.affine(
            x,
            angle=angle,
            translate=[shift_x, shift_y],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )


class CacheDataset(Dataset):
    """Images of one split from the HU cache. Rows with ok=False (corrupt files) are dropped.

    Returns (x, y, i): x float32 (3, H, W) model input, y float32 (6,) labels, i dataset index.
    Without augmentation, x equals brainct.preprocessing.window.model_input(cache_row).
    """

    def __init__(
        self,
        cache_dir: str | Path,
        labels_path: str | Path,
        split: str,
        augment: Augment | None = None,
    ) -> None:
        cache_dir = Path(cache_dir)
        index = pd.read_parquet(cache_dir / INDEX_FILE, columns=["row", "image_id", "split", "ok"])
        labels = pd.read_parquet(labels_path, columns=["image_id", "patient_id", *LABELS])
        table = index[(index["split"] == split) & index["ok"]].merge(
            labels, on="image_id", validate="1:1"
        )
        table = table.sort_values("row").reset_index(drop=True)
        if table.empty:
            raise ValueError(f"no usable images for split {split!r}")

        self.split = split
        self.augment = augment
        self.rows = table["row"].to_numpy()
        self.image_ids = table["image_id"].to_numpy()
        self.patient_ids = table["patient_id"].to_numpy()
        self.labels = np.array(table[LABELS], dtype=np.float32)  # writable copy for torch
        self._cache_path = cache_dir / CACHE_FILE
        self._cache: np.ndarray | None = None  # opened lazily, once per worker process

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self._cache is None:
            self._cache = np.load(self._cache_path, mmap_mode="r")
        hu = np.asarray(self._cache[self.rows[i]])
        x = torch.from_numpy(to_channels(hu))
        if self.augment is not None:
            x = self.augment(x)
        x = (x - _MEAN) / _STD
        return x, torch.from_numpy(self.labels[i]), i


class EpochSampler(Sampler[int]):
    """A fixed random order per (seed, epoch), optionally skipping samples already trained on.

    Resuming mid-epoch with skip = batches_done * batch_size continues in exactly the same order.
    """

    def __init__(self, n: int, seed: int, epoch: int, skip: int = 0) -> None:
        self.order = np.random.default_rng([seed, epoch]).permutation(n)[skip:]

    def __iter__(self):
        return iter(self.order.tolist())

    def __len__(self) -> int:
        return len(self.order)
