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


def _npy_layout(path: Path) -> tuple[int, tuple[int, int]]:
    """Byte offset of the data in a .npy file, and the shape of one row."""
    with open(path, "rb") as f:
        major, _ = np.lib.format.read_magic(f)
        if major == 1:
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(f)
        offset = f.tell()
    if fortran_order or dtype != np.int16 or len(shape) != 3:
        raise ValueError(f"{path}: expected a C-order int16 array [N, H, W], got {dtype} {shape}")
    return offset, (int(shape[1]), int(shape[2]))


class CacheDataset(Dataset):
    """Images of one split from the HU cache. Rows with ok=False (corrupt files) are dropped.

    Returns (x, y, i): y float32 (6,) labels, i dataset index, and x depending on `output`:
        "model_input"  float32 (3, H, W); without augmentation it equals
                       brainct.preprocessing.window.model_input(cache_row)
        "hu"           int16 (H, W), the cache row as stored (no windowing or augmentation)

    `read` chooses how a row is fetched from the .npy file:
        "direct"  one positioned read of exactly one row's bytes (default). On network storage
                  this is one request per image (compared with mmap in bench_loader.py).
        "mmap"    numpy memmap; a row read can become many small page-fault reads.
    Both return identical arrays.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        labels_path: str | Path,
        split: str,
        augment: Augment | None = None,
        output: str = "model_input",
        read: str = "direct",
    ) -> None:
        if output not in ("model_input", "hu"):
            raise ValueError(f"output must be 'model_input' or 'hu', got {output!r}")
        if read not in ("direct", "mmap"):
            raise ValueError(f"read must be 'direct' or 'mmap', got {read!r}")
        if output == "hu" and augment is not None:
            raise ValueError("augmentation needs output='model_input'")
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
        self.output = output
        self.rows = table["row"].to_numpy()
        self.image_ids = table["image_id"].to_numpy()
        self.patient_ids = table["patient_id"].to_numpy()
        self.labels = np.array(table[LABELS], dtype=np.float32)  # writable copy for torch
        self.read = read
        self._cache_path = cache_dir / CACHE_FILE
        self._offset, self._row_shape = _npy_layout(self._cache_path)
        self._row_bytes = int(np.prod(self._row_shape)) * np.dtype(np.int16).itemsize
        # Opened lazily, once per DataLoader worker process (handles can't be shared across forks).
        self._memmap: np.ndarray | None = None
        self._file = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_memmap"], state["_file"] = None, None  # reopened in each worker
        return state

    def read_row(self, row: int) -> np.ndarray:
        """One cache row as a writable int16 array."""
        if self.read == "mmap":
            if self._memmap is None:
                self._memmap = np.load(self._cache_path, mmap_mode="r")
            return np.array(self._memmap[row])
        if self._file is None:
            self._file = open(self._cache_path, "rb", buffering=0)  # kept open for the worker
        hu = np.empty(self._row_shape, dtype=np.int16)
        self._file.seek(self._offset + row * self._row_bytes)
        if self._file.readinto(memoryview(hu).cast("B")) != self._row_bytes:
            raise OSError(f"short read for cache row {row}")
        return hu

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        hu = self.read_row(int(self.rows[i]))
        y = torch.from_numpy(self.labels[i])
        if self.output == "hu":
            return torch.from_numpy(hu), y, i
        x = torch.from_numpy(to_channels(hu))
        if self.augment is not None:
            x = self.augment(x)
        x = (x - _MEAN) / _STD
        return x, y, i


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
