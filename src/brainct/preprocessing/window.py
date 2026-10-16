"""HU image -> 3-channel model input (brain, subdural and bone windows, ImageNet-normalized)."""

from __future__ import annotations

import numpy as np

# (center, width) in HU, in channel order.
WINDOWS: dict[str, tuple[float, float]] = {
    "brain": (40.0, 80.0),
    "subdural": (80.0, 200.0),
    "bone": (600.0, 2800.0),
}

# ImageNet statistics for the pretrained ResNet-50, one value per channel.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def apply_window(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """Map [center - width/2, center + width/2] HU linearly to [0, 1], clipping outside."""
    low = center - width / 2
    return np.clip((hu.astype(np.float32) - low) / width, 0.0, 1.0)


def to_channels(hu: np.ndarray) -> np.ndarray:
    """Stack the windows into a float32 array of shape (3, H, W) with values in [0, 1]."""
    return np.stack([apply_window(hu, c, w) for c, w in WINDOWS.values()]).astype(np.float32)


def model_input(hu: np.ndarray) -> np.ndarray:
    """The exact tensor the model sees: windowed channels, ImageNet-normalized, (3, H, W)."""
    channels = to_channels(hu)
    return (channels - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
