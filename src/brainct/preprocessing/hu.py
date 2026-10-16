"""Stored pixel values -> Hounsfield units (HU), padding to square and resizing.

The 12-bit sign fix is based on measured evidence (docs/results.md, Phase 2 pixel statistics):
RSNA images labelled unsigned 12-bit with a rescale intercept of 0 or 1 actually hold signed
12-bit two's-complement values - their median stored value 3097 is -999 HU (air). Values at or
above 2048 are therefore negative and need 4096 subtracted. No other group is touched.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# HU range kept after conversion. Below -1024 is scanner padding (e.g. -3024 outside the
# field of view); above 3071 is beyond the 12-bit CT range. All model windows lie inside it.
HU_MIN = -1024
HU_MAX = 3071

# Unsigned 12-bit images with an intercept above this are the mislabelled signed ones.
SIGN_FIX_MIN_INTERCEPT = -100


def needs_sign_fix(pixel_representation: int, bits_stored: int, intercept: float) -> bool:
    return pixel_representation == 0 and bits_stored == 12 and intercept > SIGN_FIX_MIN_INTERCEPT


def to_hu(
    pixels: np.ndarray,
    slope: float,
    intercept: float,
    pixel_representation: int,
    bits_stored: int,
) -> np.ndarray:
    """Convert decoded pixels to HU (float32), applying the sign fix if needed, then clip."""
    values = pixels.astype(np.int32)
    if needs_sign_fix(pixel_representation, bits_stored, intercept):
        values = values & 0x0FFF  # keep the 12 stored bits only
        values = np.where(values >= 2048, values - 4096, values)
    hu = values.astype(np.float32) * np.float32(slope) + np.float32(intercept)
    return np.clip(hu, HU_MIN, HU_MAX)


def pad_to_square(hu: np.ndarray, fill: float = HU_MIN) -> np.ndarray:
    """Center a non-square image on a square canvas filled with air, keeping proportions."""
    height, width = hu.shape
    if height == width:
        return hu
    side = max(height, width)
    canvas = np.full((side, side), fill, dtype=hu.dtype)
    top, left = (side - height) // 2, (side - width) // 2
    canvas[top : top + height, left : left + width] = hu
    return canvas


def resize(hu: np.ndarray, size: int) -> np.ndarray:
    """Resize a square float image to size x size with area averaging (Pillow BOX filter).

    For 512 -> 256 this is exactly the mean of each 2x2 block.
    """
    image = np.ascontiguousarray(hu, dtype=np.float32)
    if image.shape == (size, size):
        return image
    resized = Image.fromarray(image).resize((size, size), Image.Resampling.BOX)
    return np.asarray(resized, dtype=np.float32)
