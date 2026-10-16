"""DICOM bytes -> the cached HU representation. The single entry point for training and serving.

    hu, info = preprocess_dicom(data)    # int16 HU, shape (256, 256): what the cache stores
    x = window.model_input(hu)           # float32 (3, 256, 256): what the model sees

Training builds the cache with preprocess_dicom; the API calls the same function on uploads.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pydicom
from pydicom import config as pydicom_config
from pydicom.dataset import Dataset

from brainct.preprocessing.hu import needs_sign_fix, pad_to_square, resize, to_hu

IMAGE_SIZE = 256

REQUIRED_TAGS = (
    "PixelData",
    "RescaleSlope",
    "RescaleIntercept",
    "PixelRepresentation",
    "BitsStored",
    "Rows",
    "Columns",
)


class DicomError(ValueError):
    """The input is not a usable single-channel CT DICOM."""


@dataclass(frozen=True)
class PreprocessInfo:
    original_shape: tuple[int, int]
    sign_fixed: bool
    padded: bool


def read_dicom(data: bytes) -> Dataset:
    """Parse DICOM bytes, raising DicomError if they are unreadable or lack required tags."""
    pydicom_config.settings.reading_validation_mode = pydicom_config.IGNORE
    try:
        ds = pydicom.dcmread(io.BytesIO(data), force=True)
    except Exception as exc:
        raise DicomError(f"not a readable DICOM file: {exc}") from exc
    missing = [tag for tag in REQUIRED_TAGS if tag not in ds]
    if missing:
        raise DicomError(f"missing required DICOM tags: {', '.join(missing)}")
    if int(ds.get("SamplesPerPixel", 1)) != 1:
        raise DicomError("expected a single-channel (grayscale) image")
    return ds


def dicom_to_hu(ds: Dataset) -> tuple[np.ndarray, bool]:
    """Return the image in HU (float32, original resolution) and whether the sign fix ran."""
    try:
        pixels = ds.pixel_array
    except Exception as exc:
        raise DicomError(f"could not decode pixel data: {exc}") from exc
    if pixels.ndim != 2:
        raise DicomError(f"expected a 2D image, got shape {pixels.shape}")
    representation, bits = int(ds.PixelRepresentation), int(ds.BitsStored)
    intercept = float(ds.RescaleIntercept)
    hu = to_hu(pixels, float(ds.RescaleSlope), intercept, representation, bits)
    return hu, needs_sign_fix(representation, bits, intercept)


def preprocess_dicom(data: bytes, size: int = IMAGE_SIZE) -> tuple[np.ndarray, PreprocessInfo]:
    """DICOM bytes -> int16 HU image of shape (size, size), plus what was done to it."""
    ds = read_dicom(data)
    hu, sign_fixed = dicom_to_hu(ds)
    original_shape = (int(hu.shape[0]), int(hu.shape[1]))
    square = pad_to_square(hu)
    image = np.rint(resize(square, size)).astype(np.int16)
    info = PreprocessInfo(original_shape, sign_fixed, padded=square.shape != hu.shape)
    return image, info
