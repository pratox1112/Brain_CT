"""Read DICOM files straight out of the RSNA zip, without extracting it."""

from __future__ import annotations

import zipfile
from pathlib import Path

TRAIN_PREFIX = "rsna-intracranial-hemorrhage-detection/stage_2_train/"

# (7FE0,0010) Pixel Data tag as it appears in little-endian DICOM bytes.
PIXEL_DATA_TAG = b"\xe0\x7f\x10\x00"


def list_dicoms(zip_path: str | Path, prefix: str = TRAIN_PREFIX) -> list[str]:
    """Return the names of all .dcm files under `prefix`, in the order they are stored in the zip.

    Storage order keeps each worker's reads close together on disk.
    """
    with zipfile.ZipFile(zip_path) as zf:
        infos = [
            info
            for info in zf.infolist()
            if not info.is_dir()
            and info.filename.startswith(prefix)
            and info.filename.lower().endswith(".dcm")
        ]
    infos.sort(key=lambda info: info.header_offset)
    return [info.filename for info in infos]


def read_member(zf: zipfile.ZipFile, name: str, max_bytes: int | None = None) -> bytes:
    """Read one file from the zip: the whole file, or only its first `max_bytes` bytes."""
    with zf.open(name) as f:
        return f.read() if max_bytes is None else f.read(max_bytes)


def read_header_bytes(zf: zipfile.ZipFile, name: str, max_bytes: int = 16384) -> tuple[bytes, bool]:
    """Return bytes that contain the full DICOM header (everything before the pixel data).

    Only the first `max_bytes` are decompressed. If the Pixel Data tag is not inside them, the
    header may be longer, so the whole file is read instead.

    Returns (data, full_read) where full_read says whether the fallback was needed.
    """
    head = read_member(zf, name, max_bytes)
    if PIXEL_DATA_TAG in head or len(head) < max_bytes:
        return head, False
    return read_member(zf, name), True
