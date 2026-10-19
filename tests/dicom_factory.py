"""Builds synthetic RSNA-style CT DICOMs for tests and for the regression fixtures.

A plain module (not conftest) so scripts can import it too. No real RSNA data is involved.
"""

from __future__ import annotations

import io

import numpy as np
from pydicom import config as pydicom_config
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

# RSNA uses non-standard UIDs such as "ID_000012eaf"; allow writing and reading them.
pydicom_config.settings.writing_validation_mode = pydicom_config.IGNORE
pydicom_config.settings.reading_validation_mode = pydicom_config.IGNORE


def make_dicom_bytes(
    image_id: str,
    patient_id: str = "ID_p1",
    study_uid: str = "ID_st1",
    series_uid: str = "ID_se1",
    z: float = 0.0,
    window_center: float | list[float] = 40,
    size: int = 8,
    intercept: float = -1024,
    pixel_representation: int = 0,
    bits_stored: int = 12,
    pixels: np.ndarray | None = None,
) -> bytes:
    """Build a minimal RSNA-style CT DICOM (explicit VR little endian) and return its bytes.

    `pixels` are the stored values (before rescale); default is all zeros of shape (size, size).
    """
    dtype = np.int16 if pixel_representation == 1 else np.uint16
    pixels = np.zeros((size, size), dtype=dtype) if pixels is None else pixels.astype(dtype)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = image_id
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = image_id
    ds.PatientID = patient_id
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "CT"
    ds.ImagePositionPatient = [-125.0, -125.0, z]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [0.488281, 0.488281]
    ds.SliceThickness = 5
    ds.RescaleSlope = 1
    ds.RescaleIntercept = intercept
    ds.WindowCenter = window_center
    ds.WindowWidth = 80
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = bits_stored
    ds.HighBit = bits_stored - 1
    ds.PixelRepresentation = pixel_representation
    ds.PixelData = pixels.tobytes()

    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()
