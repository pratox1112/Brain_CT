from __future__ import annotations

from io import BytesIO

import numpy as np
import pydicom
import pytest

from brainct.preprocessing.dicom import DicomError, preprocess_dicom, read_dicom
from brainct.preprocessing.hu import HU_MAX, HU_MIN, needs_sign_fix, pad_to_square, resize, to_hu
from brainct.preprocessing.window import WINDOWS, apply_window, model_input, to_channels

# --- HU conversion ---------------------------------------------------------------------------


def test_to_hu_normal_unsigned():
    pixels = np.array([[24, 1064]], dtype=np.uint16)  # -1000 and 40 HU with intercept -1024

    hu = to_hu(pixels, slope=1.0, intercept=-1024, pixel_representation=0, bits_stored=12)

    assert hu.dtype == np.float32
    assert hu.tolist() == [[-1000.0, 40.0]]


def test_to_hu_applies_sign_fix_to_mislabelled_unsigned_12bit():
    # Stored 3096 is -1000 in 12-bit two's complement; 40 stays 40.
    pixels = np.array([[3096, 40, 4095]], dtype=np.uint16)

    hu = to_hu(pixels, slope=1.0, intercept=0, pixel_representation=0, bits_stored=12)

    assert hu.tolist() == [[-1000.0, 40.0, -1.0]]


def test_to_hu_sign_fix_keeps_intercept_1():
    pixels = np.array([[3096]], dtype=np.uint16)

    hu = to_hu(pixels, slope=1.0, intercept=1, pixel_representation=0, bits_stored=12)

    assert hu.tolist() == [[-999.0]]


@pytest.mark.parametrize(
    ("representation", "bits", "intercept", "expected"),
    [
        (0, 12, 0.0, True),
        (0, 12, 1.0, True),
        (0, 12, -1024.0, False),
        (0, 12, -1000.0, False),
        (1, 12, 0.0, False),  # signed with intercept 0 is already HU
        (1, 16, 0.0, False),
        (0, 16, 0.0, False),
    ],
)
def test_needs_sign_fix(representation, bits, intercept, expected):
    assert needs_sign_fix(representation, bits, intercept) is expected


def test_to_hu_signed_already_hu_and_clipped():
    pixels = np.array([[-3024, -1000, 40, 4000]], dtype=np.int16)

    hu = to_hu(pixels, slope=1.0, intercept=0, pixel_representation=1, bits_stored=16)

    assert hu.tolist() == [[HU_MIN, -1000.0, 40.0, HU_MAX]]


def test_to_hu_applies_slope():
    pixels = np.array([[100]], dtype=np.uint16)

    hu = to_hu(pixels, slope=2.0, intercept=-1024, pixel_representation=0, bits_stored=12)

    assert hu.tolist() == [[-824.0]]


# --- Padding and resizing --------------------------------------------------------------------


def test_pad_to_square_centers_and_fills_with_air():
    image = np.ones((2, 4), dtype=np.float32)

    square = pad_to_square(image)

    assert square.shape == (4, 4)
    assert square[1:3].tolist() == [[1.0] * 4] * 2
    assert (square[[0, 3]] == HU_MIN).all()


def test_pad_to_square_leaves_square_images_alone():
    image = np.zeros((3, 3), dtype=np.float32)

    assert pad_to_square(image) is image


def test_resize_halving_is_2x2_mean():
    rng = np.random.default_rng(0)
    image = rng.integers(-1024, 3071, size=(16, 16)).astype(np.float32)

    small = resize(image, 8)

    expected = image.reshape(8, 2, 8, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(small, expected, atol=1e-3)


def test_resize_same_size_is_noop():
    image = np.full((8, 8), 5.0, dtype=np.float32)

    np.testing.assert_array_equal(resize(image, 8), image)


# --- Windows ---------------------------------------------------------------------------------


def test_apply_window_maps_and_clips():
    hu = np.array([-100.0, 0.0, 40.0, 80.0, 500.0])

    out = apply_window(hu, center=40, width=80)

    assert out.tolist() == [0.0, 0.0, 0.5, 1.0, 1.0]


def test_to_channels_order_and_range():
    hu = np.array([[40.0, 80.0, 600.0]], dtype=np.float32)

    channels = to_channels(hu)

    assert channels.shape == (3, 1, 3)
    assert list(WINDOWS) == ["brain", "subdural", "bone"]
    assert channels[0, 0, 0] == 0.5  # 40 HU is the brain window's center
    assert channels[1, 0, 1] == 0.5  # 80 HU is the subdural window's center
    assert channels[2, 0, 2] == 0.5  # 600 HU is the bone window's center
    assert channels.min() >= 0 and channels.max() <= 1


def test_model_input_is_normalized_float32():
    x = model_input(np.zeros((4, 4), dtype=np.int16))

    assert x.shape == (3, 4, 4)
    assert x.dtype == np.float32
    assert np.isfinite(x).all()


# --- DICOM end to end ------------------------------------------------------------------------


def test_preprocess_dicom_end_to_end(dicom_bytes):
    pixels = np.full((16, 16), 1064)  # 40 HU everywhere
    data = dicom_bytes("ID_e2e", pixels=pixels)

    image, info = preprocess_dicom(data, size=8)

    assert image.shape == (8, 8)
    assert image.dtype == np.int16
    assert (image == 40).all()
    assert info.original_shape == (16, 16)
    assert not info.sign_fixed
    assert not info.padded


def test_preprocess_dicom_fixes_and_pads(dicom_bytes):
    pixels = np.full((8, 16), 3096)  # -1000 HU after the sign fix
    data = dicom_bytes("ID_fix", intercept=0, pixels=pixels)

    image, info = preprocess_dicom(data, size=8)

    assert info.sign_fixed
    assert info.padded
    assert info.original_shape == (8, 16)
    assert (image >= HU_MIN).all() and (image <= -1000).all()


def test_read_dicom_rejects_garbage():
    with pytest.raises(DicomError):
        read_dicom(b"definitely not a dicom")


def test_read_dicom_rejects_missing_tags(dicom_bytes):
    ds = pydicom.dcmread(BytesIO(dicom_bytes("ID_m")))
    del ds.RescaleIntercept
    buf = BytesIO()
    ds.save_as(buf)

    with pytest.raises(DicomError, match="RescaleIntercept"):
        read_dicom(buf.getvalue())
