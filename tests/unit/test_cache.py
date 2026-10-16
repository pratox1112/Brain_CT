from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from brainct.data.build_cache import CACHE_FILE, INDEX_FILE, build_cache
from brainct.data.check_cache import compare, verify
from brainct.data.zip_reader import TRAIN_PREFIX
from brainct.preprocessing.dicom import preprocess_dicom
from brainct.preprocessing.hu import HU_MIN

SIZE = 8  # small images keep the tests fast; the real cache uses 256


@pytest.fixture
def cache_inputs(tmp_path, dicom_bytes):
    """A zip with 11 valid DICOMs (one mislabelled, one non-square) and 1 corrupt file."""
    rng = np.random.default_rng(0)
    files = {}
    for i in range(9):
        files[f"ID_n{i}"] = dicom_bytes(f"ID_n{i}", pixels=rng.integers(0, 2000, (16, 16)))
    files["ID_fix"] = dicom_bytes("ID_fix", intercept=0, pixels=np.full((16, 16), 3096))
    files["ID_wide"] = dicom_bytes("ID_wide", pixels=np.full((8, 16), 1064))
    files["ID_bad"] = b"not a dicom file"

    zip_path = tmp_path / "rsna.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for image_id, data in files.items():
            zf.writestr(f"{TRAIN_PREFIX}{image_id}.dcm", data)

    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    for image_id in ["ID_n0", "ID_fix", "ID_wide"]:
        (sample_dir / f"{image_id}.dcm").write_bytes(files[image_id])

    splits = tmp_path / "splits.parquet"
    split_table = pd.DataFrame({"image_id": list(files), "split": ["train", "val", "test"] * 4})
    split_table.to_parquet(splits, index=False)
    return {"zip": zip_path, "splits": splits, "files": files, "samples": sample_dir}


def _build(inputs, out, **kwargs):
    return build_cache(inputs["zip"], inputs["splits"], out, size=SIZE, chunk_size=5, **kwargs)


@pytest.mark.parametrize("workers", [1, 2])
def test_build_cache_matches_preprocess_dicom(tmp_path, cache_inputs, workers):
    out = tmp_path / f"cache_{workers}"

    summary = _build(cache_inputs, out, workers=workers)

    cache = np.load(out / CACHE_FILE, mmap_mode="r")
    index = pd.read_parquet(out / INDEX_FILE)
    assert cache.shape == (12, SIZE, SIZE) and cache.dtype == np.int16
    assert index["row"].tolist() == list(range(12))
    for row, image_id, ok in index[["row", "image_id", "ok"]].itertuples(index=False):
        if ok:
            expected, _ = preprocess_dicom(cache_inputs["files"][image_id], size=SIZE)
            np.testing.assert_array_equal(cache[row], expected)
    bad = index.set_index("image_id").loc["ID_bad"]
    assert not bad["ok"] and bad["error"]
    assert (cache[int(bad["row"])] == HU_MIN).all()
    assert summary["images"] == 12 and summary["ok"] == 11 and summary["failed"] == 1
    assert summary["sign_fixed"] == 1
    assert summary["padded"] == 1
    del cache


def test_build_cache_resumes_and_skips_done_chunks(tmp_path, cache_inputs):
    out = tmp_path / "cache"
    first = _build(cache_inputs, out)
    chunk_files = sorted((out / "chunks").glob("chunk_*.parquet"))
    chunk_files[-1].unlink()  # pretend the job died before the last chunk finished

    second = _build(cache_inputs, out)

    assert first["chunks"] == second["chunks"] == 3
    assert second["chunks_skipped_resume"] == 2
    assert second["ok"] == first["ok"]


def test_build_cache_refuses_a_different_layout(tmp_path, cache_inputs):
    out = tmp_path / "cache"
    _build(cache_inputs, out)

    with pytest.raises(ValueError, match="different layout"):
        _build(cache_inputs, out, limit=4)


def test_verify_and_compare(tmp_path, cache_inputs):
    out = tmp_path / "cache"
    _build(cache_inputs, out)
    expected = tmp_path / "expected.npz"

    result = verify(
        out,
        cache_inputs["zip"],
        n_random=50,
        sample_dirs=[cache_inputs["samples"]],
        expected_out=expected,
    )

    assert result["random_rows_checked"] == 11  # all ok rows, since 50 > 11
    assert result["random_rows_mismatched"] == 0
    assert result["samples_preprocessed"] == 3
    assert result["samples_cache_mismatched"] == 0
    assert json.loads((out / "check_summary.json").read_text())["random_rows_mismatched"] == 0

    same = compare(expected, [cache_inputs["samples"]])
    assert same["compared"] == 3 and same["mismatched"] == 0

    # A changed sample file must be detected.
    (cache_inputs["samples"] / "ID_n0.dcm").write_bytes(cache_inputs["files"]["ID_n1"])
    changed = compare(expected, [cache_inputs["samples"]])
    assert changed["mismatched"] == 1


def test_verify_detects_a_corrupted_cache_row(tmp_path, cache_inputs):
    out = tmp_path / "cache"
    _build(cache_inputs, out)
    index = pd.read_parquet(out / INDEX_FILE)
    row = int(index.loc[index["image_id"] == "ID_n3", "row"].iloc[0])
    cache = np.load(out / CACHE_FILE, mmap_mode="r+")
    cache[row] += 1
    cache.flush()
    del cache

    result = verify(out, cache_inputs["zip"], n_random=50)

    assert result["random_rows_mismatched"] == 1
    assert result["mismatched_examples"] == ["ID_n3"]
