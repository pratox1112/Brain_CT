# Results

Every measured number in this project lives here. Format per entry: date, commit, run ID, hardware, software, command, result, notes. See `ROADMAP_v1.md` §7.

---

## Phase 1 – Data foundation

### Phase 1 – RSNA dataset download
- Date: 2026-10-15
- Commit: n/a (no project code yet)  Run ID: Slurm job 11112128
- Hardware: Northeastern Explorer, `short` partition CPU node (hostname in the job log), 2 CPUs, 4 GB RAM
- Software: Kaggle CLI (in uv venv `/scratch/$USER/envs/brainct`)
- Command: `sbatch ~/Brain_CT/jobs/download_rsna.sh` → `kaggle competitions download rsna-intracranial-hemorrhage-detection`
- Result:
  - Zip size: 182G as reported by `ls -lh` (binary GiB) = 194.67 GB as reported by the Open OnDemand file browser (decimal GB). Same file, different units. The Kaggle Data page lists 181 GB.
  - Download time: **2,763 s (46 min 3 s)** (script's `elapsed`; `sacct` Elapsed 00:46:04)
  - Average throughput: about 66 MB/s, computed from 182G ÷ 2,763 s (derived, not measured directly)
  - Job state: COMPLETED, exit code 0:0
- Notes: the speed varied roughly 12–155 MB/s during the run (from the progress bar).

### Phase 1 – Zip index (no extraction)
- Date: 2026-10-15
- Commit: n/a  Run ID: Slurm job 11113188 (`srun`)
- Hardware: Explorer `short` CPU node, 4 GB RAM
- Software: `zipinfo` / `unzip -l` (Info-ZIP on the cluster)
- Command: `zipinfo -1` (folder counts, train `.dcm` count) and `unzip -l | tail -1` (totals)
- Result:

| Entry | Files |
|---|---|
| `rsna-intracranial-hemorrhage-detection/stage_2_train/` | **752,803** `.dcm` |
| `rsna-intracranial-hemorrhage-detection/stage_2_test/` | 121,232 |
| `stage_2_train.csv` | 1 |
| `stage_2_sample_submission.csv` | 1 |
| **Total** | **874,037 files, 458,972,270,355 bytes uncompressed (≈459 GB)** |

- Notes:
  - The train DICOM count equals the unique image IDs in `stage_2_train.csv` (752,803), so the labels and images match one to one by count. An ID-level match is checked in header extraction.
  - The uncompressed size (≈459 GB) is about 2.5× the zip. Unzipping would have needed about 640 GB with the zip kept, so the no-unzip plan avoids that.
  - The train-only uncompressed size wasn't measured separately.

### Phase 1 – DICOM header extraction (all training images, read straight from the zip)
- Date: 2026-10-15
- Commit: n/a (code not yet committed; files as written 2026-10-15)  Run ID: Slurm job on node c0636 (log `/scratch/karthikeyan.pr/brainct/logs/extract_headers_<job>.log`)
- Hardware: Northeastern Explorer `short` partition, node c0636, 16 CPUs, 32 GB RAM
- Software: Python 3.11.17, pydicom 3.0.2
- Command: `sbatch ~/Brain_CT/jobs/extract_headers.sh` → `python -m brainct.data.headers --workers 16 --max-header-bytes 16384`
- Result:

| Metric | Value |
|---|---|
| Files listed / parsed / failed | 752,803 / **752,803** / **0** |
| Full-file fallback reads | 0 (16 KB always held the full header) |
| Unique patients | **18,938** |
| Unique studies | **21,744** |
| Unique series | 21,744 (one series per study) |
| Duplicate image IDs | 0 |
| Image ID ≠ SOPInstanceUID | 0 |
| Modality | CT: 752,803 |
| Photometric interpretation | MONOCHROME2: 752,803 |
| Rescale slope | 1.0: 752,803 |
| Rescale intercept | -1024: 740,736 · -1000: 6,681 · **0: 5,336** · **1: 50** |
| Bits stored | 16: 417,048 · 12: 335,755 |
| Pixel representation | 1 (signed): 419,360 · 0 (unsigned): 333,443 |
| Image shapes | 512×512: 752,534; 269 others (638×490: 49, 436×436: 36, 408×374: 33, 464×464: 32, 462×462: 32, 430×404: 31, 666×512: 29, 768×768: 27) |
| Elapsed | 211.7 s in-script (listing 9 s); 216 s whole job |
| Throughput | ≈3,550 files/s (derived: 752,803 ÷ 211.7 s) |

- Notes:
  - **Patients have multiple studies:** 18,938 patients vs 21,744 studies, so a study-level split would leak patients. The split must be by `PatientID`.
  - **Intercepts 0 and 1 (5,386 images) are unusual for CT.** They are candidates for the known RSNA pixel-value quirk. Verify against pixel histograms in Phase 2 before correcting anything.
  - **269 images are not 512×512.** Preprocessing resizes everything to 256×256, so this is handled, but aspect ratios differ for non-square ones (decide in Phase 2: plain resize or pad first).
  - No corrupt headers. Pixel-data corruption, if any, would show up in the Phase 2 cache build.

### Phase 1 – Label table and patient-level split (80/10/10)
- Date: 2026-10-15
- Commit: n/a (code not yet committed on the cluster; written 2026-10-15)  Run ID: Slurm job on node c0604 (log `/scratch/karthikeyan.pr/brainct/logs/make_split_<job>.log`)
- Hardware: Explorer `short` partition, node c0604, 4 CPUs, 16 GB
- Software: Python 3.11.17, pandas (cluster venv)
- Command: `sbatch ~/Brain_CT/jobs/make_split.sh` → `python -m brainct.data.split` (seed 2026, val 0.1, test 0.1)
- Elapsed: 34 s
- **Overlap check: passed.** Zero patients, studies or series shared between splits.

**Label table**

| Metric | Value |
|---|---|
| Label rows (excluding header) | 4,516,842 |
| Exact duplicate rows removed | 24 (4,516,818 = 752,803 × 6 after removal) |
| Conflicting duplicates | 0 |
| Images with all 6 labels | 752,803 |
| Images whose `any` disagrees with the subtypes | 0 |
| Label/header mismatches | 0 |

**Strata** (each patient's rarest subtype; rarity order by patient count: epidural < intraventricular < subdural < subarachnoid < intraparenchymal)

| none | intraventricular | subdural | intraparenchymal | subarachnoid | epidural |
|---|---|---|---|---|---|
| 11,286 | 3,226 | 2,272 | 993 | 851 | 310 |

**Split**

| | Train | Val | Test | Total |
|---|---|---|---|---|
| Patients | 15,150 (79.998%) | 1,894 (10.001%) | 1,894 (10.001%) | 18,938 |
| Studies | 17,346 | 2,188 | 2,210 | 21,744 |
| Images | 600,390 (79.75%) | 75,840 (10.07%) | 76,573 (10.17%) | 752,803 |

**Image-level positive rate**

| Label | Train | Val | Test | Total |
|---|---|---|---|---|
| any | 14.29% | 14.33% | 14.68% | 14.34% |
| epidural | 0.417% | 0.450% | 0.396% | 0.418% |
| intraparenchymal | 4.81% | 4.62% | 4.85% | 4.80% |
| intraventricular | 3.51% | 3.33% | 3.39% | 3.48% |
| subarachnoid | 4.67% | 5.25% | 4.79% | 4.74% |
| subdural | 6.21% | 6.31% | 6.63% | 6.27% |

**Positive images / positive patients**

| Label | Train | Val | Test |
|---|---|---|---|
| any | 85,821 / 6,122 | 10,868 / 765 | 11,244 / 765 |
| epidural | 2,501 / 248 | 341 / 31 | 303 / 31 |
| intraparenchymal | 28,902 / 3,736 | 3,505 / 458 | 3,711 / 480 |
| intraventricular | 21,086 / 2,636 | 2,526 / 334 | 2,593 / 333 |
| subarachnoid | 28,029 / 2,725 | 3,978 / 355 | 3,668 / 371 |
| subdural | 37,302 / 2,674 | 4,784 / 348 | 5,080 / 352 |

- Notes:
  - Rates in the summary tables are rounded from the report's 5-decimal values.
  - Patient fractions are exact by construction. Image fractions drift slightly because patients have different numbers of slices.
  - **Epidural is thin in val and test: 31 patients each** (341 and 303 images). Its per-subtype AUC will have the widest uncertainty. Report it with a confidence interval (bootstrap over patients) in Phase 3.
  - Outputs (cluster, not in git): `~/brainct_outputs/splits/{labels.parquet, split_images.parquet, split_patients.csv, split_report.json}`.

### Phase 1 – Test-split sample DICOMs for the laptop
- Date: 2026-10-15
- Commit: n/a (code not yet committed on the cluster; written 2026-10-15)  Run ID: Slurm job on node c0219
- Hardware: Explorer `short` partition, node c0219, 2 CPUs, 8 GB
- Command: `sbatch ~/Brain_CT/jobs/extract_samples.sh` → `python -m brainct.data.samples` (seed 2026; 60 normal, 40 per subtype, 5 per edge case)
- Elapsed: 32 s
- Result:
  - **280 images from 231 patients, all from the test split.** 146,990,556 bytes (≈147 MB).
  - Images per reason: normal 60 · each of the 5 subtypes 40 · intercept_0 5 · signed_12bit 5 · unsigned_12bit 5 · signed_16bit 5.
  - Label positives among samples: any 204 · epidural 42 · intraparenchymal 87 · intraventricular 65 · subarachnoid 73 · subdural 75. Higher than 40 because images with 2+ subtypes count under each.
  - **Not represented (zero matching test-split images): intercept_1, not_512x512, unsigned_16bit.** Whether they exist in other splits is not yet checked.
- Location: cluster `~/brainct_outputs/samples/`, copied to laptop `C:\BrainCT\samples\` (git-ignored).

---

## Phase 2 – Preprocessing

### Phase 2 – Header edge cases by split, and pixel format vs rescale intercept
- Date: 2026-10-16
- Run ID: Slurm job 11142381 (`srun`, short partition, 8 GB)
- Command: ad-hoc pandas query on `dicom_headers.parquet` joined with `split_images.parquet` (command in chat; not committed)
- Result:

| Case | Images by split | Patients |
|---|---|---|
| Rescale intercept = 1 | train 50 | 2 |
| Not 512×512 | train 269 | 8 |
| Unsigned 16-bit | none in the dataset | 0 |

Pixel format × rescale intercept (image counts):

| PixelRepresentation | BitsStored | -1024 | -1000 | 0 | 1 |
|---|---|---|---|---|---|
| 0 (unsigned) | 12 | 326,639 | 6,681 | **73** | **50** |
| 1 (signed) | 12 | 0 | 0 | 2,312 | 0 |
| 1 (signed) | 16 | 414,097 | 0 | 2,951 | 0 |

- Notes:
  - **Unsigned pixels with intercept 0 or 1 (123 images)** cannot produce negative HU, so air could never reach about -1000. These are the prime suspects for corrupted values. To verify with pixel histograms.
  - **Signed pixels with intercept 0 (5,263 images)** are plausibly already stored as HU (signed values, no offset needed). To verify that their raw values span about -1000 to +1000s.
  - Every intercept-1 and non-512×512 image is in **train**, from only 2 and 8 patients, so val and test metrics are unaffected by them.

### Phase 2 – Pixel statistics per pixel-format group (evidence for the HU correction)
- Date: 2026-10-16
- Commit: n/a on the cluster (code on branch `building_preprocessing_function`)  Run ID: Slurm job on node d0146
- Hardware: Explorer `short`, node d0146, 2 CPUs, 8 GB
- Command: `sbatch ~/Brain_CT/jobs/inspect_pixels.sh` → `python -m brainct.preprocessing.inspect_pixels` (20 images per group, seed 2026, one image per patient first)
- Elapsed: 29 s
- Result: medians across the 20 images in each group. "stored" = raw 16-bit word; "decoded" = pydicom `pixel_array`; HU = decoded × slope + intercept.

| Statistic | unsigned12 -1024 | unsigned12 -1000 | **unsigned12 0** | **unsigned12 1** | signed12 0 | signed16 -1024 | signed16 0 |
|---|---|---|---|---|---|---|---|
| stored p50 | 115.5 | 28 | **3097** | **3097** | 64536 | 829.5 | 62512 |
| stored max | 2569 | 2543 | **4095** | **4095** | 65535 | 63536 | 65535 |
| HU min | -1022.5 | -1000 | **0** | **1** | -1000 | -3024 | -3024 |
| HU p50 | -908.5 | -972 | **3097** | **3098** | -992 | -975.5 | -957 |
| HU p99 | 1026 | 1087.5 | **4079** | **4068** | 1141.5 | 1372.5 | 1219 |
| HU max | 1545 | 1543 | **4095** | **4096** | 1526 | 1821.5 | 1758 |
| frac air (< -900) | 0.504 | 0.606 | **0.000** | **0.000** | 0.605 | 0.590 | 0.545 |
| frac brain (0–80) | 0.243 | 0.169 | 0.164 | 0.132 | 0.188 | 0.190 | 0.172 |
| frac bone (> 300) | 0.075 | 0.058 | **0.812** | **0.849** | 0.050 | 0.067 | 0.064 |
| frac below -1100 | 0 | 0 | 0 | 0 | 0 | 0.213 | 0.226 |
| frac above 3000 | 0 | 0 | **0.754** | **0.811** | 0 | 0 | 0 |

- Findings:
  - **unsigned12 with intercept 0/1 is broken:** there is no air at all, and about 80% of pixels read as above 3000 HU. The median stored value of 3097 ≈ 4096 − 999, which is exactly how **-999 HU (air) looks in 12-bit two's complement**. These pixels are *signed* 12-bit values mislabelled as unsigned. Fix: treat values ≥ 2048 as negative (subtract 4096), then apply slope and intercept. Brain-range fractions (0.13–0.16) already match the normal groups, because small positive values are unaffected by the mislabel.
  - **signed12 with intercept 0 is fine:** the stored values already are HU (air -1000, about 60% air).
  - **signed16 groups:** HU min -3024 with about 21–23% of pixels below -1100. That's padding outside the scanner's circular field of view (a fixed fill value), not corruption. All three windows' lower bounds are above -1000 HU, so the padding maps to 0 either way.
  - Unsigned 16-bit doesn't occur.
- Edge samples (same job): **25 train-split DICOMs**, 5 per case with no overlap: not_512x512, intercept_1, unsigned12_intercept_0, signed12_intercept_0, signed16_intercept_0. Cluster `~/brainct_outputs/samples_edge/`; for preprocessing tests only, never for evaluation.

### Phase 2 – Verification of the 12-bit sign fix (same 20 images per group, before vs after)
- Date: 2026-10-16
- Run ID: Slurm job on node c0668 (`jobs/inspect_pixels.sh`, same seed so the same images as the first run). Elapsed 27 s.
- "fixed" = HU from the preprocessing code (`brainct.preprocessing.dicom.dicom_to_hu`: sign fix + clip to [-1024, 3071]).

| Statistic | unsigned12 -1024 | unsigned12 -1000 | **unsigned12 0** | **unsigned12 1** | signed12 0 | signed16 -1024 | signed16 0 |
|---|---|---|---|---|---|---|---|
| sign_fixed_frac | 0 | 0 | **1.0** | **1.0** | 0 | 0 | 0 |
| frac air, before → after | 0.504 → 0.504 | 0.606 → 0.606 | **0.000 → 0.527** | **0.000 → 0.688** | 0.605 → 0.605 | 0.590 → 0.590 | 0.545 → 0.545 |
| frac bone, before → after | 0.075 → 0.075 | 0.058 → 0.058 | **0.812 → 0.062** | **0.849 → 0.050** | 0.050 → 0.050 | 0.067 → 0.067 | 0.064 → 0.064 |
| frac > 3000 HU, before → after | 0 → 0 | 0 → 0 | **0.754 → 0** | **0.811 → 0** | 0 → 0 | 0 → 0 | 0 → 0 |
| frac < -1100 HU, before → after | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0.213 → 0 | 0.226 → 0 |
| HU p50, before → after | -908.5 → -908.5 | -972 → -972 | **3097 → -925.5** | **3098 → -967.5** | -992 → -992 | -975.5 → -975.5 | -957 → -957 |
| HU min, after | -1022.5 | -1000 | -1000 | -999 | -1000 | -1024 | -1024 |
| HU max, after | 1545 | 1543 | 1769.5 | 1575.5 | 1526 | 1821.5 | 1758 |

- **Conclusion: the fix works and is targeted.**
  - The two broken groups now have air, brain and bone fractions in the same range as the normal groups.
  - The sign fix ran on 100% of the broken groups and 0% of the others.
  - Normal groups are unchanged, except that clipping removed the -3024 field-of-view padding in the signed16 groups (now floored at -1024).
- Laptop: 360 tests pass (`uv run pytest`), including all 305 real sample and edge DICOMs (shape, HU range, air present) and the edge-branch test (the sign fix runs only on the broken edge images).

### Phase 2 – Cache build, test run (first 5,000 images)
- Date: 2026-10-16
- Run ID: Slurm job on node d0149, `OUT=/scratch/karthikeyan.pr/brainct/preprocessed_test`, `--limit 5000`
- Hardware: Explorer `short`, 32 CPUs, 64 GB
- Command: `sbatch --export=ALL,OUT=.../preprocessed_test ~/Brain_CT/jobs/build_cache.sh --limit 5000`
- Result:
  - 5,000 / 5,000 ok, 0 failed. Sign-fixed 2, padded 0, not 512×512: 1 (square, so no padding).
  - By split: train 3,972 · val 510 · test 518.
  - `hu256.npy` 655,360,128 bytes = 5,000 × 256 × 256 × 2 + 128-byte .npy header.
  - Elapsed 20.4 s in-script (8 s setup), 26 s job. Reported rate 249 img/s.
- Notes: with only 5 chunks of 1,000 images, at most 5 of the 32 workers had work, so 249 img/s understates the full-run throughput. It's not a usable estimate of the full build's speed.

### Phase 2 – Cache build, full run (all 752,803 training images)
- Date: 2026-10-16
- Run ID: Slurm job 11144059 on node d0142
- Hardware: Explorer `short`, 32 CPUs, 64 GB (sstat MaxRSS showed about 96 GB mid-run, very likely shared memmap pages counted once per worker; no OOM)
- Software: Python 3.11.17, NumPy 2.4.6, Pillow 12.3.0, pydicom 3.0.2
- Command: `sbatch ~/Brain_CT/jobs/build_cache.sh` → `python -m brainct.data.build_cache --workers 32` (size 256, chunk size 1,000)
- Result:

| Metric | Value |
|---|---|
| Images / ok / **failed** | 752,803 / 752,802 / **1** |
| Sign-fixed (unsigned 12-bit, intercept 0/1) | **123**, exactly as predicted from the headers (73 + 50) |
| Not 512×512 / padded (non-square) | 269 / 142 |
| OK by split | train 600,389 · val 75,840 · test 76,573 (the one failure is in train) |
| Cache file | `hu256.npy`, 98,671,394,944 bytes (92 GiB), int16 [752803, 256, 256] |
| Chunks | 753, 0 resumed |
| Elapsed | **2,049.7 s (34 min)** in-script, 2,072 s job |
| Throughput | 371 img/s overall (log); 312–432 img/s across progress intervals |

- Notes:
  - The one failed image is recorded in `~/brainct_outputs/preprocessed/cache_failures.csv`. Its row holds air (-1024) and `ok=False`, so **training must drop `ok=False` rows**.
  - Its header parsed fine in Phase 1, so the corruption is in the pixel data.
  - **The failed image is `ID_6431af929`** (cache row 295,413, train split). **Its pixel data is truncated: 153,710 of the expected 524,288 bytes** (512 × 512 × 2). This file is commonly reported as corrupt in the RSNA dataset. It's excluded from training via `ok=False`.

### Phase 2 – Cache equivalence check (cluster)
- Date: 2026-10-16
- Run ID: Slurm job on node c0603 (`jobs/check_cache.sh` → `python -m brainct.data.check_cache verify`, seed 2026). Elapsed 47 s.
- Software: Python 3.11.17, NumPy 2.4.6, Pillow 12.3.0, pydicom 3.0.2
- Result:
  - **1,000 random cache rows vs fresh `preprocess_dicom` of the zip bytes: 0 mismatches (exact equality).**
  - **All 305 laptop sample and edge DICOMs are in the cache, and all 305 cache rows equal fresh preprocessing of the sample files: 0 mismatches.**
  - Expected outputs written to `~/brainct_outputs/samples/expected_hu256.npz`, for the laptop comparison.

### Phase 2 – Cross-machine equivalence (laptop vs cluster)
- Date: 2026-10-16
- Hardware: laptop (Windows 11, Intel Core Ultra 7 155U) vs cluster (Linux, Explorer)
- Software (identical on both): Python 3.11.17, NumPy 2.4.6, Pillow 12.3.0, pydicom 3.0.2
- Command (laptop): `uv run python -m brainct.data.check_cache compare --expected samples/expected_hu256.npz --sample-dir samples/dicom --sample-dir samples_edge/dicom`
- Result: **305 / 305 sample and edge DICOMs compared, 0 missing, 0 mismatched.** The laptop's `preprocess_dicom` output is bit-identical to the cluster's, and so to the training cache.
- `uv run pytest` on the laptop: **367 passed** (9.92 s), including `test_laptop_matches_cluster_preprocessing`.
- **Phase 2 acceptance criteria met:** unit tests pass; the cache holds every image (752,802 ok plus 1 documented corrupt file); the cache matches fresh preprocessing (1,000 random rows plus 305 samples); and training and serving preprocessing are identical across machines.
