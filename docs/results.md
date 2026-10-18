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

---

## Phase 3 – Training

### Phase 3 – Speed test, AMP off vs on (500 steps, no validation)
- Date: 2026-10-16
- Code: branch `training-code` (commit pushed by the owner; the cluster copy is not a git checkout)
- Software: Python 3.11.17, torch 2.14.1+cu126, torchvision 0.29.1+cu126, CUDA 12.6
- Command: `sbatch ~/Brain_CT/jobs/train.sh --set run_id=speed_amp_{off,on} --set amp={false,true} --set max_steps=500 --set validate=false`
- Settings: ResNet-50 (ImageNet), batch 64, 256×256, channels_last, 12 DataLoader workers, 16 CPUs, augmentation on. Throughput excludes the first 20 steps.

| Run | GPU (node) | AMP | Images/s | Peak GPU memory | W&B run |
|---|---|---|---|---|---|
| speed_amp_off | A100-SXM4-**40GB** (d1026) | off (fp32 with TF32) | **365.7** | **9.85 GB** | iyfbabw0 |
| speed_amp_on | A100-SXM4-**80GB** (d1028) | bf16 | **345.8** | **4.16 GB** | wxlq7poe |

- Findings:
  - **Peak memory: −58% with bf16 AMP** (9.85 → 4.16 GB).
  - **Throughput is the same with and without AMP** (346 vs 366 img/s, within noise). The GPU is **not** the bottleneck: training is limited by the input pipeline (12 CPU workers reading cache rows from network scratch, then windowing and augmenting each image on the CPU). The AMP speed benefit can't be measured until that's fixed.
  - The two runs landed on different A100 variants (40 GB / 80 GB; same GPU chip). For the final AMP comparison, both runs should be on the same variant.
  - At about 350 img/s, one epoch (600,389 images) takes about 29 min (derived, not measured).
  - Job startup before the first step took about 3 min (imports from network scratch, dataset index, W&B init).

### Phase 3 – Input-pipeline benchmark (no model, no GPU)
- Date: 2026-10-16
- Run ID: Slurm job on node d0003 (`short`, 32 CPUs, 64 GB). Elapsed 660 s.
- Command: `sbatch ~/Brain_CT/jobs/bench_loader.sh` → `python -m brainct.training.bench_loader` (train split, batch 64, 100 timed batches after 5 warmup batches, a different random order per measurement)
- Result (images/s):

| Workers | read (cache row only) | window (+3 windows, normalize) | augment (+flip/affine) |
|---|---|---|---|
| 4 | 68.8 | 157.7 | 131.3 |
| 8 | 92.6 | 227.7 | 218.3 |
| 16 | 282.1 | 338.2 | 336.9 |
| 32 | 503.4 | 465.9 | 488.6 |

- Findings:
  - **Reading cache rows is the bottleneck, not CPU work.** Adding windowing and augmentation doesn't lower throughput: at 16 and 32 workers all three stages are about the same. The "read" stage ran first, while the page cache was cold, which is probably why it looks slower than the others at 4 and 8 workers.
  - Throughput rises almost **linearly with the number of workers**, while the data rate stays tiny (503 img/s × 128 KB ≈ 64 MB/s). That's the signature of **per-request latency, not bandwidth**: each worker spends most of its time waiting on storage.
  - Likely cause: the cache is read through `np.load(mmap_mode="r")` on network scratch. A memmap row read becomes many small page-fault reads (128 KB ≈ 32 × 4 KB pages), each a network round trip, instead of one 128 KB request.

### Phase 3 – Input-pipeline benchmark, direct reads vs memmap
- Date: 2026-10-17
- Run ID: Slurm job on node c0222 (`short`, 32 CPUs, 64 GB). Elapsed 118 s.
- Change under test: `CacheDataset(read="direct")`, one seek + readinto of exactly one row (131,072 bytes) per image, instead of a numpy memmap.
- Result (images/s, same job, same node; `read_direct` ran first with the coldest page cache):

| Workers | read_direct | read_mmap | augment (full training path, direct reads) |
|---|---|---|---|
| 4 | **1,449.8** | 754.0 | 639.7 |
| 8 | **2,433.1** | 891.2 | 959.8 |
| 16 | **2,714.3** | 576.2 | 1,929.0 |
| 32 | 1,837.1 | 944.5 | **2,553.5** |

- Findings:
  - **Direct reads are about 1.9–4.7× faster than memmap reads** within the same job (4 workers: 1,450 vs 754; 16 workers: 2,714 vs 576).
  - **The full training input path now reaches 1,929 img/s with 16 workers and 2,554 img/s with 32 workers**, compared with the about 350 img/s seen in the speed test (12 workers, memmap). Bigger samples are needed to compare these exactly, since the conditions differ.
  - With direct reads, **CPU work (windowing + augmentation) becomes the main cost at low worker counts** (4 workers: 640 augment vs 1,450 read), so the worker count matters again. At 32 workers the augment path comes close to the raw read rate.
  - **Storage speed varies between runs:** memmap reads here (576–945 img/s) were much faster than in the 2026-10-16 benchmark (69–503 img/s, node d0003, 22:05). Network scratch load changes over time, so only **same-job** comparisons are reliable.
  - These were measured on a CPU node; the GPU node's CPUs and network path may differ.

### Phase 3 – Speed test, AMP off vs on, after the input-pipeline fix (same GPU)
- Date: 2026-10-17
- Run ID: one Slurm job running both tests back to back on node d1029 (W&B runs `9lfo6gn1` = AMP off, `7pz2aeuj` = AMP on)
- Hardware: **NVIDIA A100-SXM4-80GB, the same GPU for both runs**; 32 CPUs, 64 GB
- Software: Python 3.11.17, torch 2.14.1+cu126, torchvision 0.29.1+cu126, CUDA 12.6
- Settings: ResNet-50, batch 64, 256×256, channels_last, augmentation on, **28 DataLoader workers, direct cache reads**, 500 steps, no validation. Throughput over 480 steps (the first 20 excluded).
- Command: `sbatch ... --cpus-per-task=32 --wrap '... for amp in false true; do python -m brainct.training.train --config configs/resnet50_baseline.yaml --set run_id=speed2_amp_$amp --set amp=$amp --set max_steps=500 --set validate=false --set num_workers=28; done'`

| Run | AMP | Images/s | Peak GPU memory |
|---|---|---|---|
| speed2_amp_false | off (fp32 with TF32) | **806.8** | **9.85 GB** |
| speed2_amp_true | bf16 autocast | **1,268.1** | **4.16 GB** |

- **AMP (bf16): 1.57× training throughput (806.8 → 1,268.1 img/s) and −57.7% peak GPU memory (9.85 → 4.16 GB)**, on the same A100-80GB.
- Compared with the first speed test (memmap, 12 workers, about 350 img/s on both settings), the fixed pipeline is **2.3× faster without AMP and 3.7× faster with AMP**. Caveat: different days and nodes, and the first AMP-off run was on a 40 GB A100.
- Projected epoch time at 1,268 img/s: 600,389 / 1,268 ≈ 7.9 min, so 5 epochs ≈ 40 min plus validation (derived, not measured).
- Not yet verified: whether AMP-on is now GPU-bound. 28 workers reached about 2,550 img/s in the CPU-node benchmark (above 1,268), which suggests it is, but that wasn't measured on this node.

### Phase 3 – Smoke run with forced resume, attempt 1 (FAILED: bug found)
- Date: 2026-10-17
- Run ID: `smoke`, 3 chained `gpu-short` jobs 11173509 / 11173510 / 11173511 on node d1029 (A100-SXM4-80GB). W&B run `zgkj6hg2`.
- Settings: `epochs=1`, `time_budget_hours=0.05`, defaults otherwise (bf16 AMP, 28 workers, direct reads)
- Result:
  - Job 1: trained 3,292 steps, then stopped on the time budget as designed and saved `last.pt`. Throughput **1,318.7 img/s** over 3,272 steps, peak memory 4.16 GB.
  - **Jobs 2 and 3 crashed while resuming:** `TypeError: RNG state must be a torch.ByteTensor` in `torch.set_rng_state`.
  - **Cause:** the checkpoint was loaded with `map_location=<cuda device>`, which also moved the saved RNG states to the GPU. The CPU-only laptop tests couldn't catch it.
  - **Fix:** `load_checkpoint()` always loads with `map_location="cpu"`, and `load_state_dict` moves weights and optimizer state to the parameters' device. Added a regression test (`test_checkpoint_loads_on_cpu_and_rng_state_restores`).

### Phase 3 – Smoke run with forced resume, attempt 2 (PASSED)
- Date: 2026-10-17
- Run ID: `smoke2`, 3 chained `gpu-short` jobs (first two 11174092 / 11174093) on A100-SXM4-80GB (node d1028). W&B run `i76rqmcl`, the same run continued across all jobs.
- Settings: `epochs=1`, `time_budget_hours=0.08`, defaults otherwise (bf16 AMP, 28 workers, direct reads), code including the `load_checkpoint` CPU fix
- Result:
  - Job 1: steps 1–2,879, then stopped on the time budget (707.6 img/s).
  - **Job 2: `Resumed at step 2879 (epoch 0, batch 2879)`**, exactly where job 1 stopped. Job 3 resumed again and finished the epoch (3,385 steps in that job, 637.3 img/s).
  - **Complete: 9,381 / 9,381 steps (1 epoch), validation ran once, best checkpoint saved.**
  - **Val macro AUC after 1 epoch: 0.9704** (5 subtypes). A smoke-test number, not a result: 1 epoch, with the LR schedule compressed into 1 epoch.
  - Peak memory 4.16 GB. Total train time across jobs 0.26 h.
- Notes:
  - **Acceptance check met: a run survives forced resumes.**
  - Throughput tonight was 637–708 img/s, against 1,268–1,319 earlier the same day on the same GPU type. That's consistent with shared network-scratch load varying over the day, since the input pipeline is still storage-sensitive.

### Phase 3 – Baseline training run (`resnet50_baseline`)
- Date: 2026-10-17 (20:41:58 → 21:28:02 EDT)
- Run ID: `resnet50_baseline`, W&B run `v4x48mvs`. Model artifact `resnet50_baseline-model:best` uploaded to W&B.
- Hardware: NVIDIA A100-SXM4-80GB (node d1029), 32 CPUs, 64 GB; finished in the first chained job
- Software: Python 3.11.17, torch 2.14.1+cu126, torchvision 0.29.1+cu126, CUDA 12.6
- Command: `bash ~/Brain_CT/jobs/train_chain.sh 2 --set run_id=resnet50_baseline` (config `configs/resnet50_baseline.yaml`: ResNet-50 ImageNet IMAGENET1K_V2, 6 outputs, weighted BCE any=2, AdamW lr 3e-4 wd 1e-4, batch 64, 1,000 warmup steps then cosine, bf16 AMP, augmentation, 28 workers, direct reads, seed 2026)
- Result:

| Metric | Value |
|---|---|
| Steps / epochs | 46,905 / 5 (complete) |
| **Best val macro AUC (5 subtypes)** | **0.9743** at epoch index 4 (the last epoch) |
| Training throughput | **1,109.1 img/s** (over 46,885 steps, bf16) |
| Peak GPU memory | 4.16 GB |
| Training time | **0.757 h (45.4 min)**; job wall time 2,763 s including startup and 5 validations |

- Notes:
  - Val AUC per label rose across epochs (W&B sparklines); the best epoch was the last one, so more epochs might still help. A candidate for Phase 4, judged on val only.
  - This is the val number used for model selection. Final per-subtype AUCs with confidence intervals come from `evaluate.py` (val), and the test numbers from a single test evaluation.
- Val AUC per epoch (from `metrics.jsonl`, logged at the end of each epoch):

| Epoch | any | epidural | intraparenchymal | intraventricular | subarachnoid | subdural | **macro (5 subtypes)** |
|---|---|---|---|---|---|---|---|
| 0 | 0.9665 | 0.9427 | 0.9791 | 0.9869 | 0.9525 | 0.9554 | **0.9633** |
| 1 | 0.9673 | 0.9552 | 0.9814 | 0.9889 | 0.9578 | 0.9547 | **0.9676** |
| 2 | 0.9747 | 0.9566 | 0.9843 | 0.9911 | 0.9646 | 0.9635 | **0.9720** |
| 3 | 0.9763 | 0.9544 | 0.9855 | 0.9922 | 0.9678 | 0.9666 | **0.9733** |
| 4 | 0.9765 | 0.9588 | 0.9853 | 0.9922 | 0.9680 | 0.9671 | **0.9743** |

  - Macro AUC gains per epoch: +0.0043, +0.0044, +0.0013, +0.0010. Improvement is slowing, but not flat, at the end of the cosine schedule (the LR reaches 0 at epoch 5).
  - Weakest subtypes: **epidural** (0.959, and noisy across epochs, as expected from only 31 val patients / 341 positive images), then **subdural** (0.967) and **subarachnoid** (0.968). Strongest: intraventricular (0.992) and intraparenchymal (0.985).

### Phase 3 – Baseline evaluation on val (`evaluate.py`, bf16 inference)
- Date: 2026-10-18
- Command: `sbatch ~/Brain_CT/jobs/evaluate.sh resnet50_baseline val` (best.pt = epoch index 4; inference with the run's config, i.e. bf16 autocast; 1,000 patient-level bootstrap resamples, seed 2026)
- Val set: 75,840 images, 1,894 patients
- Result:

| Label | ROC AUC | 95% CI (patient bootstrap) | F1 at val-chosen threshold | Threshold |
|---|---|---|---|---|
| any | 0.9765 | 0.9735–0.9797 | 0.838 | 0.383 |
| epidural | 0.9588 | **0.9421–0.9728** | **0.288** | 0.196 |
| intraparenchymal | 0.9853 | 0.9814–0.9886 | 0.807 | 0.440 |
| intraventricular | 0.9922 | 0.9901–0.9940 | 0.830 | 0.369 |
| subarachnoid | 0.9680 | 0.9621–0.9731 | 0.684 | 0.247 |
| subdural | 0.9670 | 0.9610–0.9727 | 0.705 | 0.369 |
| **macro (5 subtypes)** | **0.9743** | **0.9703–0.9780** | **macro F1 0.663** | — |

- Notes:
  - The AUCs match the in-training validation of epoch 4 (macro 0.97426), as expected for the same checkpoint and precision.
  - **Epidural has a high AUC but a low F1 (0.288).** At a 0.45% prevalence, even a good ranking produces many false positives per true positive at any threshold. F1 and precision-recall metrics show this; ROC AUC does not. Macro F1 is pulled down mainly by epidural.
  - **Ties from bf16 inference:** the intraventricular and subdural thresholds are bit-identical (0.36931508779525757). With bf16 autocast, the final layer outputs bf16 logits, so the probabilities fall on a coarse grid with many ties. The effect on AUC is probably small, but evaluation should use **fp32 inference** (no ties, and comparable to the Phase 5a ONNX FP32 / INT8 numbers). To redo before the test evaluation.
