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

### Phase 3 – Final model (`resnet50_baseline`) on val, fp32 inference
- Date: 2026-10-18
- Decision (owner, 2026-10-18): **the 5-epoch baseline is the final model**; no Phase 4 tuning run.
- Command: `sbatch ~/Brain_CT/jobs/evaluate.sh resnet50_baseline val` (evaluate.py now forces fp32 inference; 1,000 patient bootstrap resamples, seed 2026)
- Val set: 75,840 images, 1,894 patients
- Result:

| Label | ROC AUC | 95% CI | Average precision | F1 | Threshold (from val) |
|---|---|---|---|---|---|
| any | 0.9766 | 0.9736–0.9797 | 0.906 | 0.838 | 0.380 |
| epidural | 0.9588 | 0.9424–0.9727 | **0.220** | **0.288** | 0.184 |
| intraparenchymal | 0.9853 | 0.9814–0.9887 | 0.877 | 0.807 | 0.448 |
| intraventricular | 0.9922 | 0.9901–0.9940 | 0.872 | 0.830 | 0.376 |
| subarachnoid | 0.9681 | 0.9622–0.9732 | 0.747 | 0.685 | 0.263 |
| subdural | 0.9671 | 0.9611–0.9727 | 0.774 | 0.706 | 0.355 |
| **macro (5 subtypes)** | **0.9743** | **0.9703–0.9780** | **0.698** | **0.663** | — |

- Notes:
  - **The tie problem is gone:** all 6 thresholds are now distinct (with bf16, IVH and SDH were bit-identical).
  - **fp32 vs bf16 inference changes AUC very little:** macro 0.97431 vs 0.97426 (+0.00005); every label within ±0.0004.
  - **Epidural:** average precision 0.220 against a chance level equal to its prevalence (0.45%), about 49× better than chance, yet still the clear weak point. Its F1 of 0.288 means many false alarms per true detection at the best threshold.
  - These thresholds are the ones applied unchanged to test.

### Phase 3 – **FINAL MODEL ON TEST (evaluated once)** — headline numbers
- Date: 2026-10-18
- Model: `resnet50_baseline` best.pt (epoch index 4), W&B run `v4x48mvs`, artifact `resnet50_baseline-model`
- Command: `sbatch ~/Brain_CT/jobs/evaluate.sh resnet50_baseline test` (fp32 inference; **thresholds from val, unchanged**; 1,000 patient bootstrap resamples, seed 2026)
- Test set: **76,573 images, 1,894 patients** (patient-level split, no overlap with train or val)
- Result:

| Label | ROC AUC | 95% CI | Average precision | F1 (val threshold) |
|---|---|---|---|---|
| any | 0.9773 | 0.9741–0.9805 | 0.911 | 0.839 |
| epidural | 0.9715 | 0.9537–0.9838 | 0.283 | 0.327 |
| intraparenchymal | 0.9845 | 0.9818–0.9870 | 0.866 | 0.797 |
| intraventricular | 0.9926 | 0.9906–0.9942 | 0.897 | 0.831 |
| subarachnoid | 0.9701 | 0.9646–0.9748 | 0.746 | 0.677 |
| subdural | 0.9648 | 0.9576–0.9711 | 0.753 | 0.695 |
| **macro (5 subtypes)** | **0.9767** | **0.9727–0.9801** | **0.709** | **0.665** |

- Notes:
  - The thresholds are identical to the val ones (0.380 / 0.184 / 0.448 / 0.376 / 0.263 / 0.355), confirming that nothing was tuned on test.
  - Test vs val macro AUC: 0.9767 vs 0.9743. The intervals overlap (val 0.9703–0.9780), so the difference is within sampling noise.
  - **Epidural** test AUC 0.9715 vs val 0.9588: its intervals are the widest (31 patients per split), so the gap is consistent with noise. Epidural remains the weak class on average precision (0.283) and F1 (0.327).
  - **Weakest by AUC: subdural (0.965) and subarachnoid (0.970).** These are thin or diffuse bleeds that benefit from neighbouring-slice context, which this single-slice model doesn't have (a known limitation).
  - **Resume-line numbers:** macro AUC **0.977** (95% CI 0.973–0.980) across 5 subtypes on a patient-level held-out test set of 76,573 images from 1,894 patients.

---

## Phase 5a – ONNX export and INT8 quantization

### Phase 5a – First run (2026-10-18): export valid, but the val calibration numbers are corrupted (bug found)
- Run: `sbatch ~/Brain_CT/jobs/optimize.sh` on node d0110, **Intel Xeon Platinum 8276 (has avx512_vnni)**, `short` partition, 32 CPUs (24 ORT threads + 8 loader workers). Elapsed 1,843 s.
- **Export (valid):** TorchScript exporter, opset 17. Parity with PyTorch FP32 on 512 val images: **max abs logit diff 8.6e-5, mean 3.1e-6.**
- **File sizes (valid):** ONNX FP32 **93,999,552 B**; ONNX INT8 (entropy) **24,028,028 B** (**−74.4%**, 3.9× smaller).
- **Test comparison at val thresholds (valid, see below why):**

| Format | Test macro AUC | Δ vs PyTorch FP32 |
|---|---|---|
| PyTorch FP32 | 0.97669 | — |
| ONNX FP32 | 0.97669 | −0.000002 |
| ONNX INT8 (entropy) | 0.97539 | **−0.00130** |

  - INT8 ΔAUC by label: any −0.0018, epidural −0.0001, intraparenchymal −0.0015, intraventricular −0.0007, subarachnoid −0.0020, subdural −0.0021.
  - Decision flips vs PyTorch FP32 (out of 76,573): **ONNX FP32: 0–7 per label (≤ 0.009%)**; **ONNX INT8: 69 (epidural, 0.09%) to 603 (any, 0.79%)**.
- **BUG: the val macro AUC per calibration method is wrong** (minmax 0.746, percentile 0.800, entropy 0.900). The same entropy INT8 model scored **0.975 on test**, which is impossible for a real difference that size.
  - **Cause:** the pipeline read val images in the main process (parity check, image size), which opened the dataset's file handle there. The DataLoader then **forked** its 8 workers, which **inherited that same open file description and its shared file offset**. Concurrent `seek` + `readinto` from several processes raced, so workers read the wrong rows and the val predictions were partly garbage.
  - **Test predictions are not affected:** the test dataset was never read in the main process, so each worker opened its own handle.
  - **Training / evaluate / bench_loader / cache are not affected:** none of them read dataset rows in the main process before starting DataLoader workers.
  - **Consequence:** the calibration method choice (entropy) was made on corrupted val numbers. It must be redone after the fix.

### Phase 5a – Second run, after the fork-safety fix (2026-10-19)
- Run: `sbatch ~/Brain_CT/jobs/optimize.sh` on node d0150, **AMD EPYC 7702 (Zen 2), no VNNI instructions**, 32 CPUs (24 ORT threads + 8 workers). Elapsed **5,010 s** (vs 1,843 s on the Xeon in run 1).
- Export parity (512 val images): max abs logit diff 9.7e-5, mean 3.1e-6 (same as run 1).
- **Val macro AUC per calibration method (now valid):** minmax 0.96648 · **percentile 0.96768 (chosen)** · entropy 0.96648. PyTorch FP32 val macro AUC = 0.97431.
  - minmax and entropy are identical to 16 digits. Unexpected: to be verified by comparing the two INT8 files' hashes.
- File sizes: ONNX FP32 93,999,552 B; INT8 (percentile) 24,028,032 B.
- Test, at the val thresholds:

| Format | Test macro AUC | Δ vs PyTorch FP32 |
|---|---|---|
| PyTorch FP32 | 0.97669 | — |
| ONNX FP32 | 0.97669 | −0.000002 |
| ONNX INT8 (percentile), **on AMD EPYC without VNNI** | 0.96937 | **−0.00732** |

  - INT8 ΔAUC by label: any −0.0074, **epidural −0.0156**, intraparenchymal −0.0016, intraventricular −0.0004, subarachnoid −0.0045, **subdural −0.0145**.
  - INT8 decision flips: any 2.8%, **subdural 4.3%**, subarachnoid 1.0%, intraparenchymal 0.5%, epidural 0.4%, intraventricular 0.2%.
- **Findings:**
  - **The bug is fixed:** val and test now agree (INT8 val −0.0066, test −0.0073 relative to PyTorch).
  - **INT8 accuracy depends on the CPU.** Run 1 on a Xeon **with** VNNI: entropy INT8 lost **0.0013** test macro AUC (valid, unaffected by the bug). Run 2 on an AMD EPYC **without** VNNI: the best INT8 lost **0.0073**. This matches a known ONNX Runtime behaviour: u8s8 INT8 on AVX2 CPUs without VNNI can saturate intermediate int16 sums. ORT recommends `reduce_range=True` (7-bit weights) or u8u8 there.
  - Against the pre-agreed bar (≤ 0.005 macro, ≤ 0.01 per subtype), run 2 **fails** (macro 0.0073; epidural 0.0156; subdural 0.0145) and run 1 **passes**.
  - The laptop (Intel Core Ultra 7 155U) supports AVX-VNNI, so serving there should behave like run 1. This needs measuring on the deployment CPU, not assuming.
- **Correction (2026-10-19): entropy calibration produced a file bit-identical to minmax** (`md5sum`: int8_entropy.onnx = int8_minmax.onnx = 5f79d227…; int8_percentile.onnx differs). So in this ONNX Runtime version, our "entropy" setting effectively gave the minmax model. **Run 1's "INT8 (entropy)" result, −0.0013 test macro AUC on the VNNI Xeon, is therefore effectively the minmax model.** The number stands; the label doesn't. The cause (ORT fallback vs a missing option) is not yet known.

### Phase 5a – Decision and final INT8 model (2026-10-19)
- **Owner's decision: minimal option.** No further cluster INT8 jobs. No `reduce_range` variant for now (revisit in Phase 8 if the cloud CPUs lack VNNI).
- **Final INT8 model: `model_int8.onnx` = percentile calibration**, chosen on valid val numbers (run 2: percentile 0.96768 vs minmax 0.96648).
- Entropy is dropped from the default comparison (it was bit-identical to minmax).
- Final Phase 5a numbers, with their hardware:
  - ONNX FP32: test macro AUC 0.97669 (Δ −0.000002 vs PyTorch); ≤ 7 decision flips per label out of 76,573; 94.0 MB.
  - INT8: 24.0 MB (3.9× smaller). **Accuracy is CPU-dependent:** test macro AUC Δ **−0.0073** for percentile on AMD EPYC 7702 (no VNNI, run 2), and Δ **−0.0013** for minmax (labelled "entropy") on Xeon Platinum 8276 (VNNI, run 1). **Percentile on a VNNI CPU was not measured on the full test set.**
  - The serving laptop (Core Ultra 7 155U) has AVX-VNNI. In Phase 5b, check INT8 vs PyTorch **agreement** on the laptop sample DICOMs (a sanity check, not a selection), alongside latency.

---

## Phase 5b – Latency and throughput benchmark (laptop CPU)

### Phase 5b – Model-only latency: PyTorch FP32 vs ONNX Runtime FP32 vs INT8
- Date: 2026-10-19
- Hardware: **laptop, Intel Core Ultra 7 155U** (14 logical CPUs, AVX-VNNI capable; the script can't read CPU flags on Windows). Plugged in, "Best performance" power mode, heavy apps closed.
- Settings: **8 threads** for PyTorch and ONNX Runtime alike; input = fixed random float32 tensors of shape (N, 3, 256, 256), so preprocessing isn't timed; 10 warmup runs; 100 / 50 / 25 timed runs at batch 1 / 8 / 32; **3 repeats, medians reported**.
- Command: `uv run python -m brainct.optimization.benchmark` (code on branch `latency-benchmark`); full output in `artifacts/benchmark/benchmark.json` (local)
- Models: `artifacts/resnet50_baseline/` → best.pt, onnx/model_fp32.onnx, onnx/model_int8.onnx (percentile, final), onnx/int8_minmax.onnx

| Version | Batch | p50 (ms) | p95 (ms) | Throughput (img/s) | Test macro AUC |
|---|---|---|---|---|---|
| PyTorch FP32 | 1 | 56.7 | **61.7** | 17.5 | 0.9767 |
| PyTorch FP32 | 8 | 447.5 | 495.7 | 17.7 | |
| PyTorch FP32 | 32 | 1,959.6 | 2,423.6 | 15.9 | |
| ONNX Runtime FP32 | 1 | 28.8 | 34.4 | 33.7 | 0.9767 |
| ONNX Runtime FP32 | 8 | 243.8 | 256.6 | 32.4 | |
| ONNX Runtime FP32 | 32 | 1,001.3 | 1,114.5 | 32.0 | |
| **ONNX Runtime INT8 (percentile, final)** | 1 | **11.1** | **12.6** | 87.6 | 0.9694–0.9754* |
| ONNX Runtime INT8 (percentile, final) | 8 | 69.8 | 72.5 | 114.4 | |
| ONNX Runtime INT8 (percentile, final) | 32 | 273.1 | 281.7 | **117.0** | |
| ONNX Runtime INT8 (minmax) | 1 | 11.3 | 12.7 | 87.2 | |
| ONNX Runtime INT8 (minmax) | 8 | 70.7 | 74.0 | 112.7 | |
| ONNX Runtime INT8 (minmax) | 32 | 276.7 | 284.1 | 115.6 | |

\* The INT8 test macro AUC was measured on the cluster, not on this laptop: 0.9694 for percentile on AMD EPYC without VNNI (Δ −0.0073); 0.9754 for minmax on a Xeon with VNNI (Δ −0.0013). Percentile on a VNNI CPU was not measured on the full test set.

- **Speedups (p95, batch 1):** PyTorch FP32 61.7 ms → ONNX FP32 34.4 ms (**1.8×**) → **INT8 12.6 ms (4.9× vs PyTorch, 2.7× vs ONNX FP32)**.
- **Throughput (batch 32):** 15.9 → 32.0 → **117.0 img/s (7.4× vs PyTorch, 3.7× vs ONNX FP32)**.
- PyTorch's throughput doesn't improve with batch size on this CPU (17.5 → 15.9 img/s); ONNX Runtime INT8 gains about 34% from batching (87.6 → 117.0).
- Minmax and percentile INT8 run at the same speed (within about 1–3%).
- PyTorch FP32 batch 32 has the widest p50→p95 gap (1,960 → 2,424 ms), which may be thermal or background noise; see the per-repeat values in benchmark.json.

### Phase 5b – Agreement with PyTorch FP32 on this laptop (305 sample + edge DICOMs, val thresholds)
- Sanity check on the serving CPU, not a selection (the samples are test-split + train edge images, enriched for hemorrhages).

| Variant | Mean abs prob diff (range over labels) | Max abs prob diff | Decision flips per label (of 305) |
|---|---|---|---|
| ONNX FP32 | 2.2e-8 – 1.0e-7 | 1.9e-6 | **0** for all labels |
| **INT8 percentile (final)** | 0.0049 – 0.0123 | 0.192 | any 1 · epidural 1 · IPH 5 · IVH 2 · SAH 3 · SDH 6 |
| INT8 minmax | 0.0053 – 0.0173 | 0.291 | any 6 · epidural 4 · IPH 3 · IVH 5 · SAH 10 · SDH 5 |

- **Percentile agrees with PyTorch better than minmax on this laptop too** (smaller differences and fewer flips: 18 vs 33 in total), consistent with the val-based choice.
- The sample set is small and hemorrhage-enriched, so these flip rates (0.3–2% per label) aren't comparable to the full-test flip rates.

---

## Phase 6 – Serving API (FastAPI + Docker, laptop)

### Phase 6 – Docker image
- Date: 2026-10-19
- Image: `brainct-api:0.1.0` (`serving/Dockerfile`; base `python:3.11.11-slim-bookworm`; dependencies = base + [serve] from `serving/requirements.txt`, exported from uv.lock; **no PyTorch**)
- Command: `docker build -f serving/Dockerfile -t brainct-api:0.1.0 .`, then `docker images brainct-api:0.1.0`
- **Size reported by `docker images`: 553 MB.** Layers (`docker history`): Python packages (`/opt/venv`) 235 MB · INT8 model 24 MB · libgomp1 3.9 MB · thresholds 16 kB · the rest is the Python slim base image.
- Largest packages: onnxruntime 68 MB, numpy 45 MB + numpy.libs 28 MB, pillow.libs 14 MB + PIL 7.1 MB, pydicom 15 MB, pip 17 MB, setuptools 6.7 MB, pydantic_core 5.1 MB, pydantic 4.3 MB.
- Serving uses numpy 2.4.6, pillow 12.3.0 and pydicom 3.0.2, the same versions as the laptop and cluster, which gave bit-identical preprocessing in Phase 2.
- pip + setuptools + pkg_resources (about 26 MB) are only needed to install packages; they could be removed from the runtime image.

### Phase 6 – End-to-end load test through the API, micro-batching on vs off
- Date: 2026-10-19
- Setup: container on the laptop (Core Ultra 7 155U, Docker Desktop), `BRAINCT_THREADS=8`; batching max_batch 8 / max_wait 5 ms; warmup on. **The load generator (`brainct.serving.loadtest`, async httpx) ran on the same laptop**, so it shares the CPU with the server.
- Requests: real sample DICOMs (`samples/dicom`); 20 warmup requests, then 300 per concurrency level. Latency is client-side, end to end (HTTP upload + DICOM preprocessing + inference + JSON). 0 errors in every run.
- Commands: `docker run --rm -d -p 8000:8000 --name brainct [-e BRAINCT_BATCHING=false] brainct-api:0.1.0`, then `uv run python -m brainct.serving.loadtest --label batching_{on,off}`

| Concurrent clients | Batching | p50 (ms) | p95 (ms) | Requests/s | Batch sizes used |
|---|---|---|---|---|---|
| 1 | off | **23.9** | **38.6** | 39.5 | all 1 |
| 1 | on | 28.9 | 40.0 | 33.1 | all 1 |
| 8 | off | 118.4 | 165.9 | 67.0 | all 1 |
| 8 | on | 120.3 | 179.0 | 62.9 | mostly 3–5; 40 of 300 requests in batches of 8 |
| 32 | off | 463.0 | 544.5 | 67.8 | all 1 |
| 32 | on | 452.9 | 580.8 | 68.5 | mostly 2–3 and 8 |

- **Single-request latency through the API (batching off, 1 client): p50 23.9 ms, p95 38.6 ms.** That's consistent with Phase 5b (INT8 model alone: p50 11.1 / p95 12.6 ms) plus preprocessing (about 12.6 ms, from the server's timing field) plus HTTP overhead.
- **Micro-batching gave no throughput gain here:** the maximum was about 68 req/s either way (32 clients: 68.5 on vs 67.8 off), and it added about 5 ms of p50 latency at 1 client, which is the 5 ms wait window.
- **Why (most likely): per-request work outside the model dominates on this machine.** DICOM decoding and preprocessing (about 12.6 ms of CPU per request) and HTTP handling are not batched, and they share 14 logical CPUs with ONNX Runtime's 8 threads, Docker Desktop's VM and the load generator itself. Batching speeds up only the inference part (Phase 5b: 88 → 117 img/s for the model alone). Not yet measured: a server-side breakdown under load.
- **Not a final verdict on batching:** on a dedicated server (GKE, Phase 9), with the load generator on another machine, inference is a larger share of the work and batching may help. Re-measure there.
- **Decision (owner, 2026-10-19): micro-batching is OFF by default** (`Settings.batching = False`, Dockerfile `BRAINCT_BATCHING=false`), based on the table above. It's still switchable with `BRAINCT_BATCHING=true`. The pip/setuptools trim was not done.

---

## Phase 7 – Tests and CI

### Phase 7 – The regression tests catch a deliberate preprocessing change
- Date: 2026-10-19 (laptop)
- Expectations generated with `uv run python scripts/make_regression_expected.py --synthetic --model --reason "first version (Phase 7)"`: `tests/regression/expected_synthetic.json` (4 synthetic cases, committed) and `artifacts/regression/expected_int8_samples.json` (50 sample DICOMs, real INT8 model; local only).
- Deliberate change: brain window center 40 → **41 HU** (`src/brainct/preprocessing/window.py`), then `uv run pytest tests/regression -q`:
  - **Both regression tests FAILED**, as intended:
    - `test_preprocessing_matches_stored_fingerprints`: `unsigned12_intercept_m1024_512: model_input changed` (sha256 differs; mean −0.900741 → −0.902529).
    - `test_int8_outputs_match_stored_expectations`: `ID_000aa2bce.dcm`, epidural probability 0.000811 → 0.000669 (|Δ| 1.4e-4 > tolerance 1e-5).
  - After `git checkout -- src/brainct/preprocessing/window.py`: **both passed** again.
- **A 1-HU shift in one window is detected by both the CI-safe synthetic test and the local real-model test.**

### Phase 7 – Test suite and coverage
- Date: 2026-10-19 (laptop, Windows, Python 3.11.17)
- Command: `uv run pytest --cov --cov-report=term-missing`
- Result: **420 passed** (64.9 s). **Coverage of `src/brainct`: 86%** (2,096 statements, 301 missed).
  - **100%:** serving/app.py, serving/model.py, serving/schemas.py, serving/settings.py, preprocessing/hu.py, preprocessing/window.py, training/metrics.py, training/model.py, optimization/quantize.py, data/zip_reader.py, constants.py.
  - Lowest: serving/loadtest.py 26% (a CLI tool, only its summary maths is tested), training/evaluate.py 72%, data/check_cache.py 73%, training/bench_loader.py 73%. The missed lines are mostly `main()` / CLI argument handling and W&B logging branches.
  - CI's coverage is lower, because the real-data and real-model tests are skipped there (no RSNA data or model); not recorded.
- CI (2026-10-19): both jobs green on the Phase 7 PR (lint + tests + coverage; Docker build + container smoke test with a stand-in model). Merged into main.

---

## Phase 8 – Deployment

### Phase 8 – Local Kubernetes rehearsal (Docker Desktop, free)
- Date: 2026-10-19
- Cluster: Docker Desktop Kubernetes v1.30.5 (one node, `docker-desktop`), on the laptop
- Chart: `infra/helm/brainct` with `infra/helm/values-local.yaml`: 2 replicas, each 2 CPU (request = limit), 1–2 GiB, `BRAINCT_THREADS=2`, batching off, LoadBalancer on localhost:8080, HPA off (no metrics-server, by the owner's choice). Image `brainct-api:0.1.0` (553 MB).
- Commands: `helm lint` / `helm template` OK → `helm install brainct infra/helm/brainct -f infra/helm/values-local.yaml`
- Result:
  - **Both pods Running and Ready (1/1) within 12 s** of install. `/ready` through the Service: ready, model version ef6cea9f5cb4, batching off.
  - Load test through the Kubernetes Service (`uv run python -m brainct.serving.loadtest --url http://localhost:8080 --concurrency 1 8 --requests 100 --label k8s_local`), 0 errors:

| Concurrent clients | p50 (ms) | p95 (ms) | Requests/s |
|---|---|---|---|
| 1 | 32.8 | 87.3 | 22.9 |
| 8 | 216.2 | 412.4 | 31.9 |

  - **Self-healing:** after `kubectl delete pod brainct-6765b797dc-4b6mc`, Kubernetes created a replacement (`brainct-6765b797dc-b4hhh`) that was **Running and Ready within 5 s**, while the other pod kept serving.
- Notes:
  - Slower than the single `docker run` container (Phase 6: 1 client p50 23.9 ms, about 68 req/s maximum). Here each pod has **2 CPUs and 2 ONNX Runtime threads** (4 CPUs in total across 2 pods), against **8 threads** in one container. Docker Desktop's load-balancer path also adds a hop. The load generator again shared the laptop's CPU.
  - This run checks the mechanism (probes, Service, self-healing), not performance. Performance gets measured on GKE in Phase 9.

### Phase 8 – GKE Autopilot deployment, session 1 (2026-10-19)
- Infrastructure: `terraform -chdir=infra/terraform apply` → Autopilot cluster `brainct` + Artifact Registry repo `brainct` in **us-east1** (project in a GCP free-trial account). Helm: `helm install brainct infra/helm/brainct -f infra/helm/values-gke.yaml --set image.repository=<registry path>` (1–4 pods via the HPA, each 2 vCPU / 2 GiB, batching off, LoadBalancer on port 80).
- Image in Artifact Registry: **143,045,202 bytes compressed** (553 MB uncompressed locally).
- **Issue found: `ErrImagePull` / `ImagePullBackOff` with `403 Forbidden`** when the node fetched a registry token. Cause: the nodes' service account (the Compute Engine default SA) had **no read permission on Artifact Registry**; newer projects don't grant that account broad roles automatically. **Fix:** `roles/artifactregistry.reader` on the `brainct` repository for `<project-number>-compute@developer.gserviceaccount.com`, applied with `gcloud` during the session. Must be added to Terraform so new clusters work.
- After the fix: **the pod went from created to Ready (1/1) in 23 s** (about 10 s image pull and start, then model load and warmup).
- Node: `gk3-brainct-nap-...` (Container-Optimized OS, amd64, kubelet v1.36.4-gke.1495002), created by Autopilot when the pod was scheduled. A second node (created during the pull retries) was being removed (`NotReady,SchedulingDisabled`). Machine type not recorded.
- **Public endpoint:** `/ready` from the laptop over the internet → ready, model version ef6cea9f5cb4, batching off.
- HPA: `cpu: 0%/70%`, 1 replica (min 1 / max 4). **Metrics are available on GKE**, unlike the local cluster.
- Smoke load test from the laptop over the internet (`--concurrency 1 --requests 50 --label gke_smoke`): **p50 116.3 ms, p95 168.3 ms, 8.1 req/s, 0 errors**, all batch size 1.
  - Much slower than local (23.9 ms) **because each request uploads a ~0.5 MB DICOM from the laptop's home internet connection to us-east1**: upload bandwidth and round-trip time dominate. The server-side `timing` in each response (not captured in this run) separates server time from network time. Phase 9 should measure from a machine close to the cluster.
- **Teardown:** `helm uninstall brainct`, then `terraform -chdir=infra/terraform destroy` → **"Destroy complete! Resources: 4 destroyed."** Cluster deletion took **5 min 55 s**. The two `google_project_service` entries were removed from state, but the APIs stay enabled (`disable_on_destroy = false`; no cost). `kubectl` was switched back to `docker-desktop`.
- Session cost: to read from Billing → Reports (charges appear with a delay). **Not yet recorded.**
