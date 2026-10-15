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
