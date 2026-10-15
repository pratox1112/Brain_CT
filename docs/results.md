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
