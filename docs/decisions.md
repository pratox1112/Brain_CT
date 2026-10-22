# Design decisions

One entry per major choice: what was decided, why, and what was rejected. Numbers are from
[`results.md`](results.md).

## Data and evaluation

**Original DICOMs, read straight from the zip.** The 2025 project used JPG renders, which
have no patient IDs and fixed windowing. The RSNA DICOMs carry patient IDs and raw pixel
values, so the split can be patient-level and the windows can be chosen. Reading members
directly from the 182 GB zip avoids unpacking 752,803 files on shared scratch.

**Patient-level 80/10/10 split (seed 2026).** Slices from the same patient are highly
correlated; an image-level split would leak patients across train and test and inflate the
scores. Rejected: an image-level split, and reusing Kaggle's split. Cost: the numbers aren't
comparable to the leaderboard.

**Fix the mislabelled 12-bit images before anything else.** 123 images are stored as
unsigned 12-bit with intercept 0/1, but are really signed 12-bit two's complement: read
as-is, a brain scan becomes mostly "bone". This was found with pixel histograms per header
group before writing the fix, and the fix is applied only to that group. Rejected: clipping
or ignoring the outliers, which would have kept wrong pixels silently.

**Thresholds on val, test evaluated once.** Per-label thresholds maximise F1 on val and are
applied unchanged to test; confidence intervals come from a patient-level bootstrap. Average
precision is reported next to ROC AUC because, for rare epidural bleeds (0.45%), AUC alone
hides the false-alarm rate.

**fp32 for evaluation, bf16 for training.** bf16 inference put probabilities on a coarse
grid, and two labels' thresholds came out bit-identical. Evaluation therefore forces fp32;
training keeps bf16 AMP for speed.

## Training

**One shared preprocessing function and an HU cache.** `preprocess_dicom` produces int16 HU
at 256×256, both for the cache and in the API; the three windows are computed on the fly.
A cache of HU values (92 GiB) keeps windowing a training-time choice and is checked
bit-for-bit against fresh preprocessing. Rejected: caching windowed float images (3× larger,
windows frozen) and decoding DICOMs during training (CPU-bound).

**Direct reads instead of memmap for the cache.** Random access through `np.memmap` on
network storage was the training bottleneck. Reading each sample with positioned reads, and
a file handle opened per worker process, removed it. A shared handle inherited across
`fork` later caused corrupted reads in the quantization pipeline; that bug is why handles
are opened per process.

**ResNet-50 at 256 px, 5 epochs, as the final model.** A standard, well-understood baseline
that trains in 45 minutes on one A100 and exports cleanly to ONNX. Val macro AUC was still
rising slowly at epoch 5 (+0.001 per epoch); the owner chose to keep this run rather than
tune on val further. Rejected for now: EfficientNet, larger inputs, slice-context models.

**Chained, resumable Slurm jobs.** The GPU partition has an 8-hour limit, so training saves
full state (model, optimizer, scheduler, RNG, data position) and resumes exactly; a forced
resume test confirmed it.

## Optimization and serving

**Static INT8 quantization (QDQ, percentile calibration).** It gave a 3.9× smaller model
and 4.9× lower p95 latency than PyTorch on the laptop CPU. Calibration method was chosen on
val. Accuracy turned out CPU-dependent (−0.0013 macro AUC with VNNI, −0.0073 without); the
owner chose not to run more variants (e.g. `reduce_range`) and to document it instead.
ONNX FP32 is kept as the exact-accuracy option (Δ AUC −0.000002, 2.7× slower than INT8).

**CPU serving without PyTorch.** The API image installs ONNX Runtime and the preprocessing
code only (553 MB image, 143 MB compressed in the registry), and INT8 on CPU already gives
12.6 ms p95 per image for the model. GPU serving was not needed for this project's load.

**Micro-batching: off by default, on for GKE.** On the laptop it gave no throughput gain and
added about 5 ms. On a 2-vCPU GKE pod under overload it kept 14.7 vs 9.5 req/s (+55%), so
the GKE values turn it on.

## Deployment and monitoring

**GKE Autopilot, Terraform and Helm.** Autopilot bills per pod request and has no nodes to
manage; one cluster's management fee is covered by the free tier. Terraform makes the
cluster disposable (`destroy` after every session). Helm keeps one chart for Docker Desktop
and GKE. The node service account's Artifact Registry permission is in Terraform after a
403 on the first deployment.

**HPA on CPU (1–4 pods, 70%).** Inference is CPU-bound, so CPU utilisation tracks load. It
scaled 1 → 3 pods during the load test; new pods arrived minutes after each step began.

**Load generated inside the cluster.** The first GKE test, from a home connection, was
bounded by uploading 0.5 MB DICOMs (8 req/s). Running Locust as a Job next to the API
measures the service itself. A per-request log gives exact per-step percentiles; the
"degraded" rule (p95 above 2× the 1-user value, or over 1% errors) was fixed before
measuring.

**Prometheus and Grafana as plain manifests.** A small Prometheus that scrapes only
annotated API pods, and Grafana with the dashboard provisioned from
`monitoring/grafana/brainct.json`. The API reports its own CPU and memory, so no node
exporter or kube-state-metrics is needed. Rejected: the community Helm charts, which add
components that are awkward on Autopilot.

**Not done: drift monitoring (Evidently).** A reference built from training-split statistics
would be RSNA-derived data in the repository. Listed as a next step.
