#!/bin/bash
#SBATCH --job-name=extract_headers
#SBATCH --account=eece7268.202610
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/scratch/%u/brainct/logs/extract_headers_%j.log

# Phase 1, task 2: read every RSNA training DICOM header straight from the zip (no unzip)
# and write dicom_headers.parquet, header_failures.csv and header_summary.json.
#
# The log folder must exist before submitting:  mkdir -p /scratch/$USER/brainct/logs
# Extra arguments are passed through, e.g. a quick test run:
#   sbatch ~/Brain_CT/jobs/extract_headers.sh --limit 2000
set -euo pipefail

REPO=$HOME/Brain_CT
ZIP=/scratch/$USER/rsna/rsna-intracranial-hemorrhage-detection.zip
OUT=$HOME/brainct_outputs/metadata

source /scratch/$USER/envs/brainct/bin/activate
mkdir -p "$OUT"
cd "$REPO"

echo "START $(date -Is) on $(hostname)"
echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a (not a git checkout)')"
echo "python: $(python --version)  pydicom: $(python -c 'import pydicom; print(pydicom.__version__)')"
SECONDS=0

python -m brainct.data.headers \
    --zip "$ZIP" \
    --out "$OUT" \
    --workers "${SLURM_CPUS_PER_TASK:-16}" \
    "$@"

echo "END $(date -Is)  elapsed ${SECONDS}s"
