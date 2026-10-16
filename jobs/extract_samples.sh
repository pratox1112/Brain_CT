#!/bin/bash
#SBATCH --job-name=extract_samples
#SBATCH --account=eece7268.202610
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/%u/brainct/logs/extract_samples_%j.log

# Phase 1, task 8: pick ~300 test-split DICOMs (normals, every subtype, unusual header values)
# and extract only those files from the zip, for use on the laptop.
# Extra arguments are passed through, e.g. --n-per-subtype 50
set -euo pipefail

REPO=$HOME/Brain_CT
SPLITS=$HOME/brainct_outputs/splits
OUT=$HOME/brainct_outputs/samples

source /scratch/$USER/envs/brainct/bin/activate
cd "$REPO"

echo "START $(date -Is) on $(hostname)"
echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a (not a git checkout)')"
SECONDS=0

python -m brainct.data.samples \
    --split-images "$SPLITS/split_images.parquet" \
    --labels "$SPLITS/labels.parquet" \
    --headers "$HOME/brainct_outputs/metadata/dicom_headers.parquet" \
    --zip "/scratch/$USER/rsna/rsna-intracranial-hemorrhage-detection.zip" \
    --out "$OUT" \
    "$@"

echo "END $(date -Is)  elapsed ${SECONDS}s"
