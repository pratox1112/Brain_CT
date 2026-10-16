#!/bin/bash
#SBATCH --job-name=inspect_pixels
#SBATCH --account=eece7268.202610
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/%u/brainct/logs/inspect_pixels_%j.log

# Phase 2, step 2: measure pixel / HU statistics per pixel-format group, and extract
# edge-case DICOMs from the train split for preprocessing tests.
# Extra arguments are passed through, e.g. --n-per-group 50
set -euo pipefail

REPO=$HOME/Brain_CT

source /scratch/$USER/envs/brainct/bin/activate
cd "$REPO"

echo "START $(date -Is) on $(hostname)"
echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a (not a git checkout)')"
SECONDS=0

python -m brainct.preprocessing.inspect_pixels \
    --headers "$HOME/brainct_outputs/metadata/dicom_headers.parquet" \
    --split-images "$HOME/brainct_outputs/splits/split_images.parquet" \
    --zip "/scratch/$USER/rsna/rsna-intracranial-hemorrhage-detection.zip" \
    --stats-out "$HOME/brainct_outputs/analysis" \
    --edge-out "$HOME/brainct_outputs/samples_edge" \
    "$@"

echo "END $(date -Is)  elapsed ${SECONDS}s"
