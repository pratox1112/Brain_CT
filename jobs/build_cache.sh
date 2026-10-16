#!/bin/bash
#SBATCH --job-name=build_cache
#SBATCH --account=eece7268.202610
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/scratch/%u/brainct/logs/build_cache_%j.log

# Phase 2, step 6: preprocess every training DICOM from the zip into the int16 HU cache.
# Safe to resubmit: finished chunks are skipped.
#
# Test run on the first 5,000 images, into a separate folder:
#   sbatch --export=ALL,OUT=/scratch/$USER/brainct/preprocessed_test \
#       ~/Brain_CT/jobs/build_cache.sh --limit 5000
set -euo pipefail

REPO=$HOME/Brain_CT
OUT=${OUT:-/scratch/$USER/brainct/preprocessed}
KEEP=$HOME/brainct_outputs/$(basename "$OUT")   # small outputs copied to home

source /scratch/$USER/envs/brainct/bin/activate
cd "$REPO"

echo "START $(date -Is) on $(hostname)  out=$OUT"
echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a (not a git checkout)')"
SECONDS=0

python -m brainct.data.build_cache \
    --zip "/scratch/$USER/rsna/rsna-intracranial-hemorrhage-detection.zip" \
    --split-images "$HOME/brainct_outputs/splits/split_images.parquet" \
    --out "$OUT" \
    --workers "${SLURM_CPUS_PER_TASK:-32}" \
    "$@"

mkdir -p "$KEEP"
cp "$OUT/hu256_index.parquet" "$OUT/cache_summary.json" "$OUT/cache_failures.csv" "$KEEP/"
du -sh --apparent-size "$OUT/hu256.npy"
du -sh "$OUT/hu256.npy"

echo "END $(date -Is)  elapsed ${SECONDS}s"
