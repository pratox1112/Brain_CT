#!/bin/bash
#SBATCH --job-name=check_cache
#SBATCH --account=eece7268.202610
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/%u/brainct/logs/check_cache_%j.log

# Phase 2, step 7: check 1,000 random cache rows against fresh preprocessing of the zip bytes,
# and write the expected outputs for the 305 laptop sample DICOMs.
# Use OUT=... the same way as for build_cache.sh to check the test cache.
set -euo pipefail

REPO=$HOME/Brain_CT
OUT=${OUT:-/scratch/$USER/brainct/preprocessed}

source /scratch/$USER/envs/brainct/bin/activate
cd "$REPO"

echo "START $(date -Is) on $(hostname)  cache=$OUT"
SECONDS=0
status=0

python -m brainct.data.check_cache verify \
    --cache-dir "$OUT" \
    --zip "/scratch/$USER/rsna/rsna-intracranial-hemorrhage-detection.zip" \
    --sample-dir "$HOME/brainct_outputs/samples/dicom" \
    --sample-dir "$HOME/brainct_outputs/samples_edge/dicom" \
    --expected-out "$HOME/brainct_outputs/samples/expected_hu256.npz" \
    "$@" || status=$?

cp "$OUT/check_summary.json" "$HOME/brainct_outputs/$(basename "$OUT")/"
echo "END $(date -Is)  elapsed ${SECONDS}s  status=$status (0 = everything matched)"
exit $status
