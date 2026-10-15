#!/bin/bash
#SBATCH --job-name=make_split
#SBATCH --account=eece7268.202610
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/%u/brainct/logs/make_split_%j.log

# Phase 1, tasks 4-7: build the per-image label table, split 80/10/10 by patient,
# check that no patient / study / series spans splits, and write the split report.
# Extra arguments are passed through, e.g. --seed 2026
set -euo pipefail

REPO=$HOME/Brain_CT
LABELS_CSV=/scratch/$USER/rsna/stage_2_train.csv
HEADERS=$HOME/brainct_outputs/metadata/dicom_headers.parquet
OUT=$HOME/brainct_outputs/splits

source /scratch/$USER/envs/brainct/bin/activate
cd "$REPO"

echo "START $(date -Is) on $(hostname)"
echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a (not a git checkout)')"
SECONDS=0

python -m brainct.data.split \
    --labels-csv "$LABELS_CSV" \
    --headers "$HEADERS" \
    --out "$OUT" \
    "$@"

echo "END $(date -Is)  elapsed ${SECONDS}s"
