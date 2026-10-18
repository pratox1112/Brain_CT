#!/bin/bash
#SBATCH --job-name=bench_loader
#SBATCH --account=eece7268.202610
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/scratch/%u/brainct/logs/bench_loader_%j.log

# Phase 3: measure the input pipeline alone (no model, no GPU) at several worker counts,
# to find whether storage reads or CPU windowing/augmentation limit training throughput.
set -euo pipefail

REPO=$HOME/Brain_CT

source /scratch/$USER/envs/brainct/bin/activate
cd "$REPO"

echo "START $(date -Is) on $(hostname)  cpus=${SLURM_CPUS_PER_TASK:-?}"
SECONDS=0

python -m brainct.training.bench_loader \
    --cache-dir "/scratch/$USER/brainct/preprocessed" \
    --labels "$HOME/brainct_outputs/splits/labels.parquet" \
    --out "$HOME/brainct_outputs/analysis/loader_bench.json" \
    "$@"

echo "END $(date -Is)  elapsed ${SECONDS}s"
