#!/bin/bash
#SBATCH --job-name=optimize
#SBATCH --account=eece7268.202610
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/scratch/%u/brainct/logs/optimize_%j.log

# Phase 5a: ONNX export + INT8 quantization of the final model, and its accuracy cost on CPU.
#   sbatch ~/Brain_CT/jobs/optimize.sh
#   sbatch ~/Brain_CT/jobs/optimize.sh --methods minmax percentile   # fewer calibration methods
set -euo pipefail

REPO=$HOME/Brain_CT
RUN_ID=${RUN_ID:-resnet50_baseline}
RUN_DIR=/scratch/$USER/brainct/runs/$RUN_ID
OUT=$HOME/brainct_outputs/final/$RUN_ID/onnx
CPUS=${SLURM_CPUS_PER_TASK:-32}

source /scratch/$USER/envs/brainct/bin/activate
cd "$REPO"

echo "START $(date -Is) on $(hostname)  run=$RUN_ID  cpus=$CPUS"
lscpu | grep "Model name"
# INT8 (u8s8) accuracy can differ on CPUs without VNNI, so record whether this one has it.
grep -o -m1 -E "avx512_vnni|avx_vnni" /proc/cpuinfo || echo "no VNNI instructions"
SECONDS=0

# 8 DataLoader workers feed the model; the remaining CPUs run ONNX Runtime.
python -m brainct.optimization.evaluate_onnx \
    --run-dir "$RUN_DIR" \
    --out "$OUT" \
    --workers 8 \
    --threads $((CPUS - 8)) \
    "$@"

echo "END $(date -Is)  elapsed ${SECONDS}s"
