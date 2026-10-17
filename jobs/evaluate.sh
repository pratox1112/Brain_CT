#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --account=eece7268.202610
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/scratch/%u/brainct/logs/evaluate_%j.log

# Phase 3: evaluate a run's best checkpoint on val or test (test only once per run).
#   sbatch ~/Brain_CT/jobs/evaluate.sh resnet50_baseline val
#   sbatch ~/Brain_CT/jobs/evaluate.sh resnet50_baseline test
set -euo pipefail

RUN_ID=$1
SPLIT=$2
shift 2
REPO=$HOME/Brain_CT
RUN_DIR=/scratch/$USER/brainct/runs/$RUN_ID

source /scratch/$USER/envs/brainct/bin/activate
cd "$REPO"

echo "START $(date -Is) on $(hostname)  run=$RUN_ID  split=$SPLIT"
SECONDS=0

python -m brainct.training.evaluate --run-dir "$RUN_DIR" --split "$SPLIT" "$@"

mkdir -p "$HOME/brainct_outputs/runs/$RUN_ID"
cp "$RUN_DIR/eval/metrics_$SPLIT.json" "$HOME/brainct_outputs/runs/$RUN_ID/"
echo "END $(date -Is)  elapsed ${SECONDS}s"
