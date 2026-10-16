#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=eece7268.202610
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/scratch/%u/brainct/logs/train_%j.log

# Phase 3: train (or resume) one run. Arguments are passed to brainct.training.train, e.g.
#   sbatch ~/Brain_CT/jobs/train.sh --set run_id=speed_amp_on --set max_steps=500 --set validate=false
# For a full run that needs several 8 h jobs, use jobs/train_chain.sh instead.
# CONFIG=... overrides the config file (default: configs/resnet50_baseline.yaml).
set -euo pipefail

REPO=$HOME/Brain_CT
CONFIG=${CONFIG:-configs/resnet50_baseline.yaml}

source /scratch/$USER/envs/brainct/bin/activate
cd "$REPO"
export TORCH_HOME=/scratch/$USER/cache/torch   # pretrained weights download here

echo "START $(date -Is) on $(hostname)  config=$CONFIG  args: $*"
echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a (not a git checkout)')"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
SECONDS=0

python -m brainct.training.train --config "$CONFIG" "$@"

echo "END $(date -Is)  elapsed ${SECONDS}s"
