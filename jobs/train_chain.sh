#!/bin/bash
# Submit N chained train.sh jobs for one run. Each job starts after the previous one ends
# (afterany), resumes from checkpoints/last.pt, and exits at once if the run is already DONE.
#
# Run on the login node (this script submits jobs; it is not a job itself):
#   bash ~/Brain_CT/jobs/train_chain.sh 3 --set run_id=resnet50_baseline
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: bash $0 <number_of_jobs> [--set key=value ...]" >&2
    exit 1
fi
N=$1
shift

prev=$(sbatch --parsable "$HOME/Brain_CT/jobs/train.sh" "$@")
echo "job 1/$N: $prev"
for i in $(seq 2 "$N"); do
    prev=$(sbatch --parsable --dependency=afterany:"$prev" "$HOME/Brain_CT/jobs/train.sh" "$@")
    echo "job $i/$N: $prev (after the previous one)"
done
