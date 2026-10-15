#!/bin/bash
#SBATCH --job-name=rsna_dl
#SBATCH --account=eece7268.202610
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/scratch/%u/rsna/download_%j.log

# Downloads the full RSNA Intracranial Hemorrhage dataset (single zip, ~181 GB)
# into /scratch/$USER/rsna. Does not unzip.
set -euo pipefail

DEST=/scratch/$USER/rsna
ZIP=$DEST/rsna-intracranial-hemorrhage-detection.zip

source /scratch/$USER/envs/brainct/bin/activate
mkdir -p "$DEST"
cd "$DEST"

echo "START $(date -Is) on $(hostname)"
SECONDS=0

kaggle competitions download rsna-intracranial-hemorrhage-detection -p "$DEST"

echo "END $(date -Is)  elapsed ${SECONDS}s"
ls -lh "$ZIP"
