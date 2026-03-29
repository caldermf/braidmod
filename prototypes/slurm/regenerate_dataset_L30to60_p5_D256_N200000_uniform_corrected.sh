#!/usr/bin/env bash
# Regenerate the L30..60 p=5 D=256 uniform dataset using the corrected Burau normalization.

#SBATCH --job-name=braidmod-gen-p5-L30to60-200k-D256-corrected
#SBATCH --partition=scavenge
#SBATCH --time=01:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/nfs/roberts/project/pi_com36/com36/burau-experiments/braidmod"
PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"
OUT_PATH="$REPO_ROOT/data/burau_gnf_L30to60_p5_D256_N200000_uniform.json"
TMP_PATH="$OUT_PATH.tmp.${SLURM_JOB_ID:-manual}"

mkdir -p "$REPO_ROOT/slurm_logs" "$REPO_ROOT/data"

cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi

cleanup() {
  if [[ -f "$TMP_PATH" ]]; then
    echo "Leaving partial dataset at $TMP_PATH"
  fi
}
trap cleanup EXIT

echo "Starting corrected dataset regeneration at $(date)"
echo "Host: $(hostname)"
echo "Python: $PYTHON_PATH"
echo "Output: $OUT_PATH"
echo "Temporary output: $TMP_PATH"

"$PYTHON_PATH" generate_mod7_varlen_dataset.py \
  --output-path "$TMP_PATH" \
  --num-samples 200000 \
  --length-min 30 \
  --length-max 60 \
  --length-mode balanced_uniform \
  --p 5 \
  --D 256 \
  --d-min 0 \
  --d-max 0 \
  --seed 42 \
  --progress-every 1000 \
  --heartbeat-secs 10

mv "$TMP_PATH" "$OUT_PATH"
trap - EXIT

echo "Finished at $(date)"
ls -lh "$OUT_PATH"
