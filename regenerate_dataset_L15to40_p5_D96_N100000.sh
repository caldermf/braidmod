#!/usr/bin/env bash
#SBATCH --job-name=braidmod-gen-p5-L15to40-100k
#SBATCH --partition=scavenge
#SBATCH --time=01:00:00
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
PYTHON_PATH="$REPO_ROOT/.venv/bin/python"
OUT_PATH="$REPO_ROOT/data/burau_gnf_L15to40_p5_D96_N100000_corrected.json"
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

echo "Starting dataset regeneration at $(date)"
echo "Host: $(hostname)"
echo "Python: $PYTHON_PATH"
echo "Output: $OUT_PATH"
echo "Temporary output: $TMP_PATH"

"$PYTHON_PATH" generate_mod7_varlen_dataset.py \
  --output-path "$TMP_PATH" \
  --num-samples 100000 \
  --length-min 15 \
  --length-max 40 \
  --p 5 \
  --D 96 \
  --d-min 0 \
  --d-max 0 \
  --seed 42 \
  --progress-every 1000 \
  --heartbeat-secs 10

mv "$TMP_PATH" "$OUT_PATH"
trap - EXIT

echo "Finished at $(date)"
ls -lh "$OUT_PATH"
