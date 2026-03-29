#!/usr/bin/env bash
#SBATCH --job-name=braidmod-gen-p5-L20
#SBATCH --partition=scavenge
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/nfs/roberts/project/pi_com36/com36/burau-experiments/braidmod"
PYTHON_PATH="$REPO_ROOT/.venv/bin/python"
OUT_PATH="$REPO_ROOT/data/burau_gnf_L20_p5_D96_N20000_corrected.json"
TMP_PATH="$OUT_PATH.tmp.$SLURM_JOB_ID"

mkdir -p "$REPO_ROOT/slurm_logs" "$REPO_ROOT/data"

cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi

echo "Starting dataset regeneration at $(date)"
echo "Host: $(hostname)"
echo "Python: $PYTHON_PATH"
echo "Output: $OUT_PATH"

"$PYTHON_PATH" generate_mod7_varlen_dataset.py \
  --output-path "$TMP_PATH" \
  --num-samples 20000 \
  --length-min 20 \
  --length-max 20 \
  --p 5 \
  --D 96 \
  --d-min 0 \
  --d-max 0 \
  --seed 42 \
  --progress-every 1000 \
  --heartbeat-secs 10

mv "$TMP_PATH" "$OUT_PATH"

echo "Finished at $(date)"
ls -lh "$OUT_PATH"
