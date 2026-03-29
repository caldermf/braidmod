#!/usr/bin/env bash
# Run the public search configuration that combines projlen with model
# confusion through frontier scoring.

#SBATCH --job-name=braidmod-public-search
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_PATH="${PYTHON_PATH:-/home/com36/.conda/envs/burau_gpu/bin/python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/checkpoints/best_transformer/best_model.pt}"
OUT_JSON="${OUT_JSON:-$REPO_ROOT/artifacts/public_frontier_search.json}"

mkdir -p "$REPO_ROOT/slurm_logs" "$REPO_ROOT/artifacts"
cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint not found at $CHECKPOINT_PATH" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

"$PYTHON_PATH" -u reservoir_search_braidmod.py \
  --p 5 \
  --max-length 60 \
  --bucket-size 100000 \
  --use-best 300000 \
  --bootstrap-length 5 \
  --num-buckets 100 \
  --score-type frontier_target_xent \
  --checkpoint "$CHECKPOINT_PATH" \
  --device cuda \
  --out-json "$OUT_JSON"
