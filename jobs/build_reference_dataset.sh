#!/usr/bin/env bash
# Build the main public reference dataset: p=5, lengths 30..60, depth 140.

#SBATCH --job-name=braidmod-build-dataset
#SBATCH --partition=scavenge
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_PATH="${PYTHON_PATH:-$REPO_ROOT/.venv/bin/python}"
OUT_PATH="${OUT_PATH:-$REPO_ROOT/data/generated/burau_gnf_L30to60_p5_D140_N200000_uniform_corrected.json}"

mkdir -p "$REPO_ROOT/slurm_logs" "$(dirname "$OUT_PATH")"
cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

"$PYTHON_PATH" -u generate_dataset.py \
  --output-path "$OUT_PATH" \
  --num-samples 200000 \
  --length-min 30 \
  --length-max 60 \
  --p 5 \
  --D 140 \
  --seed 42 \
  --progress-every 1000
