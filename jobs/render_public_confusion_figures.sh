#!/usr/bin/env bash
# Render the public model-confusion figures from tracked figure data.

#SBATCH --job-name=braidmod-public-figures
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_PATH="${PYTHON_PATH:-/home/com36/.conda/envs/burau_gpu/bin/python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/checkpoints/best_transformer/best_model.pt}"
SUITE_DIR="${SUITE_DIR:-$REPO_ROOT/figure_data/confusion_suite_tuned}"
SEARCH_JSON="${SEARCH_JSON:-$REPO_ROOT/figure_data/search/kernel_hits_len60.json}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/figures/generated}"

mkdir -p "$REPO_ROOT/slurm_logs" "$OUT_DIR"
cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint not found at $CHECKPOINT_PATH" >&2
  exit 1
fi
if [[ ! -d "$SUITE_DIR" ]]; then
  echo "Suite dir not found at $SUITE_DIR" >&2
  exit 1
fi
if [[ ! -f "$SEARCH_JSON" ]]; then
  echo "Search JSON not found at $SEARCH_JSON" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

for window in 5 7 10 13 15 20; do
  "$PYTHON_PATH" -u render_average_kernel_random_xent_overlay.py \
    --search-json "$SEARCH_JSON" \
    --checkpoint "$CHECKPOINT_PATH" \
    --suite-dir "$SUITE_DIR" \
    --out-png "$OUT_DIR/kernel_avg_first5_vs_random_avg_target_xent_avg${window}.png" \
    --device cuda \
    --mode avg5 \
    --window "$window" \
    --max-length 60 \
    --num-kernels 5
done
