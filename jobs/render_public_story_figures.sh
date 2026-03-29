#!/usr/bin/env bash
# Render the curated public figure set used by the README.

#SBATCH --job-name=braidmod-public-story
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_PATH="${PYTHON_PATH:-/home/com36/.conda/envs/burau_gpu/bin/python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/checkpoints/best_transformer/best_model.pt}"
SUITE_DIR="${SUITE_DIR:-$REPO_ROOT/figure_data/confusion_suite_tuned}"
SEARCH_JSON="${SEARCH_JSON:-$REPO_ROOT/figure_data/search/kernel_hits_len60.json}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/figures}"

mkdir -p "$REPO_ROOT/slurm_logs" "$OUT_DIR"
cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

"$PYTHON_PATH" plot_public_training_story.py \
  --mlp-log "$REPO_ROOT/checkpoints/original_mlp/train.log" \
  --transformer-log "$REPO_ROOT/checkpoints/best_transformer/train.log" \
  --out "$OUT_DIR/mlp_and_transformer_training_curves.png"

"$PYTHON_PATH" render_public_gwy_case_study.py \
  --suite-dir "$SUITE_DIR" \
  --out-dir "$OUT_DIR" \
  --window 5

"$PYTHON_PATH" render_average_kernel_random_xent_overlay.py \
  --search-json "$SEARCH_JSON" \
  --checkpoint "$CHECKPOINT_PATH" \
  --suite-dir "$SUITE_DIR" \
  --out-png "$OUT_DIR/known_kernel_elements_vs_random_target_cross_entropy_running_average_15_steps.png" \
  --device cuda \
  --mode avg5 \
  --window 15 \
  --max-length 60 \
  --num-kernels 5

"$PYTHON_PATH" render_kernel_random_xent_overlay.py \
  --search-json "$SEARCH_JSON" \
  --checkpoint "$CHECKPOINT_PATH" \
  --suite-dir "$SUITE_DIR" \
  --out-png "$OUT_DIR/individual_known_kernel_elements_vs_random_target_cross_entropy_running_average_15_steps.png" \
  --device cuda \
  --mode avg5 \
  --window 15 \
  --max-length 60
