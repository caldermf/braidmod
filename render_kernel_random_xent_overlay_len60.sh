#!/usr/bin/env bash
# Render the combined 6-kernel-hit plus 5-random avg5 target-xent overlay on GPU.

#SBATCH --job-name=braidmod-render-kernel-random-overlay
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/nfs/roberts/project/pi_com36/com36/burau-experiments/braidmod"
PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"
SEARCH_JSON="$REPO_ROOT/artifacts/reservoir_search_projlen_exact_kernelhits_gpu_len60_b200k_u500k_boot5.json"
CHECKPOINT_PATH="$REPO_ROOT/artifacts/garside_mlp_L30to60_p5_D140_N200000_e20/best_model.pt"
SUITE_DIR="$REPO_ROOT/artifacts/length54_confusion_suite_D140_e20"
OUT_PNG="$REPO_ROOT/artifacts/reservoir_search_projlen_exact_kernelhits_gpu_len60_b200k_u500k_boot5_kernel_target_xent_avg5_overlay.png"

mkdir -p "$REPO_ROOT/slurm_logs" "$REPO_ROOT/artifacts"
cd "$REPO_ROOT"

"$PYTHON_PATH" -u render_kernel_random_xent_overlay.py \
  --search-json "$SEARCH_JSON" \
  --checkpoint "$CHECKPOINT_PATH" \
  --suite-dir "$SUITE_DIR" \
  --out-png "$OUT_PNG" \
  --device cuda \
  --window 5 \
  --max-length 60
