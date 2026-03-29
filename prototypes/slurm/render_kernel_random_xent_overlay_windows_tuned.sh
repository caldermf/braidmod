#!/usr/bin/env bash
# Render extra smoothed kernel-vs-random target-xent overlays for the tuned
# factor-only transformer using the saved tuned confusion suite.

#SBATCH --job-name=braidmod-xent-windows
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

REPO_ROOT="/nfs/roberts/project/pi_com36/com36/burau-experiments/braidmod"
PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"
CHECKPOINT_PATH="$REPO_ROOT/artifacts/garside_transformer_tuned_factoronly_base_noema_L30to60_p5_D140_N200000_e30/best_model.pt"
SUITE_DIR="$REPO_ROOT/artifacts/length54_confusion_suite_transformer_tuned_factoronly_base_noema"
SEARCH_JSON="$REPO_ROOT/artifacts/reservoir_search_projlen_exact_kernelhits_gpu_len60_b200k_u500k_boot5.json"

mkdir -p "$REPO_ROOT/slurm_logs" "$SUITE_DIR"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1

echo "Starting windowed overlay render at $(date)"
echo "Host: $(hostname)"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Suite dir: $SUITE_DIR"

for window in 7 10 13 15 20; do
  "$PYTHON_PATH" -u render_kernel_random_xent_overlay.py \
    --search-json "$SEARCH_JSON" \
    --checkpoint "$CHECKPOINT_PATH" \
    --suite-dir "$SUITE_DIR" \
    --out-png "$SUITE_DIR/kernel_hits_vs_random_target_xent_avg${window}_overlay.png" \
    --device cuda \
    --mode avg5 \
    --window "$window" \
    --max-length 60
done

echo "Finished at $(date)"
ls -lh "$SUITE_DIR"/kernel_hits_vs_random_target_xent_avg*_overlay.png
