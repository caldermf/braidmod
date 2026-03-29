#!/usr/bin/env bash
# Run pure projlen search with exact projlen buckets, save many confirmed kernel hits,
# then overlay avg5 target-xent curves for all saved kernel elements.

#SBATCH --job-name=braidmod-kernelhits-projlen-l60-b200k-u500k
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/nfs/roberts/project/pi_com36/com36/burau-experiments/braidmod"
PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"
CHECKPOINT_PATH="$REPO_ROOT/artifacts/garside_mlp_L30to60_p5_D140_N200000_e20/best_model.pt"

OUT_JSON="$REPO_ROOT/artifacts/reservoir_search_projlen_exact_kernelhits_gpu_len60_b200k_u500k_boot5.json"
OUT_PNG="$REPO_ROOT/artifacts/reservoir_search_projlen_exact_kernelhits_gpu_len60_b200k_u500k_boot5_kernel_target_xent_avg5_overlay.png"
RANDOM_SUITE_DIR="$REPO_ROOT/artifacts/length54_confusion_suite_D140_e20"

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

echo "Starting pure-projlen exact-bucket kernel-hit search at $(date)"
echo "Host: $(hostname)"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "JSON out: $OUT_JSON"

"$PYTHON_PATH" -u reservoir_search_braidmod.py \
  --p 5 \
  --max-length 60 \
  --bucket-size 200000 \
  --use-best 500000 \
  --bootstrap-length 5 \
  --num-buckets 200 \
  --score-type projlen \
  --projlen-bucket-mode exact \
  --checkpoint "$CHECKPOINT_PATH" \
  --save-kernel-hits 5000 \
  --device cuda \
  --out-json "$OUT_JSON"

"$PYTHON_PATH" -u render_kernel_random_xent_overlay.py \
  --search-json "$OUT_JSON" \
  --checkpoint "$CHECKPOINT_PATH" \
  --suite-dir "$RANDOM_SUITE_DIR" \
  --out-png "$OUT_PNG" \
  --device cuda \
  --window 5 \
  --max-length 60

echo "Finished at $(date)"
ls -lh "$OUT_JSON" "$OUT_PNG"
