#!/usr/bin/env bash
# Pure projlen search with periodic raw-target-xent pruning checkpoints.

#SBATCH --job-name=braidmod-projlen-xprune-l60-b200k-u500k
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=01:45:00
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

OUT_JSON="$REPO_ROOT/artifacts/reservoir_search_projlen_xentprune_gpu_len60_b200k_u500k_boot5_t10_l10-25-40.json"
OUT_PNG="$REPO_ROOT/artifacts/reservoir_search_projlen_xentprune_gpu_len60_b200k_u500k_boot5_t10_l10-25-40_best_projlen.png"

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

echo "Starting projlen + periodic xent-prune search at $(date)"
echo "Host: $(hostname)"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "JSON out: $OUT_JSON"
echo "Plot out: $OUT_PNG"

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
  --xent-prune-levels 10,25,40 \
  --xent-prune-threshold 10 \
  --save-kernel-hits 5000 \
  --device cuda \
  --out-json "$OUT_JSON"

"$PYTHON_PATH" -u plot_search_best_projlen.py \
  --search-json "$OUT_JSON" \
  --out-png "$OUT_PNG" \
  --title "Best Projlen By Level: Projlen Search With Xent Prune"

echo "Finished at $(date)"
ls -lh "$OUT_JSON" "$OUT_PNG"
