#!/usr/bin/env bash
# Run pure projlen reservoir search with exact projlen buckets at length 100.

#SBATCH --job-name=braidmod-search-projlen-l100-b200k-u500k-nb200
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/nfs/roberts/project/pi_com36/com36/burau-experiments/braidmod"
PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"
OUT_JSON="$REPO_ROOT/artifacts/reservoir_search_projlen_exact_gpu_len100_b200k_u500k_nb200_boot5.json"
OUT_PNG="$REPO_ROOT/artifacts/reservoir_search_projlen_exact_gpu_len100_b200k_u500k_nb200_boot5_best_projlen.png"

mkdir -p "$REPO_ROOT/slurm_logs" "$REPO_ROOT/artifacts"
cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

echo "Starting pure projlen exact-bucket search at $(date)"
echo "Host: $(hostname)"
echo "JSON out: $OUT_JSON"

"$PYTHON_PATH" -u reservoir_search_braidmod.py \
  --p 5 \
  --max-length 100 \
  --bucket-size 200000 \
  --use-best 500000 \
  --bootstrap-length 5 \
  --num-buckets 200 \
  --score-type projlen \
  --projlen-bucket-mode exact \
  --save-kernel-hits 5000 \
  --device cuda \
  --out-json "$OUT_JSON"

"$PYTHON_PATH" -u plot_search_best_projlen.py \
  --search-json "$OUT_JSON" \
  --out-png "$OUT_PNG" \
  --title "Best Projlen by Length (pure projlen exact-bucket search)"

echo "Finished at $(date)"
ls -lh "$OUT_JSON" "$OUT_PNG"
