#!/usr/bin/env bash
# Generate a pilot random B_4 length-25 p=2 corpus for validation.

#SBATCH --job-name=b4-l25-p2-pilot
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --array=0-3%4
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --output=interp/slurm_logs/%x-%A_%a.out
#SBATCH --error=interp/slurm_logs/%x-%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi

OUT_DIR="${OUT_DIR:-$REPO_ROOT/interp/data/generated/b4_l25_p2_pilot_n262144}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.7.1-foss-2024a-CUDA-12.8.0}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
SPOT_CHECK="${SPOT_CHECK:-8}"

mkdir -p "$REPO_ROOT/interp/slurm_logs" "$OUT_DIR"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/generate_b4_dataset.py \
  --out-dir "$OUT_DIR" \
  --length 25 \
  --num-samples 262144 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --num-shards 4 \
  --batch-size "$BATCH_SIZE" \
  --device cuda \
  --spot-check "$SPOT_CHECK" \
  --overwrite
