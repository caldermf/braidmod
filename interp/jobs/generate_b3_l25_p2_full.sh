#!/usr/bin/env bash
# Generate one shard per array task for the exhaustive B_3 length-25 p=2 corpus.

#SBATCH --job-name=b3-l25-p2-gen
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --array=0-63%8
#SBATCH --time=02:00:00
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
OUT_DIR="${OUT_DIR:-$REPO_ROOT/interp/data/generated/b3_l25_p2_full}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.7.1-foss-2024a-CUDA-12.8.0}"
BATCH_SIZE="${BATCH_SIZE:-131072}"
SPOT_CHECK="${SPOT_CHECK:-16}"

mkdir -p "$REPO_ROOT/interp/slurm_logs" "$OUT_DIR"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/generate_b3_dataset.py \
  --out-dir "$OUT_DIR" \
  --length 25 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --num-shards 64 \
  --batch-size "$BATCH_SIZE" \
  --device cuda \
  --spot-check "$SPOT_CHECK"
