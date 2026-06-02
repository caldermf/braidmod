#!/usr/bin/env bash
# Train a simple one-hidden-layer MLP baseline on absolute-degree B_3 tokens.

#SBATCH --job-name=b3-l25-p2-mlp1
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --output=interp/slurm_logs/%x-%j.out
#SBATCH --error=interp/slurm_logs/%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi

DATA_DIR="${DATA_DIR:-$REPO_ROOT/interp/data/generated/b3_l25_p2_full}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/interp/artifacts/b3_l25_p2_mlp1_abs_h128}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.7.1-foss-2024a-CUDA-12.8.0}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
EPOCHS="${EPOCHS:-8}"
TRAIN_EXAMPLES_PER_EPOCH="${TRAIN_EXAMPLES_PER_EPOCH:-16777216}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-1048576}"

mkdir -p "$REPO_ROOT/interp/slurm_logs" "$OUT_DIR"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/train_b3_mlp.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --length 25 \
  --num-shards 64 \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --train-examples-per-epoch "$TRAIN_EXAMPLES_PER_EPOCH" \
  --eval-examples "$EVAL_EXAMPLES" \
  --test-examples "$EVAL_EXAMPLES" \
  --lr 3e-4 \
  --weight-decay 1e-2 \
  --grad-clip 1.0 \
  --seed 42 \
  --device cuda \
  --hidden-dim 128 \
  --num-hidden-layers 1 \
  --dropout 0.0
