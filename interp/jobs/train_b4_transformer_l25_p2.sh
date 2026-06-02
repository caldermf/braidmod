#!/usr/bin/env bash
# Train the first B_4 absolute-degree transformer for descent-set prediction.

#SBATCH --job-name=b4-l25-p2-xfmr3
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=interp/slurm_logs/%x-%j.out
#SBATCH --error=interp/slurm_logs/%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi

DATA_DIR="${DATA_DIR:-$REPO_ROOT/interp/data/generated/b4_l25_p2_n16777216}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/interp/artifacts/b4_l25_p2_xfmr3_abs}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.7.1-foss-2024a-CUDA-12.8.0}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
EPOCHS="${EPOCHS:-32}"
TRAIN_EXAMPLES_PER_EPOCH="${TRAIN_EXAMPLES_PER_EPOCH:-1048576}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-262144}"
TEST_EXAMPLES="${TEST_EXAMPLES:-524288}"
SEED="${SEED:-42}"
D_MODEL="${D_MODEL:-192}"
NUM_LAYERS="${NUM_LAYERS:-3}"
NUM_HEADS="${NUM_HEADS:-6}"
FFN_MULT="${FFN_MULT:-4}"
DROPOUT="${DROPOUT:-0.0}"

mkdir -p "$REPO_ROOT/interp/slurm_logs" "$OUT_DIR"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/train_b4_transformer.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --num-shards 64 \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --train-examples-per-epoch "$TRAIN_EXAMPLES_PER_EPOCH" \
  --eval-examples "$EVAL_EXAMPLES" \
  --test-examples "$TEST_EXAMPLES" \
  --resume \
  --lr 3e-4 \
  --weight-decay 1e-2 \
  --grad-clip 1.0 \
  --seed "$SEED" \
  --device cuda \
  --num-workers 0 \
  --d-model "$D_MODEL" \
  --num-layers "$NUM_LAYERS" \
  --num-heads "$NUM_HEADS" \
  --ffn-mult "$FFN_MULT" \
  --dropout "$DROPOUT"
