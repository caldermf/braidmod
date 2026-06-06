#!/usr/bin/env bash
# Train a deliberately constrained B_4 Z[v] sign-token boundary-only transformer.

#SBATCH --job-name=b4-zsign-bdry-small
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=02:00:00
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
OUT_DIR="${OUT_DIR:-$REPO_ROOT/interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.7.1-foss-2024a-CUDA-12.8.0}"

mkdir -p "$REPO_ROOT/interp/slurm_logs" "$OUT_DIR"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/train_b4_z_sign_transformer.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --num-shards "${NUM_SHARDS:-64}" \
  --batch-size "${BATCH_SIZE:-2048}" \
  --epochs "${EPOCHS:-12}" \
  --train-examples-per-epoch "${TRAIN_EXAMPLES_PER_EPOCH:-1048576}" \
  --eval-examples "${EVAL_EXAMPLES:-262144}" \
  --test-examples "${TEST_EXAMPLES:-524288}" \
  --lr "${LR:-3e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-2}" \
  --grad-clip "${GRAD_CLIP:-1.0}" \
  --seed "${SEED:-42}" \
  --device "${DEVICE:-cuda}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --d-model "${D_MODEL:-96}" \
  --num-layers "${NUM_LAYERS:-2}" \
  --num-heads "${NUM_HEADS:-4}" \
  --ffn-mult "${FFN_MULT:-4}" \
  --dropout "${DROPOUT:-0.0}" \
  --boundary-only-radius "${BOUNDARY_ONLY_RADIUS:-8}"
