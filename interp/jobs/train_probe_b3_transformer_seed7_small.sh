#!/usr/bin/env bash
# Robustness replicate: train a smaller seed-7 B_3 transformer and probe late CLS.

#SBATCH --job-name=b3-xfmr-s7-probe
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=00:45:00
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

PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.7.1-foss-2024a-CUDA-12.8.0}"
OUT_DIR="$REPO_ROOT/interp/artifacts/b3_l25_p2_xfmr2_abs_seed7_small"

mkdir -p "$REPO_ROOT/interp/slurm_logs" "$OUT_DIR"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/train_b3_transformer.py \
  --data-dir "$REPO_ROOT/interp/data/generated/b3_l25_p2_full" \
  --out-dir "$OUT_DIR" \
  --length 25 \
  --num-shards 64 \
  --batch-size 4096 \
  --epochs 5 \
  --train-examples-per-epoch 8388608 \
  --eval-examples 262144 \
  --test-examples 262144 \
  --lr 0.0003 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --seed 7 \
  --device cuda \
  --num-workers 0 \
  --d-model 96 \
  --num-layers 2 \
  --num-heads 4 \
  --ffn-mult 4 \
  --dropout 0.0

python -u interp/run_b3_semantic_probes.py \
  --data-dir "$REPO_ROOT/interp/data/generated/b3_l25_p2_full" \
  --checkpoint "$OUT_DIR/best_model.pt" \
  --out "$OUT_DIR/semantic_probes.json" \
  --length 25 \
  --num-shards 64 \
  --train-examples 8192 \
  --eval-examples 8192 \
  --ridge 0.01 \
  --device cuda
