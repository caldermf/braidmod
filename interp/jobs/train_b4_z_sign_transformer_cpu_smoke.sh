#!/usr/bin/env bash
# Tiny CPU-only smoke test for the B_4 Z-sign transformer trainer.

#SBATCH --job-name=b4-zsign-cpu
#SBATCH --partition=scavenge
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=interp/slurm_logs/%x-%j.out
#SBATCH --error=interp/slurm_logs/%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi

DATA_DIR="${DATA_DIR:-$REPO_ROOT/interp/data/generated/b4_l25_p2_n16777216}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/interp/artifacts/b4_l25_zsign_xfmr3_abs_cpu_smoke}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.7.1-foss-2024a-CUDA-12.8.0}"

mkdir -p "$REPO_ROOT/interp/slurm_logs" "$OUT_DIR"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/train_b4_z_sign_transformer.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --num-shards 64 \
  --batch-size 64 \
  --epochs 1 \
  --train-examples-per-epoch 256 \
  --eval-examples 128 \
  --test-examples 128 \
  --lr 3e-4 \
  --weight-decay 1e-2 \
  --grad-clip 1.0 \
  --seed 42 \
  --device cpu \
  --num-workers 0 \
  --d-model 48 \
  --num-layers 1 \
  --num-heads 2 \
  --ffn-mult 2 \
  --dropout 0.0
