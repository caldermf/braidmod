#!/usr/bin/env bash
# First-pass TransformerLens-style interp experiments for the B_3 transformer.

#SBATCH --job-name=b3-l25-interp1
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=00:30:00
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
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/interp/artifacts/b3_l25_p2_xfmr2_abs/best_model.pt}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/interp/artifacts/b3_l25_p2_interp_firstpass}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.7.1-foss-2024a-CUDA-12.8.0}"

mkdir -p "$REPO_ROOT/interp/slurm_logs" "$OUT_DIR"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/run_b3_interp_experiments.py \
  --data-dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$OUT_DIR" \
  --length 25 \
  --num-shards 64 \
  --batch-size 8192 \
  --feature-radius 2 \
  --feature-train-examples 2097152 \
  --feature-eval-examples 671088 \
  --transform-eval-examples 262144 \
  --cache-check-examples 512 \
  --patch-pairs 64 \
  --device cuda
