#!/usr/bin/env bash
# Boundary-token flip counterfactuals for the trained B_3 transformer and MLP.

#SBATCH --job-name=b3-l25-boundary-cf
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
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
mkdir -p "$REPO_ROOT/interp/slurm_logs" "$REPO_ROOT/interp/artifacts/b3_l25_p2_boundary_counterfactuals"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/run_b3_boundary_counterfactuals.py \
  --data-dir "$REPO_ROOT/interp/data/generated/b3_l25_p2_full" \
  --transformer-checkpoint "$REPO_ROOT/interp/artifacts/b3_l25_p2_xfmr2_abs/best_model.pt" \
  --mlp-checkpoint "$REPO_ROOT/interp/artifacts/b3_l25_p2_mlp1_abs_h128/best_model.pt" \
  --out "$REPO_ROOT/interp/artifacts/b3_l25_p2_boundary_counterfactuals/results.json" \
  --length 25 \
  --num-shards 64 \
  --batch-size 8192 \
  --examples 1048576 \
  --device cuda
