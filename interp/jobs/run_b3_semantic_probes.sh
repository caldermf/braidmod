#!/usr/bin/env bash
# Linear semantic probes for the B_3 transformer interp harness.

#SBATCH --job-name=b3-sem-probes
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=00:05:00
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
mkdir -p "$REPO_ROOT/interp/slurm_logs" "$REPO_ROOT/interp/artifacts/b3_l25_p2_semantic_probes"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/run_b3_semantic_probes.py \
  --data-dir "$REPO_ROOT/interp/data/generated/b3_l25_p2_full" \
  --checkpoint "$REPO_ROOT/interp/artifacts/b3_l25_p2_xfmr2_abs/best_model.pt" \
  --out "$REPO_ROOT/interp/artifacts/b3_l25_p2_semantic_probes/results.json" \
  --length 25 \
  --num-shards 64 \
  --batch-size 8192 \
  --train-examples 8192 \
  --eval-examples 8192 \
  --ridge 0.01 \
  --device cuda
