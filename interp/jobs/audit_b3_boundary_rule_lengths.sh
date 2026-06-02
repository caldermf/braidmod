#!/usr/bin/env bash
# Audit B_3 boundary-slice rule across a range of Garside lengths.

#SBATCH --job-name=b3-boundary-lengths
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=00:20:00
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
mkdir -p "$REPO_ROOT/interp/slurm_logs" "$REPO_ROOT/interp/artifacts/b3_boundary_rule_lengths"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/audit_b3_boundary_rule_lengths.py \
  --max-length 20 \
  --batch-size 65536 \
  --exhaustive-limit 1048576 \
  --sample-limit 1048576 \
  --device cuda \
  --out "$REPO_ROOT/interp/artifacts/b3_boundary_rule_lengths/results.json"
