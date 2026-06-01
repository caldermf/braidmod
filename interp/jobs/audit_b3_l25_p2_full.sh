#!/usr/bin/env bash
# Audit the exhaustive B_3 length-25 p=2 corpus after all array shards finish.

#SBATCH --job-name=b3-l25-p2-audit
#SBATCH --partition=scavenge
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.7.1-foss-2024a-CUDA-12.8.0}"

mkdir -p "$REPO_ROOT/interp/slurm_logs"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/audit_b3_dataset.py \
  --data-dir "$DATA_DIR" \
  --length 25 \
  --num-shards 64 \
  --spot-check-per-shard 4
