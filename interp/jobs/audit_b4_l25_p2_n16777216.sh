#!/usr/bin/env bash
# Audit the main random B_4 length-25 p=2 corpus.

#SBATCH --job-name=b4-audit-full
#SBATCH --partition=scavenge
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

PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.7.1-foss-2024a-CUDA-12.8.0}"
OUT="$REPO_ROOT/interp/artifacts/b4_l25_p2_n16777216_audit"
mkdir -p "$REPO_ROOT/interp/slurm_logs" "$OUT"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/audit_b4_dataset.py \
  --data-dir "$REPO_ROOT/interp/data/generated/b4_l25_p2_n16777216" \
  --length 25 \
  --num-samples 16777216 \
  --num-shards 64 \
  --spot-check-per-shard 4 \
  | tee "$OUT/results.json"
