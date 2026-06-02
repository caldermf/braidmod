#!/usr/bin/env bash
# Audit the pilot random B_4 length-25 p=2 corpus.

#SBATCH --job-name=b4-audit-pilot
#SBATCH --partition=scavenge
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
OUT="$REPO_ROOT/interp/artifacts/b4_l25_p2_pilot_audit"
mkdir -p "$REPO_ROOT/interp/slurm_logs" "$OUT"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/audit_b4_dataset.py \
  --data-dir "$REPO_ROOT/interp/data/generated/b4_l25_p2_pilot_n262144" \
  --length 25 \
  --num-samples 262144 \
  --num-shards 4 \
  --spot-check-per-shard 8 \
  | tee "$OUT/results.json"
