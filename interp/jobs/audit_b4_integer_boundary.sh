#!/usr/bin/env bash
# Audit signed/integer B_4 Burau boundary rules over Z[v].

#SBATCH --job-name=b4-zboundary
#SBATCH --partition=scavenge
#SBATCH --time=01:30:00
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
OUT="$REPO_ROOT/interp/artifacts/${ARTIFACT_NAME:-b4_l25_p2_integer_boundary}"
mkdir -p "$REPO_ROOT/interp/slurm_logs" "$OUT"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/audit_b4_integer_boundary.py \
  --data-dir "$REPO_ROOT/interp/data/generated/b4_l25_p2_n16777216" \
  --num-shards 64 \
  --length "${LENGTH:-25}" \
  --train-examples "${TRAIN_EXAMPLES:-65536}" \
  --eval-examples "${EVAL_EXAMPLES:-16384}" \
  --batch-size "${BATCH_SIZE:-2048}" \
  --radius "${RADIUS:-4}" \
  --coeff-clip "${COEFF_CLIP:-3}" \
  --validate-examples "${VALIDATE_EXAMPLES:-16}" \
  --out "$OUT/results.json" \
  | tee "$OUT/stdout.json"
