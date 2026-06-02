#!/usr/bin/env bash
# Audit explicit column-support frontier rules for B_4.

#SBATCH --job-name=b4-colrule
#SBATCH --partition=scavenge
#SBATCH --time=00:45:00
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
OUT="$REPO_ROOT/interp/artifacts/${ARTIFACT_NAME:-b4_l25_p2_column_filtration}"
mkdir -p "$REPO_ROOT/interp/slurm_logs" "$OUT"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/audit_b4_column_filtration.py \
  --data-dir "$REPO_ROOT/interp/data/generated/b4_l25_p2_n16777216" \
  --num-shards 64 \
  --split "${SPLIT:-test}" \
  --examples "${EXAMPLES:-262144}" \
  --batch-size "${BATCH_SIZE:-8192}" \
  --radius "${RADIUS:-8}" \
  --out "$OUT/results.json" \
  | tee "$OUT/stdout.json"
