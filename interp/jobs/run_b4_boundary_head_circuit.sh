#!/usr/bin/env bash
# Targeted final-boundary-head circuit analysis for B_4.

#SBATCH --job-name=b4-bdry-head
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=interp/slurm_logs/%x-%j.out
#SBATCH --error=interp/slurm_logs/%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi

DATA_DIR="${DATA_DIR:-$REPO_ROOT/interp/data/generated/b4_l25_p2_n16777216}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/interp/artifacts/b4_l25_p2_xfmr3_abs/best_model.pt}"
OUT="${OUT:-$REPO_ROOT/interp/artifacts/b4_l25_p2_boundary_head_circuit/results.json}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.7.1-foss-2024a-CUDA-12.8.0}"

mkdir -p "$REPO_ROOT/interp/slurm_logs" "$(dirname "$OUT")"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/run_b4_boundary_head_circuit.py \
  --data-dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --out "$OUT" \
  --num-shards 64 \
  --batch-size "${BATCH_SIZE:-4096}" \
  --examples "${EXAMPLES:-4096}" \
  --max-scan-examples "${MAX_SCAN_EXAMPLES:-524288}" \
  --matched-pairs "${MATCHED_PAIRS:-256}" \
  --boundary-radius "${BOUNDARY_RADIUS:-8}" \
  --seed "${SEED:-42}" \
  --device cuda
