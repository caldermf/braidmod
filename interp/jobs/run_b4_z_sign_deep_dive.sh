#!/usr/bin/env bash
# Deeper B_4 Z[v] sign-token interpretability experiments.

#SBATCH --job-name=b4-zsign-deep
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
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
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/interp/artifacts/b4_l25_zsign_xfmr3_abs/best_model.pt}"
OUT="${OUT:-$REPO_ROOT/interp/artifacts/b4_l25_zsign_deep_dive/results.json}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.7.1-foss-2024a-CUDA-12.8.0}"

mkdir -p "$REPO_ROOT/interp/slurm_logs" "$(dirname "$OUT")"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/run_b4_z_sign_deep_dive.py \
  --data-dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --out "$OUT" \
  --num-shards "${NUM_SHARDS:-64}" \
  --batch-size "${BATCH_SIZE:-2048}" \
  --probe-train-examples "${PROBE_TRAIN_EXAMPLES:-32768}" \
  --probe-eval-examples "${PROBE_EVAL_EXAMPLES:-8192}" \
  --quotient-train-examples "${QUOTIENT_TRAIN_EXAMPLES:-32768}" \
  --quotient-eval-examples "${QUOTIENT_EVAL_EXAMPLES:-8192}" \
  --max-scan-examples "${MAX_SCAN_EXAMPLES:-262144}" \
  --prefix-pairs "${PREFIX_PAIRS:-512}" \
  --chunk-size "${CHUNK_SIZE:-512}" \
  --ridge "${RIDGE:-1e-2}" \
  --seed "${SEED:-20260602}" \
  --device "${DEVICE:-cuda}" \
  --model-input-boundary-radius "${MODEL_INPUT_BOUNDARY_RADIUS:--2}"
