#!/usr/bin/env bash
# Search for theorem-shaped B_4 Burau/descent rules over Z[v].

#SBATCH --job-name=b4-hidden-thm
#SBATCH --partition=scavenge
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
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
ARTIFACT_NAME="${ARTIFACT_NAME:-b4_l25_z_hidden_theorem_search}"
OUT="$REPO_ROOT/interp/artifacts/$ARTIFACT_NAME"
mkdir -p "$REPO_ROOT/interp/slurm_logs" "$OUT"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

EXTRA_ARGS=()
if [[ "${SIMPLE_QUOTIENTS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--simple-quotients)
fi
if [[ "${SHUFFLE:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--shuffle)
fi

python -u interp/search_b4_hidden_theorem.py \
  --data-dir "$REPO_ROOT/interp/data/generated/b4_l25_p2_n16777216" \
  --num-shards "${NUM_SHARDS:-64}" \
  --length "${LENGTH:-25}" \
  --train-examples "${TRAIN_EXAMPLES:-131072}" \
  --eval-examples "${EVAL_EXAMPLES:-32768}" \
  --batch-size "${BATCH_SIZE:-2048}" \
  --radius "${RADIUS:-6}" \
  --device "${DEVICE:-auto}" \
  --seed "${SEED:-20260602}" \
  --out "$OUT/results.json" \
  "${EXTRA_ARGS[@]}" \
  | tee "$OUT/stdout.json"
