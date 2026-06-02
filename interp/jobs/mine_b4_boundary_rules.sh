#!/usr/bin/env bash
# Mine algebraic B_4 boundary/frontier rules on the generated corpus.

#SBATCH --job-name=b4-rule-mine
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
ARTIFACT_NAME="${ARTIFACT_NAME:-b4_l25_p2_boundary_rule_mining}"
OUT="$REPO_ROOT/interp/artifacts/$ARTIFACT_NAME"
mkdir -p "$REPO_ROOT/interp/slurm_logs" "$OUT"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/mine_b4_boundary_rules.py \
  --data-dir "$REPO_ROOT/interp/data/generated/b4_l25_p2_n16777216" \
  --num-shards 64 \
  --train-examples "${TRAIN_EXAMPLES:-262144}" \
  --eval-examples "${EVAL_EXAMPLES:-65536}" \
  --batch-size "${BATCH_SIZE:-8192}" \
  --max-radius "${MAX_RADIUS:-8}" \
  --top-k "${TOP_K:-16}" \
  --max-pair-keys "${MAX_PAIR_KEYS:-20000}" \
  --out "$OUT/results.json" \
  | tee "$OUT/stdout.json"
