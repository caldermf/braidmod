#!/usr/bin/env bash
# Stress-test B_4 SAE feature claims with random/permutation controls.

#SBATCH --job-name=b4-sae-stress
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=02:00:00
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
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small/best_model.pt}"
SAE_DIR="${SAE_DIR:-$REPO_ROOT/interp/artifacts/b4_l25_zsign_boundary_r8_sae_seed42_v2}"
RESULTS_JSON="${RESULTS_JSON:-$SAE_DIR/results.json}"
OUT="${OUT:-$SAE_DIR/stress_controls.json}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.7.1-foss-2024a-CUDA-12.8.0}"

mkdir -p "$REPO_ROOT/interp/slurm_logs" "$(dirname "$OUT")"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/stress_b4_sae_controls.py \
  --data-dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --sae-dir "$SAE_DIR" \
  --results-json "$RESULTS_JSON" \
  --out "$OUT" \
  --num-shards "${NUM_SHARDS:-64}" \
  --batch-size "${BATCH_SIZE:-2048}" \
  --eval-examples "${EVAL_EXAMPLES:-8192}" \
  --prefix-pairs "${PREFIX_PAIRS:-512}" \
  --chunk-size "${CHUNK_SIZE:-512}" \
  --feature-counts "${FEATURE_COUNTS:-1,2,4,8,16,32,64,128}" \
  --random-trials "${RANDOM_TRIALS:-20}" \
  --sites "${SITES:-l1_resid_post_cls+final_hidden_cls+l1_attn_out_cls}" \
  --seed "${SEED:-20260606}" \
  --device "${DEVICE:-cuda}" \
  --model-input-boundary-radius "${MODEL_INPUT_BOUNDARY_RADIUS:--2}"
