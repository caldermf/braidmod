#!/usr/bin/env bash
# Final B_4 Z-sign SAE feature analysis.

#SBATCH --job-name=b4-sae-final
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
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

PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.7.1-foss-2024a-CUDA-12.8.0}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/interp/artifacts/b4_l25_zsign_boundary_r8_sae_final}"

mkdir -p "$REPO_ROOT/interp/slurm_logs" "$OUT_DIR"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/analyze_b4_sae_final.py \
  --data-dir "${DATA_DIR:-$REPO_ROOT/interp/data/generated/b4_l25_p2_n16777216}" \
  --out-dir "$OUT_DIR" \
  --seed42-checkpoint "${SEED42_CHECKPOINT:-$REPO_ROOT/interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small/best_model.pt}" \
  --seed42-sae-dir "${SEED42_SAE_DIR:-$REPO_ROOT/interp/artifacts/b4_l25_zsign_boundary_r8_sae_seed42_v2}" \
  --seed7-checkpoint "${SEED7_CHECKPOINT:-$REPO_ROOT/interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small_seed7/best_model.pt}" \
  --seed7-sae-dir "${SEED7_SAE_DIR:-$REPO_ROOT/interp/artifacts/b4_l25_zsign_boundary_r8_sae_seed7_v2}" \
  --sites "${SITES:-final_hidden_cls+l1_resid_post_cls+l1_attn_out_cls}" \
  --atlas-sites "${ATLAS_SITES:-final_hidden_cls+l1_resid_post_cls}" \
  --classifier-sites "${CLASSIFIER_SITES:-final_hidden_cls+l1_resid_post_cls}" \
  --path-sites "${PATH_SITES:-final_hidden_cls+l1_resid_post_cls}" \
  --num-shards "${NUM_SHARDS:-64}" \
  --batch-size "${BATCH_SIZE:-2048}" \
  --eval-examples "${EVAL_EXAMPLES:-8192}" \
  --train-examples "${TRAIN_EXAMPLES:-32768}" \
  --prefix-pairs "${PREFIX_PAIRS:-512}" \
  --chunk-size "${CHUNK_SIZE:-512}" \
  --max-atlas-features "${MAX_ATLAS_FEATURES:-32}" \
  --classifier-steps "${CLASSIFIER_STEPS:-500}" \
  --classifier-lr "${CLASSIFIER_LR:-0.05}" \
  --random-classifier-trials "${RANDOM_CLASSIFIER_TRIALS:-10}" \
  --top-examples "${TOP_EXAMPLES:-8}" \
  --seed "${SEED:-20260606}" \
  --device "${DEVICE:-cuda}" \
  --model-input-boundary-radius "${MODEL_INPUT_BOUNDARY_RADIUS:--2}"
