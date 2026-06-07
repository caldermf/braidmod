#!/usr/bin/env bash
# Train and evaluate SAEs on B_4 Z[v] boundary-transformer activations.

#SBATCH --job-name=b4-zsign-sae
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

DATA_DIR="${DATA_DIR:-$REPO_ROOT/interp/data/generated/b4_l25_p2_n16777216}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small/best_model.pt}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/interp/artifacts/b4_l25_zsign_boundary_r8_sae_suite}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.7.1-foss-2024a-CUDA-12.8.0}"

mkdir -p "$REPO_ROOT/interp/slurm_logs" "$OUT_DIR"
cd "$REPO_ROOT"

module purge || true
module load "$PYTORCH_MODULE"

python -u interp/run_b4_sae_experiments.py \
  --data-dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$OUT_DIR" \
  --num-shards "${NUM_SHARDS:-64}" \
  --batch-size "${BATCH_SIZE:-2048}" \
  --train-examples "${TRAIN_EXAMPLES:-262144}" \
  --eval-examples "${EVAL_EXAMPLES:-32768}" \
  --prefix-pairs "${PREFIX_PAIRS:-512}" \
  --epochs "${SAE_EPOCHS:-8}" \
  --expansion "${EXPANSION:-16}" \
  --top-k "${TOP_K:-16}" \
  --lr "${LR:-1e-3}" \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --chunk-size "${CHUNK_SIZE:-512}" \
  --top-label-examples "${TOP_LABEL_EXAMPLES:-256}" \
  --max-labeled-features "${MAX_LABELED_FEATURES:-32}" \
  --sites "${SITES:-l1_resid_post_cls+final_hidden_cls+l1_attn_out_cls+l0_mlp_out_cls}" \
  --seed "${SEED:-20260606}" \
  --device "${DEVICE:-cuda}" \
  --model-input-boundary-radius "${MODEL_INPUT_BOUNDARY_RADIUS:--2}"
