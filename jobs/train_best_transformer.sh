#!/usr/bin/env bash
# Train the best public transformer: factor-only hierarchical encoder,
# selected by validation loss.

#SBATCH --job-name=braidmod-public-transformer
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_PATH="${PYTHON_PATH:-/home/com36/.conda/envs/burau_gpu/bin/python}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/burau_gnf_L30to60_p5_D140_N200000_uniform_corrected.json}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/public_best_transformer}"

mkdir -p "$REPO_ROOT/slurm_logs" "$OUT_DIR"
cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi
if [[ ! -f "$DATA_PATH" ]]; then
  echo "Dataset not found at $DATA_PATH" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

"$PYTHON_PATH" -u train_garside_mlp.py \
  --data-path "$DATA_PATH" \
  --p 5 \
  --model-type transformer \
  --task final_factor \
  --batch-size 256 \
  --epochs 30 \
  --lr 3e-4 \
  --weight-decay 5e-3 \
  --val-fraction 0.1 \
  --seed 42 \
  --d-model 256 \
  --ffn-mult 4 \
  --num-local-blocks 2 \
  --num-local-heads 4 \
  --num-global-blocks 6 \
  --num-global-heads 8 \
  --dropout 0.05 \
  --label-smoothing 0.03 \
  --selection-objective loss \
  --num-workers 4 \
  --grad-clip 1.0 \
  --out-dir "$OUT_DIR" \
  --device cuda \
  2>&1 | tee "$OUT_DIR/train.log"

"$PYTHON_PATH" plot_training_curves.py \
  --log "$OUT_DIR/train.log" \
  --out "$OUT_DIR/training_curves.png" \
  --title "Best Transformer (factor-only, L30to60, p=5, D=140, N=200000)"
