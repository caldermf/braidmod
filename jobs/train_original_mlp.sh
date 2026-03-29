#!/usr/bin/env bash
# Train the original MLP baseline used as the main public comparison.

#SBATCH --job-name=braidmod-public-mlp
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_PATH="${PYTHON_PATH:-/home/com36/.conda/envs/burau_gpu/bin/python}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/generated/burau_gnf_L30to60_p5_D140_N200000_uniform_corrected.json}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/public_original_mlp}"

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
  --task multitask \
  --batch-size 512 \
  --epochs 40 \
  --lr 3e-4 \
  --weight-decay 1e-2 \
  --val-fraction 0.1 \
  --seed 42 \
  --embed-dim 32 \
  --hidden-dim 1024 \
  --blocks 3 \
  --dropout 0.1 \
  --aux-weight 0.2 \
  --num-workers 4 \
  --grad-clip 1.0 \
  --out-dir "$OUT_DIR" \
  --device cuda \
  2>&1 | tee "$OUT_DIR/train.log"

"$PYTHON_PATH" plot_training_curves.py \
  --log "$OUT_DIR/train.log" \
  --out "$OUT_DIR/training_curves.png" \
  --title "Original MLP Baseline (L30to60, p=5, D=140, N=200000)"
