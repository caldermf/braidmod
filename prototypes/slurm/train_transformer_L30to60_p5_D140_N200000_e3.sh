#!/usr/bin/env bash
# Short transformer run to produce a clean early-epoch checkpoint on the L30..60, p=5, D=140, N=200000 dataset.

#SBATCH --job-name=braidmod-xfmr-p5-L30to60-D140-e3
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/nfs/roberts/project/pi_com36/com36/burau-experiments/braidmod"
PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"
DATA_PATH="$REPO_ROOT/data/burau_gnf_L30to60_p5_D140_N200000_uniform.json"
OUT_DIR="$REPO_ROOT/artifacts/garside_transformer_L30to60_p5_D140_N200000_e3"
LOG_PATH="$OUT_DIR/train.log"
PLOT_PATH="$OUT_DIR/training_curves.png"

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

echo "Starting training at $(date)"
echo "Host: $(hostname)"
echo "Python: $PYTHON_PATH"
echo "Dataset: $DATA_PATH"
echo "Out dir: $OUT_DIR"

"$PYTHON_PATH" -u train_garside_mlp.py \
  --data-path "$DATA_PATH" \
  --p 5 \
  --model-type transformer \
  --task multitask \
  --batch-size 256 \
  --epochs 3 \
  --lr 3e-4 \
  --weight-decay 1e-2 \
  --val-fraction 0.1 \
  --seed 42 \
  --d-model 256 \
  --ffn-mult 4 \
  --num-local-blocks 2 \
  --num-local-heads 4 \
  --num-global-blocks 6 \
  --num-global-heads 8 \
  --dropout 0.1 \
  --aux-weight 0.2 \
  --num-workers 4 \
  --grad-clip 1.0 \
  --out-dir "$OUT_DIR" \
  --device cuda \
  2>&1 | tee "$LOG_PATH"

"$PYTHON_PATH" plot_training_curves.py \
  --log "$LOG_PATH" \
  --out "$PLOT_PATH" \
  --title "Transformer Training Curves (L30to60, p=5, D=140, N=200000, 3 epochs)"

echo "Finished at $(date)"
ls -lh "$OUT_DIR"/best_model.pt "$LOG_PATH" "$PLOT_PATH"
