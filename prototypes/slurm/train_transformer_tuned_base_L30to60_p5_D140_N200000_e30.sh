#!/usr/bin/env bash
# Train the original Burau transformer with a slightly better-calibrated
# optimization setup, still keeping the architecture unchanged.

#SBATCH --job-name=braidmod-xfmr-tuned-base
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

REPO_ROOT="/nfs/roberts/project/pi_com36/com36/burau-experiments/braidmod"
PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"
DATA_PATH="$REPO_ROOT/data/burau_gnf_L30to60_p5_D140_N200000_uniform.json"
OUT_DIR="$REPO_ROOT/artifacts/garside_transformer_tuned_base_noema_L30to60_p5_D140_N200000_e30"
LOG_PATH="$OUT_DIR/train.log"
PLOT_PATH="$OUT_DIR/training_curves.png"

mkdir -p "$REPO_ROOT/slurm_logs" "$OUT_DIR"
cd "$REPO_ROOT"

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
  --aux-weight 0.2 \
  --num-workers 4 \
  --grad-clip 1.0 \
  --out-dir "$OUT_DIR" \
  --device cuda \
  2>&1 | tee "$LOG_PATH"

"$PYTHON_PATH" plot_training_curves.py \
  --log "$LOG_PATH" \
  --out "$PLOT_PATH" \
  --title "Tuned Burau Transformer Curves (base arch, no EMA, L30to60, p=5, D=140, N=200000, 30 epochs)"

echo "Finished at $(date)"
ls -lh "$OUT_DIR"/best_model.pt "$OUT_DIR"/best_loss_model.pt "$OUT_DIR"/best_metric_model.pt "$LOG_PATH" "$PLOT_PATH"
