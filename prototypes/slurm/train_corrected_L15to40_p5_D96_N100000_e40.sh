#!/usr/bin/env bash
#SBATCH --job-name=braidmod-train-p5-L15to40-e40
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=00:18:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/nfs/roberts/project/pi_com36/com36/burau-experiments/braidmod"
PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"
DATA_PATH="$REPO_ROOT/data/burau_gnf_L15to40_p5_D96_N100000_corrected.json"
OUT_DIR="$REPO_ROOT/artifacts/garside_mlp_L15to40_p5_D96_N100000_corrected_e40"
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
  2>&1 | tee "$LOG_PATH"

"$PYTHON_PATH" plot_training_curves.py \
  --log "$LOG_PATH" \
  --out "$PLOT_PATH" \
  --title "Corrected Burau Training Curves (L15to40, p=5, D=96, N=100000, 40 epochs)"

echo "Finished at $(date)"
ls -lh "$OUT_DIR"/best_model.pt "$LOG_PATH" "$PLOT_PATH"
