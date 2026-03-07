#!/usr/bin/env bash
# Render Geordie plus five random length-54 confusion plots for the D140 e20 model.

#SBATCH --job-name=braidmod-confusion-L54-D140-e20
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/nfs/roberts/project/pi_com36/com36/burau-experiments/braidmod"
PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"
CHECKPOINT_PATH="$REPO_ROOT/artifacts/garside_mlp_L30to60_p5_D140_N200000_e20/best_model.pt"
GEORDIE_GNF_PATH="$REPO_ROOT/data/geordie_kernel_gnf.json"
OUT_DIR="$REPO_ROOT/artifacts/length54_confusion_suite_D140_e20"

mkdir -p "$REPO_ROOT/slurm_logs" "$OUT_DIR"
cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint not found at $CHECKPOINT_PATH" >&2
  exit 1
fi
if [[ ! -f "$GEORDIE_GNF_PATH" ]]; then
  echo "Geordie GNF JSON not found at $GEORDIE_GNF_PATH" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

echo "Starting confusion suite render at $(date)"
echo "Host: $(hostname)"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Out dir: $OUT_DIR"

"$PYTHON_PATH" -u generate_length54_confusion_suite.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --device cuda \
  --geordie-gnf-path "$GEORDIE_GNF_PATH" \
  --out-dir "$OUT_DIR" \
  --seed 42 \
  --topk 3

echo "Finished at $(date)"
ls -lh "$OUT_DIR"
