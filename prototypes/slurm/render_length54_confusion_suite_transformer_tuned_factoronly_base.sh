#!/usr/bin/env bash
# Render Geordie plus saved kernel/random confusion plots for the tuned
# factor-only transformer checkpoint selected by validation loss.

#SBATCH --job-name=braidmod-confusion-xfmr-tuned
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/nfs/roberts/project/pi_com36/com36/burau-experiments/braidmod"
CHECKPOINT_PATH="$REPO_ROOT/artifacts/garside_transformer_tuned_factoronly_base_noema_L30to60_p5_D140_N200000_e30/best_model.pt"
OUT_DIR="$REPO_ROOT/artifacts/length54_confusion_suite_transformer_tuned_factoronly_base_noema"

mkdir -p "$REPO_ROOT/slurm_logs" "$OUT_DIR"
cd "$REPO_ROOT"

export REPO_ROOT
export CHECKPOINT_PATH
export OUT_DIR

./run_transformer_prefix_gpu_eval.sh
