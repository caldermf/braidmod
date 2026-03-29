#!/usr/bin/env bash
# Render Geordie plus saved kernel/random target-xent plots for the length-conditioned transformer.

#SBATCH --job-name=braidmod-confusion-xfmr-len
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/nfs/roberts/project/pi_com36/com36/burau-experiments/braidmod"

mkdir -p "$REPO_ROOT/slurm_logs"
cd "$REPO_ROOT"

export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/artifacts/garside_transformer_with_length_L30to60_p5_D140_N200000_e20/best_model_epoch6_snapshot.pt}"
export OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/length54_confusion_suite_transformer_with_length_epoch6}"

"$REPO_ROOT/run_transformer_with_length_prefix_gpu_eval.sh"
