#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/nfs/roberts/project/pi_com36/com36/burau-experiments/braidmod"

export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/artifacts/garside_transformer_with_length_L30to60_p5_D140_N200000_e20/best_model.pt}"
export OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/length54_confusion_suite_transformer_with_length}"

"$REPO_ROOT/run_transformer_prefix_gpu_eval.sh"
