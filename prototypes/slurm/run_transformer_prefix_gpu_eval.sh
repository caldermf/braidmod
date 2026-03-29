#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/nfs/roberts/project/pi_com36/com36/burau-experiments/braidmod}"
PYTHON_PATH="${PYTHON_PATH:-/home/com36/.conda/envs/burau_gpu/bin/python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/artifacts/garside_transformer_L30to60_p5_D140_N200000_e20/best_model_epoch6_snapshot.pt}"
SOURCE_SUITE_DIR="${SOURCE_SUITE_DIR:-$REPO_ROOT/artifacts/length54_confusion_suite_D140_e20}"
SEARCH_JSON="${SEARCH_JSON:-$REPO_ROOT/artifacts/reservoir_search_projlen_exact_kernelhits_gpu_len60_b200k_u500k_boot5.json}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/length54_confusion_suite_transformer_epoch6}"
AVG5_OVERLAY_PNG="${AVG5_OVERLAY_PNG:-$OUT_DIR/kernel_hits_vs_random_target_xent_avg5_overlay.png}"
RAW_OVERLAY_PNG="${RAW_OVERLAY_PNG:-$OUT_DIR/kernel_hits_vs_random_target_xent_raw_overlay.png}"

cd "$REPO_ROOT"
mkdir -p "$OUT_DIR"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint not found at $CHECKPOINT_PATH" >&2
  exit 1
fi
if [[ ! -d "$SOURCE_SUITE_DIR" ]]; then
  echo "Source suite dir not found at $SOURCE_SUITE_DIR" >&2
  exit 1
fi
if [[ ! -f "$SEARCH_JSON" ]]; then
  echo "Search JSON not found at $SEARCH_JSON" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

echo "Starting transformer prefix evaluation at $(date)"
echo "Host: $(hostname)"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Source suite: $SOURCE_SUITE_DIR"
echo "Out dir: $OUT_DIR"

"$PYTHON_PATH" -u rescore_saved_prefix_suite.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --device cuda \
  --source-suite-dir "$SOURCE_SUITE_DIR" \
  --out-dir "$OUT_DIR" \
  --topk 3

"$PYTHON_PATH" -u render_smoothed_xent_suite.py \
  --suite-dir "$OUT_DIR" \
  --window 5

"$PYTHON_PATH" -u render_kernel_random_xent_overlay.py \
  --search-json "$SEARCH_JSON" \
  --checkpoint "$CHECKPOINT_PATH" \
  --suite-dir "$OUT_DIR" \
  --out-png "$AVG5_OVERLAY_PNG" \
  --device cuda \
  --mode avg5 \
  --window 5 \
  --max-length 60

"$PYTHON_PATH" -u render_kernel_random_xent_overlay.py \
  --search-json "$SEARCH_JSON" \
  --checkpoint "$CHECKPOINT_PATH" \
  --suite-dir "$OUT_DIR" \
  --out-png "$RAW_OVERLAY_PNG" \
  --device cuda \
  --mode raw \
  --window 5 \
  --max-length 60

echo "Finished at $(date)"
ls -lh "$OUT_DIR"
