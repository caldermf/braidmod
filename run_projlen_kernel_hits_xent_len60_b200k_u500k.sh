#!/usr/bin/env bash
# Run pure projlen search with exact projlen buckets, save many confirmed kernel hits,
# then overlay avg5 target-xent curves for all saved kernel elements.

#SBATCH --job-name=braidmod-kernelhits-projlen-l60-b200k-u500k
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/nfs/roberts/project/pi_com36/com36/burau-experiments/braidmod"
PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"
CHECKPOINT_PATH="$REPO_ROOT/artifacts/garside_mlp_L30to60_p5_D140_N200000_e20/best_model.pt"

OUT_JSON="$REPO_ROOT/artifacts/reservoir_search_projlen_exact_kernelhits_gpu_len60_b200k_u500k_boot5.json"
OUT_PNG="$REPO_ROOT/artifacts/reservoir_search_projlen_exact_kernelhits_gpu_len60_b200k_u500k_boot5_kernel_target_xent_avg5_overlay.png"

mkdir -p "$REPO_ROOT/slurm_logs" "$REPO_ROOT/artifacts"
cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint not found at $CHECKPOINT_PATH" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

echo "Starting pure-projlen exact-bucket kernel-hit search at $(date)"
echo "Host: $(hostname)"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "JSON out: $OUT_JSON"

"$PYTHON_PATH" -u reservoir_search_braidmod.py \
  --p 5 \
  --max-length 60 \
  --bucket-size 200000 \
  --use-best 500000 \
  --bootstrap-length 5 \
  --num-buckets 200 \
  --score-type projlen \
  --projlen-bucket-mode exact \
  --checkpoint "$CHECKPOINT_PATH" \
  --save-kernel-hits 5000 \
  --device cuda \
  --out-json "$OUT_JSON"

"$PYTHON_PATH" - <<'PY'
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_prefix_confusion import build_progression
from render_smoothed_xent_suite import running_average

checkpoint_path = "artifacts/garside_mlp_L30to60_p5_D140_N200000_e20/best_model.pt"
out_json = Path("artifacts/reservoir_search_projlen_exact_kernelhits_gpu_len60_b200k_u500k_boot5.json")
out_png = Path("artifacts/reservoir_search_projlen_exact_kernelhits_gpu_len60_b200k_u500k_boot5_kernel_target_xent_avg5_overlay.png")

payload = json.loads(out_json.read_text(encoding="utf-8"))
kernel_hits = payload.get("kernel_hits", [])
if not kernel_hits:
    raise ValueError("Search did not save any kernel hits to plot")

fig, ax = plt.subplots(figsize=(11, 6))
ymax = 1.0
num_plotted = 0
for idx, hit in enumerate(kernel_hits):
    factors = [tuple(int(v) for v in factor) for factor in hit["gnf_factors"]]
    progression = build_progression(
        checkpoint_path=checkpoint_path,
        device_arg="cuda",
        d=0,
        factors=factors,
        topk=1,
        truncate_overflow=False,
    )["progression"]
    prefix_lens = [int(item["prefix_len"]) for item in progression]
    xent = [float(item["target_cross_entropy"]) for item in progression]
    avg5 = running_average(xent, window=5)
    ymax = max(ymax, max(avg5) if avg5 else 1.0)

    label = None
    if idx < 12:
        label = f"hit {idx + 1} {hit.get('kernel_type', 'kernel')} len={len(factors)}"
    ax.plot(prefix_lens, avg5, linewidth=1.2, alpha=0.35, label=label)
    num_plotted += 1

ax.set_xlabel("Garside Prefix Length")
ax.set_ylabel("Target Cross-Entropy Avg5")
ax.set_title(f"Avg5 Target Cross-Entropy for Saved Kernel Hits (n={num_plotted})")
ax.set_xlim(1, 60)
ax.set_ylim(0.0, 1.05 * ymax)
ax.grid(True, alpha=0.3)
if num_plotted <= 12:
    ax.legend()
fig.tight_layout()
fig.savefig(out_png, dpi=180)
plt.close(fig)

print(json.dumps(
    {
        "num_kernel_hits_plotted": num_plotted,
        "png": str(out_png),
    },
    indent=2,
))
PY

echo "Finished at $(date)"
ls -lh "$OUT_JSON" "$OUT_PNG"
