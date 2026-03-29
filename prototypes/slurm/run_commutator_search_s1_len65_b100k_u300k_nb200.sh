#!/usr/bin/env bash
# Run separate commutator-projlen reservoir search for [s_1, b] up to Garside length 65.

#SBATCH --job-name=braidmod-comm-s1-l65-b100k-u300k-nb200
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=08:00:00
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
OUT_JSON="$REPO_ROOT/artifacts/reservoir_search_commutator_s1_gpu_len65_b100k_u300k_nb200_boot5.json"
OUT_PNG="$REPO_ROOT/artifacts/reservoir_search_commutator_s1_gpu_len65_b100k_u300k_nb200_boot5_best_projlen.png"

mkdir -p "$REPO_ROOT/slurm_logs" "$REPO_ROOT/artifacts"
cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

echo "Starting commutator reservoir search at $(date)"
echo "Host: $(hostname)"
echo "Reference braid word: [1]"
echo "JSON out: $OUT_JSON"

"$PYTHON_PATH" -u reservoir_search_commutator.py \
  --reference-braid-word 1 \
  --p 5 \
  --max-length 65 \
  --bucket-size 100000 \
  --use-best 300000 \
  --bootstrap-length 5 \
  --num-buckets 200 \
  --bucket-mode exact \
  --device cuda \
  --out-json "$OUT_JSON"

"$PYTHON_PATH" - <<'PY'
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_json = Path("artifacts/reservoir_search_commutator_s1_gpu_len65_b100k_u300k_nb200_boot5.json")
out_png = Path("artifacts/reservoir_search_commutator_s1_gpu_len65_b100k_u300k_nb200_boot5_best_projlen.png")
payload = json.loads(out_json.read_text(encoding="utf-8"))

levels = []
best_projlen = []
for item in payload["level_summaries"]:
    candidate = item.get("best_candidate")
    if candidate is None:
        continue
    levels.append(int(item["level"]))
    best_projlen.append(int(candidate["commutator_projlen"]))

if not levels:
    raise RuntimeError("No level summaries with best candidates were found in the commutator search output")

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(levels, best_projlen, marker="o", markersize=2.8, linewidth=1.8, color="#1f4e79")
ax.set_xlabel("Garside Length of b")
ax.set_ylabel("Best projlen([s_1, b]) Found")
ax.set_title("Best Commutator Projlen by Garside Length")
ax.set_xlim(1, max(levels))
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(out_png, dpi=180)
plt.close(fig)
PY

echo "Finished at $(date)"
ls -lh "$OUT_JSON" "$OUT_PNG"
