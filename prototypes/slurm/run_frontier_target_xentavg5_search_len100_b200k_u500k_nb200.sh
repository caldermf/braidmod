#!/usr/bin/env bash
# Run frontier-distance avg5 target-xent reservoir search at length 100.

#SBATCH --job-name=braidmod-search-frontierxentavg5-l100-b200k-u500k-nb200
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/nfs/roberts/project/pi_com36/com36/burau-experiments/braidmod"
PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"
CHECKPOINT_PATH="$REPO_ROOT/artifacts/garside_mlp_L30to60_p5_D140_N200000_e20/best_model.pt"
OUT_JSON="$REPO_ROOT/artifacts/reservoir_search_frontier_target_xentavg5_gpu_len100_b200k_u500k_nb200_boot5.json"
OUT_PNG="$REPO_ROOT/artifacts/reservoir_search_frontier_target_xentavg5_gpu_len100_b200k_u500k_nb200_boot5_best_projlen.png"

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

echo "Starting frontier_target_xent avg5 search at $(date)"
echo "Host: $(hostname)"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "JSON out: $OUT_JSON"

"$PYTHON_PATH" -u reservoir_search_braidmod.py \
  --p 5 \
  --max-length 100 \
  --bucket-size 200000 \
  --use-best 500000 \
  --bootstrap-length 5 \
  --num-buckets 200 \
  --score-type frontier_target_xent \
  --checkpoint "$CHECKPOINT_PATH" \
  --projlen-weight 1.0 \
  --confusion-weight 1.0 \
  --device cuda \
  --out-json "$OUT_JSON"

"$PYTHON_PATH" - <<'PY'
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from braid_data import GNF, GarsideFactor, burau_mod_p_projective_tensor_from_gnf

out_json = Path("artifacts/reservoir_search_frontier_target_xentavg5_gpu_len100_b200k_u500k_nb200_boot5.json")
out_png = Path("artifacts/reservoir_search_frontier_target_xentavg5_gpu_len100_b200k_u500k_nb200_boot5_best_projlen.png")
payload = json.loads(out_json.read_text(encoding="utf-8"))
levels = [int(item["level"]) for item in payload["level_summaries"]]

vals = []
for item in payload["level_summaries"]:
    candidate = item["best_candidate"]
    gnf = GNF(0, [GarsideFactor(tuple(f)) for f in candidate["gnf_factors"]])
    tensor, _ = burau_mod_p_projective_tensor_from_gnf(gnf, p=5, D=4 * len(candidate["gnf_factors"]) + 1)
    t = torch.tensor(tensor, dtype=torch.int64)
    deg_has = t.ne(0).any(dim=(-1, -2))
    lo = int(deg_has.int().argmax().item())
    hi = int(t.shape[0] - 1 - deg_has.flip(dims=[0]).int().argmax().item())
    vals.append(hi - lo + 1)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(levels, vals, marker="o", markersize=2.5, linewidth=1.9, color="#2f855a")
ax.set_xlabel("Garside Length")
ax.set_ylabel("Best Projlen Found")
ax.set_title("Best Projlen by Length (frontier_target_xent avg5 search)")
ax.set_xlim(1, max(levels))
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(out_png, dpi=180)
plt.close(fig)
PY

echo "Finished at $(date)"
ls -lh "$OUT_JSON" "$OUT_PNG"
