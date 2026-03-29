#!/usr/bin/env bash
# Run pure target-cross-entropy reservoir search with max-so-far scoring.

#SBATCH --job-name=braidmod-search-xentmax-l60-b100k-u300k-nb100
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
PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"
CHECKPOINT_PATH="$REPO_ROOT/artifacts/garside_mlp_L30to60_p5_D140_N200000_e20/best_model.pt"
OUT_JSON="$REPO_ROOT/artifacts/reservoir_search_target_xentmax_gpu_len60_b100k_u300k_nb100_boot5.json"
OUT_PNG="$REPO_ROOT/artifacts/reservoir_search_target_xentmax_gpu_len60_b100k_u300k_nb100_boot5_best_projlen.png"

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

echo "Starting target_xent max-so-far search at $(date)"
echo "Host: $(hostname)"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "JSON out: $OUT_JSON"

"$PYTHON_PATH" -u reservoir_search_braidmod.py \
  --p 5 \
  --max-length 60 \
  --bucket-size 100000 \
  --use-best 300000 \
  --bootstrap-length 5 \
  --num-buckets 100 \
  --score-type target_xent_max \
  --checkpoint "$CHECKPOINT_PATH" \
  --device cuda \
  --out-json "$OUT_JSON"

"$PYTHON_PATH" - <<'PY'
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_json = Path("artifacts/reservoir_search_target_xentmax_gpu_len60_b100k_u300k_nb100_boot5.json")
out_png = Path("artifacts/reservoir_search_target_xentmax_gpu_len60_b100k_u300k_nb100_boot5_best_projlen.png")
payload = json.loads(out_json.read_text(encoding="utf-8"))
levels = [int(item["level"]) for item in payload["level_summaries"]]

from braid_data import GNF, GarsideFactor, burau_mod_p_projective_tensor_from_gnf
vals = []
for item in payload["level_summaries"]:
    candidate = item["best_candidate"]
    gnf = GNF(0, [GarsideFactor(tuple(f)) for f in candidate["gnf_factors"]])
    tensor, _ = burau_mod_p_projective_tensor_from_gnf(gnf, p=5, D=4 * len(candidate["gnf_factors"]) + 1)
    import torch
    t = torch.tensor(tensor, dtype=torch.int64)
    deg_has = t.ne(0).any(dim=(-1, -2))
    lo = int(deg_has.int().argmax().item())
    hi = int(t.shape[0] - 1 - deg_has.flip(dims=[0]).int().argmax().item())
    vals.append(hi - lo + 1)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(levels, vals, marker="o", markersize=3, linewidth=1.8, color="#1f4e79")
ax.set_xlabel("Garside Length")
ax.set_ylabel("Best Projlen Found")
ax.set_title("Best Projlen by Length (target_xent max-so-far search)")
ax.set_xlim(1, max(levels))
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(out_png, dpi=180)
plt.close(fig)
PY

echo "Finished at $(date)"
ls -lh "$OUT_JSON" "$OUT_PNG"
