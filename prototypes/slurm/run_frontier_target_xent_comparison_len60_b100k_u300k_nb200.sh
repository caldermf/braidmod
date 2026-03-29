#!/usr/bin/env bash
# Compare pure projlen, pure avg5 target_xent maximize, and frontier-distance avg5 target_xent search at length 60.

#SBATCH --job-name=braidmod-search-frontiercmp-l60-b100k-u300k-nb200
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

PROJLEN_JSON="$REPO_ROOT/artifacts/reservoir_search_projlen_gpu_len60_b100k_u300k_nb200_boot5.json"
XENT_JSON="$REPO_ROOT/artifacts/reservoir_search_target_xentavg5max_gpu_len60_b100k_u300k_nb200_boot5.json"
FRONTIER_JSON="$REPO_ROOT/artifacts/reservoir_search_frontier_target_xentavg5_gpu_len60_b100k_u300k_nb200_boot5.json"
OUT_PNG="$REPO_ROOT/artifacts/reservoir_search_frontier_target_xentavg5_gpu_len60_b100k_u300k_nb200_boot5_best_projlen_compare.png"

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

COMMON_ARGS=(
  --p 5
  --max-length 60
  --bucket-size 100000
  --use-best 300000
  --bootstrap-length 5
  --num-buckets 200
  --device cuda
)

echo "Starting length-60 comparison run at $(date)"
echo "Host: $(hostname)"
echo "Checkpoint: $CHECKPOINT_PATH"

echo
echo "[1/3] Pure projlen"
"$PYTHON_PATH" -u reservoir_search_braidmod.py \
  "${COMMON_ARGS[@]}" \
  --score-type projlen \
  --out-json "$PROJLEN_JSON"

echo
echo "[2/3] Pure avg5 target_xent maximize"
"$PYTHON_PATH" -u reservoir_search_braidmod.py \
  "${COMMON_ARGS[@]}" \
  --score-type target_xent_maximize \
  --checkpoint "$CHECKPOINT_PATH" \
  --out-json "$XENT_JSON"

echo
echo "[3/3] Frontier-distance avg5 target_xent"
"$PYTHON_PATH" -u reservoir_search_braidmod.py \
  "${COMMON_ARGS[@]}" \
  --score-type frontier_target_xent \
  --checkpoint "$CHECKPOINT_PATH" \
  --projlen-weight 1.0 \
  --confusion-weight 1.0 \
  --out-json "$FRONTIER_JSON"

"$PYTHON_PATH" - <<'PY'
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from braid_data import GNF, GarsideFactor, burau_mod_p_projective_tensor_from_gnf


def best_projlen_series(path_str: str) -> tuple[list[int], list[int]]:
    payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
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
    return levels, vals


projlen_json = "artifacts/reservoir_search_projlen_gpu_len60_b100k_u300k_nb200_boot5.json"
xent_json = "artifacts/reservoir_search_target_xentavg5max_gpu_len60_b100k_u300k_nb200_boot5.json"
frontier_json = "artifacts/reservoir_search_frontier_target_xentavg5_gpu_len60_b100k_u300k_nb200_boot5.json"
out_png = Path("artifacts/reservoir_search_frontier_target_xentavg5_gpu_len60_b100k_u300k_nb200_boot5_best_projlen_compare.png")

proj_levels, proj_vals = best_projlen_series(projlen_json)
xent_levels, xent_vals = best_projlen_series(xent_json)
frontier_levels, frontier_vals = best_projlen_series(frontier_json)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(proj_levels, proj_vals, marker="o", markersize=2.5, linewidth=1.7, color="#1f4e79", label="projlen")
ax.plot(xent_levels, xent_vals, marker="o", markersize=2.5, linewidth=1.7, color="#c05621", label="target_xent_maximize avg5")
ax.plot(frontier_levels, frontier_vals, marker="o", markersize=2.5, linewidth=1.9, color="#2f855a", label="frontier_target_xent avg5")
ax.set_xlabel("Garside Length")
ax.set_ylabel("Best Projlen Found")
ax.set_title("Best Projlen by Length: projlen vs avg5 target_xent vs frontier-distance")
ax.set_xlim(1, max(frontier_levels))
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(out_png, dpi=180)
plt.close(fig)
PY

echo "Finished at $(date)"
ls -lh "$PROJLEN_JSON" "$XENT_JSON" "$FRONTIER_JSON" "$OUT_PNG"
