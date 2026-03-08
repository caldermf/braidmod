#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot best projlen by search level from a saved reservoir-search JSON.")
    parser.add_argument("--search-json", required=True, help="Path to reservoir search JSON output")
    parser.add_argument("--out-png", required=True, help="Output PNG path")
    parser.add_argument("--title", default="Best Projlen By Search Level", help="Plot title")
    args = parser.parse_args()

    payload = json.loads(Path(args.search_json).read_text(encoding="utf-8"))
    levels = []
    best_projlen = []
    prune_levels = []
    prune_kept = []
    prune_discarded = []
    for summary in payload.get("level_summaries", []):
        best_candidate = summary.get("best_candidate")
        if best_candidate is None or best_candidate.get("score") is None:
            continue
        levels.append(int(summary["level"]))
        best_projlen.append(float(best_candidate["score"]))
        prune = summary.get("xent_prune")
        if prune is not None:
            prune_levels.append(int(summary["level"]))
            prune_kept.append(int(prune["kept"]))
            prune_discarded.append(int(prune["discarded"]))

    if not levels:
        raise ValueError(f"No best-candidate trajectory found in {args.search_json}")

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(levels, best_projlen, color="#1f4e79", linewidth=2.1, marker="o", markersize=3.5)
    for level, kept, discarded in zip(prune_levels, prune_kept, prune_discarded):
        ax.axvline(level, color="#b4442a", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(
            level + 0.2,
            max(best_projlen),
            f"discard {discarded}\nkeep {kept}",
            color="#b4442a",
            fontsize=8,
            va="top",
        )

    ax.set_xlabel("Garside Length")
    ax.set_ylabel("Best Projlen")
    ax.set_title(args.title)
    ax.set_xlim(1, max(levels))
    ax.set_ylim(0.0, 1.05 * max(best_projlen))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
