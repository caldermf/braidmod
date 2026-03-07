#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def running_average(values: List[float], window: int) -> List[float]:
    out = []
    total = 0.0
    for idx, value in enumerate(values):
        total += value
        if idx >= window:
            total -= values[idx - window]
        denom = min(idx + 1, window)
        out.append(total / float(denom))
    return out


def cumulative_average(values: List[float]) -> List[float]:
    out = []
    total = 0.0
    for idx, value in enumerate(values, start=1):
        total += value
        out.append(total / float(idx))
    return out


def running_max(values: List[float]) -> List[float]:
    out = []
    best = float("-inf")
    for value in values:
        best = max(best, value)
        out.append(best)
    return out


def render_plot(json_path: Path, window: int) -> Path:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    progression = payload["progression"]
    prefix_lens = [item["prefix_len"] for item in progression]
    xent = [float(item["target_cross_entropy"]) for item in progression]
    smooth = running_average(xent, window=window)

    stem = json_path.name.replace("_confusion.json", "")
    out_path = json_path.parent / f"{stem}_target_cross_entropy_avg{window}.png"
    title = f"{stem.replace('_', ' ').title()} Target Cross-Entropy (Running Avg {window})"

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(prefix_lens, smooth, marker="o", markersize=3, linewidth=1.8, color="#1f4e79")
    ax.set_xlabel("Garside Prefix Length")
    ax.set_ylabel("Target Cross-Entropy")
    ax.set_title(title)
    ax.set_xlim(1, max(prefix_lens))
    ymax = max(smooth) if smooth else 1.0
    ax.set_ylim(0.0, max(1.0, 1.05 * ymax))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def render_max_plot(json_path: Path) -> Path:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    progression = payload["progression"]
    prefix_lens = [item["prefix_len"] for item in progression]
    xent = [float(item["target_cross_entropy"]) for item in progression]
    max_seen = running_max(xent)

    stem = json_path.name.replace("_confusion.json", "")
    out_path = json_path.parent / f"{stem}_target_cross_entropy_max_so_far.png"
    title = f"{stem.replace('_', ' ').title()} Max Target Cross-Entropy So Far"

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(prefix_lens, max_seen, marker="o", markersize=3, linewidth=1.8, color="#1f4e79")
    ax.set_xlabel("Garside Prefix Length")
    ax.set_ylabel("Max Target Cross-Entropy So Far")
    ax.set_title(title)
    ax.set_xlim(1, max(prefix_lens))
    ymax = max(max_seen) if max_seen else 1.0
    ax.set_ylim(0.0, max(1.0, 1.05 * ymax))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def render_combined_cumulative_average_plot(json_paths: List[Path]) -> Path:
    if not json_paths:
        raise ValueError("Need at least one JSON path")

    out_dir = json_paths[0].parent
    out_path = out_dir / "all_cases_target_cross_entropy_cumavg.png"

    fig, ax = plt.subplots(figsize=(11, 6))
    ymax = 1.0
    for json_path in json_paths:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        progression = payload["progression"]
        prefix_lens = [item["prefix_len"] for item in progression]
        xent = [float(item["target_cross_entropy"]) for item in progression]
        cumavg = cumulative_average(xent)
        ymax = max(ymax, max(cumavg) if cumavg else 1.0)

        stem = json_path.name.replace("_confusion.json", "")
        label = stem.replace("_", " ")
        linewidth = 2.4 if stem == "geordie_kernel" else 1.7
        alpha = 1.0 if stem == "geordie_kernel" else 0.8
        ax.plot(prefix_lens, cumavg, linewidth=linewidth, alpha=alpha, label=label)

    ax.set_xlabel("Garside Prefix Length")
    ax.set_ylabel("Average Target Cross-Entropy So Far")
    ax.set_title("Cumulative Average Target Cross-Entropy by Prefix")
    ax.set_xlim(1, max(prefix_lens))
    ax.set_ylim(0.0, 1.05 * ymax)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Render smoothed target-cross-entropy plots from saved confusion JSON files.")
    parser.add_argument("--suite-dir", required=True, help="Directory containing *_confusion.json files")
    parser.add_argument("--window", type=int, default=5, help="Running-average window")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir)
    json_paths = sorted(suite_dir.glob("*_confusion.json"))
    if not json_paths:
        raise ValueError(f"No *_confusion.json files found in {suite_dir}")

    written = []
    for path in json_paths:
        written.append(str(render_plot(path, window=args.window)))
        written.append(str(render_max_plot(path)))
    written.append(str(render_combined_cumulative_average_plot(json_paths)))
    print(json.dumps({"written": written}, indent=2))


if __name__ == "__main__":
    main()
