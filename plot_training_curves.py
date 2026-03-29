#!/usr/bin/env python3
import argparse
import os
import re
import tempfile
from pathlib import Path

cache_root = Path(tempfile.gettempdir()) / "braidmod-cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LINE_RE = re.compile(
    r"epoch=(\d+)\s+"
    r"train_loss=([0-9.]+)\s+"
    r"train_([A-Za-z0-9_]+)=([0-9.]+)\s+"
    r"val_loss=([0-9.]+)\s+"
    r"val_([A-Za-z0-9_]+)=([0-9.]+)"
)


def parse_log(text: str):
    epochs = []
    train_loss = []
    val_loss = []
    train_metric = []
    val_metric = []
    metric_name = None

    for line in text.splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        train_metric_name = m.group(3)
        val_metric_name = m.group(6)
        if train_metric_name != val_metric_name:
            raise ValueError(
                f"Mismatched train/val metric names in line: {train_metric_name} vs {val_metric_name}"
            )
        if metric_name is None:
            metric_name = train_metric_name
        elif metric_name != train_metric_name:
            raise ValueError(
                f"Found multiple metric names in one log: {metric_name} and {train_metric_name}"
            )
        epochs.append(int(m.group(1)))
        train_loss.append(float(m.group(2)))
        train_metric.append(float(m.group(4)))
        val_loss.append(float(m.group(5)))
        val_metric.append(float(m.group(7)))

    if not epochs:
        raise ValueError("No epoch lines found in log.")
    return epochs, train_loss, val_loss, train_metric, val_metric, metric_name


def main():
    parser = argparse.ArgumentParser(description="Plot train/val loss and accuracy curves from training logs.")
    parser.add_argument("--log", required=True, help="Path to log file containing epoch=... lines")
    parser.add_argument("--out", default="training_curves.png", help="Output PNG path")
    parser.add_argument("--title", default="Training Curves", help="Figure title")
    args = parser.parse_args()

    log_text = Path(args.log).read_text(encoding="utf-8")
    epochs, train_loss, val_loss, train_metric, val_metric, metric_name = parse_log(log_text)

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    loss_colors = {"train": "#355070", "val": "#b56576"}
    metric_colors = {"train": "#2a9d8f", "val": "#e76f51"}

    best_loss_idx = min(range(len(val_loss)), key=val_loss.__getitem__)
    best_metric_idx = max(range(len(val_metric)), key=val_metric.__getitem__)

    axes[0].plot(epochs, train_loss, label="train_loss", color=loss_colors["train"], linewidth=2.4)
    axes[0].plot(epochs, val_loss, label="val_loss", color=loss_colors["val"], linewidth=2.4)
    axes[0].scatter(
        [epochs[best_loss_idx]],
        [val_loss[best_loss_idx]],
        color=loss_colors["val"],
        s=42,
        zorder=3,
        label=f"best val_loss={val_loss[best_loss_idx]:.4f}",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        train_metric,
        label=f"train_{metric_name}",
        color=metric_colors["train"],
        linewidth=2.4,
    )
    axes[1].plot(
        epochs,
        val_metric,
        label=f"val_{metric_name}",
        color=metric_colors["val"],
        linewidth=2.4,
    )
    axes[1].scatter(
        [epochs[best_metric_idx]],
        [val_metric[best_metric_idx]],
        color=metric_colors["val"],
        s=42,
        zorder=3,
        label=f"best val_{metric_name}={val_metric[best_metric_idx]:.4f}",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(metric_name)
    axes[1].set_title(metric_name)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
