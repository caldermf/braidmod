#!/usr/bin/env python3
import argparse
import os
import tempfile
from pathlib import Path

cache_root = Path(tempfile.gettempdir()) / "braidmod-cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_training_curves import parse_log


def parse_labeled_log(spec: str):
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=PATH, got: {spec}")
    label, path = spec.split("=", 1)
    if not label or not path:
        raise ValueError(f"Expected LABEL=PATH, got: {spec}")
    return label, Path(path)


def main():
    parser = argparse.ArgumentParser(
        description="Overlay train/val curves from multiple training logs."
    )
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        help="Run spec in the form LABEL=/abs/path/to/train.log. Repeat for multiple runs.",
    )
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--title", default="Training Comparison", help="Figure title")
    parser.add_argument(
        "--include-train",
        action="store_true",
        help="Also plot train curves for each run. Defaults to validation-only for readability.",
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    cmap = plt.get_cmap("tab10")
    metric_name = None

    for idx, spec in enumerate(args.log):
        label, path = parse_labeled_log(spec)
        epochs, train_loss, val_loss, train_metric, val_metric, current_metric_name = parse_log(
            path.read_text(encoding="utf-8")
        )
        if metric_name is None:
            metric_name = current_metric_name
        elif metric_name != current_metric_name:
            raise ValueError(f"Mismatched metric names: {metric_name} vs {current_metric_name}")

        color = cmap(idx % 10)
        axes[0].plot(epochs, val_loss, label=f"{label} val", color=color, linewidth=2.4)
        axes[1].plot(epochs, val_metric, label=f"{label} val", color=color, linewidth=2.4)

        if args.include_train:
            axes[0].plot(epochs, train_loss, color=color, linewidth=1.5, alpha=0.35, linestyle="--")
            axes[1].plot(epochs, train_metric, color=color, linewidth=1.5, alpha=0.35, linestyle="--")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Validation Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(metric_name or "metric")
    axes[1].set_title(f"Validation {metric_name or 'metric'}")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
