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


def _metric_label(metric_name: str) -> str:
    if metric_name == "factor_acc":
        return "Factor accuracy"
    return metric_name.replace("_", " ")


def _plot_model_column(ax_loss, ax_metric, title: str, log_path: Path, loss_colors, metric_colors) -> None:
    epochs, train_loss, val_loss, train_metric, val_metric, metric_name = parse_log(
        log_path.read_text(encoding="utf-8")
    )

    best_loss_idx = min(range(len(val_loss)), key=val_loss.__getitem__)
    best_metric_idx = max(range(len(val_metric)), key=val_metric.__getitem__)

    ax_loss.plot(epochs, train_loss, color=loss_colors["train"], linewidth=2.0, label="train loss")
    ax_loss.plot(epochs, val_loss, color=loss_colors["val"], linewidth=2.4, label="validation loss")
    ax_loss.scatter(
        [epochs[best_loss_idx]],
        [val_loss[best_loss_idx]],
        color=loss_colors["val"],
        s=34,
        zorder=3,
    )
    ax_loss.set_title(title)
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(frameon=False, fontsize=9)
    ax_loss.text(
        0.98,
        0.97,
        f"best val loss = {val_loss[best_loss_idx]:.4f}",
        transform=ax_loss.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=loss_colors["val"],
    )

    ax_metric.plot(epochs, train_metric, color=metric_colors["train"], linewidth=2.0, label=f"train {_metric_label(metric_name).lower()}")
    ax_metric.plot(epochs, val_metric, color=metric_colors["val"], linewidth=2.4, label=f"validation {_metric_label(metric_name).lower()}")
    ax_metric.scatter(
        [epochs[best_metric_idx]],
        [val_metric[best_metric_idx]],
        color=metric_colors["val"],
        s=34,
        zorder=3,
    )
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel(_metric_label(metric_name))
    ax_metric.grid(alpha=0.3)
    ax_metric.legend(frameon=False, fontsize=9)
    ax_metric.text(
        0.98,
        0.03,
        f"best val accuracy = {val_metric[best_metric_idx]:.4f}",
        transform=ax_metric.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=metric_colors["val"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a clean side-by-side training story for the public MLP baseline and transformer."
    )
    parser.add_argument("--mlp-log", required=True, help="Path to the public MLP training log")
    parser.add_argument("--transformer-log", required=True, help="Path to the public transformer training log")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument(
        "--title",
        default="MLP baseline vs hierarchical transformer",
        help="Figure title",
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 7.8), sharex="col")

    mlp_loss_colors = {"train": "#4d6a9a", "val": "#c06c84"}
    mlp_metric_colors = {"train": "#3f7f6b", "val": "#de6b48"}
    xfmr_loss_colors = {"train": "#355070", "val": "#b56576"}
    xfmr_metric_colors = {"train": "#2a9d8f", "val": "#e76f51"}

    _plot_model_column(
        axes[0, 0],
        axes[1, 0],
        "Original MLP baseline",
        Path(args.mlp_log),
        mlp_loss_colors,
        mlp_metric_colors,
    )
    _plot_model_column(
        axes[0, 1],
        axes[1, 1],
        "Hierarchical transformer",
        Path(args.transformer_log),
        xfmr_loss_colors,
        xfmr_metric_colors,
    )

    fig.suptitle(args.title)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
