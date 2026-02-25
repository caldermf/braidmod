#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


LINE_RE = re.compile(
    r"epoch=(\d+)\s+train_loss=([0-9.]+)\s+train_acc=([0-9.]+)\s+val_loss=([0-9.]+)\s+val_acc=([0-9.]+)"
)


def parse_log(text: str):
    epochs = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for line in text.splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        epochs.append(int(m.group(1)))
        train_loss.append(float(m.group(2)))
        train_acc.append(float(m.group(3)))
        val_loss.append(float(m.group(4)))
        val_acc.append(float(m.group(5)))

    if not epochs:
        raise ValueError("No epoch lines found in log.")
    return epochs, train_loss, val_loss, train_acc, val_acc


def main():
    parser = argparse.ArgumentParser(description="Plot train/val loss and accuracy curves from training logs.")
    parser.add_argument("--log", required=True, help="Path to log file containing epoch=... lines")
    parser.add_argument("--out", default="training_curves.png", help="Output PNG path")
    parser.add_argument("--title", default="Training Curves", help="Figure title")
    args = parser.parse_args()

    log_text = Path(args.log).read_text(encoding="utf-8")
    epochs, train_loss, val_loss, train_acc, val_acc = parse_log(log_text)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train_acc")
    axes[1].plot(epochs, val_acc, label="val_acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
