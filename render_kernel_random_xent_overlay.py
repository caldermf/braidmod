#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from plot_prefix_confusion import PERM_TO_ID
from predict_garside_mlp import build_model, resolve_device
from reservoir_search_braidmod import _build_garside_tables, _right_multiply_simple_batch
from render_smoothed_xent_suite import running_average


def maybe_smooth(values: List[float], mode: str, window: int) -> List[float]:
    if mode == "avg5":
        return running_average(values, window=window)
    if mode == "raw":
        return values
    raise ValueError(f"Unsupported mode: {mode}")


def load_random_series(suite_dir: Path, mode: str, window: int) -> List[Tuple[str, List[int], List[float]]]:
    series = []
    for json_path in sorted(suite_dir.glob("random_*_confusion.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        progression = payload["progression"]
        prefix_lens = [int(item["prefix_len"]) for item in progression]
        xent = [float(item["target_cross_entropy"]) for item in progression]
        label = json_path.stem.replace("_confusion", "").replace("_", " ")
        series.append((label, prefix_lens, maybe_smooth(xent, mode=mode, window=window)))
    if not series:
        raise ValueError(f"No random_*_confusion.json files found in {suite_dir}")
    return series


def build_kernel_series(
    search_json: Path,
    checkpoint_path: str,
    device: str,
    mode: str,
    window: int,
) -> List[Tuple[str, List[int], List[float]]]:
    resolved_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    model = build_model(checkpoint, resolved_device)
    depth = int(checkpoint["D"])
    p = int(checkpoint["p"])
    tables = _build_garside_tables(p=p, device=resolved_device)

    payload = json.loads(search_json.read_text(encoding="utf-8"))
    kernel_hits = payload.get("kernel_hits", [])
    if not kernel_hits:
        raise ValueError(f"No kernel hits found in {search_json}")

    series = []
    with torch.no_grad():
        for idx, hit in enumerate(kernel_hits, start=1):
            factor_ids = [int(v) for v in hit.get("factor_ids", [])]
            if not factor_ids:
                factor_ids = [PERM_TO_ID[tuple(int(v) for v in factor)] for factor in hit["gnf_factors"]]

            current_tensor = torch.zeros(1, depth, 3, 3, dtype=torch.int16, device=resolved_device)
            current_tensor[0, 0, 0, 0] = 1
            current_tensor[0, 0, 1, 1] = 1
            current_tensor[0, 0, 2, 2] = 1
            current_min_degree = torch.zeros(1, dtype=torch.int32, device=resolved_device)

            prefix_tensors = []
            prefix_min_degrees = []
            for factor_id in factor_ids:
                suffix_ids = torch.tensor([factor_id], dtype=torch.long, device=resolved_device)
                current_tensor, current_min_degree = _right_multiply_simple_batch(
                    current_tensor,
                    current_min_degree,
                    suffix_ids,
                    tables.simple_shift_mats,
                    p,
                )
                prefix_tensors.append(current_tensor[0].clone())
                prefix_min_degrees.append(current_min_degree[0].clone())

            batch_tensors = torch.stack(prefix_tensors, dim=0).to(dtype=torch.long)
            batch_min_degrees = torch.stack(prefix_min_degrees, dim=0).to(dtype=torch.float32)
            targets = torch.tensor(factor_ids, dtype=torch.long, device=resolved_device)
            factor_logits, _ = model(batch_tensors, min_degree=batch_min_degrees)
            xent = F.cross_entropy(factor_logits, targets, reduction="none").tolist()
            prefix_lens = list(range(1, len(factor_ids) + 1))

            kernel_type = hit.get("kernel_type", "kernel")
            label = f"kernel {idx} {kernel_type}"
            series.append((label, prefix_lens, maybe_smooth(xent, mode=mode, window=window)))
    return series


def plot_overlay(
    kernel_series: Sequence[Tuple[str, List[int], List[float]]],
    random_series: Sequence[Tuple[str, List[int], List[float]]],
    out_png: Path,
    max_length: int,
    mode: str,
    window: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ymax = 1.0

    kernel_colors = list(plt.cm.tab10.colors)
    random_colors = list(plt.cm.Set2.colors)

    for idx, (label, prefix_lens, avg) in enumerate(kernel_series):
        ymax = max(ymax, max(avg) if avg else 1.0)
        ax.plot(
            prefix_lens,
            avg,
            label=label,
            linewidth=2.1,
            alpha=0.95,
            color=kernel_colors[idx % len(kernel_colors)],
        )

    for idx, (label, prefix_lens, avg) in enumerate(random_series):
        ymax = max(ymax, max(avg) if avg else 1.0)
        ax.plot(
            prefix_lens,
            avg,
            label=label,
            linewidth=1.8,
            alpha=0.9,
            linestyle="--",
            color=random_colors[idx % len(random_colors)],
        )

    ax.set_xlabel("Garside Prefix Length")
    if mode == "avg5":
        ylabel = f"Target Cross-Entropy Avg{window}"
        title = "Kernel Hits vs Random Length-54 Elements: Target Cross-Entropy Avg5"
    else:
        ylabel = "Target Cross-Entropy"
        title = "Kernel Hits vs Random Length-54 Elements: Target Cross-Entropy"
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(1, max_length)
    ax.set_ylim(0.0, 1.05 * ymax)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay kernel-hit avg5 target-cross-entropy curves with random confusion-suite curves."
    )
    parser.add_argument("--search-json", required=True, help="Reservoir search JSON containing kernel_hits")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint used to score kernel hits")
    parser.add_argument("--suite-dir", required=True, help="Directory containing random_*_confusion.json files")
    parser.add_argument("--out-png", required=True, help="Output PNG path")
    parser.add_argument("--device", default="auto", help="Device for scoring kernel hits")
    parser.add_argument("--mode", choices=("avg5", "raw"), default="avg5", help="Plot raw xent or running-average xent")
    parser.add_argument("--window", type=int, default=5, help="Running-average window")
    parser.add_argument("--max-length", type=int, default=60, help="X-axis upper bound")
    args = parser.parse_args()

    kernel_series = build_kernel_series(
        search_json=Path(args.search_json),
        checkpoint_path=args.checkpoint,
        device=args.device,
        mode=args.mode,
        window=args.window,
    )
    random_series = load_random_series(Path(args.suite_dir), mode=args.mode, window=args.window)
    plot_overlay(
        kernel_series=kernel_series,
        random_series=random_series,
        out_png=Path(args.out_png),
        max_length=args.max_length,
        mode=args.mode,
        window=args.window,
    )
    print(
        json.dumps(
            {
                "num_kernel_series": len(kernel_series),
                "num_random_series": len(random_series),
                "out_png": args.out_png,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
