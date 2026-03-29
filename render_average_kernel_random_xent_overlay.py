#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from render_kernel_random_xent_overlay import build_kernel_series, load_random_series
from render_kernel_random_xent_overlay import smoothing_phrase


Series = Tuple[str, List[int], List[float]]


def average_series(series_group: Sequence[Series]) -> Tuple[List[int], List[float]]:
    if not series_group:
        raise ValueError("Need at least one series to average")

    buckets: Dict[int, List[float]] = {}
    for _, prefix_lens, values in series_group:
        for prefix_len, value in zip(prefix_lens, values):
            buckets.setdefault(int(prefix_len), []).append(float(value))

    prefix_lens = sorted(buckets)
    averaged = [sum(buckets[prefix_len]) / float(len(buckets[prefix_len])) for prefix_len in prefix_lens]
    return prefix_lens, averaged


def plot_average_overlay(
    kernel_avg: Tuple[List[int], List[float]],
    random_avg: Tuple[List[int], List[float]],
    out_png: Path,
    max_length: int,
    mode: str,
    window: int,
    num_kernels: int,
    num_randoms: int,
    title: str | None = None,
    ylabel: str | None = None,
    kernel_label: str | None = None,
    random_label: str | None = None,
) -> None:
    kernel_prefix_lens, kernel_values = kernel_avg
    random_prefix_lens, random_values = random_avg

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ymax = max(
        1.0,
        max(kernel_values) if kernel_values else 1.0,
        max(random_values) if random_values else 1.0,
    )

    ax.plot(
        kernel_prefix_lens,
        kernel_values,
        color="#b23a48",
        linewidth=3.0,
        label=kernel_label or f"mean of first {num_kernels} known kernel elements",
    )
    ax.plot(
        random_prefix_lens,
        random_values,
        color="#3a86a8",
        linewidth=3.0,
        label=random_label or f"mean of {num_randoms} random braids",
    )

    ax.set_xlabel("Prefix length in Garside factors")
    if ylabel is None:
        if mode == "avg5":
            ylabel = f"Target cross-entropy ({smoothing_phrase(mode, window)})"
        else:
            ylabel = "Target cross-entropy"
    if title is None:
        if mode == "avg5":
            title = f"Known kernel elements vs random braids: mean target cross-entropy ({smoothing_phrase(mode, window)})"
        else:
            title = "Known kernel elements vs random braids: mean target cross-entropy"
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(1, max_length)
    ax.set_ylim(0.0, 1.05 * ymax)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a two-line public overlay: mean known-kernel curve vs mean random-control curve."
    )
    parser.add_argument("--search-json", required=True, help="Reservoir search JSON containing kernel_hits")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint used to score kernel hits")
    parser.add_argument("--suite-dir", required=True, help="Directory containing random_*_confusion.json files")
    parser.add_argument("--out-png", required=True, help="Output PNG path")
    parser.add_argument("--device", default="auto", help="Device for scoring kernel hits")
    parser.add_argument("--mode", choices=("avg5", "raw"), default="avg5", help="Plot raw target cross-entropy or a running average")
    parser.add_argument("--window", type=int, default=5, help="Running-average window")
    parser.add_argument("--max-length", type=int, default=60, help="X-axis upper bound")
    parser.add_argument(
        "--num-kernels",
        type=int,
        default=5,
        help="Number of kernel-hit series to average from the front of the saved kernel-hit list",
    )
    parser.add_argument("--title", help="Optional plot title override")
    parser.add_argument("--ylabel", help="Optional y-axis label override")
    parser.add_argument("--kernel-label", help="Optional legend label for the kernel mean curve")
    parser.add_argument("--random-label", help="Optional legend label for the random mean curve")
    args = parser.parse_args()

    kernel_series = build_kernel_series(
        search_json=Path(args.search_json),
        checkpoint_path=args.checkpoint,
        device=args.device,
        mode=args.mode,
        window=args.window,
    )
    if len(kernel_series) < args.num_kernels:
        raise ValueError(
            f"Requested {args.num_kernels} kernel series, but only found {len(kernel_series)}"
        )
    kernel_series = kernel_series[: args.num_kernels]

    random_series = load_random_series(Path(args.suite_dir), mode=args.mode, window=args.window)
    kernel_avg = average_series(kernel_series)
    random_avg = average_series(random_series)
    plot_average_overlay(
        kernel_avg=kernel_avg,
        random_avg=random_avg,
        out_png=Path(args.out_png),
        max_length=args.max_length,
        mode=args.mode,
        window=args.window,
        num_kernels=len(kernel_series),
        num_randoms=len(random_series),
        title=args.title,
        ylabel=args.ylabel,
        kernel_label=args.kernel_label,
        random_label=args.random_label,
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
