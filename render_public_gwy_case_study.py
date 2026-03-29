#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from render_smoothed_xent_suite import cumulative_average, running_average


Series = Tuple[List[int], List[float]]


def _load_progression(json_path: Path) -> List[dict]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return payload["progression"]


def _prefix_and_metric(progression: Sequence[dict], metric_key: str) -> Series:
    prefix_lens = [int(item["prefix_len"]) for item in progression]
    values = [float(item[metric_key]) for item in progression]
    return prefix_lens, values


def _average_series(series_group: Iterable[Series]) -> Series:
    buckets: Dict[int, List[float]] = {}
    for prefix_lens, values in series_group:
        for prefix_len, value in zip(prefix_lens, values):
            buckets.setdefault(prefix_len, []).append(value)
    prefix_lens = sorted(buckets)
    averaged = [sum(buckets[p]) / float(len(buckets[p])) for p in prefix_lens]
    return prefix_lens, averaged


def _render_two_line_plot(
    lhs: Series,
    rhs: Series,
    out_png: Path,
    title: str,
    ylabel: str,
    lhs_label: str,
    rhs_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    lhs_x, lhs_y = lhs
    rhs_x, rhs_y = rhs
    ymax = max(1.0, max(lhs_y) if lhs_y else 1.0, max(rhs_y) if rhs_y else 1.0)

    ax.plot(lhs_x, lhs_y, color="#b23a48", linewidth=3.0, label=lhs_label)
    ax.plot(rhs_x, rhs_y, color="#3a86a8", linewidth=3.0, label=rhs_label)
    ax.set_xlabel("Prefix length in Garside factors")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(1, max(lhs_x[-1] if lhs_x else 1, rhs_x[-1] if rhs_x else 1))
    ax.set_ylim(0.0, 1.05 * ymax)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the public GWY kernel-element case-study figures from saved confusion JSON files."
    )
    parser.add_argument("--suite-dir", required=True, help="Directory containing the saved GWY and random confusion JSON files")
    parser.add_argument("--out-dir", required=True, help="Output directory for the rendered PNGs")
    parser.add_argument("--window", type=int, default=5, help="Running-average window for the smoothed target-cross-entropy plot")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir)
    out_dir = Path(args.out_dir)

    gwy_progression = _load_progression(suite_dir / "geordie_kernel_confusion.json")
    random_progressions = [
        _load_progression(path) for path in sorted(suite_dir.glob("random_*_confusion.json"))
    ]
    if not random_progressions:
        raise ValueError(f"No random_*_confusion.json files found in {suite_dir}")

    gwy_prefixes, gwy_xent = _prefix_and_metric(gwy_progression, "target_cross_entropy")
    gwy_entropy = _prefix_and_metric(gwy_progression, "entropy_confusion_score")

    random_xent_series = [_prefix_and_metric(prog, "target_cross_entropy") for prog in random_progressions]
    random_entropy_series = [_prefix_and_metric(prog, "entropy_confusion_score") for prog in random_progressions]

    gwy_cumavg = (gwy_prefixes, cumulative_average(gwy_xent))
    random_cumavg = _average_series(
        (prefix_lens, cumulative_average(values)) for prefix_lens, values in random_xent_series
    )
    _render_two_line_plot(
        gwy_cumavg,
        random_cumavg,
        out_dir / "gwy_kernel_element_vs_random_cumulative_target_cross_entropy.png",
        "GWY kernel element vs random braids: cumulative target cross-entropy",
        "Average target cross-entropy so far",
        "GWY kernel element",
        f"mean of {len(random_progressions)} random braids",
    )

    gwy_running = (gwy_prefixes, running_average(gwy_xent, window=args.window))
    random_running = _average_series(
        (prefix_lens, running_average(values, window=args.window)) for prefix_lens, values in random_xent_series
    )
    _render_two_line_plot(
        gwy_running,
        random_running,
        out_dir / f"gwy_kernel_element_vs_random_target_cross_entropy_running_average_{args.window}_steps.png",
        f"GWY kernel element vs random braids: target cross-entropy ({args.window}-step running average)",
        f"Target cross-entropy ({args.window}-step running average)",
        "GWY kernel element",
        f"mean of {len(random_progressions)} random braids",
    )

    random_entropy = _average_series(random_entropy_series)
    _render_two_line_plot(
        gwy_entropy,
        random_entropy,
        out_dir / "gwy_kernel_element_vs_random_entropy_confusion.png",
        "GWY kernel element vs random braids: entropy confusion",
        "Entropy confusion",
        "GWY kernel element",
        f"mean of {len(random_progressions)} random braids",
    )

    print(
        json.dumps(
            {
                "written": [
                    str(out_dir / "gwy_kernel_element_vs_random_cumulative_target_cross_entropy.png"),
                    str(out_dir / f"gwy_kernel_element_vs_random_target_cross_entropy_running_average_{args.window}_steps.png"),
                    str(out_dir / "gwy_kernel_element_vs_random_entropy_confusion.png"),
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
