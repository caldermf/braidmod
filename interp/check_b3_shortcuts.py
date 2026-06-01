#!/usr/bin/env python3
"""Audit simple scalar shortcuts for the B_3 Burau/descent corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.b3_data import discover_shards, load_shard  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check min-degree and metadata baselines.")
    parser.add_argument("--data-dir", default="interp/data/generated/b3_l25_p2_full")
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--max-shards", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_paths = discover_shards(args.data_dir, num_shards=args.num_shards, allow_partial=args.allow_partial)
    if args.max_shards > 0:
        shard_paths = shard_paths[: args.max_shards]

    total = 0
    label_counts = torch.zeros(2, dtype=torch.long)
    factor_counts = torch.zeros(4, dtype=torch.long)
    parity_even_correct = 0
    parity_odd_correct = 0
    min_degree_counts: dict[int, list[int]] = {}

    for path in shard_paths:
        payload = load_shard(path)
        labels = payload["label"].long()
        min_degree = payload["burau_min_degree"].long()
        final_factor_id = payload["final_factor_id"].long()
        total += int(labels.numel())
        label_counts += torch.bincount(labels, minlength=2)
        factor_counts += torch.bincount(final_factor_id, minlength=4)
        parity_even_correct += int(((min_degree % 2 == 0).long() == labels).sum().item())
        parity_odd_correct += int(((min_degree % 2 == 1).long() == labels).sum().item())

        for value in torch.unique(min_degree).tolist():
            mask = min_degree == int(value)
            counts = torch.bincount(labels[mask], minlength=2)
            slot = min_degree_counts.setdefault(int(value), [0, 0])
            slot[0] += int(counts[0].item())
            slot[1] += int(counts[1].item())

    by_min_degree = {
        str(k): {
            "label_counts": v,
            "label_1_fraction": v[1] / max(1, v[0] + v[1]),
        }
        for k, v in sorted(min_degree_counts.items())
    }
    print(
        json.dumps(
            {
                "data_dir": args.data_dir,
                "shards_checked": len(shard_paths),
                "examples": total,
                "label_counts": label_counts.tolist(),
                "final_factor_counts": factor_counts.tolist(),
                "baseline_label_1_if_min_degree_even_accuracy": parity_even_correct / max(1, total),
                "baseline_label_1_if_min_degree_odd_accuracy": parity_odd_correct / max(1, total),
                "by_min_degree": by_min_degree,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
