#!/usr/bin/env python3
"""Audit the B_3 boundary-slice rule across Garside lengths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.analyze_b3_boundary_rule import nonzero_column_for_unit_token, slice_token_at_relative_degree  # noqa: E402
from interp.generate_b3_dataset import generate_batch, iter_ranges, total_examples  # noqa: E402
from interp.train_b3_transformer import atomic_json_dump, resolve_device, set_seed  # noqa: E402


def deterministic_sample_ids(total: int, count: int, seed: int) -> torch.Tensor:
    if count >= total:
        return torch.arange(total, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(total, generator=generator)[:count].sort().values.to(torch.long)


def audit_length(
    length: int,
    *,
    batch_size: int,
    exhaustive_limit: int,
    sample_limit: int,
    seed: int,
    device: torch.device,
) -> dict:
    total = total_examples(length)
    exhaustive = total <= exhaustive_limit
    sample_ids = None if exhaustive else deterministic_sample_ids(total, min(sample_limit, total), seed + length)
    count = total if exhaustive else int(sample_ids.numel())

    leading_valid = 0
    trailing_valid = 0
    leading_correct = 0
    trailing_correct = 0
    leading_counts = torch.zeros(16, 2, dtype=torch.long)
    trailing_counts = torch.zeros(16, 2, dtype=torch.long)
    trailing_degree = 2 * length

    if exhaustive:
        range_iter = iter_ranges(0, total, batch_size)
    else:
        range_iter = ((start, min(batch_size, count - start)) for start in range(0, count, batch_size))

    for start, n in range_iter:
        if exhaustive:
            ids = torch.arange(start, start + n, dtype=torch.long)
        else:
            ids = sample_ids[start : start + n]
        batch = generate_batch(ids, length=length, device=device)
        labels = batch["label"].long()
        leading = slice_token_at_relative_degree(batch["matrix_bits"], 0)
        trailing = slice_token_at_relative_degree(batch["matrix_bits"], trailing_degree)
        leading_col = nonzero_column_for_unit_token(leading)
        trailing_col = nonzero_column_for_unit_token(trailing)
        leading_pred = (leading_col == 0).to(torch.long)
        trailing_pred = (trailing_col == 1).to(torch.long)
        leading_valid += int((leading_col >= 0).sum().item())
        trailing_valid += int((trailing_col >= 0).sum().item())
        leading_correct += int((leading_pred == labels).sum().item())
        trailing_correct += int((trailing_pred == labels).sum().item())
        for token in range(16):
            mask = leading == token
            if bool(mask.any()):
                leading_counts[token] += torch.bincount(labels[mask], minlength=2)
            mask = trailing == token
            if bool(mask.any()):
                trailing_counts[token] += torch.bincount(labels[mask], minlength=2)

    return {
        "length": length,
        "total_examples": total,
        "checked_examples": count,
        "mode": "exhaustive" if exhaustive else "sample",
        "leading_rule_accuracy": leading_correct / max(1, count),
        "trailing_rule_accuracy": trailing_correct / max(1, count),
        "leading_unit_token_fraction": leading_valid / max(1, count),
        "trailing_unit_token_fraction": trailing_valid / max(1, count),
        "leading_token_counts": {str(i): leading_counts[i].tolist() for i in range(16) if int(leading_counts[i].sum())},
        "trailing_token_counts": {str(i): trailing_counts[i].tolist() for i in range(16) if int(trailing_counts[i].sum())},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit B_3 boundary rule across lengths.")
    parser.add_argument("--max-length", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--exhaustive-limit", type=int, default=1_048_576)
    parser.add_argument("--sample-limit", type=int, default=1_048_576)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="interp/artifacts/b3_boundary_rule_lengths/results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    results = {
        "device": str(device),
        "max_length": args.max_length,
        "exhaustive_limit": args.exhaustive_limit,
        "sample_limit": args.sample_limit,
        "label_convention": {"0": "{s_1}", "1": "{s_2}"},
        "leading_rule": "At min occupied degree, predict {s_2} iff the unique nonzero entry is in column 0.",
        "trailing_rule": "At max occupied degree, predict {s_2} iff the unique nonzero entry is in column 1.",
        "lengths": [],
    }
    for length in range(1, args.max_length + 1):
        row = audit_length(
            length,
            batch_size=args.batch_size,
            exhaustive_limit=args.exhaustive_limit,
            sample_limit=args.sample_limit,
            seed=args.seed,
            device=device,
        )
        results["lengths"].append(row)
        print(json.dumps(row, indent=2))
    atomic_json_dump(results, Path(args.out))
    print(f"results={args.out}")


if __name__ == "__main__":
    main()
