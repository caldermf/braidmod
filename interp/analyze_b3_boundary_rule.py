#!/usr/bin/env python3
"""Audit the B_3 boundary-slice descent rule on generated shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.b3_data import discover_shards, relative_depth_for_length  # noqa: E402
from interp.train_b3_transformer import atomic_json_dump  # noqa: E402


def slice_token_at_relative_degree(matrix_bits: torch.Tensor, degree: int) -> torch.Tensor:
    bits = matrix_bits.to(torch.long)
    return (
        ((bits[:, 0] >> degree) & 1)
        | (((bits[:, 1] >> degree) & 1) << 1)
        | (((bits[:, 2] >> degree) & 1) << 2)
        | (((bits[:, 3] >> degree) & 1) << 3)
    ).to(torch.long)


def token_to_matrix(token: int) -> list[list[int]]:
    return [
        [int(token & 1), int((token >> 1) & 1)],
        [int((token >> 2) & 1), int((token >> 3) & 1)],
    ]


def nonzero_column_for_unit_token(token: torch.Tensor) -> torch.Tensor:
    """Return 0 for E00/E10, 1 for E01/E11, and -1 otherwise."""
    out = torch.full_like(token, -1)
    out[(token == 1) | (token == 4)] = 0
    out[(token == 2) | (token == 8)] = 1
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit exact B_3 boundary-slice descent rules.")
    parser.add_argument("--data-dir", default="interp/data/generated/b3_l25_p2_full")
    parser.add_argument("--out", default="interp/artifacts/b3_l25_p2_boundary_rule/results.json")
    parser.add_argument("--length", type=int, default=25)
    parser.add_argument("--num-shards", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = discover_shards(args.data_dir, num_shards=args.num_shards)
    trailing_degree = relative_depth_for_length(args.length) - 1
    leading_counts = torch.zeros(16, 2, dtype=torch.long)
    trailing_counts = torch.zeros(16, 2, dtype=torch.long)
    leading_factor_counts = torch.zeros(16, 4, dtype=torch.long)
    trailing_factor_counts = torch.zeros(16, 4, dtype=torch.long)
    leading_rule_correct = 0
    trailing_rule_correct = 0
    leading_valid = 0
    trailing_valid = 0
    total = 0

    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        labels = payload["label"].long()
        factors = payload["final_factor_id"].long()
        leading = slice_token_at_relative_degree(payload["matrix_bits"], 0)
        trailing = slice_token_at_relative_degree(payload["matrix_bits"], trailing_degree)
        total += int(labels.numel())

        leading_col = nonzero_column_for_unit_token(leading)
        trailing_col = nonzero_column_for_unit_token(trailing)
        leading_pred = (leading_col == 0).to(torch.long)
        trailing_pred = (trailing_col == 1).to(torch.long)
        leading_valid += int((leading_col >= 0).sum().item())
        trailing_valid += int((trailing_col >= 0).sum().item())
        leading_rule_correct += int((leading_pred == labels).sum().item())
        trailing_rule_correct += int((trailing_pred == labels).sum().item())

        for token in range(16):
            mask = leading == token
            if bool(mask.any()):
                leading_counts[token] += torch.bincount(labels[mask], minlength=2)
                leading_factor_counts[token] += torch.bincount(factors[mask], minlength=4)
            mask = trailing == token
            if bool(mask.any()):
                trailing_counts[token] += torch.bincount(labels[mask], minlength=2)
                trailing_factor_counts[token] += torch.bincount(factors[mask], minlength=4)

    result = {
        "data_dir": args.data_dir,
        "length": args.length,
        "num_shards": len(paths),
        "examples": total,
        "label_convention": {"0": "{s_1}", "1": "{s_2}"},
        "token_bit_convention": {
            "bit0": "entry (0,0)",
            "bit1": "entry (0,1)",
            "bit2": "entry (1,0)",
            "bit3": "entry (1,1)",
        },
        "leading_rule": {
            "description": "At the minimum occupied degree, predict {s_2} iff the unique nonzero entry is in column 0.",
            "valid_unit_token_fraction": leading_valid / max(1, total),
            "accuracy": leading_rule_correct / max(1, total),
        },
        "trailing_rule": {
            "description": "At the maximum occupied degree, predict {s_2} iff the unique nonzero entry is in column 1.",
            "valid_unit_token_fraction": trailing_valid / max(1, total),
            "accuracy": trailing_rule_correct / max(1, total),
        },
        "leading_tokens": {
            str(token): {
                "matrix": token_to_matrix(token),
                "label_counts": leading_counts[token].tolist(),
                "final_factor_counts": leading_factor_counts[token].tolist(),
            }
            for token in range(16)
            if int(leading_counts[token].sum()) > 0
        },
        "trailing_tokens": {
            str(token): {
                "matrix": token_to_matrix(token),
                "label_counts": trailing_counts[token].tolist(),
                "final_factor_counts": trailing_factor_counts[token].tolist(),
            }
            for token in range(16)
            if int(trailing_counts[token].sum()) > 0
        },
    }
    atomic_json_dump(result, Path(args.out))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
