#!/usr/bin/env python3
"""Audit column-support filtration rules for the B_4 descent task."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.b4_data import B4ShardBatchIterable, discover_shards  # noqa: E402
from interp.generate_b4_dataset import PROPER_FACTOR_PERMS, RIGHT_DESC_MASK, SIMPLE_MATS  # noqa: E402
from interp.mine_b4_boundary_rules import (  # noqa: E402
    gather_backward,
    gather_forward,
    mask_to_bits,
    pack_token_bits,
    precompute_token_attrs,
    support_features,
)


def collect_examples(
    *,
    data_dir: Path,
    num_shards: int,
    split: str,
    max_examples: int,
    batch_size: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    dataset = B4ShardBatchIterable(
        discover_shards(data_dir, num_shards=num_shards),
        split=split,
        batch_size=batch_size,
        seed=seed,
        shuffle_shards=True,
        shuffle_rows=True,
        max_examples=max_examples,
    )
    pieces: dict[str, list[torch.Tensor]] = defaultdict(list)
    for batch in dataset:
        for key in ("tokens", "descent_mask", "final_factor_id", "min_degree", "max_degree"):
            pieces[key].append(batch[key])
    if not pieces:
        raise RuntimeError("no examples loaded")
    return {key: torch.cat(values, dim=0)[:max_examples] for key, values in pieces.items()}


def metrics_from_mask_prediction(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> dict:
    pred_mask = pred_mask.cpu().to(torch.long)
    true_mask = true_mask.cpu().to(torch.long)
    pred_bits = mask_to_bits(pred_mask)
    true_bits = mask_to_bits(true_mask)
    eq_bits = pred_bits.eq(true_bits)
    false_pos = ((pred_bits == 1) & (true_bits == 0)).sum(dim=0)
    false_neg = ((pred_bits == 0) & (true_bits == 1)).sum(dim=0)
    positives = true_bits.sum(dim=0).clamp_min(1)
    negatives = (true_bits.shape[0] - true_bits.sum(dim=0)).clamp_min(1)
    return {
        "exact_accuracy": float((pred_mask == true_mask).float().mean().item()),
        "bit_accuracy": float(eq_bits.float().mean().item()),
        "per_label_accuracy": [float(x) for x in eq_bits.float().mean(dim=0).tolist()],
        "false_positive_rate_by_label": [float(x) for x in (false_pos / negatives).tolist()],
        "false_negative_rate_by_label": [float(x) for x in (false_neg / positives).tolist()],
        "pred_mask_counts": torch.bincount(pred_mask, minlength=8).tolist(),
    }


def bitwise_or_columns(col_masks: torch.Tensor, radius: int) -> torch.Tensor:
    out = torch.zeros(col_masks.shape[0], dtype=torch.long)
    for k in range(radius + 1):
        out |= col_masks[:, k].to(torch.long)
    return out


def bitwise_and_columns(col_masks: torch.Tensor, radius: int) -> torch.Tensor:
    out = torch.full((col_masks.shape[0],), 7, dtype=torch.long)
    for k in range(radius + 1):
        out &= col_masks[:, k].to(torch.long)
    return out


def explicit_column_rules(batch: dict[str, torch.Tensor], attrs: dict[str, torch.Tensor], radius: int) -> dict:
    tokens = batch["tokens"]
    true = batch["descent_mask"]
    feats = support_features(tokens)
    lead = gather_forward(tokens, feats["first"], radius)
    trail = gather_backward(tokens, feats["last"], radius)
    lead_cols = attrs["col_mask"][lead]
    trail_cols = attrs["col_mask"][trail]
    lead_rows = attrs["row_mask"][lead]
    trail_rows = attrs["row_mask"][trail]

    rules = {}
    for k in range(radius + 1):
        rules[f"leading_col_k{k}"] = lead_cols[:, k]
        rules[f"trailing_col_k{k}"] = trail_cols[:, k]
        rules[f"leading_row_k{k}"] = lead_rows[:, k]
        rules[f"trailing_row_k{k}"] = trail_rows[:, k]
    for r in range(radius + 1):
        lead_or = bitwise_or_columns(lead_cols, r)
        trail_or = bitwise_or_columns(trail_cols, r)
        lead_and = bitwise_and_columns(lead_cols, r)
        trail_and = bitwise_and_columns(trail_cols, r)
        rules[f"leading_col_or_r{r}"] = lead_or
        rules[f"trailing_col_or_r{r}"] = trail_or
        rules[f"both_col_or_r{r}"] = lead_or | trail_or
        rules[f"leading_col_and_r{r}"] = lead_and
        rules[f"trailing_col_and_r{r}"] = trail_and
        rules[f"both_col_and_r{r}"] = lead_and & trail_and

    return {name: metrics_from_mask_prediction(pred, true) for name, pred in sorted(rules.items())}


def rows_as_keys(x: torch.Tensor) -> list[tuple[int, ...]]:
    x = x.cpu().to(torch.long)
    if x.ndim == 1:
        return [(int(v),) for v in x.tolist()]
    return [tuple(int(v) for v in row.tolist()) for row in x]


def feature_purity(feature: torch.Tensor, true_mask: torch.Tensor, *, top_conflicts: int) -> dict:
    keys = rows_as_keys(feature)
    masks = [int(x) for x in true_mask.cpu().to(torch.long).tolist()]
    table: dict[tuple[int, ...], Counter] = defaultdict(Counter)
    for key, mask in zip(keys, masks, strict=True):
        table[key][mask] += 1

    total = len(keys)
    pure_keys = 0
    examples_on_pure_keys = 0
    majority_correct = 0
    conflicts = []
    for key, counter in table.items():
        count = sum(counter.values())
        majority_mask, majority_count = counter.most_common(1)[0]
        majority_correct += majority_count
        if len(counter) == 1:
            pure_keys += 1
            examples_on_pure_keys += count
        else:
            conflicts.append(
                {
                    "key": list(key),
                    "count": count,
                    "majority_mask": int(majority_mask),
                    "majority_fraction": majority_count / count,
                    "mask_counts": {str(int(mask)): int(c) for mask, c in sorted(counter.items())},
                }
            )
    conflicts.sort(key=lambda item: (item["count"], -item["majority_fraction"]), reverse=True)
    return {
        "num_keys": len(table),
        "pure_keys": pure_keys,
        "pure_key_fraction": pure_keys / max(1, len(table)),
        "examples_on_pure_keys": examples_on_pure_keys,
        "examples_on_pure_keys_fraction": examples_on_pure_keys / max(1, total),
        "majority_accuracy_on_same_sample": majority_correct / max(1, total),
        "top_conflicts": conflicts[:top_conflicts],
    }


def column_feature_tensors(batch: dict[str, torch.Tensor], attrs: dict[str, torch.Tensor], radius: int) -> dict[str, torch.Tensor]:
    tokens = batch["tokens"]
    feats = support_features(tokens)
    lead = gather_forward(tokens, feats["first"], radius)
    trail = gather_backward(tokens, feats["last"], radius)
    lead_cols = attrs["col_mask"][lead]
    trail_cols = attrs["col_mask"][trail]
    lead_kernels = attrs["kernel"][lead]
    trail_kernels = attrs["kernel"][trail]
    return {
        "lead_col_r2": lead_cols[:, :3],
        "trail_col_r2": trail_cols[:, :3],
        "both_col_r2": torch.cat([lead_cols[:, :3], trail_cols[:, :3]], dim=1),
        "lead_col_r4": lead_cols[:, :5],
        "trail_col_r4": trail_cols[:, :5],
        "both_kernel_r2": torch.cat([lead_kernels[:, :3], trail_kernels[:, :3]], dim=1),
    }


def simple_factor_column_table(attrs: dict[str, torch.Tensor]) -> list[dict]:
    rows = []
    for factor_id, perm in enumerate(PROPER_FACTOR_PERMS):
        mat = SIMPLE_MATS[factor_id].to(torch.bool)
        tokens = pack_token_bits(mat.view(mat.shape[0], 9)).cpu()
        support = tokens.ne(0)
        first = int(support.to(torch.long).argmax().item())
        last = int(tokens.shape[0] - 1 - support.flip(dims=[0]).to(torch.long).argmax().item())
        lead = int(tokens[first].item())
        trail = int(tokens[last].item())
        rmask = int(RIGHT_DESC_MASK[factor_id].item())
        trail_col = int(attrs["col_mask"][trail].item())
        rows.append(
            {
                "factor_id": int(factor_id),
                "perm": list(perm),
                "right_descent_mask": rmask,
                "trailing_col_mask": trail_col,
                "trailing_col_is_subset": bool((trail_col & rmask) == trail_col),
                "trailing_col_equals_right_descent": bool(trail_col == rmask),
                "leading_col_mask": int(attrs["col_mask"][lead].item()),
                "min_degree": first,
                "max_degree": last,
                "leading_token": lead,
                "trailing_token": trail,
            }
        )
    return rows


def atomic_json_dump(payload: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit explicit B_4 column-support descent rules.")
    parser.add_argument("--data-dir", default="interp/data/generated/b4_l25_p2_n16777216")
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--split", default="test")
    parser.add_argument("--examples", type=int, default=262_144)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--top-conflicts", type=int, default=20)
    parser.add_argument("--out", default="interp/artifacts/b4_l25_p2_column_filtration/results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attrs = precompute_token_attrs()
    batch = collect_examples(
        data_dir=Path(args.data_dir),
        num_shards=args.num_shards,
        split=args.split,
        max_examples=args.examples,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    features = column_feature_tensors(batch, attrs, args.radius)
    result = {
        "config": vars(args),
        "n": int(batch["descent_mask"].numel()),
        "mask_counts": torch.bincount(batch["descent_mask"].to(torch.long), minlength=8).tolist(),
        "simple_factor_column_table": simple_factor_column_table(attrs),
        "explicit_column_rules": explicit_column_rules(batch, attrs, args.radius),
        "feature_purity": {
            name: feature_purity(feature, batch["descent_mask"], top_conflicts=args.top_conflicts)
            for name, feature in features.items()
        },
    }
    atomic_json_dump(result, Path(args.out))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
