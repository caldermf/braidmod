#!/usr/bin/env python3
"""Mine algebraic boundary/frontier rules for the B_4 Burau descent task.

This script is deliberately not a neural-net experiment.  It looks for small
finite invariants of the reduced Burau matrix over F_2[v] that predict the
right descent set of the final Garside factor.

The most important family is the right-division frontier test.  Since
``s_i`` is in the final right descent set exactly when the positive braid is
right-divisible by the Artin generator sigma_i, we can multiply the Burau
matrix on the right by rho(sigma_i)^{-1} and ask how the low/high-degree
frontier changes.  Over F_2 this is not a complete theorem by itself, but it
is an algebraically natural place to search for a B_4 analogue of the B_3
boundary-column rule.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from braid_data import GarsideFactor  # noqa: E402
from interp.b4_data import B4ShardBatchIterable, discover_shards  # noqa: E402
from interp.generate_b4_dataset import (  # noqa: E402
    LEFT_DESC_MASK,
    PROPER_FACTOR_PERMS,
    RIGHT_DESC_MASK,
    SIMPLE_MATS,
)


BIT_WEIGHTS_9 = torch.tensor([1 << i for i in range(9)], dtype=torch.long)


def mask_to_bits(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(torch.long)
    return torch.stack([((mask >> i) & 1) for i in range(3)], dim=1).to(torch.long)


def support_features(tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    support = tokens.ne(0)
    any_support = support.any(dim=1)
    first = support.to(torch.long).argmax(dim=1)
    last = tokens.shape[1] - 1 - support.flip(dims=[1]).to(torch.long).argmax(dim=1)
    first = torch.where(any_support, first, torch.zeros_like(first))
    last = torch.where(any_support, last, torch.zeros_like(last))
    rows = torch.arange(tokens.shape[0], device=tokens.device)
    return {
        "first": first,
        "last": last,
        "width": last - first + 1,
        "leading_token": tokens[rows, first],
        "trailing_token": tokens[rows, last],
    }


def gather_forward(tokens: torch.Tensor, starts: torch.Tensor, radius: int) -> torch.Tensor:
    offsets = torch.arange(radius + 1, dtype=torch.long, device=tokens.device)
    idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
    valid = (idx >= 0) & (idx < tokens.shape[1])
    idx = idx.clamp(0, tokens.shape[1] - 1)
    out = tokens.gather(1, idx)
    return torch.where(valid, out, torch.zeros_like(out))


def gather_backward(tokens: torch.Tensor, starts: torch.Tensor, radius: int) -> torch.Tensor:
    offsets = torch.arange(radius + 1, dtype=torch.long, device=tokens.device)
    idx = starts.unsqueeze(1) - offsets.unsqueeze(0)
    valid = (idx >= 0) & (idx < tokens.shape[1])
    idx = idx.clamp(0, tokens.shape[1] - 1)
    out = tokens.gather(1, idx)
    return torch.where(valid, out, torch.zeros_like(out))


def matrix_rank_f2(token: int) -> int:
    rows = []
    for i in range(3):
        row = 0
        for j in range(3):
            if (token >> (3 * i + j)) & 1:
                row |= 1 << j
        rows.append(row)

    rank = 0
    pivot = 0
    for col in range(3):
        found = None
        for r in range(pivot, 3):
            if (rows[r] >> col) & 1:
                found = r
                break
        if found is None:
            continue
        rows[pivot], rows[found] = rows[found], rows[pivot]
        for r in range(3):
            if r != pivot and ((rows[r] >> col) & 1):
                rows[r] ^= rows[pivot]
        rank += 1
        pivot += 1
    return rank


def mat_vec_mul_token(token: int, vector: int) -> int:
    out = 0
    for i in range(3):
        bit = 0
        for j in range(3):
            bit ^= ((token >> (3 * i + j)) & 1) & ((vector >> j) & 1)
        out |= bit << i
    return out


def transpose_token(token: int) -> int:
    out = 0
    for i in range(3):
        for j in range(3):
            if (token >> (3 * i + j)) & 1:
                out |= 1 << (3 * j + i)
    return out


def precompute_token_attrs() -> dict[str, torch.Tensor]:
    rank = []
    row_mask = []
    col_mask = []
    diag_mask = []
    popcount = []
    image = []
    kernel = []
    rowspace = []
    for token in range(512):
        rank.append(matrix_rank_f2(token))
        rows = 0
        cols = 0
        diag = 0
        count = 0
        for i in range(3):
            for j in range(3):
                bit = (token >> (3 * i + j)) & 1
                if bit:
                    rows |= 1 << i
                    cols |= 1 << j
                    count += 1
                    if i == j:
                        diag |= 1 << i
        row_mask.append(rows)
        col_mask.append(cols)
        diag_mask.append(diag)
        popcount.append(count)
        img = 0
        ker = 0
        for vector in range(8):
            mv = mat_vec_mul_token(token, vector)
            img |= 1 << mv
            if mv == 0:
                ker |= 1 << vector
        image.append(img)
        kernel.append(ker)
        row_img = 0
        t_token = transpose_token(token)
        for vector in range(8):
            row_img |= 1 << mat_vec_mul_token(t_token, vector)
        rowspace.append(row_img)

    return {
        "rank": torch.tensor(rank, dtype=torch.long),
        "row_mask": torch.tensor(row_mask, dtype=torch.long),
        "col_mask": torch.tensor(col_mask, dtype=torch.long),
        "diag_mask": torch.tensor(diag_mask, dtype=torch.long),
        "popcount": torch.tensor(popcount, dtype=torch.long),
        "image": torch.tensor(image, dtype=torch.long),
        "kernel": torch.tensor(kernel, dtype=torch.long),
        "rowspace": torch.tensor(rowspace, dtype=torch.long),
    }


def token_bits(tokens: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(9, dtype=torch.long, device=tokens.device)
    return ((tokens.to(torch.long).unsqueeze(-1) >> shifts) & 1).to(torch.bool)


def pack_token_bits(bits: torch.Tensor) -> torch.Tensor:
    weights = BIT_WEIGHTS_9.to(device=bits.device)
    view_shape = [1] * bits.ndim
    view_shape[-1] = 9
    return (bits.to(torch.long) * weights.view(*view_shape)).sum(dim=-1)


def assert_token_range(tokens: torch.Tensor, name: str) -> None:
    if tokens.numel() == 0:
        raise RuntimeError(f"{name} unexpectedly has no elements")
    low = int(tokens.min().item())
    high = int(tokens.max().item())
    if low < 0 or high >= 512:
        raise RuntimeError(f"{name} is not a packed 3x3 F2 token: min={low}, max={high}")


def cumulative_token(band: torch.Tensor, radius: int, *, mode: str) -> torch.Tensor:
    bits = token_bits(band[:, : radius + 1])
    if mode == "or":
        out = pack_token_bits(bits.any(dim=1))
        assert_token_range(out, f"cumulative_{mode}_r{radius}")
        return out
    if mode == "xor":
        out = pack_token_bits(bits.to(torch.long).sum(dim=1).remainder(2).to(torch.bool))
        assert_token_range(out, f"cumulative_{mode}_r{radius}")
        return out
    raise ValueError(f"unknown cumulative mode {mode!r}")


def right_divide_by_generator_tokens(tokens: torch.Tensor, generator_idx: int) -> torch.Tensor:
    """Return tokens for M * rho(sigma_i)^-1 over F_2[v, v^-1].

    The output degree axis is shifted by +2: output index 0 is Laurent degree
    -2, index 2 is degree 0, and index D+1 is degree D-1.
    """
    if generator_idx not in (0, 1, 2):
        raise ValueError("generator_idx must be 0, 1, or 2")
    bits = token_bits(tokens)
    batch, depth = tokens.shape
    out = torch.zeros(batch, depth + 2, 9, dtype=torch.bool, device=tokens.device)

    if generator_idx == 0:
        terms = {
            0: [(0, -2)],
            1: [(0, -1), (1, 0)],
            2: [(2, 0)],
        }
    elif generator_idx == 1:
        terms = {
            0: [(0, 0), (1, -1)],
            1: [(1, -2)],
            2: [(1, -1), (2, 0)],
        }
    else:
        terms = {
            0: [(0, 0)],
            1: [(1, 0), (2, -1)],
            2: [(2, -2)],
        }

    for dest_col, pieces in terms.items():
        for src_col, shift in pieces:
            start = 2 + shift
            for row in range(3):
                src_entry = 3 * row + src_col
                dst_entry = 3 * row + dest_col
                out[:, start : start + depth, dst_entry] ^= bits[:, :, src_entry]
    return pack_token_bits(out)


def collect_split(
    *,
    data_dir: Path,
    num_shards: int,
    split: str,
    max_examples: int,
    batch_size: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    paths = discover_shards(data_dir, num_shards=num_shards)
    dataset = B4ShardBatchIterable(
        paths,
        split=split,
        batch_size=batch_size,
        seed=seed,
        shuffle_shards=True,
        shuffle_rows=True,
        max_examples=max_examples,
    )
    pieces: dict[str, list[torch.Tensor]] = defaultdict(list)
    count = 0
    for batch in dataset:
        for key in ("tokens", "descent_mask", "label_bits", "min_degree", "max_degree", "final_factor_id", "sample_id"):
            pieces[key].append(batch[key])
        count += int(batch["descent_mask"].numel())
        if count >= max_examples:
            break
    if count == 0:
        raise RuntimeError(f"no examples loaded for split={split}")
    return {key: torch.cat(values, dim=0)[:max_examples] for key, values in pieces.items()}


def rows_as_tuples(x: torch.Tensor) -> list[tuple[int, ...]]:
    x = x.cpu().to(torch.long)
    if x.ndim == 1:
        return [(int(v),) for v in x.tolist()]
    return [tuple(int(v) for v in row) for row in x.tolist()]


def lookup_score(train_feature: torch.Tensor, train_masks: torch.Tensor, eval_feature: torch.Tensor, eval_masks: torch.Tensor) -> dict:
    train_keys = rows_as_tuples(train_feature)
    eval_keys = rows_as_tuples(eval_feature)
    train_masks_list = [int(x) for x in train_masks.cpu().to(torch.long).tolist()]
    eval_masks_cpu = eval_masks.cpu().to(torch.long)
    eval_bits = mask_to_bits(eval_masks_cpu)

    mask_counts: dict[tuple[int, ...], Counter] = defaultdict(Counter)
    bit_counts: dict[tuple[int, ...], torch.Tensor] = {}
    total_counts: dict[tuple[int, ...], int] = defaultdict(int)
    global_counter: Counter = Counter()
    global_bits = torch.zeros(3, dtype=torch.long)

    for key, mask in zip(train_keys, train_masks_list, strict=True):
        mask_counts[key][mask] += 1
        total_counts[key] += 1
        if key not in bit_counts:
            bit_counts[key] = torch.zeros(3, dtype=torch.long)
        bits = [(mask >> i) & 1 for i in range(3)]
        bit_counts[key] += torch.tensor(bits, dtype=torch.long)
        global_counter[mask] += 1
        global_bits += torch.tensor(bits, dtype=torch.long)

    fallback_mask = int(global_counter.most_common(1)[0][0])
    fallback_bits = (global_bits * 2 >= len(train_masks_list)).to(torch.long)
    mask_table = {key: int(counter.most_common(1)[0][0]) for key, counter in mask_counts.items()}
    bit_table = {
        key: (counts * 2 >= total_counts[key]).to(torch.long)
        for key, counts in bit_counts.items()
    }

    pred_masks = []
    pred_bits = []
    seen = 0
    for key in eval_keys:
        if key in mask_table:
            seen += 1
        pred_masks.append(mask_table.get(key, fallback_mask))
        pred_bits.append(bit_table.get(key, fallback_bits))
    pred_masks_t = torch.tensor(pred_masks, dtype=torch.long)
    pred_bits_t = torch.stack(pred_bits, dim=0)
    exact_bits_from_mask = mask_to_bits(pred_masks_t)

    return {
        "unique_train_keys": int(len(mask_table)),
        "coverage": float(seen / max(1, len(eval_keys))),
        "mask_majority_exact_accuracy": float((pred_masks_t == eval_masks_cpu).float().mean().item()),
        "mask_majority_bit_accuracy": float(exact_bits_from_mask.eq(eval_bits).float().mean().item()),
        "bit_majority_exact_accuracy": float(pred_bits_t.eq(eval_bits).all(dim=1).float().mean().item()),
        "bit_majority_bit_accuracy": float(pred_bits_t.eq(eval_bits).float().mean().item()),
        "bit_majority_per_label_accuracy": [float(x) for x in pred_bits_t.eq(eval_bits).float().mean(dim=0).tolist()],
        "fallback_mask": fallback_mask,
    }


def append_feature(
    features: dict[str, torch.Tensor],
    name: str,
    values: torch.Tensor,
    *,
    max_columns: int = 64,
) -> None:
    values = values.cpu().to(torch.long)
    if values.ndim == 1:
        values = values.unsqueeze(1)
    if values.shape[1] > max_columns:
        return
    features[name] = values


def build_feature_tensors(tokens: torch.Tensor, max_radius: int, attrs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    features: dict[str, torch.Tensor] = {}
    feats = support_features(tokens)
    first = feats["first"]
    last = feats["last"]
    width = feats["width"]
    lead = gather_forward(tokens, first, max_radius)
    trail = gather_backward(tokens, last, max_radius)

    append_feature(features, "degree_width", width)
    append_feature(features, "boundary_degrees", torch.stack([first, last, width], dim=1))
    append_feature(features, "leading_token", feats["leading_token"])
    append_feature(features, "trailing_token", feats["trailing_token"])
    append_feature(features, "boundary_tokens", torch.stack([feats["leading_token"], feats["trailing_token"]], dim=1))
    append_feature(features, "boundary_degrees_tokens", torch.stack([first, last, width, feats["leading_token"], feats["trailing_token"]], dim=1))

    for k in range(max_radius + 1):
        append_feature(features, f"lead_token_k{k}", lead[:, k])
        append_feature(features, f"trail_token_k{k}", trail[:, k])
        append_feature(features, f"leadtrail_token_k{k}", torch.stack([lead[:, k], trail[:, k]], dim=1))

    for radius in range(max_radius + 1):
        append_feature(features, f"lead_raw_r{radius}", lead[:, : radius + 1])
        append_feature(features, f"trail_raw_r{radius}", trail[:, : radius + 1])
        append_feature(features, f"both_raw_r{radius}", torch.cat([lead[:, : radius + 1], trail[:, : radius + 1]], dim=1))

        for mode in ("or", "xor"):
            lead_cum = cumulative_token(lead, radius, mode=mode)
            trail_cum = cumulative_token(trail, radius, mode=mode)
            append_feature(features, f"lead_{mode}_r{radius}", lead_cum)
            append_feature(features, f"trail_{mode}_r{radius}", trail_cum)
            append_feature(features, f"both_{mode}_r{radius}", torch.stack([lead_cum, trail_cum], dim=1))
            append_feature(
                features,
                f"both_{mode}_lin_r{radius}",
                torch.stack(
                    [
                        attrs["rank"][lead_cum],
                        attrs["col_mask"][lead_cum],
                        attrs["row_mask"][lead_cum],
                        attrs["image"][lead_cum],
                        attrs["kernel"][lead_cum],
                        attrs["rank"][trail_cum],
                        attrs["col_mask"][trail_cum],
                        attrs["row_mask"][trail_cum],
                        attrs["image"][trail_cum],
                        attrs["kernel"][trail_cum],
                    ],
                    dim=1,
                ),
            )

        for attr_name in ("rank", "row_mask", "col_mask", "diag_mask", "popcount", "image", "kernel", "rowspace"):
            table = attrs[attr_name]
            lead_attr = table[lead[:, : radius + 1]]
            trail_attr = table[trail[:, : radius + 1]]
            append_feature(features, f"lead_{attr_name}_r{radius}", lead_attr)
            append_feature(features, f"trail_{attr_name}_r{radius}", trail_attr)
            append_feature(features, f"both_{attr_name}_r{radius}", torch.cat([lead_attr, trail_attr], dim=1))

    div_summary_parts = []
    for gen_idx in range(3):
        divided = right_divide_by_generator_tokens(tokens, gen_idx)
        div_feats = support_features(divided)
        div_first_exp = div_feats["first"] - 2
        div_last_exp = div_feats["last"] - 2
        min_delta = div_first_exp - first
        max_delta = div_last_exp - last
        width_delta = div_feats["width"] - width
        lead_token = div_feats["leading_token"]
        trail_token = div_feats["trailing_token"]
        core = torch.stack([min_delta, max_delta, width_delta, lead_token, trail_token], dim=1)
        div_summary_parts.append(torch.stack([min_delta, max_delta, width_delta], dim=1))
        append_feature(features, f"right_div_s{gen_idx + 1}_frontier", core)
        append_feature(
            features,
            f"right_div_s{gen_idx + 1}_frontier_lin",
            torch.stack(
                [
                    min_delta,
                    max_delta,
                    width_delta,
                    attrs["rank"][lead_token],
                    attrs["col_mask"][lead_token],
                    attrs["row_mask"][lead_token],
                    attrs["image"][lead_token],
                    attrs["kernel"][lead_token],
                    attrs["rank"][trail_token],
                    attrs["col_mask"][trail_token],
                    attrs["row_mask"][trail_token],
                    attrs["image"][trail_token],
                    attrs["kernel"][trail_token],
                ],
                dim=1,
            ),
        )
        for radius in range(min(max_radius, 4) + 1):
            div_lead = gather_forward(divided, div_feats["first"], radius)
            div_trail = gather_backward(divided, div_feats["last"], radius)
            append_feature(features, f"right_div_s{gen_idx + 1}_lead_raw_r{radius}", div_lead)
            append_feature(features, f"right_div_s{gen_idx + 1}_trail_raw_r{radius}", div_trail)
            append_feature(features, f"right_div_s{gen_idx + 1}_both_raw_r{radius}", torch.cat([div_lead, div_trail], dim=1))
    append_feature(features, "right_div_all_deltas", torch.cat(div_summary_parts, dim=1))
    return features


def score_features(
    train_features: dict[str, torch.Tensor],
    eval_features: dict[str, torch.Tensor],
    train_masks: torch.Tensor,
    eval_masks: torch.Tensor,
    *,
    top_k: int,
    max_pair_keys: int,
) -> dict:
    individual = {}
    names = sorted(train_features)
    for name in names:
        individual[name] = lookup_score(train_features[name], train_masks, eval_features[name], eval_masks)

    sorted_by_exact = sorted(
        individual.items(),
        key=lambda item: (
            item[1]["bit_majority_exact_accuracy"],
            item[1]["bit_majority_bit_accuracy"],
            -item[1]["unique_train_keys"],
        ),
        reverse=True,
    )
    sorted_by_bit = sorted(
        individual.items(),
        key=lambda item: (
            item[1]["bit_majority_bit_accuracy"],
            item[1]["bit_majority_exact_accuracy"],
            -item[1]["unique_train_keys"],
        ),
        reverse=True,
    )

    pair_pool = [
        name
        for name, stats in sorted_by_exact[: max(2 * top_k, top_k)]
        if stats["unique_train_keys"] <= max_pair_keys
    ][:top_k]
    pairs = {}
    for i, left in enumerate(pair_pool):
        for right in pair_pool[i + 1 :]:
            name = f"{left}__PLUS__{right}"
            train_values = torch.cat([train_features[left], train_features[right]], dim=1)
            eval_values = torch.cat([eval_features[left], eval_features[right]], dim=1)
            pairs[name] = lookup_score(train_values, train_masks, eval_values, eval_masks)

    sorted_pairs = sorted(
        pairs.items(),
        key=lambda item: (
            item[1]["bit_majority_exact_accuracy"],
            item[1]["bit_majority_bit_accuracy"],
            -item[1]["unique_train_keys"],
        ),
        reverse=True,
    )

    per_label_top = {}
    for label_idx in range(3):
        per_label_top[f"s{label_idx + 1}"] = [
            {"feature": name, **stats}
            for name, stats in sorted(
                individual.items(),
                key=lambda item: item[1]["bit_majority_per_label_accuracy"][label_idx],
                reverse=True,
            )[:20]
        ]

    return {
        "top_individual_by_exact": [{"feature": name, **stats} for name, stats in sorted_by_exact[:50]],
        "top_individual_by_bit": [{"feature": name, **stats} for name, stats in sorted_by_bit[:50]],
        "top_pairs_by_exact": [{"feature": name, **stats} for name, stats in sorted_pairs[:50]],
        "per_label_top_individual": per_label_top,
        "num_individual_features": len(individual),
        "num_pair_features": len(pairs),
        "pair_pool": pair_pool,
    }


def simple_factor_diagnostics(attrs: dict[str, torch.Tensor]) -> list[dict]:
    table = []
    for factor_id, perm in enumerate(PROPER_FACTOR_PERMS):
        mat = SIMPLE_MATS[factor_id].to(torch.bool)
        tokens = pack_token_bits(mat.view(mat.shape[0], 9)).cpu()
        support = tokens.ne(0)
        first = int(support.to(torch.long).argmax().item())
        last = int(tokens.shape[0] - 1 - support.flip(dims=[0]).to(torch.long).argmax().item())
        lead = int(tokens[first].item())
        trail = int(tokens[last].item())
        factor = GarsideFactor(perm)
        table.append(
            {
                "factor_id": int(factor_id),
                "perm": list(perm),
                "artin_word_1_based": [int(x + 1) for x in factor.artin_factors()],
                "left_descent_mask": int(LEFT_DESC_MASK[factor_id].item()),
                "right_descent_mask": int(RIGHT_DESC_MASK[factor_id].item()),
                "min_degree": first,
                "max_degree": last,
                "leading_token": lead,
                "trailing_token": trail,
                "leading_rank": int(attrs["rank"][lead].item()),
                "trailing_rank": int(attrs["rank"][trail].item()),
                "leading_col_mask": int(attrs["col_mask"][lead].item()),
                "leading_row_mask": int(attrs["row_mask"][lead].item()),
                "trailing_col_mask": int(attrs["col_mask"][trail].item()),
                "trailing_row_mask": int(attrs["row_mask"][trail].item()),
            }
        )
    return table


def dataset_summary(batch: dict[str, torch.Tensor]) -> dict:
    masks = batch["descent_mask"].to(torch.long)
    final_ids = batch["final_factor_id"].to(torch.long)
    return {
        "n": int(masks.numel()),
        "mask_counts": torch.bincount(masks, minlength=8).tolist(),
        "final_factor_counts": torch.bincount(final_ids, minlength=len(PROPER_FACTOR_PERMS)).tolist(),
        "min_degree_range": [int(batch["min_degree"].min().item()), int(batch["min_degree"].max().item())],
        "max_degree_range": [int(batch["max_degree"].min().item()), int(batch["max_degree"].max().item())],
        "width_range": [
            int((batch["max_degree"] - batch["min_degree"] + 1).min().item()),
            int((batch["max_degree"] - batch["min_degree"] + 1).max().item()),
        ],
    }


def atomic_json_dump(payload: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine B_4 Burau frontier rules for final descent.")
    parser.add_argument("--data-dir", default="interp/data/generated/b4_l25_p2_n16777216")
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--train-examples", type=int, default=262_144)
    parser.add_argument("--eval-examples", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-radius", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--max-pair-keys", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--out", default="interp/artifacts/b4_l25_p2_boundary_rule_mining/results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attrs = precompute_token_attrs()
    for name, table in attrs.items():
        if tuple(table.shape) != (512,):
            raise RuntimeError(f"token attribute table {name!r} has shape {tuple(table.shape)}, expected (512,)")
    data_dir = Path(args.data_dir)

    train = collect_split(
        data_dir=data_dir,
        num_shards=args.num_shards,
        split="train",
        max_examples=args.train_examples,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    eval_batch = collect_split(
        data_dir=data_dir,
        num_shards=args.num_shards,
        split="test",
        max_examples=args.eval_examples,
        batch_size=args.batch_size,
        seed=args.seed + 1,
    )

    train_features = build_feature_tensors(train["tokens"], max_radius=args.max_radius, attrs=attrs)
    eval_features = build_feature_tensors(eval_batch["tokens"], max_radius=args.max_radius, attrs=attrs)
    scores = score_features(
        train_features,
        eval_features,
        train["descent_mask"],
        eval_batch["descent_mask"],
        top_k=args.top_k,
        max_pair_keys=args.max_pair_keys,
    )

    result = {
        "config": vars(args),
        "interpretation": {
            "right_division_frontier": (
                "For each generator sigma_i, multiply the Burau matrix on the right by "
                "rho(sigma_i)^-1 over F_2[v,v^-1].  Because right descent is right "
                "divisibility, shifts and cancellations in the new degree frontier are "
                "natural candidate invariants."
            ),
            "boundary_features": (
                "Leading bands are degrees min..min+r; trailing bands are degrees max..max-r. "
                "Features include raw tokens, finite linear-algebra invariants of each 3x3 "
                "slice, cumulative OR/XOR frontier summaries, and generator-division frontiers."
            ),
        },
        "train_summary": dataset_summary(train),
        "eval_summary": dataset_summary(eval_batch),
        "simple_factor_diagnostics": simple_factor_diagnostics(attrs),
        "scores": scores,
    }
    atomic_json_dump(result, Path(args.out))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
