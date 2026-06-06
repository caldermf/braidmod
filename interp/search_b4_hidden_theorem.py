#!/usr/bin/env python3
"""Search for theorem-shaped B4 Burau/descent rules over Z[v].

This is deliberately an algebraic audit, not a neural-net experiment.  The
main question is whether right descent can be read from a clean quotient or
frontier invariant of the integer Burau matrix.

The most natural test is right division: sigma_i is in the right descent set
iff beta * sigma_i^{-1} is a positive braid.  In a faithful, positivity-aware
matrix representation, this might show up as a simple condition on the Laurent
polynomial matrix.  Reduced Burau in B4 is subtler, so this script measures
exactly how far those candidate conditions get and where they collide.
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
from interp.b4_data import discover_shards, load_shard, split_mask  # noqa: E402
from interp.b4_z_sign import dense_burau_z_for_factor_ids, simple_mats_z  # noqa: E402
from interp.generate_b4_dataset import (  # noqa: E402
    LEFT_DESC_MASK,
    MATRIX_SIZE,
    PROPER_FACTOR_PERMS,
    RIGHT_DESC_MASK,
    absolute_depth_for_length,
)
from interp.train_b4_transformer import atomic_json_dump, resolve_device, set_seed  # noqa: E402


def bits_from_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(torch.long)
    return torch.stack([((mask >> idx) & 1) for idx in range(3)], dim=1).to(torch.long)


def mask_from_bits(bits: torch.Tensor) -> torch.Tensor:
    bits = bits.to(torch.long)
    weights = torch.tensor([1, 2, 4], dtype=torch.long, device=bits.device).view(1, 3)
    return (bits * weights).sum(dim=1)


def collect_factor_rows(
    *,
    data_dir: Path,
    num_shards: int,
    split: str,
    max_examples: int,
    shuffle: bool,
    seed: int,
) -> dict[str, torch.Tensor]:
    paths = discover_shards(data_dir, num_shards=num_shards)
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(paths), generator=g).tolist()
        paths = [paths[idx] for idx in order]
    pieces: dict[str, list[torch.Tensor]] = defaultdict(list)
    emitted = 0
    for path in paths:
        payload = load_shard(path)
        meta = payload["metadata"]
        count = int(meta["sample_id_count"])
        start = int(meta["sample_id_start"])
        sample_ids = torch.arange(start, start + count, dtype=torch.long)
        rows = torch.nonzero(split_mask(sample_ids, split), as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        if shuffle:
            g = torch.Generator().manual_seed(seed + int(start))
            rows = rows[torch.randperm(rows.numel(), generator=g)]
        take = rows[: max(0, max_examples - emitted)]
        if take.numel() == 0:
            break
        for key in ("factor_ids", "final_factor_id", "descent_mask", "label_bits"):
            pieces[key].append(payload[key][take])
        pieces["sample_id"].append(sample_ids[take])
        emitted += int(take.numel())
        if emitted >= max_examples:
            break
    if emitted == 0:
        raise RuntimeError(f"no rows collected for split={split}")
    out = {key: torch.cat(values, dim=0)[:max_examples] for key, values in pieces.items()}
    out["label_bits"] = out["label_bits"].to(torch.long)
    out["descent_mask"] = out["descent_mask"].to(torch.long)
    out["final_factor_id"] = out["final_factor_id"].to(torch.long)
    out["factor_ids"] = out["factor_ids"].to(torch.long)
    return out


def support_features_z(mat: torch.Tensor, *, offset: int = 0) -> dict[str, torch.Tensor]:
    occupied = mat.ne(0).any(dim=(-1, -2))
    any_support = occupied.any(dim=1)
    first_idx = occupied.to(torch.long).argmax(dim=1)
    last_idx = mat.shape[1] - 1 - occupied.flip(dims=[1]).to(torch.long).argmax(dim=1)
    first_idx = torch.where(any_support, first_idx, torch.zeros_like(first_idx))
    last_idx = torch.where(any_support, last_idx, torch.zeros_like(last_idx))
    return {
        "first_idx": first_idx,
        "last_idx": last_idx,
        "first_exp": first_idx + int(offset),
        "last_exp": last_idx + int(offset),
        "width": last_idx - first_idx + 1,
    }


def gather_window_mats(mat: torch.Tensor, centers: torch.Tensor, radius: int, *, direction: str) -> torch.Tensor:
    if direction == "forward":
        offsets = torch.arange(0, radius + 1, dtype=torch.long, device=mat.device)
    elif direction == "backward":
        offsets = -torch.arange(0, radius + 1, dtype=torch.long, device=mat.device)
    else:
        raise ValueError(f"unknown direction {direction!r}")
    idx = centers.unsqueeze(1) + offsets.unsqueeze(0)
    valid = (idx >= 0) & (idx < mat.shape[1])
    idx = idx.clamp(0, mat.shape[1] - 1)
    idx4 = idx.view(idx.shape[0], idx.shape[1], 1, 1).expand(-1, -1, MATRIX_SIZE, MATRIX_SIZE)
    out = mat.gather(1, idx4)
    return torch.where(valid.view(valid.shape[0], valid.shape[1], 1, 1), out, torch.zeros_like(out))


def sign_token_from_mats(band: torch.Tensor) -> torch.Tensor:
    signs = torch.zeros_like(band, dtype=torch.long)
    signs = torch.where(band.lt(0), torch.ones_like(signs), signs)
    signs = torch.where(band.gt(0), torch.full_like(signs, 2), signs)
    digits = signs.view(signs.shape[0], signs.shape[1], MATRIX_SIZE * MATRIX_SIZE)
    weights = torch.tensor([3**i for i in range(MATRIX_SIZE * MATRIX_SIZE)], dtype=torch.long, device=band.device)
    return (digits * weights.view(1, 1, -1)).sum(dim=-1)


def support_token_from_mats(band: torch.Tensor) -> torch.Tensor:
    bits = band.ne(0).view(band.shape[0], band.shape[1], MATRIX_SIZE * MATRIX_SIZE).to(torch.long)
    weights = torch.tensor([1 << i for i in range(MATRIX_SIZE * MATRIX_SIZE)], dtype=torch.long, device=band.device)
    return (bits * weights.view(1, 1, -1)).sum(dim=-1)


def clipped_coeff_token_from_mats(band: torch.Tensor, clip: int) -> torch.Tensor:
    clipped = band.clamp(min=-clip, max=clip).to(torch.long) + int(clip)
    digits = clipped.view(clipped.shape[0], clipped.shape[1], MATRIX_SIZE * MATRIX_SIZE)
    base = 2 * int(clip) + 1
    weights = torch.tensor([base**i for i in range(MATRIX_SIZE * MATRIX_SIZE)], dtype=torch.long, device=band.device)
    return (digits * weights.view(1, 1, -1)).sum(dim=-1)


def signed_column_masks_from_mats(band: torch.Tensor) -> dict[str, torch.Tensor]:
    weights = torch.tensor([1, 2, 4], dtype=torch.long, device=band.device).view(1, 1, 3)
    any_cols = band.ne(0).any(dim=2).to(torch.long)
    pos_cols = band.gt(0).any(dim=2).to(torch.long)
    neg_cols = band.lt(0).any(dim=2).to(torch.long)
    return {
        "any_col": (any_cols * weights).sum(dim=-1),
        "pos_col": (pos_cols * weights).sum(dim=-1),
        "neg_col": (neg_cols * weights).sum(dim=-1),
    }


def signed_row_masks_from_mats(band: torch.Tensor) -> dict[str, torch.Tensor]:
    weights = torch.tensor([1, 2, 4], dtype=torch.long, device=band.device).view(1, 1, 3)
    any_rows = band.ne(0).any(dim=3).to(torch.long)
    pos_rows = band.gt(0).any(dim=3).to(torch.long)
    neg_rows = band.lt(0).any(dim=3).to(torch.long)
    return {
        "any_row": (any_rows * weights).sum(dim=-1),
        "pos_row": (pos_rows * weights).sum(dim=-1),
        "neg_row": (neg_rows * weights).sum(dim=-1),
    }


def right_inverse_generator_terms(generator_idx: int) -> list[tuple[int, int, int, int]]:
    if generator_idx == 0:
        return [(-2, 0, 0, -1), (-1, 0, 1, -1), (0, 1, 1, 1), (0, 2, 2, 1)]
    if generator_idx == 1:
        return [(0, 0, 0, 1), (-1, 1, 0, -1), (-2, 1, 1, -1), (-1, 1, 2, -1), (0, 2, 2, 1)]
    if generator_idx == 2:
        return [(0, 0, 0, 1), (0, 1, 1, 1), (-1, 2, 1, -1), (-2, 2, 2, -1)]
    raise ValueError("generator_idx must be 0, 1, or 2")


def right_multiply_inverse_generator_laurent(
    mat: torch.Tensor,
    *,
    offset: int,
    generator_idx: int,
) -> tuple[torch.Tensor, int]:
    """Right multiply by rho(sigma_i)^-1.

    ``offset`` is the Laurent exponent represented by matrix index 0.  The
    inverse generator has minimum exponent -2 and maximum exponent 0, so the
    new offset is ``offset - 2`` and the depth grows by two.
    """
    batch_size, depth = mat.shape[:2]
    out = torch.zeros(batch_size, depth + 2, MATRIX_SIZE, MATRIX_SIZE, dtype=mat.dtype, device=mat.device)
    for shift, src_col, dst_col, coeff in right_inverse_generator_terms(generator_idx):
        start = shift + 2
        out[:, start : start + depth, :, dst_col] += int(coeff) * mat[:, :, :, src_col]
    return out, int(offset) - 2


def right_divide_generators(mat: torch.Tensor) -> tuple[torch.Tensor, int]:
    quotients = []
    for generator_idx in range(3):
        q, offset = right_multiply_inverse_generator_laurent(mat, offset=0, generator_idx=generator_idx)
        quotients.append(q)
    return torch.stack(quotients, dim=1), offset


def simple_factor_inverse_words() -> list[list[int]]:
    return [
        list(reversed([int(idx) for idx in GarsideFactor(perm).artin_factors()]))
        for perm in PROPER_FACTOR_PERMS
    ]


def right_divide_simple_factors(mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return negative-support summaries for quotient by each proper simple.

    The full quotient tensors are intentionally not retained.  We only keep a
    few membership-like statistics: number of nonzero entries in negative
    Laurent degrees, first exponent, last exponent, and final width.
    """
    inverse_words = simple_factor_inverse_words()
    neg_counts = []
    first_exps = []
    widths = []
    for inv_word in inverse_words:
        q = mat
        offset = 0
        for generator_idx in inv_word:
            q, offset = right_multiply_inverse_generator_laurent(q, offset=offset, generator_idx=generator_idx)
        neg_width = max(0, -offset)
        if neg_width:
            neg_count = q[:, :neg_width].ne(0).sum(dim=(1, 2, 3))
        else:
            neg_count = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
        feats = support_features_z(q, offset=offset)
        neg_counts.append(neg_count.to(torch.long))
        first_exps.append(feats["first_exp"].to(torch.long))
        widths.append(feats["width"].to(torch.long))
    return torch.stack(neg_counts, dim=1), torch.stack(first_exps, dim=1), torch.stack(widths, dim=1)


def feature_rows(x: torch.Tensor) -> list[tuple[int, ...]]:
    x = x.detach().cpu().to(torch.long)
    if x.ndim == 1:
        return [(int(v),) for v in x.tolist()]
    return [tuple(int(v) for v in row) for row in x.tolist()]


def lookup_tables(train_feature: torch.Tensor, train_masks: torch.Tensor) -> dict:
    keys = feature_rows(train_feature)
    masks = [int(x) for x in train_masks.detach().cpu().to(torch.long).tolist()]
    mask_counts: dict[tuple[int, ...], Counter] = defaultdict(Counter)
    bit_counts: dict[tuple[int, ...], torch.Tensor] = {}
    key_total: dict[tuple[int, ...], int] = defaultdict(int)
    global_mask = Counter()
    global_bits = torch.zeros(3, dtype=torch.long)
    for key, mask in zip(keys, masks, strict=True):
        mask_counts[key][mask] += 1
        key_total[key] += 1
        if key not in bit_counts:
            bit_counts[key] = torch.zeros(3, dtype=torch.long)
        bits = torch.tensor([(mask >> idx) & 1 for idx in range(3)], dtype=torch.long)
        bit_counts[key] += bits
        global_mask[mask] += 1
        global_bits += bits

    fallback_mask = int(global_mask.most_common(1)[0][0])
    fallback_bits = (global_bits * 2 >= len(masks)).to(torch.long)
    mask_table = {key: int(counter.most_common(1)[0][0]) for key, counter in mask_counts.items()}
    bit_table = {key: (counts * 2 >= key_total[key]).to(torch.long) for key, counts in bit_counts.items()}
    majority_mass = 0
    pure_mass = 0
    singleton_mass = 0
    conflict_rows = []
    for key, counter in mask_counts.items():
        total = key_total[key]
        majority = counter.most_common(1)[0][1]
        majority_mass += majority
        if len(counter) == 1:
            pure_mass += total
        if total == 1:
            singleton_mass += total
        elif len(counter) > 1:
            entropy = 0.0
            for count in counter.values():
                p = count / total
                entropy -= p * math.log2(p)
            conflict_rows.append(
                {
                    "key": list(key[:24]),
                    "key_length": len(key),
                    "count": int(total),
                    "entropy": float(entropy),
                    "mask_counts": {str(mask): int(count) for mask, count in sorted(counter.items())},
                }
            )
    return {
        "mask_table": mask_table,
        "bit_table": bit_table,
        "fallback_mask": fallback_mask,
        "fallback_bits": fallback_bits,
        "num_train": len(keys),
        "unique_train_keys": len(mask_counts),
        "train_majority_exact": majority_mass / max(1, len(keys)),
        "train_pure_example_fraction": pure_mass / max(1, len(keys)),
        "train_singleton_example_fraction": singleton_mass / max(1, len(keys)),
        "top_conflicts": sorted(conflict_rows, key=lambda row: (row["count"], row["entropy"]), reverse=True)[:12],
    }


def lookup_score(
    train_feature: torch.Tensor,
    train_masks: torch.Tensor,
    eval_feature: torch.Tensor,
    eval_masks: torch.Tensor,
) -> dict:
    tables = lookup_tables(train_feature, train_masks)
    keys = feature_rows(eval_feature)
    true_masks = eval_masks.detach().cpu().to(torch.long)
    true_bits = bits_from_mask(true_masks)
    pred_masks = []
    pred_bits = []
    seen = 0
    for key in keys:
        if key in tables["mask_table"]:
            seen += 1
        pred_masks.append(tables["mask_table"].get(key, tables["fallback_mask"]))
        pred_bits.append(tables["bit_table"].get(key, tables["fallback_bits"]))
    pred_masks_t = torch.tensor(pred_masks, dtype=torch.long)
    pred_bits_t = torch.stack(pred_bits, dim=0)
    pred_bits_from_mask = bits_from_mask(pred_masks_t)
    bit_eq = pred_bits_t.eq(true_bits)
    return {
        "unique_train_keys": int(tables["unique_train_keys"]),
        "coverage": float(seen / max(1, len(keys))),
        "train_majority_exact": float(tables["train_majority_exact"]),
        "train_pure_example_fraction": float(tables["train_pure_example_fraction"]),
        "train_singleton_example_fraction": float(tables["train_singleton_example_fraction"]),
        "mask_majority_exact_accuracy": float((pred_masks_t == true_masks).float().mean().item()),
        "mask_majority_bit_accuracy": float(pred_bits_from_mask.eq(true_bits).float().mean().item()),
        "bit_majority_exact_accuracy": float(bit_eq.all(dim=1).float().mean().item()),
        "bit_majority_bit_accuracy": float(bit_eq.float().mean().item()),
        "bit_majority_per_label_accuracy": [float(x) for x in bit_eq.float().mean(dim=0).tolist()],
        "fallback_mask": int(tables["fallback_mask"]),
        "top_conflicts": tables["top_conflicts"],
    }


def binary_lookup_score(
    train_feature: torch.Tensor,
    train_labels: torch.Tensor,
    eval_feature: torch.Tensor,
    eval_labels: torch.Tensor,
) -> dict:
    tables = lookup_tables(train_feature, train_labels.to(torch.long))
    keys = feature_rows(eval_feature)
    true = eval_labels.detach().cpu().to(torch.long)
    pred = []
    seen = 0
    for key in keys:
        if key in tables["mask_table"]:
            seen += 1
        pred.append(tables["mask_table"].get(key, tables["fallback_mask"]))
    pred_t = torch.tensor(pred, dtype=torch.long)
    tp = int(((pred_t == 1) & (true == 1)).sum().item())
    fp = int(((pred_t == 1) & (true == 0)).sum().item())
    fn = int(((pred_t == 0) & (true == 1)).sum().item())
    f1 = 0.0 if (2 * tp + fp + fn) == 0 else (2 * tp) / (2 * tp + fp + fn)
    out = {
        "unique_train_keys": int(tables["unique_train_keys"]),
        "coverage": float(seen / max(1, len(keys))),
        "train_majority_accuracy": float(tables["train_majority_exact"]),
        "train_pure_example_fraction": float(tables["train_pure_example_fraction"]),
        "train_singleton_example_fraction": float(tables["train_singleton_example_fraction"]),
        "binary_accuracy": float((pred_t == true).float().mean().item()),
        "binary_f1": float(f1),
        "pred_positive_rate": float(pred_t.float().mean().item()),
        "true_positive_rate": float(true.float().mean().item()),
        "fallback_label": int(tables["fallback_mask"]),
        "top_conflicts": tables["top_conflicts"],
    }
    if true.numel() % 3 == 0:
        pred_bits = pred_t.view(-1, 3)
        true_bits = true.view(-1, 3)
        set_metrics = metrics_from_bits(pred_bits, true_bits)
        out.update(
            {
                "set_exact_accuracy": set_metrics["exact_accuracy"],
                "set_bit_accuracy": set_metrics["bit_accuracy"],
                "set_per_label_accuracy": set_metrics["per_label_accuracy"],
                "set_pred_mask_counts": set_metrics["pred_mask_counts"],
                "set_true_mask_counts": set_metrics["true_mask_counts"],
            }
        )
    return out


def metrics_from_bits(pred_bits: torch.Tensor, true_bits: torch.Tensor) -> dict:
    pred_bits = pred_bits.detach().cpu().to(torch.long)
    true_bits = true_bits.detach().cpu().to(torch.long)
    eq = pred_bits.eq(true_bits)
    pred_masks = mask_from_bits(pred_bits)
    true_masks = mask_from_bits(true_bits)
    tp = ((pred_bits == 1) & (true_bits == 1)).sum(dim=0)
    fp = ((pred_bits == 1) & (true_bits == 0)).sum(dim=0)
    fn = ((pred_bits == 0) & (true_bits == 1)).sum(dim=0)
    denom = 2 * tp + fp + fn
    f1 = torch.where(denom > 0, 2 * tp / denom.clamp_min(1), torch.zeros_like(denom, dtype=torch.float32))
    return {
        "exact_accuracy": float((pred_masks == true_masks).float().mean().item()),
        "bit_accuracy": float(eq.float().mean().item()),
        "per_label_accuracy": [float(x) for x in eq.float().mean(dim=0).tolist()],
        "per_label_f1": [float(x) for x in f1.tolist()],
        "pred_mask_counts": torch.bincount(pred_masks, minlength=8).tolist(),
        "true_mask_counts": torch.bincount(true_masks, minlength=8).tolist(),
    }


def build_boundary_features(mat: torch.Tensor, *, radius: int) -> dict[str, torch.Tensor]:
    feats = support_features_z(mat, offset=0)
    lead = gather_window_mats(mat, feats["first_idx"], radius, direction="forward")
    trail = gather_window_mats(mat, feats["last_idx"], radius, direction="backward")
    lead_sign = sign_token_from_mats(lead)
    trail_sign = sign_token_from_mats(trail)
    lead_support = support_token_from_mats(lead)
    trail_support = support_token_from_mats(trail)
    lead_clip3 = clipped_coeff_token_from_mats(lead, 3)
    trail_clip3 = clipped_coeff_token_from_mats(trail, 3)
    lead_clip7 = clipped_coeff_token_from_mats(lead, 7)
    trail_clip7 = clipped_coeff_token_from_mats(trail, 7)
    lead_cols = signed_column_masks_from_mats(lead)
    trail_cols = signed_column_masks_from_mats(trail)
    lead_rows = signed_row_masks_from_mats(lead)
    trail_rows = signed_row_masks_from_mats(trail)
    out = {
        "z_degree_bounds": torch.stack([feats["first_exp"], feats["last_exp"], feats["width"]], dim=1),
    }
    for r in range(radius + 1):
        out[f"z_lead_sign_r{r}"] = lead_sign[:, : r + 1]
        out[f"z_trail_sign_r{r}"] = trail_sign[:, : r + 1]
        out[f"z_both_sign_r{r}"] = torch.cat([lead_sign[:, : r + 1], trail_sign[:, : r + 1]], dim=1)
        out[f"z_both_support_r{r}"] = torch.cat([lead_support[:, : r + 1], trail_support[:, : r + 1]], dim=1)
        out[f"z_trail_clip3_r{r}"] = trail_clip3[:, : r + 1]
        out[f"z_both_clip3_r{r}"] = torch.cat([lead_clip3[:, : r + 1], trail_clip3[:, : r + 1]], dim=1)
        out[f"z_trail_clip7_r{r}"] = trail_clip7[:, : r + 1]
        out[f"z_both_clip7_r{r}"] = torch.cat([lead_clip7[:, : r + 1], trail_clip7[:, : r + 1]], dim=1)
        out[f"z_both_pos_col_r{r}"] = torch.cat([lead_cols["pos_col"][:, : r + 1], trail_cols["pos_col"][:, : r + 1]], dim=1)
        out[f"z_both_neg_col_r{r}"] = torch.cat([lead_cols["neg_col"][:, : r + 1], trail_cols["neg_col"][:, : r + 1]], dim=1)
        out[f"z_both_any_col_r{r}"] = torch.cat([lead_cols["any_col"][:, : r + 1], trail_cols["any_col"][:, : r + 1]], dim=1)
        out[f"z_both_signed_rows_cols_r{r}"] = torch.cat(
            [
                lead_rows["pos_row"][:, : r + 1],
                lead_rows["neg_row"][:, : r + 1],
                lead_cols["pos_col"][:, : r + 1],
                lead_cols["neg_col"][:, : r + 1],
                trail_rows["pos_row"][:, : r + 1],
                trail_rows["neg_row"][:, : r + 1],
                trail_cols["pos_col"][:, : r + 1],
                trail_cols["neg_col"][:, : r + 1],
            ],
            dim=1,
        )
    return out


def build_generator_quotient_features(mat: torch.Tensor, *, radius: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    q, offset = right_divide_generators(mat)
    batch, num_gen, depth = q.shape[:3]
    flat = q.reshape(batch * num_gen, depth, MATRIX_SIZE, MATRIX_SIZE)
    feats = support_features_z(flat, offset=offset)
    neg_width = max(0, -offset)
    neg_band = flat[:, :neg_width] if neg_width else flat[:, :0]
    neg_count = neg_band.ne(0).sum(dim=(1, 2, 3)) if neg_width else torch.zeros(batch * num_gen, dtype=torch.long, device=mat.device)
    neg_sign = sign_token_from_mats(neg_band) if neg_width else torch.zeros(batch * num_gen, 0, dtype=torch.long, device=mat.device)
    lead = gather_window_mats(flat, feats["first_idx"], radius, direction="forward")
    trail = gather_window_mats(flat, feats["last_idx"], radius, direction="backward")
    lead_sign = sign_token_from_mats(lead)
    trail_sign = sign_token_from_mats(trail)
    lead_cols = signed_column_masks_from_mats(lead)
    trail_cols = signed_column_masks_from_mats(trail)
    gen_ids = torch.arange(num_gen, dtype=torch.long, device=mat.device).view(1, num_gen).expand(batch, num_gen).reshape(-1)

    bit_features = {
        "q_negative_count": torch.stack([gen_ids, neg_count.clamp_max(32)], dim=1),
        "q_negative_sign_tokens": torch.cat([gen_ids.unsqueeze(1), neg_sign], dim=1),
        "q_degree_frontier": torch.stack([gen_ids, feats["first_exp"], feats["last_exp"], feats["width"], neg_count.clamp_max(32)], dim=1),
    }
    for r in range(radius + 1):
        bit_features[f"q_leading_sign_r{r}"] = torch.cat([gen_ids.unsqueeze(1), lead_sign[:, : r + 1]], dim=1)
        bit_features[f"q_trailing_sign_r{r}"] = torch.cat([gen_ids.unsqueeze(1), trail_sign[:, : r + 1]], dim=1)
        bit_features[f"q_both_sign_r{r}"] = torch.cat([gen_ids.unsqueeze(1), lead_sign[:, : r + 1], trail_sign[:, : r + 1]], dim=1)
        bit_features[f"q_both_signed_col_r{r}"] = torch.cat(
            [
                gen_ids.unsqueeze(1),
                lead_cols["pos_col"][:, : r + 1],
                lead_cols["neg_col"][:, : r + 1],
                trail_cols["pos_col"][:, : r + 1],
                trail_cols["neg_col"][:, : r + 1],
            ],
            dim=1,
        )

    explicit_bits = {
        "q_no_negative_terms": neg_count.eq(0).view(batch, num_gen).to(torch.long),
        "q_first_exp_nonnegative": feats["first_exp"].ge(0).view(batch, num_gen).to(torch.long),
        "q_negative_count_le_1": neg_count.le(1).view(batch, num_gen).to(torch.long),
        "q_negative_count_le_3": neg_count.le(3).view(batch, num_gen).to(torch.long),
    }
    return bit_features, explicit_bits


def process_split(
    rows: dict[str, torch.Tensor],
    *,
    length: int,
    absolute_depth: int,
    batch_size: int,
    radius: int,
    simple_quotients: bool,
    device: torch.device,
) -> dict:
    simple_mats = simple_mats_z(device)
    feature_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    qbit_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    explicit_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    simple_neg_chunks = []
    simple_first_chunks = []
    simple_width_chunks = []
    max_abs_coeff = 0
    total = int(rows["factor_ids"].shape[0])
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        factor_ids = rows["factor_ids"][start:stop].to(device)
        mat = dense_burau_z_for_factor_ids(
            factor_ids,
            length=length,
            absolute_depth=absolute_depth,
            simple_mats=simple_mats,
        )
        max_abs_coeff = max(max_abs_coeff, int(mat.abs().max().item()))
        boundary_features = build_boundary_features(mat, radius=radius)
        for name, value in boundary_features.items():
            feature_chunks[name].append(value.cpu())
        q_features, explicit = build_generator_quotient_features(mat, radius=min(radius, 4))
        for name, value in q_features.items():
            qbit_chunks[name].append(value.cpu())
        for name, value in explicit.items():
            explicit_chunks[name].append(value.cpu())
        if simple_quotients:
            neg_counts, first_exps, widths = right_divide_simple_factors(mat)
            simple_neg_chunks.append(neg_counts.cpu())
            simple_first_chunks.append(first_exps.cpu())
            simple_width_chunks.append(widths.cpu())
    out = {
        "boundary_features": {name: torch.cat(values, dim=0) for name, values in feature_chunks.items()},
        "q_bit_features": {name: torch.cat(values, dim=0) for name, values in qbit_chunks.items()},
        "q_explicit_bits": {name: torch.cat(values, dim=0) for name, values in explicit_chunks.items()},
        "max_abs_coeff": max_abs_coeff,
    }
    if simple_quotients:
        out["simple_neg_counts"] = torch.cat(simple_neg_chunks, dim=0)
        out["simple_first_exps"] = torch.cat(simple_first_chunks, dim=0)
        out["simple_widths"] = torch.cat(simple_width_chunks, dim=0)
    return out


def score_boundary_features(train_proc: dict, eval_proc: dict, train_masks: torch.Tensor, eval_masks: torch.Tensor) -> dict:
    scores = {}
    for name, train_feature in sorted(train_proc["boundary_features"].items()):
        scores[name] = lookup_score(train_feature, train_masks, eval_proc["boundary_features"][name], eval_masks)
    top_exact = sorted(
        ({"feature": name, **stats} for name, stats in scores.items()),
        key=lambda item: (item["bit_majority_exact_accuracy"], item["bit_majority_bit_accuracy"]),
        reverse=True,
    )[:40]
    top_pure = sorted(
        ({"feature": name, **stats} for name, stats in scores.items()),
        key=lambda item: (item["train_pure_example_fraction"], item["bit_majority_exact_accuracy"]),
        reverse=True,
    )[:20]
    return {"features": scores, "top_by_exact": top_exact, "top_by_train_purity": top_pure}


def score_generator_quotients(train_proc: dict, eval_proc: dict, train_rows: dict, eval_rows: dict) -> dict:
    train_bit_labels = train_rows["label_bits"].reshape(-1).to(torch.long)
    eval_bit_labels = eval_rows["label_bits"].reshape(-1).to(torch.long)
    bit_scores = {}
    set_scores = {}
    for name, train_feature in sorted(train_proc["q_bit_features"].items()):
        bit_scores[name] = binary_lookup_score(
            train_feature,
            train_bit_labels,
            eval_proc["q_bit_features"][name],
            eval_bit_labels,
        )
        if train_feature.shape[0] == train_rows["descent_mask"].numel() * 3:
            train_set = train_feature.reshape(train_rows["descent_mask"].numel(), -1)
            eval_set = eval_proc["q_bit_features"][name].reshape(eval_rows["descent_mask"].numel(), -1)
            set_scores[name] = lookup_score(train_set, train_rows["descent_mask"], eval_set, eval_rows["descent_mask"])

    explicit = {}
    eval_true_bits = eval_rows["label_bits"].to(torch.long)
    for name, eval_bits in sorted(eval_proc["q_explicit_bits"].items()):
        explicit[name] = metrics_from_bits(eval_bits, eval_true_bits)
    top_bit = sorted(
        ({"feature": name, **stats} for name, stats in bit_scores.items()),
        key=lambda item: (item.get("set_exact_accuracy", 0.0), item["binary_accuracy"]),
        reverse=True,
    )[:40]
    top_set = sorted(
        ({"feature": name, **stats} for name, stats in set_scores.items()),
        key=lambda item: (item["bit_majority_exact_accuracy"], item["bit_majority_bit_accuracy"]),
        reverse=True,
    )[:40]
    return {
        "bit_lookup_features": bit_scores,
        "set_lookup_features": set_scores,
        "top_bit_lookup_features": top_bit,
        "top_set_lookup_features": top_set,
        "explicit_predicates": explicit,
    }


def score_simple_quotients(train_proc: dict, eval_proc: dict, train_rows: dict, eval_rows: dict) -> dict:
    if "simple_neg_counts" not in train_proc:
        return {}
    right_masks = RIGHT_DESC_MASK.to(torch.long)
    train_neg = train_proc["simple_neg_counts"].to(torch.long)
    eval_neg = eval_proc["simple_neg_counts"].to(torch.long)
    train_first = train_proc["simple_first_exps"].to(torch.long)
    eval_first = eval_proc["simple_first_exps"].to(torch.long)
    train_width = train_proc["simple_widths"].to(torch.long)
    eval_width = eval_proc["simple_widths"].to(torch.long)

    zero_flags_train = train_neg.eq(0).to(torch.long)
    zero_flags_eval = eval_neg.eq(0).to(torch.long)
    clipped_train = train_neg.clamp_max(7)
    clipped_eval = eval_neg.clamp_max(7)
    first_clipped_train = train_first.clamp(min=-12, max=12) + 12
    first_clipped_eval = eval_first.clamp(min=-12, max=12) + 12
    width_delta_train = (train_width - train_width[:, :1]).clamp(min=-12, max=12) + 12
    width_delta_eval = (eval_width - eval_width[:, :1]).clamp(min=-12, max=12) + 12

    features = {
        "simple_zero_negative_flags": (zero_flags_train, zero_flags_eval),
        "simple_negative_count_clip7": (clipped_train, clipped_eval),
        "simple_first_exp_clip": (first_clipped_train, first_clipped_eval),
        "simple_neg_count_plus_first": (
            torch.cat([clipped_train, first_clipped_train], dim=1),
            torch.cat([clipped_eval, first_clipped_eval], dim=1),
        ),
        "simple_neg_count_plus_width_delta": (
            torch.cat([clipped_train, width_delta_train], dim=1),
            torch.cat([clipped_eval, width_delta_eval], dim=1),
        ),
    }
    lookup = {
        name: lookup_score(train_feature, train_rows["descent_mask"], eval_feature, eval_rows["descent_mask"])
        for name, (train_feature, eval_feature) in features.items()
    }

    # Interpretable direct rules from matrix-positive simple quotient flags.
    eval_zero = zero_flags_eval.to(torch.bool)
    desc_bits_by_factor = bits_from_mask(right_masks).to(torch.long)
    union_bits = torch.zeros(eval_zero.shape[0], 3, dtype=torch.long)
    intersection_bits = torch.ones(eval_zero.shape[0], 3, dtype=torch.long)
    has_any = eval_zero.any(dim=1)
    for row in range(eval_zero.shape[0]):
        active = torch.nonzero(eval_zero[row], as_tuple=False).flatten()
        if active.numel() == 0:
            intersection_bits[row] = 0
            continue
        bits = desc_bits_by_factor[active]
        union_bits[row] = bits.any(dim=0).to(torch.long)
        intersection_bits[row] = bits.all(dim=0).to(torch.long)
    min_neg_idx = eval_neg.argmin(dim=1)
    min_neg_bits = desc_bits_by_factor[min_neg_idx]
    direct = {
        "matrix_positive_simple_union_descent": {
            "examples_with_any_zero_negative_simple": int(has_any.sum().item()),
            **metrics_from_bits(union_bits, eval_rows["label_bits"]),
        },
        "matrix_positive_simple_intersection_descent": {
            "examples_with_any_zero_negative_simple": int(has_any.sum().item()),
            **metrics_from_bits(intersection_bits, eval_rows["label_bits"]),
        },
        "minimum_negative_count_simple_descent": metrics_from_bits(min_neg_bits, eval_rows["label_bits"]),
    }

    return {
        "lookup_features": lookup,
        "top_lookup_features": sorted(
            ({"feature": name, **stats} for name, stats in lookup.items()),
            key=lambda item: (item["bit_majority_exact_accuracy"], item["bit_majority_bit_accuracy"]),
            reverse=True,
        ),
        "direct_rules": direct,
        "simple_factor_right_descent_masks": [int(x) for x in right_masks.tolist()],
    }


def select_feature_names(ranked: list[dict], *, limit: int, max_columns: int = 64) -> list[str]:
    names = []
    for item in ranked:
        name = item["feature"]
        if name not in names:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def score_combined_features(
    train_proc: dict,
    eval_proc: dict,
    train_rows: dict,
    eval_rows: dict,
    boundary_scores: dict,
    generator_scores: dict,
    simple_scores: dict,
) -> dict:
    pools: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    boundary_names = select_feature_names(boundary_scores["top_by_exact"], limit=10)
    for name in boundary_names:
        pools[f"boundary::{name}"] = (
            train_proc["boundary_features"][name],
            eval_proc["boundary_features"][name],
        )

    q_names = select_feature_names(generator_scores["top_set_lookup_features"], limit=8)
    for name in q_names:
        train_feature = train_proc["q_bit_features"][name].reshape(train_rows["descent_mask"].numel(), -1)
        eval_feature = eval_proc["q_bit_features"][name].reshape(eval_rows["descent_mask"].numel(), -1)
        pools[f"quotient_set::{name}"] = (train_feature, eval_feature)

    if simple_scores:
        for name in select_feature_names(simple_scores["top_lookup_features"], limit=3):
            if name == "simple_zero_negative_flags":
                train_feature = train_proc["simple_neg_counts"].eq(0).to(torch.long)
                eval_feature = eval_proc["simple_neg_counts"].eq(0).to(torch.long)
            elif name == "simple_negative_count_clip7":
                train_feature = train_proc["simple_neg_counts"].to(torch.long).clamp_max(7)
                eval_feature = eval_proc["simple_neg_counts"].to(torch.long).clamp_max(7)
            elif name == "simple_first_exp_clip":
                train_feature = train_proc["simple_first_exps"].to(torch.long).clamp(min=-12, max=12) + 12
                eval_feature = eval_proc["simple_first_exps"].to(torch.long).clamp(min=-12, max=12) + 12
            elif name == "simple_neg_count_plus_first":
                train_feature = torch.cat(
                    [
                        train_proc["simple_neg_counts"].to(torch.long).clamp_max(7),
                        train_proc["simple_first_exps"].to(torch.long).clamp(min=-12, max=12) + 12,
                    ],
                    dim=1,
                )
                eval_feature = torch.cat(
                    [
                        eval_proc["simple_neg_counts"].to(torch.long).clamp_max(7),
                        eval_proc["simple_first_exps"].to(torch.long).clamp(min=-12, max=12) + 12,
                    ],
                    dim=1,
                )
            elif name == "simple_neg_count_plus_width_delta":
                train_width = train_proc["simple_widths"].to(torch.long)
                eval_width = eval_proc["simple_widths"].to(torch.long)
                train_feature = torch.cat(
                    [
                        train_proc["simple_neg_counts"].to(torch.long).clamp_max(7),
                        (train_width - train_width[:, :1]).clamp(min=-12, max=12) + 12,
                    ],
                    dim=1,
                )
                eval_feature = torch.cat(
                    [
                        eval_proc["simple_neg_counts"].to(torch.long).clamp_max(7),
                        (eval_width - eval_width[:, :1]).clamp(min=-12, max=12) + 12,
                    ],
                    dim=1,
                )
            else:
                continue
            pools[f"simple::{name}"] = (train_feature, eval_feature)

    names = sorted(pools)
    pair_scores = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            train_feature = torch.cat([pools[left][0], pools[right][0]], dim=1)
            eval_feature = torch.cat([pools[left][1], pools[right][1]], dim=1)
            pair_scores[f"{left} + {right}"] = lookup_score(
                train_feature,
                train_rows["descent_mask"],
                eval_feature,
                eval_rows["descent_mask"],
            )

    triple_scores = {}
    seed_names = []
    for prefix in ("boundary::", "quotient_set::", "simple::"):
        seed_names.extend([name for name in names if name.startswith(prefix)][:3])
    seed_names = sorted(set(seed_names))
    for i, a in enumerate(seed_names):
        for j, b in enumerate(seed_names[i + 1 :], start=i + 1):
            for c in seed_names[j + 1 :]:
                train_feature = torch.cat([pools[a][0], pools[b][0], pools[c][0]], dim=1)
                eval_feature = torch.cat([pools[a][1], pools[b][1], pools[c][1]], dim=1)
                triple_scores[f"{a} + {b} + {c}"] = lookup_score(
                    train_feature,
                    train_rows["descent_mask"],
                    eval_feature,
                    eval_rows["descent_mask"],
                )

    def top(scores: dict[str, dict]) -> list[dict]:
        return sorted(
            ({"feature": name, **stats} for name, stats in scores.items()),
            key=lambda item: (
                item["bit_majority_exact_accuracy"],
                item["bit_majority_bit_accuracy"],
                -item["unique_train_keys"],
            ),
            reverse=True,
        )[:40]

    return {
        "pool_names": names,
        "top_pairs": top(pair_scores),
        "top_triples": top(triple_scores),
        "num_pairs": len(pair_scores),
        "num_triples": len(triple_scores),
    }


def final_simple_oracle() -> dict:
    simple_mats = simple_mats_z(torch.device("cpu"))
    rows = []
    top2_to_mask = {}
    conflicts = defaultdict(Counter)
    for factor_id in range(simple_mats.shape[0]):
        mat = simple_mats[factor_id]
        occ = mat.ne(0).any(dim=(-1, -2))
        last = int(torch.nonzero(occ, as_tuple=False).flatten().max().item())
        top = mat[last : last + 1]
        prev = mat[last - 1 : last] if last > 0 else torch.zeros_like(top)
        top_cols = signed_column_masks_from_mats(top.unsqueeze(0))["any_col"][0, 0]
        prev_cols = signed_column_masks_from_mats(prev.unsqueeze(0))["any_col"][0, 0]
        key = (int(top_cols.item()), int(prev_cols.item()))
        mask = int(RIGHT_DESC_MASK[factor_id].item())
        top2_to_mask[key] = mask
        conflicts[key][mask] += 1
        rows.append(
            {
                "factor_id": factor_id,
                "artin_word_1_based": [int(x + 1) for x in GarsideFactor(PROPER_FACTOR_PERMS[factor_id]).artin_factors()],
                "left_descent_mask": int(LEFT_DESC_MASK[factor_id].item()),
                "right_descent_mask": mask,
                "top_col_mask": key[0],
                "previous_col_mask": key[1],
            }
        )
    return {
        "statement": "For an isolated final simple factor, (top column mask, previous column mask) determines right descent.",
        "num_top2_keys": len(top2_to_mask),
        "has_conflicts": any(len(counter) > 1 for counter in conflicts.values()),
        "simple_factors": rows,
    }


def summarize_rows(rows: dict[str, torch.Tensor]) -> dict:
    return {
        "n": int(rows["descent_mask"].numel()),
        "mask_counts": torch.bincount(rows["descent_mask"].to(torch.long), minlength=8).tolist(),
        "final_factor_counts": torch.bincount(rows["final_factor_id"].to(torch.long), minlength=len(PROPER_FACTOR_PERMS)).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search for theorem-shaped B4 Burau/descent rules over Z[v].")
    parser.add_argument("--data-dir", default="interp/data/generated/b4_l25_p2_n16777216")
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--length", type=int, default=25)
    parser.add_argument("--train-examples", type=int, default=131_072)
    parser.add_argument("--eval-examples", type=int, default=32_768)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=6)
    parser.add_argument("--simple-quotients", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="interp/artifacts/b4_l25_z_hidden_theorem_search/results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    absolute_depth = absolute_depth_for_length(args.length)
    train_rows = collect_factor_rows(
        data_dir=data_dir,
        num_shards=args.num_shards,
        split="train",
        max_examples=args.train_examples,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    eval_rows = collect_factor_rows(
        data_dir=data_dir,
        num_shards=args.num_shards,
        split="test",
        max_examples=args.eval_examples,
        shuffle=args.shuffle,
        seed=args.seed + 1,
    )
    train_proc = process_split(
        train_rows,
        length=args.length,
        absolute_depth=absolute_depth,
        batch_size=args.batch_size,
        radius=args.radius,
        simple_quotients=args.simple_quotients,
        device=device,
    )
    eval_proc = process_split(
        eval_rows,
        length=args.length,
        absolute_depth=absolute_depth,
        batch_size=args.batch_size,
        radius=args.radius,
        simple_quotients=args.simple_quotients,
        device=device,
    )
    boundary_scores = score_boundary_features(train_proc, eval_proc, train_rows["descent_mask"], eval_rows["descent_mask"])
    generator_scores = score_generator_quotients(train_proc, eval_proc, train_rows, eval_rows)
    simple_scores = score_simple_quotients(train_proc, eval_proc, train_rows, eval_rows) if args.simple_quotients else {}
    result = {
        "config": vars(args),
        "device": str(device),
        "interpretation": {
            "right_quotient_test": (
                "For each generator sigma_i, compute rho(beta) rho(sigma_i)^-1 over Z[v,v^-1]. "
                "If reduced Burau exposed positive right divisibility perfectly, simple membership-like "
                "features of this quotient would determine the i-th descent bit."
            ),
            "simple_quotient_test": (
                "Optionally quotient by all 22 proper simple factors and ask whether the vector of "
                "matrix-positive/simple quotient signatures determines the final descent set."
            ),
        },
        "train_summary": summarize_rows(train_rows),
        "eval_summary": summarize_rows(eval_rows),
        "max_abs_coeff": {
            "train": int(train_proc["max_abs_coeff"]),
            "eval": int(eval_proc["max_abs_coeff"]),
        },
        "final_simple_oracle": final_simple_oracle(),
        "boundary_lookup": boundary_scores,
        "generator_quotient": generator_scores,
        "combined_lookup": score_combined_features(
            train_proc,
            eval_proc,
            train_rows,
            eval_rows,
            boundary_scores,
            generator_scores,
            simple_scores,
        ),
    }
    if args.simple_quotients:
        result["simple_quotient"] = simple_scores
    atomic_json_dump(result, Path(args.out))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
