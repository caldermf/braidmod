#!/usr/bin/env python3
"""Audit B_4 boundary rules over Z[v].

The generated B_4 dataset stores factor ids, so we can replay the braid over
the integer Burau representation without regenerating normal forms.  This
script computes dense coefficient matrices over Z[v] with int64 arithmetic,
validates a small sample against the existing Python big-int implementation,
and then scores simple signed/support boundary invariants.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from braid_data import GarsideFactor, burau_polynomial_matrix  # noqa: E402
from interp.b4_data import discover_shards, load_shard, split_mask  # noqa: E402
from interp.generate_b4_dataset import (  # noqa: E402
    MATRIX_SIZE,
    PROPER_FACTOR_PERMS,
    RIGHT_DESC_MASK,
    absolute_depth_for_length,
)
from interp.mine_b4_boundary_rules import mask_to_bits  # noqa: E402


def build_simple_mats_z() -> torch.Tensor:
    sparse = []
    max_degree = 0
    for perm in PROPER_FACTOR_PERMS:
        word = [idx + 1 for idx in GarsideFactor(perm).artin_factors()]
        mat = burau_polynomial_matrix(word, n=4)
        factor_terms = []
        for i in range(MATRIX_SIZE):
            for j in range(MATRIX_SIZE):
                for exp, coeff in mat[i][j].items():
                    if exp < 0:
                        raise RuntimeError("positive simple factor produced negative degree")
                    if coeff:
                        max_degree = max(max_degree, exp)
                        factor_terms.append((exp, i, j, int(coeff)))
        sparse.append(factor_terms)
    out = torch.zeros(len(PROPER_FACTOR_PERMS), max_degree + 1, MATRIX_SIZE, MATRIX_SIZE, dtype=torch.long)
    for factor_id, terms in enumerate(sparse):
        for exp, i, j, coeff in terms:
            out[factor_id, exp, i, j] = coeff
    return out


SIMPLE_MATS_Z = build_simple_mats_z()


def collect_factor_rows(
    *,
    data_dir: Path,
    num_shards: int,
    split: str,
    max_examples: int,
) -> dict[str, torch.Tensor]:
    pieces: dict[str, list[torch.Tensor]] = defaultdict(list)
    emitted = 0
    for path in discover_shards(data_dir, num_shards=num_shards):
        payload = load_shard(path)
        meta = payload["metadata"]
        count = int(meta["sample_id_count"])
        start = int(meta["sample_id_start"])
        sample_ids = torch.arange(start, start + count, dtype=torch.long)
        rows = torch.nonzero(split_mask(sample_ids, split), as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        if max_examples > 0:
            rows = rows[: max(0, max_examples - emitted)]
        if rows.numel() == 0:
            break
        for key in ("factor_ids", "final_factor_id", "descent_mask", "label_bits"):
            pieces[key].append(payload[key][rows])
        pieces["sample_id"].append(sample_ids[rows])
        emitted += int(rows.numel())
        if max_examples > 0 and emitted >= max_examples:
            break
    if emitted == 0:
        raise RuntimeError(f"no examples loaded for split={split}")
    return {key: torch.cat(values, dim=0) for key, values in pieces.items()}


def right_multiply_dense_z(mat: torch.Tensor, factor_ids: torch.Tensor, simple_mats: torch.Tensor) -> torch.Tensor:
    batch_size, depth = mat.shape[:2]
    coeff = simple_mats[factor_ids.to(torch.long)]
    out = torch.zeros_like(mat)
    for shift in range(simple_mats.shape[1]):
        if shift >= depth:
            break
        for k in range(MATRIX_SIZE):
            src = mat[:, : depth - shift, :, k]
            for j in range(MATRIX_SIZE):
                active = coeff[:, shift, k, j]
                if bool(active.ne(0).any().item()):
                    out[:, shift:, :, j] += src * active.view(batch_size, 1, 1)
    return out


def dense_burau_z_for_factor_ids(factor_ids: torch.Tensor, *, length: int, absolute_depth: int) -> torch.Tensor:
    batch_size = factor_ids.shape[0]
    mat = torch.zeros(batch_size, absolute_depth, MATRIX_SIZE, MATRIX_SIZE, dtype=torch.long)
    for i in range(MATRIX_SIZE):
        mat[:, 0, i, i] = 1
    for pos in range(length):
        mat = right_multiply_dense_z(mat, factor_ids[:, pos], SIMPLE_MATS_Z)
    return mat


def exact_python_matrix_for_factor_ids(factor_ids: Iterable[int]) -> dict[tuple[int, int, int], int]:
    word: list[int] = []
    for factor_id in factor_ids:
        perm = PROPER_FACTOR_PERMS[int(factor_id)]
        word.extend(idx + 1 for idx in GarsideFactor(perm).artin_factors())
    mat = burau_polynomial_matrix(word, n=4)
    out = {}
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            for exp, coeff in mat[i][j].items():
                if coeff:
                    out[(int(exp), i, j)] = int(coeff)
    return out


def tensor_matrix_to_sparse(mat: torch.Tensor, row: int) -> dict[tuple[int, int, int], int]:
    one = mat[row]
    out = {}
    nz = torch.nonzero(one.ne(0), as_tuple=False)
    for exp, i, j in nz.tolist():
        out[(int(exp), int(i), int(j))] = int(one[exp, i, j].item())
    return out


def validate_against_python(factor_ids: torch.Tensor, mat: torch.Tensor, *, count: int) -> dict:
    mismatches = []
    checked = min(int(count), int(factor_ids.shape[0]))
    for row in range(checked):
        exact = exact_python_matrix_for_factor_ids(factor_ids[row].tolist())
        got = tensor_matrix_to_sparse(mat, row)
        if exact != got:
            missing = sorted(set(exact) - set(got))[:10]
            extra = sorted(set(got) - set(exact))[:10]
            wrong = sorted(key for key in set(exact).intersection(got) if exact[key] != got[key])[:10]
            mismatches.append(
                {
                    "row": row,
                    "missing": [list(x) for x in missing],
                    "extra": [list(x) for x in extra],
                    "wrong": [{"key": list(key), "exact": exact[key], "got": got[key]} for key in wrong],
                }
            )
            if len(mismatches) >= 5:
                break
    return {
        "checked": checked,
        "passed": not mismatches,
        "mismatches": mismatches,
    }


def support_features_z(mat: torch.Tensor) -> dict[str, torch.Tensor]:
    occupied = mat.ne(0).any(dim=(-1, -2))
    any_support = occupied.any(dim=1)
    first = occupied.to(torch.long).argmax(dim=1)
    last = mat.shape[1] - 1 - occupied.flip(dims=[1]).to(torch.long).argmax(dim=1)
    first = torch.where(any_support, first, torch.zeros_like(first))
    last = torch.where(any_support, last, torch.zeros_like(last))
    return {"first": first, "last": last, "width": last - first + 1}


def gather_forward_mats(mat: torch.Tensor, starts: torch.Tensor, radius: int) -> torch.Tensor:
    offsets = torch.arange(radius + 1, dtype=torch.long)
    idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
    valid = (idx >= 0) & (idx < mat.shape[1])
    idx = idx.clamp(0, mat.shape[1] - 1)
    idx4 = idx.view(idx.shape[0], idx.shape[1], 1, 1).expand(-1, -1, MATRIX_SIZE, MATRIX_SIZE)
    out = mat.gather(1, idx4)
    return torch.where(valid.view(valid.shape[0], valid.shape[1], 1, 1), out, torch.zeros_like(out))


def gather_backward_mats(mat: torch.Tensor, starts: torch.Tensor, radius: int) -> torch.Tensor:
    offsets = torch.arange(radius + 1, dtype=torch.long)
    idx = starts.unsqueeze(1) - offsets.unsqueeze(0)
    valid = (idx >= 0) & (idx < mat.shape[1])
    idx = idx.clamp(0, mat.shape[1] - 1)
    idx4 = idx.view(idx.shape[0], idx.shape[1], 1, 1).expand(-1, -1, MATRIX_SIZE, MATRIX_SIZE)
    out = mat.gather(1, idx4)
    return torch.where(valid.view(valid.shape[0], valid.shape[1], 1, 1), out, torch.zeros_like(out))


def support_token_from_mats(band: torch.Tensor) -> torch.Tensor:
    bits = band.ne(0).view(band.shape[0], band.shape[1], 9).to(torch.long)
    weights = torch.tensor([1 << i for i in range(9)], dtype=torch.long).view(1, 1, 9)
    return (bits * weights).sum(dim=-1)


def sign_token_from_mats(band: torch.Tensor) -> torch.Tensor:
    signs = torch.zeros_like(band, dtype=torch.long)
    signs = torch.where(band.lt(0), torch.ones_like(signs), signs)
    signs = torch.where(band.gt(0), torch.full_like(signs, 2), signs)
    digits = signs.view(signs.shape[0], signs.shape[1], 9)
    weights = torch.tensor([3**i for i in range(9)], dtype=torch.long).view(1, 1, 9)
    return (digits * weights).sum(dim=-1)


def clipped_coeff_token_from_mats(band: torch.Tensor, clip: int) -> torch.Tensor:
    clipped = band.clamp(min=-clip, max=clip) + clip
    digits = clipped.view(clipped.shape[0], clipped.shape[1], 9).to(torch.long)
    base = 2 * clip + 1
    weights = torch.tensor([base**i for i in range(9)], dtype=torch.long).view(1, 1, 9)
    return (digits * weights).sum(dim=-1)


def column_masks_from_mats(band: torch.Tensor) -> dict[str, torch.Tensor]:
    any_cols = band.ne(0).any(dim=2).to(torch.long)
    pos_cols = band.gt(0).any(dim=2).to(torch.long)
    neg_cols = band.lt(0).any(dim=2).to(torch.long)
    weights = torch.tensor([1, 2, 4], dtype=torch.long).view(1, 1, 3)
    return {
        "any_col": (any_cols * weights).sum(dim=-1),
        "pos_col": (pos_cols * weights).sum(dim=-1),
        "neg_col": (neg_cols * weights).sum(dim=-1),
    }


def row_masks_from_mats(band: torch.Tensor) -> dict[str, torch.Tensor]:
    any_rows = band.ne(0).any(dim=3).to(torch.long)
    pos_rows = band.gt(0).any(dim=3).to(torch.long)
    neg_rows = band.lt(0).any(dim=3).to(torch.long)
    weights = torch.tensor([1, 2, 4], dtype=torch.long).view(1, 1, 3)
    return {
        "any_row": (any_rows * weights).sum(dim=-1),
        "pos_row": (pos_rows * weights).sum(dim=-1),
        "neg_row": (neg_rows * weights).sum(dim=-1),
    }


def right_divide_z(mat: torch.Tensor, generator_idx: int) -> torch.Tensor:
    batch, depth = mat.shape[:2]
    out = torch.zeros(batch, depth + 2, MATRIX_SIZE, MATRIX_SIZE, dtype=torch.long)

    if generator_idx == 0:
        terms = {
            0: [(0, -2, -1)],
            1: [(0, -1, -1), (1, 0, 1)],
            2: [(2, 0, 1)],
        }
    elif generator_idx == 1:
        terms = {
            0: [(0, 0, 1), (1, -1, -1)],
            1: [(1, -2, -1)],
            2: [(1, -1, -1), (2, 0, 1)],
        }
    elif generator_idx == 2:
        terms = {
            0: [(0, 0, 1)],
            1: [(1, 0, 1), (2, -1, -1)],
            2: [(2, -2, -1)],
        }
    else:
        raise ValueError("generator_idx must be 0, 1, or 2")

    for dest_col, pieces in terms.items():
        for src_col, shift, coeff in pieces:
            start = 2 + shift
            out[:, start : start + depth, :, dest_col] += coeff * mat[:, :, :, src_col]
    return out


def rows_as_tuples(x: torch.Tensor) -> list[tuple[int, ...]]:
    x = x.cpu().to(torch.long)
    if x.ndim == 1:
        return [(int(v),) for v in x.tolist()]
    return [tuple(int(v) for v in row.tolist()) for row in x]


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
        bits = torch.tensor([(mask >> i) & 1 for i in range(3)], dtype=torch.long)
        bit_counts[key] += bits
        global_counter[mask] += 1
        global_bits += bits

    fallback_mask = int(global_counter.most_common(1)[0][0])
    fallback_bits = (global_bits * 2 >= len(train_masks_list)).to(torch.long)
    mask_table = {key: int(counter.most_common(1)[0][0]) for key, counter in mask_counts.items()}
    bit_table = {key: (counts * 2 >= total_counts[key]).to(torch.long) for key, counts in bit_counts.items()}

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
    pred_bits_from_mask = mask_to_bits(pred_masks_t)
    return {
        "unique_train_keys": int(len(mask_table)),
        "coverage": float(seen / max(1, len(eval_keys))),
        "mask_majority_exact_accuracy": float((pred_masks_t == eval_masks_cpu).float().mean().item()),
        "mask_majority_bit_accuracy": float(pred_bits_from_mask.eq(eval_bits).float().mean().item()),
        "bit_majority_exact_accuracy": float(pred_bits_t.eq(eval_bits).all(dim=1).float().mean().item()),
        "bit_majority_bit_accuracy": float(pred_bits_t.eq(eval_bits).float().mean().item()),
        "bit_majority_per_label_accuracy": [float(x) for x in pred_bits_t.eq(eval_bits).float().mean(dim=0).tolist()],
    }


def metrics_from_mask_prediction(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> dict:
    pred_mask = pred_mask.cpu().to(torch.long)
    true_mask = true_mask.cpu().to(torch.long)
    pred_bits = mask_to_bits(pred_mask)
    true_bits = mask_to_bits(true_mask)
    eq_bits = pred_bits.eq(true_bits)
    return {
        "exact_accuracy": float((pred_mask == true_mask).float().mean().item()),
        "bit_accuracy": float(eq_bits.float().mean().item()),
        "per_label_accuracy": [float(x) for x in eq_bits.float().mean(dim=0).tolist()],
        "pred_mask_counts": torch.bincount(pred_mask, minlength=8).tolist(),
    }


def build_features_from_mat(mat: torch.Tensor, *, radius: int, coeff_clip: int) -> tuple[dict[str, torch.Tensor], dict]:
    feats = support_features_z(mat)
    lead = gather_forward_mats(mat, feats["first"], radius)
    trail = gather_backward_mats(mat, feats["last"], radius)

    lead_support = support_token_from_mats(lead)
    trail_support = support_token_from_mats(trail)
    lead_sign = sign_token_from_mats(lead)
    trail_sign = sign_token_from_mats(trail)
    lead_clip = clipped_coeff_token_from_mats(lead, coeff_clip)
    trail_clip = clipped_coeff_token_from_mats(trail, coeff_clip)
    lead_cols = column_masks_from_mats(lead)
    trail_cols = column_masks_from_mats(trail)
    lead_rows = row_masks_from_mats(lead)
    trail_rows = row_masks_from_mats(trail)

    features: dict[str, torch.Tensor] = {
        "z_width": feats["width"].unsqueeze(1),
        "z_degree_bounds": torch.stack([feats["first"], feats["last"], feats["width"]], dim=1),
    }
    for r in range(radius + 1):
        features[f"z_lead_support_r{r}"] = lead_support[:, : r + 1]
        features[f"z_trail_support_r{r}"] = trail_support[:, : r + 1]
        features[f"z_both_support_r{r}"] = torch.cat([lead_support[:, : r + 1], trail_support[:, : r + 1]], dim=1)
        features[f"z_lead_sign_r{r}"] = lead_sign[:, : r + 1]
        features[f"z_trail_sign_r{r}"] = trail_sign[:, : r + 1]
        features[f"z_both_sign_r{r}"] = torch.cat([lead_sign[:, : r + 1], trail_sign[:, : r + 1]], dim=1)
        features[f"z_lead_clip_r{r}"] = lead_clip[:, : r + 1]
        features[f"z_trail_clip_r{r}"] = trail_clip[:, : r + 1]
        features[f"z_both_clip_r{r}"] = torch.cat([lead_clip[:, : r + 1], trail_clip[:, : r + 1]], dim=1)
        features[f"z_lead_any_col_r{r}"] = lead_cols["any_col"][:, : r + 1]
        features[f"z_trail_any_col_r{r}"] = trail_cols["any_col"][:, : r + 1]
        features[f"z_both_any_col_r{r}"] = torch.cat(
            [lead_cols["any_col"][:, : r + 1], trail_cols["any_col"][:, : r + 1]],
            dim=1,
        )
        features[f"z_lead_pos_col_r{r}"] = lead_cols["pos_col"][:, : r + 1]
        features[f"z_trail_pos_col_r{r}"] = trail_cols["pos_col"][:, : r + 1]
        features[f"z_both_pos_col_r{r}"] = torch.cat(
            [lead_cols["pos_col"][:, : r + 1], trail_cols["pos_col"][:, : r + 1]],
            dim=1,
        )
        features[f"z_lead_neg_col_r{r}"] = lead_cols["neg_col"][:, : r + 1]
        features[f"z_trail_neg_col_r{r}"] = trail_cols["neg_col"][:, : r + 1]
        features[f"z_both_neg_col_r{r}"] = torch.cat(
            [lead_cols["neg_col"][:, : r + 1], trail_cols["neg_col"][:, : r + 1]],
            dim=1,
        )
        features[f"z_lead_any_row_r{r}"] = lead_rows["any_row"][:, : r + 1]
        features[f"z_trail_any_row_r{r}"] = trail_rows["any_row"][:, : r + 1]
        features[f"z_both_any_row_r{r}"] = torch.cat(
            [lead_rows["any_row"][:, : r + 1], trail_rows["any_row"][:, : r + 1]],
            dim=1,
        )

    div_parts = []
    for gen_idx in range(3):
        qmat = right_divide_z(mat, gen_idx)
        qfeats = support_features_z(qmat)
        qlead = gather_forward_mats(qmat, qfeats["first"], min(radius, 4))
        qtrail = gather_backward_mats(qmat, qfeats["last"], min(radius, 4))
        qlead_sign = sign_token_from_mats(qlead)
        qtrail_sign = sign_token_from_mats(qtrail)
        qlead_support = support_token_from_mats(qlead)
        qtrail_support = support_token_from_mats(qtrail)
        min_delta = qfeats["first"] - 2 - feats["first"]
        max_delta = qfeats["last"] - 2 - feats["last"]
        width_delta = qfeats["width"] - feats["width"]
        div_parts.append(torch.stack([min_delta, max_delta, width_delta], dim=1))
        features[f"z_right_div_s{gen_idx + 1}_frontier"] = torch.stack(
            [min_delta, max_delta, width_delta, qlead_sign[:, 0], qtrail_sign[:, 0]],
            dim=1,
        )
        for r in range(min(radius, 4) + 1):
            features[f"z_right_div_s{gen_idx + 1}_both_sign_r{r}"] = torch.cat(
                [qlead_sign[:, : r + 1], qtrail_sign[:, : r + 1]],
                dim=1,
            )
            features[f"z_right_div_s{gen_idx + 1}_both_support_r{r}"] = torch.cat(
                [qlead_support[:, : r + 1], qtrail_support[:, : r + 1]],
                dim=1,
            )
    features["z_right_div_all_deltas"] = torch.cat(div_parts, dim=1)

    explicit = {
        "z_trailing_top_any_col": trail_cols["any_col"][:, 0],
        "z_trailing_top_pos_col": trail_cols["pos_col"][:, 0],
        "z_trailing_top_neg_col": trail_cols["neg_col"][:, 0],
        "z_leading_top_any_col": lead_cols["any_col"][:, 0],
    }
    summary = {
        "min_degree_range": [int(feats["first"].min().item()), int(feats["first"].max().item())],
        "max_degree_range": [int(feats["last"].min().item()), int(feats["last"].max().item())],
        "width_range": [int(feats["width"].min().item()), int(feats["width"].max().item())],
        "max_abs_coeff": int(mat.abs().max().item()),
    }
    return features, {"explicit_predictions": explicit, "summary": summary}


def score_feature_set(train_features: dict[str, torch.Tensor], eval_features: dict[str, torch.Tensor], train_masks: torch.Tensor, eval_masks: torch.Tensor) -> dict:
    individual = {}
    for name in sorted(train_features):
        individual[name] = lookup_score(train_features[name], train_masks, eval_features[name], eval_masks)
    sorted_exact = sorted(
        individual.items(),
        key=lambda item: (
            item[1]["bit_majority_exact_accuracy"],
            item[1]["bit_majority_bit_accuracy"],
            -item[1]["unique_train_keys"],
        ),
        reverse=True,
    )
    sorted_bit = sorted(
        individual.items(),
        key=lambda item: (
            item[1]["bit_majority_bit_accuracy"],
            item[1]["bit_majority_exact_accuracy"],
            -item[1]["unique_train_keys"],
        ),
        reverse=True,
    )
    return {
        "top_by_exact": [{"feature": name, **stats} for name, stats in sorted_exact[:50]],
        "top_by_bit": [{"feature": name, **stats} for name, stats in sorted_bit[:50]],
        "num_features": len(individual),
    }


def process_split(batch: dict[str, torch.Tensor], *, length: int, absolute_depth: int, batch_size: int, radius: int, coeff_clip: int, validate_examples: int) -> dict:
    feature_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    explicit_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    summaries = []
    validation = None
    total = int(batch["factor_ids"].shape[0])
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        factor_ids = batch["factor_ids"][start:stop].to(torch.long)
        mat = dense_burau_z_for_factor_ids(factor_ids, length=length, absolute_depth=absolute_depth)
        if validation is None and validate_examples > 0:
            validation = validate_against_python(factor_ids, mat, count=validate_examples)
        features, extra = build_features_from_mat(mat, radius=radius, coeff_clip=coeff_clip)
        for name, values in features.items():
            feature_chunks[name].append(values.cpu())
        for name, values in extra["explicit_predictions"].items():
            explicit_chunks[name].append(values.cpu())
        summaries.append(extra["summary"])
    features_out = {name: torch.cat(values, dim=0) for name, values in feature_chunks.items()}
    explicit_out = {name: torch.cat(values, dim=0) for name, values in explicit_chunks.items()}
    max_abs = max(item["max_abs_coeff"] for item in summaries)
    summary = {
        "max_abs_coeff": int(max_abs),
        "min_degree_range": [
            min(item["min_degree_range"][0] for item in summaries),
            max(item["min_degree_range"][1] for item in summaries),
        ],
        "max_degree_range": [
            min(item["max_degree_range"][0] for item in summaries),
            max(item["max_degree_range"][1] for item in summaries),
        ],
        "width_range": [
            min(item["width_range"][0] for item in summaries),
            max(item["width_range"][1] for item in summaries),
        ],
    }
    return {
        "features": features_out,
        "explicit_predictions": explicit_out,
        "summary": summary,
        "validation": validation,
    }


def atomic_json_dump(payload: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit signed B_4 Burau boundary features over Z[v].")
    parser.add_argument("--data-dir", default="interp/data/generated/b4_l25_p2_n16777216")
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--length", type=int, default=25)
    parser.add_argument("--train-examples", type=int, default=65_536)
    parser.add_argument("--eval-examples", type=int, default=16_384)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--coeff-clip", type=int, default=3)
    parser.add_argument("--validate-examples", type=int, default=16)
    parser.add_argument("--out", default="interp/artifacts/b4_l25_p2_integer_boundary/results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    absolute_depth = absolute_depth_for_length(args.length)
    train = collect_factor_rows(
        data_dir=Path(args.data_dir),
        num_shards=args.num_shards,
        split="train",
        max_examples=args.train_examples,
    )
    eval_batch = collect_factor_rows(
        data_dir=Path(args.data_dir),
        num_shards=args.num_shards,
        split="test",
        max_examples=args.eval_examples,
    )
    train_proc = process_split(
        train,
        length=args.length,
        absolute_depth=absolute_depth,
        batch_size=args.batch_size,
        radius=args.radius,
        coeff_clip=args.coeff_clip,
        validate_examples=args.validate_examples,
    )
    eval_proc = process_split(
        eval_batch,
        length=args.length,
        absolute_depth=absolute_depth,
        batch_size=args.batch_size,
        radius=args.radius,
        coeff_clip=args.coeff_clip,
        validate_examples=0,
    )
    scores = score_feature_set(
        train_proc["features"],
        eval_proc["features"],
        train["descent_mask"].to(torch.long),
        eval_batch["descent_mask"].to(torch.long),
    )
    explicit_scores = {
        name: metrics_from_mask_prediction(pred, eval_batch["descent_mask"])
        for name, pred in eval_proc["explicit_predictions"].items()
    }
    result = {
        "config": vars(args),
        "validation": train_proc["validation"],
        "train_summary": {
            "n": int(train["descent_mask"].numel()),
            "mask_counts": torch.bincount(train["descent_mask"].to(torch.long), minlength=8).tolist(),
            **train_proc["summary"],
        },
        "eval_summary": {
            "n": int(eval_batch["descent_mask"].numel()),
            "mask_counts": torch.bincount(eval_batch["descent_mask"].to(torch.long), minlength=8).tolist(),
            **eval_proc["summary"],
        },
        "explicit_column_rules": explicit_scores,
        "scores": scores,
    }
    atomic_json_dump(result, Path(args.out))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
