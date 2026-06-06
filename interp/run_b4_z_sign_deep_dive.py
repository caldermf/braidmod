#!/usr/bin/env python3
"""Deeper B4 Z[v] sign-token interpretability experiments.

This script is meant to push beyond first-pass "boundary attention" evidence.
It runs three experiments against the trained sign-token transformer:

1. circuit/probe classifiers from internal activations;
2. prefix-fixed final-factor counterfactual patching;
3. right-division quotient frontier lookup rules.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.b4_data import discover_shards  # noqa: E402
from interp.b4_interp import (  # noqa: E402
    gather_relative_window,
    load_transformer_checkpoint,
    metrics_from_logits,
    multilabel_logit_score,
    run_with_cache,
    support_features,
    zero_except_windows,
    zero_head_positions_hook,
    zero_hook,
)
from interp.b4_z_sign import (  # noqa: E402
    B4FactorBatchIterable,
    dense_burau_z_for_factor_ids,
    factor_ids_to_z_sign_tokens,
    simple_mats_z_cpu,
    simple_mats_z,
    sign_tokens_from_dense,
)
from interp.generate_b4_dataset import (  # noqa: E402
    LEFT_DESC_MASK,
    MATRIX_SIZE,
    PROPER_FACTOR_PERMS,
    RIGHT_DESC_MASK,
)
from interp.train_b4_transformer import atomic_json_dump, resolve_device, set_seed  # noqa: E402
from braid_data import GarsideFactor  # noqa: E402


TOP_ZSIGN_HEADS = [(1, 4), (2, 2), (0, 5), (0, 2)]


def apply_model_input_transform(tokens: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 0:
        return tokens
    return zero_except_windows(tokens, radius, leading=True, trailing=True)


def apply_batch_input_transform(batch: dict, radius: int) -> dict:
    if radius < 0:
        return batch
    out = dict(batch)
    out["tokens"] = apply_model_input_transform(batch["tokens"], radius)
    return out


def infer_boundary_radius_from_checkpoint(checkpoint: dict) -> int:
    transform = str(checkpoint.get("input_transform", "none"))
    prefix = "zero_except_leading_and_trailing_windows_radius_"
    if transform.startswith(prefix):
        return int(transform[len(prefix) :])
    return -1


def make_factor_loader(
    shard_paths: list[Path],
    *,
    split: str,
    batch_size: int,
    seed: int,
    shuffle: bool,
    max_examples: int,
) -> DataLoader:
    dataset = B4FactorBatchIterable(
        shard_paths,
        split=split,
        batch_size=batch_size,
        seed=seed,
        epoch=0,
        shuffle_shards=shuffle,
        shuffle_rows=shuffle,
        max_examples=max_examples,
    )
    return DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=False)


def bits_from_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(torch.long)
    return torch.stack([((mask >> idx) & 1) for idx in range(3)], dim=1).to(torch.float32)


def mask_from_bits(bits: torch.Tensor) -> torch.Tensor:
    bits = bits.to(torch.long)
    weights = torch.tensor([1, 2, 4], dtype=torch.long, device=bits.device).view(1, 3)
    return (bits * weights).sum(dim=1)


def label_bits_from_final_ids(final_ids: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    masks = RIGHT_DESC_MASK.to(device=device)[final_ids.to(device=device, dtype=torch.long)]
    return masks.to(torch.long), bits_from_mask(masks).to(device)


@torch.no_grad()
def collect_factor_batch(
    loader: DataLoader,
    *,
    device: torch.device,
    length: int,
    absolute_depth: int,
    min_examples: int,
) -> dict:
    simple_mats = simple_mats_z(device)
    pieces: dict[str, list[torch.Tensor]] = defaultdict(list)
    count = 0
    for batch in loader:
        factor_ids = batch["factor_ids"].to(device)
        tokens = factor_ids_to_z_sign_tokens(
            factor_ids,
            length=length,
            absolute_depth=absolute_depth,
            simple_mats=simple_mats,
        )
        pieces["factor_ids"].append(factor_ids.cpu())
        pieces["tokens"].append(tokens.cpu())
        pieces["label_bits"].append(batch["label_bits"].cpu())
        pieces["descent_mask"].append(batch["descent_mask"].cpu())
        pieces["final_factor_id"].append(batch["final_factor_id"].cpu())
        pieces["sample_id"].append(batch["sample_id"].cpu())
        count += int(tokens.shape[0])
        if count >= min_examples:
            break
    out = {key: torch.cat(values, dim=0)[:min_examples] for key, values in pieces.items()}
    for key in ("factor_ids", "tokens", "label_bits", "descent_mask", "final_factor_id", "sample_id"):
        out[key] = out[key].to(device)
    out["label_bits"] = out["label_bits"].to(torch.float32)
    return out


def add_bias(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)], dim=1)


def normalize_pair(train_x: torch.Tensor, eval_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-4)
    return (train_x - mean) / std, (eval_x - mean) / std


def ridge_solve(x: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    x = add_bias(x.to(torch.float32))
    y = y.to(torch.float32)
    gram = x.T @ x
    reg = ridge * torch.eye(gram.shape[0], dtype=x.dtype, device=x.device)
    reg[-1, -1] = 0.0
    return torch.linalg.solve(gram + reg, x.T @ y)


@torch.no_grad()
def collect_activation_reps(
    model,
    tokens: torch.Tensor,
    *,
    chunk_size: int,
) -> dict[str, torch.Tensor]:
    resid_sites = ["hook_resid_embed"]
    resid_sites.extend(f"blocks.{layer}.hook_resid_post" for layer in range(len(model.blocks)))
    resid_sites.append("hook_final_hidden")
    names_filter = set(resid_sites) | {
        f"blocks.{layer}.hook_attn_head_out"
        for layer in range(len(model.blocks))
    }
    pieces: dict[str, list[torch.Tensor]] = defaultdict(list)
    for start in range(0, tokens.shape[0], chunk_size):
        chunk = tokens[start : start + chunk_size]
        _, cache = run_with_cache(model, chunk, names_filter=names_filter)
        feats = support_features(chunk)
        rows = torch.arange(chunk.shape[0], device=chunk.device)
        leading = feats["first"] + 1
        trailing = feats["last"] + 1
        support = torch.cat(
            [torch.zeros(chunk.shape[0], 1, dtype=torch.bool, device=chunk.device), feats["support"]],
            dim=1,
        )
        support_weight = support.to(torch.float32)
        support_weight = support_weight / support_weight.sum(dim=1, keepdim=True).clamp_min(1.0)

        for site in resid_sites:
            if site not in cache:
                continue
            value = cache[site]
            short = site.replace("blocks.", "l").replace(".hook_", "_").replace("hook_", "")
            pieces[f"{short}_cls"].append(value[:, 0].detach().cpu())
            pieces[f"{short}_leading"].append(value[rows, leading].detach().cpu())
            pieces[f"{short}_trailing"].append(value[rows, trailing].detach().cpu())
            pieces[f"{short}_support_mean"].append((value * support_weight.unsqueeze(-1)).sum(dim=1).detach().cpu())

        for layer, head in TOP_ZSIGN_HEADS:
            key = f"blocks.{layer}.hook_attn_head_out"
            if key not in cache:
                continue
            value = cache[key]
            if head < value.shape[1]:
                pieces[f"l{layer}h{head}_headout_cls"].append(value[:, head, 0].detach().cpu())
                pieces[f"l{layer}h{head}_headout_leading"].append(value[rows, head, leading].detach().cpu())
                pieces[f"l{layer}h{head}_headout_trailing"].append(value[rows, head, trailing].detach().cpu())
    return {key: torch.cat(values, dim=0).to(tokens.device) for key, values in pieces.items()}


def multilabel_metrics_from_bits(pred_bits: torch.Tensor, true_bits: torch.Tensor, model_bits: torch.Tensor | None = None) -> dict:
    pred_bits = pred_bits.to(torch.float32)
    true_bits = true_bits.to(torch.float32)
    out = {
        "exact_set_accuracy": float(pred_bits.eq(true_bits).all(dim=1).float().mean().item()),
        "bit_accuracy": float(pred_bits.eq(true_bits).float().mean().item()),
        "per_label_accuracy": [float(x) for x in pred_bits.eq(true_bits).float().mean(dim=0).tolist()],
        "pred_mask_counts": torch.bincount(mask_from_bits(pred_bits), minlength=8).cpu().tolist(),
        "true_mask_counts": torch.bincount(mask_from_bits(true_bits), minlength=8).cpu().tolist(),
        "n": int(true_bits.shape[0]),
    }
    if model_bits is not None:
        out["agreement_with_model_exact"] = float(pred_bits.eq(model_bits).all(dim=1).float().mean().item())
        out["agreement_with_model_bit"] = float(pred_bits.eq(model_bits).float().mean().item())
    return out


def binary_probe_summary(
    train_x: torch.Tensor,
    train_bits: torch.Tensor,
    eval_x: torch.Tensor,
    eval_bits: torch.Tensor,
    model_bits: torch.Tensor,
    *,
    ridge: float,
) -> dict:
    train_x, eval_x = normalize_pair(train_x, eval_x)
    train_signed = train_bits.to(torch.float32).mul(2).sub(1)
    weights = ridge_solve(train_x, train_signed, ridge=ridge)
    scores = add_bias(eval_x.to(torch.float32)) @ weights
    pred_bits = (scores >= 0).to(torch.float32)
    out = multilabel_metrics_from_bits(pred_bits, eval_bits, model_bits=model_bits)
    out["probe_kind"] = "ridge_binary_descent_bits"
    return out


def final_factor_probe_summary(
    train_x: torch.Tensor,
    train_final_ids: torch.Tensor,
    eval_x: torch.Tensor,
    eval_final_ids: torch.Tensor,
    eval_bits: torch.Tensor,
    model_bits: torch.Tensor,
    *,
    ridge: float,
) -> dict:
    train_x, eval_x = normalize_pair(train_x, eval_x)
    train_y = train_final_ids.to(torch.long)
    eval_y = eval_final_ids.to(torch.long)
    y_onehot = F.one_hot(train_y, num_classes=int(RIGHT_DESC_MASK.numel())).to(torch.float32)
    weights = ridge_solve(train_x, y_onehot, ridge=ridge)
    scores = add_bias(eval_x.to(torch.float32)) @ weights
    pred_final = scores.argmax(dim=1)
    pred_masks = RIGHT_DESC_MASK.to(device=eval_x.device)[pred_final]
    pred_bits = bits_from_mask(pred_masks).to(eval_x.device)
    out = multilabel_metrics_from_bits(pred_bits, eval_bits, model_bits=model_bits)
    out["final_factor_accuracy"] = float((pred_final == eval_y).float().mean().item())
    out["probe_kind"] = "ridge_final_factor_then_exact_descent_lookup"
    out["pred_final_factor_counts"] = torch.bincount(pred_final, minlength=int(RIGHT_DESC_MASK.numel())).cpu().tolist()
    return out


def simple_factor_latent_targets(device: torch.device) -> dict[str, torch.Tensor]:
    mats = simple_mats_z_cpu()
    top_col_masks = []
    prev_col_masks = []
    artin_lengths = []
    for factor_id in range(mats.shape[0]):
        occupied = mats[factor_id].ne(0).any(dim=(-1, -2))
        max_degree = int(torch.nonzero(occupied, as_tuple=False).flatten().max().item())
        top_cols = mats[factor_id, max_degree].ne(0).any(dim=0)
        top_mask = sum((1 << idx) for idx, value in enumerate(top_cols.tolist()) if value)
        if max_degree > 0:
            prev_cols = mats[factor_id, max_degree - 1].ne(0).any(dim=0)
            prev_mask = sum((1 << idx) for idx, value in enumerate(prev_cols.tolist()) if value)
        else:
            prev_mask = 0
        top_col_masks.append(top_mask)
        prev_col_masks.append(prev_mask)
        artin_lengths.append(len(GarsideFactor(PROPER_FACTOR_PERMS[factor_id]).artin_factors()))

    top = torch.tensor(top_col_masks, dtype=torch.long)
    prev = torch.tensor(prev_col_masks, dtype=torch.long)
    left = LEFT_DESC_MASK.to(torch.long)
    right = RIGHT_DESC_MASK.to(torch.long)
    artin = torch.tensor(artin_lengths, dtype=torch.long)
    return {
        "final_factor_id": torch.arange(len(PROPER_FACTOR_PERMS), dtype=torch.long, device=device),
        "right_descent_mask": right.to(device),
        "left_descent_mask": left.to(device),
        "left_right_descent_masks": (left + 8 * right).to(device),
        "top_col_mask": top.to(device),
        "top2_col_masks": (top + 8 * prev).to(device),
        "left_top2_col_masks": (left + 8 * top + 64 * prev).to(device),
        "artin_length": artin.to(device),
    }


def remap_train_eval_targets(train_raw: torch.Tensor, eval_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    classes = torch.unique(train_raw.detach().cpu().to(torch.long), sorted=True).to(train_raw.device)
    train_cls = torch.bucketize(train_raw.to(torch.long), classes)
    eval_cls = torch.bucketize(eval_raw.to(torch.long), classes)
    in_range = eval_cls < classes.numel()
    eval_matches = torch.zeros_like(in_range)
    eval_matches[in_range] = classes[eval_cls[in_range]] == eval_raw.to(torch.long)[in_range]
    eval_cls = torch.where(eval_matches, eval_cls, torch.zeros_like(eval_cls))
    return classes, train_cls, eval_cls


def latent_rule_probe_summary(
    train_x: torch.Tensor,
    train_raw: torch.Tensor,
    train_bits: torch.Tensor,
    eval_x: torch.Tensor,
    eval_raw: torch.Tensor,
    eval_bits: torch.Tensor,
    model_bits: torch.Tensor,
    *,
    ridge: float,
) -> dict:
    train_x, eval_x = normalize_pair(train_x, eval_x)
    classes, train_cls, eval_cls = remap_train_eval_targets(train_raw, eval_raw)
    train_onehot = F.one_hot(train_cls.to(torch.long), num_classes=int(classes.numel())).to(torch.float32)
    weights = ridge_solve(train_x, train_onehot, ridge=ridge)
    scores = add_bias(eval_x.to(torch.float32)) @ weights
    pred_cls = scores.argmax(dim=1)
    pred_raw = classes[pred_cls]
    target_accuracy = float((pred_raw == eval_raw.to(torch.long)).float().mean().item())

    majority_by_raw: dict[int, Counter] = defaultdict(Counter)
    for raw, mask in zip(train_raw.detach().cpu().tolist(), mask_from_bits(train_bits).detach().cpu().tolist(), strict=True):
        majority_by_raw[int(raw)][int(mask)] += 1
    raw_to_mask = {raw: counter.most_common(1)[0][0] for raw, counter in majority_by_raw.items()}
    fallback = Counter(mask_from_bits(train_bits).detach().cpu().tolist()).most_common(1)[0][0]
    pred_masks = torch.tensor([raw_to_mask.get(int(raw), int(fallback)) for raw in pred_raw.detach().cpu().tolist()], device=eval_bits.device)
    pred_bits = bits_from_mask(pred_masks).to(eval_bits.device)
    out = multilabel_metrics_from_bits(pred_bits, eval_bits, model_bits=model_bits)
    out.update(
        {
            "probe_kind": "ridge_latent_multiclass_then_majority_descent_lookup",
            "latent_target_accuracy": target_accuracy,
            "num_latent_classes": int(classes.numel()),
            "latent_values": [int(x) for x in classes.detach().cpu().tolist()],
        }
    )
    return out


@torch.no_grad()
def semantic_probe_experiment(
    model,
    train_batch: dict,
    eval_batch: dict,
    *,
    chunk_size: int,
    ridge: float,
) -> dict:
    train_reps = collect_activation_reps(model, train_batch["tokens"], chunk_size=chunk_size)
    eval_reps = collect_activation_reps(model, eval_batch["tokens"], chunk_size=chunk_size)
    logits = []
    for start in range(0, eval_batch["tokens"].shape[0], chunk_size):
        logits.append(model(eval_batch["tokens"][start : start + chunk_size]))
    eval_logits = torch.cat(logits, dim=0)
    model_bits = (eval_logits >= 0).to(torch.float32)

    reps = {}
    latent_by_factor = simple_factor_latent_targets(train_batch["tokens"].device)
    train_latents = {
        name: values[train_batch["final_factor_id"].to(torch.long)]
        for name, values in latent_by_factor.items()
    }
    eval_latents = {
        name: values[eval_batch["final_factor_id"].to(torch.long)]
        for name, values in latent_by_factor.items()
    }
    for name, train_x in train_reps.items():
        eval_x = eval_reps[name]
        latent_results = {}
        for latent_name in train_latents:
            latent_results[latent_name] = latent_rule_probe_summary(
                train_x,
                train_latents[latent_name],
                train_batch["label_bits"],
                eval_x,
                eval_latents[latent_name],
                eval_batch["label_bits"],
                model_bits,
                ridge=ridge,
            )
        reps[name] = {
            "dim": int(train_x.shape[1]),
            "descent_bit_probe": binary_probe_summary(
                train_x,
                train_batch["label_bits"],
                eval_x,
                eval_batch["label_bits"],
                model_bits,
                ridge=ridge,
            ),
            "final_factor_probe_then_rule": final_factor_probe_summary(
                train_x,
                train_batch["final_factor_id"],
                eval_x,
                eval_batch["final_factor_id"],
                eval_batch["label_bits"],
                model_bits,
                ridge=ridge,
            ),
            "latent_rule_probes": latent_results,
        }
    top_final = sorted(
        (
            {
                "representation": name,
                **value["final_factor_probe_then_rule"],
            }
            for name, value in reps.items()
        ),
        key=lambda item: item["exact_set_accuracy"],
        reverse=True,
    )[:12]
    top_descent = sorted(
        (
            {
                "representation": name,
                **value["descent_bit_probe"],
            }
            for name, value in reps.items()
        ),
        key=lambda item: item["exact_set_accuracy"],
        reverse=True,
    )[:12]
    top_latent = sorted(
        (
            {
                "representation": rep_name,
                "latent": latent_name,
                **latent_value,
            }
            for rep_name, rep_value in reps.items()
            for latent_name, latent_value in rep_value["latent_rule_probes"].items()
        ),
        key=lambda item: item["exact_set_accuracy"],
        reverse=True,
    )[:20]
    return {
        "train_examples": int(train_batch["tokens"].shape[0]),
        "eval_examples": int(eval_batch["tokens"].shape[0]),
        "model_metrics": metrics_from_logits(eval_logits, eval_batch["label_bits"]),
        "ridge": ridge,
        "representations": reps,
        "top_final_factor_rule_probes": top_final,
        "top_descent_bit_probes": top_descent,
        "top_latent_rule_probes": top_latent,
    }


def valid_next_final_table(device: torch.device) -> list[list[int]]:
    left_masks = LEFT_DESC_MASK.tolist()
    right_masks = RIGHT_DESC_MASK.tolist()
    table: list[list[int]] = []
    for prev_id in range(len(right_masks)):
        prev_right = right_masks[prev_id]
        valid = [idx for idx, left in enumerate(left_masks) if (left & prev_right) == left]
        table.append(valid)
    return table


@torch.no_grad()
def collect_prefix_fixed_pairs(
    loader: DataLoader,
    *,
    device: torch.device,
    length: int,
    absolute_depth: int,
    num_pairs: int,
) -> dict:
    table = valid_next_final_table(device)
    right_masks = RIGHT_DESC_MASK.to(device)
    simple_mats = simple_mats_z(device)
    clean_factors: list[torch.Tensor] = []
    corrupt_factors: list[torch.Tensor] = []
    scanned = 0
    for batch in loader:
        factor_ids = batch["factor_ids"].to(device)
        sample_ids = batch["sample_id"].to(device)
        scanned += int(factor_ids.shape[0])
        for row_idx in range(factor_ids.shape[0]):
            factors = factor_ids[row_idx]
            prev_id = int(factors[-2].item())
            current_final = int(factors[-1].item())
            current_mask = int(right_masks[current_final].item())
            candidates = [x for x in table[prev_id] if int(right_masks[x].item()) != current_mask]
            if not candidates:
                continue
            choice = candidates[int(sample_ids[row_idx].item()) % len(candidates)]
            corrupt = factors.clone()
            corrupt[-1] = int(choice)
            clean_factors.append(factors.detach().cpu())
            corrupt_factors.append(corrupt.detach().cpu())
            if len(clean_factors) >= num_pairs:
                break
        if len(clean_factors) >= num_pairs:
            break
    if len(clean_factors) < num_pairs:
        raise RuntimeError(f"Only found {len(clean_factors)} prefix-fixed pairs after scanning {scanned} examples")

    clean_factor_ids = torch.stack(clean_factors, dim=0).to(device)
    corrupt_factor_ids = torch.stack(corrupt_factors, dim=0).to(device)
    clean_tokens = factor_ids_to_z_sign_tokens(
        clean_factor_ids,
        length=length,
        absolute_depth=absolute_depth,
        simple_mats=simple_mats,
    )
    corrupt_tokens = factor_ids_to_z_sign_tokens(
        corrupt_factor_ids,
        length=length,
        absolute_depth=absolute_depth,
        simple_mats=simple_mats,
    )
    clean_masks, clean_bits = label_bits_from_final_ids(clean_factor_ids[:, -1], device)
    corrupt_masks, corrupt_bits = label_bits_from_final_ids(corrupt_factor_ids[:, -1], device)
    return {
        "clean_factor_ids": clean_factor_ids,
        "corrupt_factor_ids": corrupt_factor_ids,
        "clean_tokens": clean_tokens,
        "corrupt_tokens": corrupt_tokens,
        "clean_labels": clean_bits,
        "corrupt_labels": corrupt_bits,
        "clean_descent_mask": clean_masks,
        "corrupt_descent_mask": corrupt_masks,
        "clean_final_factor_id": clean_factor_ids[:, -1].to(torch.long),
        "corrupt_final_factor_id": corrupt_factor_ids[:, -1].to(torch.long),
        "matched_pairs": len(clean_factors),
        "scanned_examples": scanned,
    }


def score_against_clean(logits: torch.Tensor, clean_labels: torch.Tensor) -> torch.Tensor:
    return multilabel_logit_score(logits, clean_labels).mean(dim=1)


def patch_metrics(logits: torch.Tensor, clean_labels: torch.Tensor, clean_score: torch.Tensor, corrupt_score: torch.Tensor) -> dict:
    score = score_against_clean(logits, clean_labels)
    denom = (clean_score.mean() - corrupt_score.mean()).clamp_min(1e-6)
    recovery = (score.mean() - corrupt_score.mean()) / denom
    direct = metrics_from_logits(logits, clean_labels)
    direct["mean_score_against_clean"] = float(score.mean().item())
    direct["normalized_score_recovery"] = float(recovery.item())
    return direct


def window_mask(tokens: torch.Tensor, radius: int, *, leading: bool, trailing: bool, include_cls: bool) -> torch.Tensor:
    feats = support_features(tokens)
    seq_len = tokens.shape[1] + (1 if include_cls else 0)
    mask = torch.zeros(tokens.shape[0], seq_len, dtype=torch.bool, device=tokens.device)
    if include_cls:
        offset = 1
    else:
        offset = 0
    offsets = torch.arange(-radius, radius + 1, device=tokens.device)
    centers = []
    if leading:
        centers.append(feats["first"])
    if trailing:
        centers.append(feats["last"])
    for center in centers:
        idx = center.unsqueeze(1) + offsets.unsqueeze(0)
        valid = (idx >= 0) & (idx < tokens.shape[1])
        idx = (idx + offset).clamp(0, seq_len - 1)
        mask.scatter_(1, idx, valid)
    return mask


def patch_input_by_mask(clean_tokens: torch.Tensor, corrupt_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    patched = corrupt_tokens.clone()
    patched[mask] = clean_tokens[mask]
    return patched


def patch_activation_mask_hook(source: torch.Tensor, mask: torch.Tensor):
    def hook(value: torch.Tensor, _: str) -> torch.Tensor:
        patched = value.clone()
        src = source.to(device=value.device, dtype=value.dtype)
        local_mask = mask.to(value.device)
        patched[local_mask] = src[local_mask]
        return patched

    return hook


@torch.no_grad()
def prefix_counterfactual_experiment(model, pairs: dict, *, radii: list[int]) -> dict:
    clean_tokens = pairs["clean_tokens"]
    corrupt_tokens = pairs["corrupt_tokens"]
    clean_labels = pairs["clean_labels"]
    clean_logits, clean_cache = run_with_cache(
        model,
        clean_tokens,
        names_filter={"hook_resid_embed", "hook_final_hidden"}
        | {f"blocks.{layer}.hook_resid_post" for layer in range(len(model.blocks))},
    )
    corrupt_logits, _ = run_with_cache(model, corrupt_tokens, names_filter=set())
    clean_score = score_against_clean(clean_logits, clean_labels)
    corrupt_score = score_against_clean(corrupt_logits, clean_labels)

    token_variants = {}
    for radius in radii:
        masks = {
            f"leading_r{radius}": window_mask(clean_tokens, radius, leading=True, trailing=False, include_cls=False),
            f"trailing_r{radius}": window_mask(clean_tokens, radius, leading=False, trailing=True, include_cls=False),
            f"both_boundaries_r{radius}": window_mask(clean_tokens, radius, leading=True, trailing=True, include_cls=False),
        }
        both = masks[f"both_boundaries_r{radius}"]
        masks[f"interior_except_boundaries_r{radius}"] = ~both
        for name, mask in masks.items():
            patched_tokens = patch_input_by_mask(clean_tokens, corrupt_tokens, mask)
            logits = model(patched_tokens)
            token_variants[name] = {
                "patched_degree_entries": int(mask.sum().item()),
                **patch_metrics(logits, clean_labels, clean_score, corrupt_score),
            }

    activation_variants = {}
    sites = ["hook_resid_embed"]
    sites.extend(f"blocks.{layer}.hook_resid_post" for layer in range(len(model.blocks)))
    sites.append("hook_final_hidden")
    for site in sites:
        if site not in clean_cache:
            continue
        source = clean_cache[site]
        cls_mask = torch.zeros(clean_tokens.shape[0], clean_tokens.shape[1] + 1, dtype=torch.bool, device=clean_tokens.device)
        cls_mask[:, 0] = True
        for name, mask in {"cls": cls_mask}.items():
            logits, _ = run_with_cache(
                model,
                corrupt_tokens,
                hooks={site: patch_activation_mask_hook(source, mask)},
                names_filter=set(),
            )
            activation_variants[f"{site}::{name}"] = {
                "site": site,
                "position_set": name,
                "patched_positions": int(mask.sum().item()),
                **patch_metrics(logits, clean_labels, clean_score, corrupt_score),
            }
        for radius in [0, 2, 5, 8]:
            if radius not in radii:
                continue
            both = window_mask(clean_tokens, radius, leading=True, trailing=True, include_cls=True)
            cls_plus = both.clone()
            cls_plus[:, 0] = True
            for name, mask in {
                f"both_boundaries_r{radius}": both,
                f"cls_plus_boundaries_r{radius}": cls_plus,
            }.items():
                logits, _ = run_with_cache(
                    model,
                    corrupt_tokens,
                    hooks={site: patch_activation_mask_hook(source, mask)},
                    names_filter=set(),
                )
                activation_variants[f"{site}::{name}"] = {
                    "site": site,
                    "position_set": name,
                    "patched_positions": int(mask.sum().item()),
                    **patch_metrics(logits, clean_labels, clean_score, corrupt_score),
                }

    return {
        "matched_pairs": int(pairs["matched_pairs"]),
        "scanned_examples": int(pairs["scanned_examples"]),
        "clean": metrics_from_logits(clean_logits, clean_labels),
        "corrupt_scored_as_clean": patch_metrics(corrupt_logits, clean_labels, clean_score, corrupt_score),
        "clean_final_factor_counts": torch.bincount(pairs["clean_final_factor_id"], minlength=int(RIGHT_DESC_MASK.numel())).cpu().tolist(),
        "corrupt_final_factor_counts": torch.bincount(pairs["corrupt_final_factor_id"], minlength=int(RIGHT_DESC_MASK.numel())).cpu().tolist(),
        "clean_mask_counts": torch.bincount(pairs["clean_descent_mask"], minlength=8).cpu().tolist(),
        "corrupt_mask_counts": torch.bincount(pairs["corrupt_descent_mask"], minlength=8).cpu().tolist(),
        "token_patching": token_variants,
        "top_token_patches": sorted(token_variants.values(), key=lambda item: item["normalized_score_recovery"], reverse=True)[:12],
        "activation_patching": activation_variants,
        "top_activation_patches": sorted(
            activation_variants.values(),
            key=lambda item: item["normalized_score_recovery"],
            reverse=True,
        )[:20],
    }


@torch.no_grad()
def eval_tokens_with_hooks(model, tokens: torch.Tensor, labels: torch.Tensor, *, hooks: dict | None, chunk_size: int) -> dict:
    logits = []
    for start in range(0, tokens.shape[0], chunk_size):
        chunk = tokens[start : start + chunk_size]
        if hooks:
            chunk_logits, _ = run_with_cache(model, chunk, hooks=hooks, names_filter=set())
        else:
            chunk_logits = model(chunk)
        logits.append(chunk_logits)
    return metrics_from_logits(torch.cat(logits, dim=0), labels)


@torch.no_grad()
def head_ablation_experiment(model, eval_batch: dict, *, chunk_size: int) -> dict:
    tokens = eval_batch["tokens"]
    labels = eval_batch["label_bits"]
    variants = {
        "full": eval_tokens_with_hooks(model, tokens, labels, hooks=None, chunk_size=chunk_size),
    }
    for layer in range(len(model.blocks)):
        variants[f"zero_l{layer}_attention"] = eval_tokens_with_hooks(
            model,
            tokens,
            labels,
            hooks={f"blocks.{layer}.hook_attn_head_out": zero_hook},
            chunk_size=chunk_size,
        )
        variants[f"zero_l{layer}_mlp"] = eval_tokens_with_hooks(
            model,
            tokens,
            labels,
            hooks={f"blocks.{layer}.hook_mlp_out": zero_hook},
            chunk_size=chunk_size,
        )
        for head in range(model.blocks[layer].attn.num_heads):
            variants[f"zero_l{layer}h{head}_cls"] = eval_tokens_with_hooks(
                model,
                tokens,
                labels,
                hooks={f"blocks.{layer}.hook_attn_head_out": zero_head_positions_hook(head_idx=head, token_indices=[0])},
                chunk_size=chunk_size,
            )
    ranking = []
    full_exact = variants["full"]["exact_set_accuracy"]
    full_bit = variants["full"]["bit_accuracy"]
    for name, value in variants.items():
        if name == "full":
            continue
        ranking.append(
            {
                "variant": name,
                "exact_drop": float(full_exact - value["exact_set_accuracy"]),
                "bit_drop": float(full_bit - value["bit_accuracy"]),
                **value,
            }
        )
    return {
        "variants": variants,
        "top_accuracy_drops": sorted(ranking, key=lambda item: item["exact_drop"], reverse=True)[:20],
    }


@torch.no_grad()
def attention_boundary_summary(model, eval_batch: dict, *, chunk_size: int) -> dict:
    tokens = eval_batch["tokens"]
    names_filter = {f"blocks.{layer}.hook_pattern" for layer in range(len(model.blocks))}
    totals: dict[str, dict[str, float]] = {}
    n_total = 0
    for start in range(0, tokens.shape[0], chunk_size):
        chunk = tokens[start : start + chunk_size]
        _, cache = run_with_cache(model, chunk, names_filter=names_filter)
        feats = support_features(chunk)
        first = feats["first"] + 1
        last = feats["last"] + 1
        support = feats["support"]
        batch_n = int(chunk.shape[0])
        n_total += batch_n
        rows = torch.arange(batch_n, device=chunk.device)
        for layer in range(len(model.blocks)):
            pattern = cache[f"blocks.{layer}.hook_pattern"][:, :, 0, :]
            for head in range(pattern.shape[1]):
                key = f"L{layer}H{head}"
                rec = totals.setdefault(
                    key,
                    {
                        "sum_leading": 0.0,
                        "sum_trailing": 0.0,
                        "sum_boundaries": 0.0,
                        "sum_support": 0.0,
                        "sum_entropy": 0.0,
                    },
                )
                head_pattern = pattern[:, head]
                leading = head_pattern[rows, first]
                trailing = head_pattern[rows, last]
                seq_support = torch.cat([torch.ones(batch_n, 1, dtype=torch.bool, device=chunk.device), support], dim=1)
                support_mass = (head_pattern * seq_support.to(head_pattern.dtype)).sum(dim=1)
                entropy = -(head_pattern.clamp_min(1e-8) * head_pattern.clamp_min(1e-8).log()).sum(dim=1)
                rec["sum_leading"] += float(leading.sum().item())
                rec["sum_trailing"] += float(trailing.sum().item())
                rec["sum_boundaries"] += float((leading + trailing).sum().item())
                rec["sum_support"] += float(support_mass.sum().item())
                rec["sum_entropy"] += float(entropy.sum().item())
    heads = {
        key: {
            "mean_cls_to_leading": value["sum_leading"] / n_total,
            "mean_cls_to_trailing": value["sum_trailing"] / n_total,
            "mean_cls_to_boundaries": value["sum_boundaries"] / n_total,
            "mean_cls_to_support": value["sum_support"] / n_total,
            "mean_entropy": value["sum_entropy"] / n_total,
        }
        for key, value in totals.items()
    }
    return {
        "n": n_total,
        "top_by_boundary_attention": [
            {"head": key, **value}
            for key, value in sorted(heads.items(), key=lambda item: item[1]["mean_cls_to_boundaries"], reverse=True)[:12]
        ],
        "top_by_leading_attention": [
            {"head": key, **value}
            for key, value in sorted(heads.items(), key=lambda item: item[1]["mean_cls_to_leading"], reverse=True)[:12]
        ],
        "top_by_trailing_attention": [
            {"head": key, **value}
            for key, value in sorted(heads.items(), key=lambda item: item[1]["mean_cls_to_trailing"], reverse=True)[:12]
        ],
    }


def right_inverse_generator_terms() -> list[list[tuple[int, int, int, int]]]:
    return [
        [(-2, 0, 0, -1), (-1, 0, 1, -1), (0, 1, 1, 1), (0, 2, 2, 1)],
        [(0, 0, 0, 1), (-1, 1, 0, -1), (-2, 1, 1, -1), (-1, 1, 2, -1), (0, 2, 2, 1)],
        [(0, 0, 0, 1), (0, 1, 1, 1), (-1, 2, 1, -1), (-2, 2, 2, -1)],
    ]


def right_quotient_sign_tokens(mat: torch.Tensor) -> torch.Tensor:
    batch_size, depth = mat.shape[:2]
    out_tokens = []
    for terms in right_inverse_generator_terms():
        out = torch.zeros(batch_size, depth + 2, MATRIX_SIZE, MATRIX_SIZE, dtype=mat.dtype, device=mat.device)
        for exp, k, j, coeff in terms:
            start = exp + 2
            out[:, start : start + depth, :, j] += mat[:, :, :, k] * coeff
        out_tokens.append(sign_tokens_from_dense(out))
    return torch.stack(out_tokens, dim=1)


def rows_as_tuples(x: torch.Tensor) -> list[tuple]:
    return [tuple(int(v) for v in row.tolist()) for row in x.detach().cpu()]


def quotient_feature_keys(tokens_by_gen: torch.Tensor, *, radii: list[int]) -> dict[str, list[tuple]]:
    batch_size, num_gen, depth = tokens_by_gen.shape
    flat = tokens_by_gen.reshape(batch_size * num_gen, depth)
    gen_ids = torch.arange(num_gen, device=tokens_by_gen.device).view(1, num_gen).expand(batch_size, num_gen).reshape(-1)
    feats = support_features(flat)
    keys: dict[str, list[tuple]] = {
        "q_min_degree": [(int(g), int(v)) for g, v in zip(gen_ids.cpu().tolist(), feats["first"].cpu().tolist(), strict=True)],
        "q_max_degree": [(int(g), int(v)) for g, v in zip(gen_ids.cpu().tolist(), feats["last"].cpu().tolist(), strict=True)],
        "q_width": [(int(g), int(v)) for g, v in zip(gen_ids.cpu().tolist(), feats["width"].cpu().tolist(), strict=True)],
        "q_low_tokens_0_1_2": [
            (int(g), int(row[0]), int(row[1]), int(row[2]))
            for g, row in zip(gen_ids.cpu().tolist(), flat[:, :3].cpu().tolist(), strict=True)
        ],
        "q_leading_token": [
            (int(g), int(v)) for g, v in zip(gen_ids.cpu().tolist(), feats["leading_token"].cpu().tolist(), strict=True)
        ],
        "q_trailing_token": [
            (int(g), int(v)) for g, v in zip(gen_ids.cpu().tolist(), feats["trailing_token"].cpu().tolist(), strict=True)
        ],
        "q_boundary_tokens": [
            (int(g), int(a), int(b))
            for g, a, b in zip(
                gen_ids.cpu().tolist(),
                feats["leading_token"].cpu().tolist(),
                feats["trailing_token"].cpu().tolist(),
                strict=True,
            )
        ],
    }
    for radius in radii:
        lead = gather_relative_window(flat, feats["first"], radius)
        trail = gather_relative_window(flat, feats["last"], radius)
        lead_rows = rows_as_tuples(lead)
        trail_rows = rows_as_tuples(trail)
        gen_list = gen_ids.cpu().tolist()
        keys[f"q_leading_window_r{radius}"] = [(int(g),) + row for g, row in zip(gen_list, lead_rows, strict=True)]
        keys[f"q_trailing_window_r{radius}"] = [(int(g),) + row for g, row in zip(gen_list, trail_rows, strict=True)]
        keys[f"q_both_windows_r{radius}"] = [
            (int(g),) + lrow + trow
            for g, lrow, trow in zip(gen_list, lead_rows, trail_rows, strict=True)
        ]
    return keys


def majority_lookup(train_keys: Iterable[tuple], train_labels: Iterable[int]) -> tuple[dict[tuple, int], int]:
    counts: dict[tuple, Counter] = defaultdict(Counter)
    global_counts: Counter = Counter()
    for key, label in zip(train_keys, train_labels, strict=True):
        label = int(label)
        counts[key][label] += 1
        global_counts[label] += 1
    fallback = global_counts.most_common(1)[0][0]
    table = {key: counter.most_common(1)[0][0] for key, counter in counts.items()}
    return table, fallback


def binary_lookup_metrics(table: dict[tuple, int], fallback: int, eval_keys: Iterable[tuple], eval_labels: torch.Tensor) -> dict:
    preds = []
    seen = 0
    for key in eval_keys:
        if key in table:
            seen += 1
        preds.append(table.get(key, fallback))
    pred = torch.tensor(preds, dtype=torch.long)
    labels = eval_labels.detach().cpu().to(torch.long)
    tp = int(((pred == 1) & (labels == 1)).sum().item())
    fp = int(((pred == 1) & (labels == 0)).sum().item())
    fn = int(((pred == 0) & (labels == 1)).sum().item())
    f1 = 0.0 if (2 * tp + fp + fn) == 0 else (2 * tp) / (2 * tp + fp + fn)
    out = {
        "accuracy": float((pred == labels).float().mean().item()),
        "coverage": seen / max(1, len(preds)),
        "positive_rate": float(pred.float().mean().item()),
        "true_positive_rate": float(labels.float().mean().item()),
        "f1": float(f1),
        "n": int(labels.numel()),
        "pred_counts": torch.bincount(pred, minlength=2).tolist(),
    }
    if labels.numel() % 3 == 0:
        pred_bits = pred.view(-1, 3).to(torch.float32)
        true_bits = labels.view(-1, 3).to(torch.float32)
        out["set_exact_accuracy"] = float(pred_bits.eq(true_bits).all(dim=1).float().mean().item())
        out["set_bit_accuracy"] = float(pred_bits.eq(true_bits).float().mean().item())
        out["set_pred_mask_counts"] = torch.bincount(mask_from_bits(pred_bits), minlength=8).tolist()
        out["set_true_mask_counts"] = torch.bincount(mask_from_bits(true_bits), minlength=8).tolist()
        out["set_n"] = int(true_bits.shape[0])
    return out


@torch.no_grad()
def quotient_lookup_experiment(
    train_batch: dict,
    eval_batch: dict,
    *,
    length: int,
    absolute_depth: int,
    radii: list[int],
) -> dict:
    simple_mats = simple_mats_z(train_batch["factor_ids"].device)
    train_mat = dense_burau_z_for_factor_ids(
        train_batch["factor_ids"],
        length=length,
        absolute_depth=absolute_depth,
        simple_mats=simple_mats,
    )
    eval_mat = dense_burau_z_for_factor_ids(
        eval_batch["factor_ids"],
        length=length,
        absolute_depth=absolute_depth,
        simple_mats=simple_mats,
    )
    train_q = right_quotient_sign_tokens(train_mat)
    eval_q = right_quotient_sign_tokens(eval_mat)
    train_labels = train_batch["label_bits"].to(torch.long).reshape(-1).cpu()
    eval_labels = eval_batch["label_bits"].to(torch.long).reshape(-1).cpu()
    train_keys = quotient_feature_keys(train_q, radii=radii)
    eval_keys = quotient_feature_keys(eval_q, radii=radii)
    summary = {}
    for name, keys in train_keys.items():
        table, fallback = majority_lookup(keys, train_labels.tolist())
        summary[name] = {
            "unique_train_keys": len(table),
            "fallback": int(fallback),
            **binary_lookup_metrics(table, fallback, eval_keys[name], eval_labels),
        }
    top = sorted(({"feature": key, **value} for key, value in summary.items()), key=lambda item: item["accuracy"], reverse=True)
    return {
        "train_examples": int(train_batch["factor_ids"].shape[0]),
        "eval_examples": int(eval_batch["factor_ids"].shape[0]),
        "binary_examples_per_split": {
            "train": int(train_labels.numel()),
            "eval": int(eval_labels.numel()),
        },
        "features": summary,
        "top_features": top[:16],
        "quotient_degree_offset": 2,
        "quotient_generators": ["s_1^{-1}", "s_2^{-1}", "s_3^{-1}"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run B4 Z-sign deeper interpretability experiments.")
    parser.add_argument("--data-dir", default="interp/data/generated/b4_l25_p2_n16777216")
    parser.add_argument("--checkpoint", default="interp/artifacts/b4_l25_zsign_xfmr3_abs/best_model.pt")
    parser.add_argument("--out", default="interp/artifacts/b4_l25_zsign_deep_dive/results.json")
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--probe-train-examples", type=int, default=32768)
    parser.add_argument("--probe-eval-examples", type=int, default=8192)
    parser.add_argument("--quotient-train-examples", type=int, default=32768)
    parser.add_argument("--quotient-eval-examples", type=int, default=8192)
    parser.add_argument("--max-scan-examples", type=int, default=262144)
    parser.add_argument("--prefix-pairs", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-input-boundary-radius",
        type=int,
        default=-2,
        help="Use -2 to infer from checkpoint, -1 for no transform, or nonnegative to keep only boundary windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    loaded = load_transformer_checkpoint(args.checkpoint, device=device)
    model = loaded.model
    cfg = model.config
    if args.model_input_boundary_radius == -2:
        model_input_boundary_radius = infer_boundary_radius_from_checkpoint(loaded.checkpoint)
    else:
        model_input_boundary_radius = int(args.model_input_boundary_radius)
    shard_paths = discover_shards(args.data_dir, num_shards=args.num_shards)

    train_loader = make_factor_loader(
        shard_paths,
        split="train",
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=True,
        max_examples=max(args.probe_train_examples, args.quotient_train_examples),
    )
    eval_loader = make_factor_loader(
        shard_paths,
        split="test",
        batch_size=args.batch_size,
        seed=args.seed + 1,
        shuffle=True,
        max_examples=max(args.probe_eval_examples, args.quotient_eval_examples),
    )
    train_batch = collect_factor_batch(
        train_loader,
        device=device,
        length=cfg.length,
        absolute_depth=cfg.absolute_depth,
        min_examples=max(args.probe_train_examples, args.quotient_train_examples),
    )
    eval_batch = collect_factor_batch(
        eval_loader,
        device=device,
        length=cfg.length,
        absolute_depth=cfg.absolute_depth,
        min_examples=max(args.probe_eval_examples, args.quotient_eval_examples),
    )

    pair_loader = make_factor_loader(
        shard_paths,
        split="test",
        batch_size=args.batch_size,
        seed=args.seed + 2,
        shuffle=True,
        max_examples=args.max_scan_examples,
    )
    pairs = collect_prefix_fixed_pairs(
        pair_loader,
        device=device,
        length=cfg.length,
        absolute_depth=cfg.absolute_depth,
        num_pairs=args.prefix_pairs,
    )
    train_model_batch = apply_batch_input_transform(train_batch, model_input_boundary_radius)
    eval_model_batch = apply_batch_input_transform(eval_batch, model_input_boundary_radius)
    model_pairs = dict(pairs)
    model_pairs["clean_tokens"] = apply_model_input_transform(pairs["clean_tokens"], model_input_boundary_radius)
    model_pairs["corrupt_tokens"] = apply_model_input_transform(pairs["corrupt_tokens"], model_input_boundary_radius)

    result = {
        "config": vars(args),
        "effective_model_input_boundary_radius": int(model_input_boundary_radius),
        "checkpoint_model_config": cfg.to_dict(),
        "checkpoint_input_transform": loaded.checkpoint.get("input_transform", "none"),
        "semantic_probes": semantic_probe_experiment(
            model,
            {key: value[: args.probe_train_examples] for key, value in train_model_batch.items()},
            {key: value[: args.probe_eval_examples] for key, value in eval_model_batch.items()},
            chunk_size=args.chunk_size,
            ridge=args.ridge,
        ),
        "prefix_fixed_counterfactuals": prefix_counterfactual_experiment(
            model,
            model_pairs,
            radii=[0, 1, 2, 3, 5, 8],
        ),
        "head_ablation": head_ablation_experiment(
            model,
            {key: value[: args.probe_eval_examples] for key, value in eval_model_batch.items()},
            chunk_size=args.chunk_size,
        ),
        "attention_boundary_summary": attention_boundary_summary(
            model,
            {key: value[: min(args.probe_eval_examples, 2048)] for key, value in eval_model_batch.items()},
            chunk_size=min(args.chunk_size, 256),
        ),
        "right_quotient_lookup": quotient_lookup_experiment(
            {key: value[: args.quotient_train_examples] for key, value in train_batch.items()},
            {key: value[: args.quotient_eval_examples] for key, value in eval_batch.items()},
            length=cfg.length,
            absolute_depth=cfg.absolute_depth,
            radii=[0, 1, 2, 3],
        ),
    }
    atomic_json_dump(result, Path(args.out))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
