#!/usr/bin/env python3
"""Sparse-autoencoder experiments for the B4 Z[v] boundary transformer.

The goal is not just to find correlated sparse features.  For each activation
site, this script trains a TopK SAE, labels its features against algebraic
variables, and tests whether the feature basis is causal by reconstructing,
ablating, and prefix-fixed patching through the trained transformer.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from braid_data import GarsideFactor  # noqa: E402
from interp.b4_interp import (  # noqa: E402
    load_transformer_checkpoint,
    metrics_from_logits,
    multilabel_logit_score,
    run_with_cache,
    support_features,
    zero_except_windows,
)
from interp.b4_z_sign import (  # noqa: E402
    B4FactorBatchIterable,
    discover_b4_shards,
    factor_ids_to_z_sign_tokens,
    simple_mats_z,
    simple_mats_z_cpu,
)
from interp.generate_b4_dataset import (  # noqa: E402
    LEFT_DESC_MASK,
    MATRIX_SIZE,
    PROPER_FACTOR_PERMS,
    RIGHT_DESC_MASK,
    absolute_depth_for_length,
)
from interp.train_b4_transformer import atomic_json_dump, load_shard, resolve_device, set_seed  # noqa: E402


@dataclass(frozen=True)
class SiteSpec:
    key: str
    hook: str
    selector: str


SITE_SPECS = {
    "final_hidden_cls": SiteSpec("final_hidden_cls", "hook_final_hidden", "cls"),
    "l1_resid_post_cls": SiteSpec("l1_resid_post_cls", "blocks.1.hook_resid_post", "cls"),
    "l1_attn_out_cls": SiteSpec("l1_attn_out_cls", "blocks.1.hook_attn_out", "cls"),
    "l0_mlp_out_cls": SiteSpec("l0_mlp_out_cls", "blocks.0.hook_mlp_out", "cls"),
    "l0_mlp_out_leading": SiteSpec("l0_mlp_out_leading", "blocks.0.hook_mlp_out", "leading"),
    "l0_mlp_out_trailing": SiteSpec("l0_mlp_out_trailing", "blocks.0.hook_mlp_out", "trailing"),
}


def parse_site_list(text: str) -> list[SiteSpec]:
    names = [name.strip() for name in re.split(r"[,+:;]", text) if name.strip()]
    unknown = [name for name in names if name not in SITE_SPECS]
    if unknown:
        raise ValueError(f"Unknown SAE site(s): {unknown}; expected one of {sorted(SITE_SPECS)}")
    return [SITE_SPECS[name] for name in names]


class TopKSAE(nn.Module):
    def __init__(self, d_in: int, n_features: int, top_k: int):
        super().__init__()
        self.d_in = int(d_in)
        self.n_features = int(n_features)
        self.top_k = int(top_k)
        self.pre_bias = nn.Parameter(torch.zeros(d_in))
        self.encoder = nn.Linear(d_in, n_features)
        self.decoder = nn.Linear(n_features, d_in, bias=False)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.normal_(self.decoder.weight, mean=0.0, std=1.0 / math.sqrt(n_features))
        self.normalize_decoder_()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        acts = F.relu(self.encoder(x - self.pre_bias))
        if 0 < self.top_k < acts.shape[-1]:
            values, indices = torch.topk(acts, k=self.top_k, dim=-1)
            sparse = torch.zeros_like(acts)
            sparse.scatter_(1, indices, values)
            acts = sparse
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return self.decoder(acts) + self.pre_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = self.encode(x)
        return self.decode(acts), acts

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        weight = self.decoder.weight.data
        norms = weight.norm(dim=0, keepdim=True).clamp_min(1e-6)
        self.decoder.weight.data = weight / norms


def infer_boundary_radius_from_checkpoint(checkpoint: dict) -> int:
    transform = str(checkpoint.get("input_transform", "none"))
    prefix = "zero_except_leading_and_trailing_windows_radius_"
    if transform.startswith(prefix):
        return int(transform[len(prefix) :])
    return -1


def infer_data_config(shard_paths: list[Path]) -> dict:
    meta = load_shard(shard_paths[0])["metadata"]
    length = int(meta["length"])
    return {
        "length": length,
        "absolute_depth": int(meta.get("absolute_depth", absolute_depth_for_length(length))),
    }


def make_loader(
    shard_paths: list[Path],
    *,
    split: str,
    batch_size: int,
    seed: int,
    epoch: int,
    shuffle: bool,
    max_examples: int,
) -> DataLoader:
    dataset = B4FactorBatchIterable(
        shard_paths,
        split=split,
        batch_size=batch_size,
        seed=seed,
        epoch=epoch,
        shuffle_shards=shuffle,
        shuffle_rows=shuffle,
        max_examples=max_examples,
    )
    return DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=False)


def apply_input_transform(tokens: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 0:
        return tokens
    return zero_except_windows(tokens, radius, leading=True, trailing=True)


def batch_to_tokens(
    batch: dict,
    *,
    device: torch.device,
    length: int,
    absolute_depth: int,
    simple_mats: torch.Tensor,
    boundary_radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    factor_ids = batch["factor_ids"].to(device)
    labels = batch["label_bits"].to(device)
    tokens = factor_ids_to_z_sign_tokens(
        factor_ids,
        length=length,
        absolute_depth=absolute_depth,
        simple_mats=simple_mats,
    )
    tokens = apply_input_transform(tokens, boundary_radius)
    return tokens, labels


def select_site(value: torch.Tensor, tokens: torch.Tensor, spec: SiteSpec) -> torch.Tensor:
    if spec.selector == "cls":
        return value[:, 0]
    feats = support_features(tokens)
    rows = torch.arange(tokens.shape[0], device=tokens.device)
    if spec.selector == "leading":
        return value[rows, feats["first"] + 1]
    if spec.selector == "trailing":
        return value[rows, feats["last"] + 1]
    if spec.selector == "support_mean":
        support = torch.cat(
            [torch.zeros(tokens.shape[0], 1, dtype=torch.bool, device=tokens.device), feats["support"]],
            dim=1,
        )
        weight = support.to(torch.float32)
        weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (value * weight.unsqueeze(-1)).sum(dim=1)
    raise ValueError(f"Unsupported selector {spec.selector!r}")


def patch_site(value: torch.Tensor, tokens: torch.Tensor, spec: SiteSpec, replacement: torch.Tensor) -> torch.Tensor:
    patched = value.clone()
    replacement = replacement.to(device=value.device, dtype=value.dtype)
    if spec.selector == "cls":
        patched[:, 0] = replacement
        return patched
    feats = support_features(tokens)
    rows = torch.arange(tokens.shape[0], device=tokens.device)
    if spec.selector == "leading":
        patched[rows, feats["first"] + 1] = replacement
        return patched
    if spec.selector == "trailing":
        patched[rows, feats["last"] + 1] = replacement
        return patched
    raise ValueError(f"Cannot patch selector {spec.selector!r}")


@torch.no_grad()
def collect_site_activation(
    model,
    tokens: torch.Tensor,
    spec: SiteSpec,
    *,
    chunk_size: int,
) -> torch.Tensor:
    pieces = []
    for start in range(0, tokens.shape[0], chunk_size):
        chunk = tokens[start : start + chunk_size]
        _, cache = run_with_cache(model, chunk, names_filter={spec.hook})
        pieces.append(select_site(cache[spec.hook], chunk, spec).detach().cpu())
    return torch.cat(pieces, dim=0).to(tokens.device)


def r2_score(x: torch.Tensor, recon: torch.Tensor) -> float:
    x = x.to(torch.float32)
    recon = recon.to(torch.float32)
    sse = (x - recon).pow(2).sum()
    centered = x - x.mean(dim=0, keepdim=True)
    sst = centered.pow(2).sum().clamp_min(1e-8)
    return float((1.0 - sse / sst).item())


def sae_batch_summary(sae: TopKSAE, x: torch.Tensor) -> dict:
    with torch.no_grad():
        recon, acts = sae(x)
        active = acts > 0
    return {
        "mse": float(F.mse_loss(recon, x).item()),
        "r2": r2_score(x, recon),
        "mean_active_features": float(active.sum(dim=1).float().mean().item()),
        "feature_active_fraction_mean": float(active.float().mean(dim=0).mean().item()),
        "dead_feature_fraction": float(active.any(dim=0).logical_not().float().mean().item()),
    }


def train_sae_for_site(
    model,
    loader: DataLoader,
    *,
    spec: SiteSpec,
    device: torch.device,
    length: int,
    absolute_depth: int,
    simple_mats: torch.Tensor,
    boundary_radius: int,
    d_in: int,
    n_features: int,
    top_k: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    chunk_size: int,
) -> tuple[TopKSAE, list[dict]]:
    sae = TopKSAE(d_in=d_in, n_features=n_features, top_k=top_k).to(device)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict] = []
    model.eval()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_count = 0
        for batch in loader:
            tokens, _ = batch_to_tokens(
                batch,
                device=device,
                length=length,
                absolute_depth=absolute_depth,
                simple_mats=simple_mats,
                boundary_radius=boundary_radius,
            )
            with torch.no_grad():
                acts = collect_site_activation(model, tokens, spec, chunk_size=chunk_size)
            recon, _ = sae(acts)
            loss = F.mse_loss(recon, acts)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sae.normalize_decoder_()
            total_loss += float(loss.detach().item()) * int(acts.shape[0])
            total_count += int(acts.shape[0])
        row = {
            "epoch": epoch,
            "mse": total_loss / max(total_count, 1),
            "examples": total_count,
        }
        history.append(row)
        print(f"site={spec.key} epoch={epoch:03d} sae_mse={row['mse']:.6f} examples={total_count}", flush=True)
    return sae, history


@torch.no_grad()
def collect_eval_batch(
    loader: DataLoader,
    *,
    device: torch.device,
    length: int,
    absolute_depth: int,
    simple_mats: torch.Tensor,
    boundary_radius: int,
    examples: int,
) -> dict:
    pieces: dict[str, list[torch.Tensor]] = defaultdict(list)
    total = 0
    for batch in loader:
        tokens, labels = batch_to_tokens(
            batch,
            device=device,
            length=length,
            absolute_depth=absolute_depth,
            simple_mats=simple_mats,
            boundary_radius=boundary_radius,
        )
        pieces["tokens"].append(tokens.detach().cpu())
        pieces["label_bits"].append(labels.detach().cpu())
        for key in ("factor_ids", "descent_mask", "final_factor_id", "sample_id"):
            pieces[key].append(batch[key].detach().cpu())
        total += int(tokens.shape[0])
        if total >= examples:
            break
    out = {key: torch.cat(value, dim=0)[:examples].to(device) for key, value in pieces.items()}
    out["label_bits"] = out["label_bits"].to(torch.float32)
    return out


@torch.no_grad()
def logits_with_sae_reconstruction(
    model,
    tokens: torch.Tensor,
    *,
    spec: SiteSpec,
    sae: TopKSAE,
    chunk_size: int,
    keep_features: torch.Tensor | None = None,
    ablate_features: torch.Tensor | None = None,
) -> torch.Tensor:
    outs = []
    for start in range(0, tokens.shape[0], chunk_size):
        chunk = tokens[start : start + chunk_size]

        def hook(value: torch.Tensor, _: str) -> torch.Tensor:
            x = select_site(value, chunk, spec)
            acts = sae.encode(x)
            if keep_features is not None:
                mask = torch.zeros(acts.shape[1], dtype=torch.bool, device=acts.device)
                mask[keep_features.to(acts.device)] = True
                acts = torch.where(mask.view(1, -1), acts, torch.zeros_like(acts))
            if ablate_features is not None:
                acts = acts.clone()
                acts[:, ablate_features.to(acts.device)] = 0
            recon = sae.decode(acts)
            return patch_site(value, chunk, spec, recon)

        logits, _ = run_with_cache(model, chunk, hooks={spec.hook: hook}, names_filter=set())
        outs.append(logits.detach().cpu())
    return torch.cat(outs, dim=0).to(tokens.device)


def base3_digits(tokens: torch.Tensor) -> torch.Tensor:
    powers = torch.tensor([3**i for i in range(MATRIX_SIZE * MATRIX_SIZE)], dtype=torch.long, device=tokens.device)
    return (tokens.unsqueeze(-1).to(torch.long) // powers.view(1, 1, -1)) % 3


def column_mask_from_digits(digits: torch.Tensor, sign_digit: int) -> torch.Tensor:
    mats = digits.view(digits.shape[0], digits.shape[1], MATRIX_SIZE, MATRIX_SIZE)
    cols = mats.eq(sign_digit).any(dim=2)
    weights = torch.tensor([1, 2, 4], dtype=torch.long, device=digits.device).view(1, 1, MATRIX_SIZE)
    return (cols.to(torch.long) * weights).sum(dim=-1)


def rowwise_bitwise_or(values: torch.Tensor) -> torch.Tensor:
    out = values[:, 0].clone()
    for idx in range(1, values.shape[1]):
        out = torch.bitwise_or(out, values[:, idx])
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


def make_label_tables(batch: dict) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    device = batch["tokens"].device
    bits = batch["label_bits"].to(torch.long)
    binary = {
        "descent_s1": bits[:, 0],
        "descent_s2": bits[:, 1],
        "descent_s3": bits[:, 2],
    }
    categorical = {
        "descent_mask": batch["descent_mask"].to(torch.long),
        "final_factor_id": batch["final_factor_id"].to(torch.long),
    }
    latent_by_factor = simple_factor_latent_targets(device)
    final_ids = batch["final_factor_id"].to(torch.long)
    for name, values in latent_by_factor.items():
        categorical[name] = values[final_ids]

    tokens = batch["tokens"]
    feats = support_features(tokens)
    rows = torch.arange(tokens.shape[0], device=device)
    categorical["leading_token"] = feats["leading_token"].to(torch.long)
    categorical["trailing_token"] = feats["trailing_token"].to(torch.long)
    categorical["support_width"] = feats["width"].to(torch.long)
    digits = base3_digits(tokens)
    neg_cols = column_mask_from_digits(digits, 1)
    pos_cols = column_mask_from_digits(digits, 2)
    categorical["leading_neg_col_mask"] = neg_cols[rows, feats["first"]]
    categorical["trailing_neg_col_mask"] = neg_cols[rows, feats["last"]]
    categorical["leading_pos_col_mask"] = pos_cols[rows, feats["first"]]
    categorical["trailing_pos_col_mask"] = pos_cols[rows, feats["last"]]
    for radius in (1, 3):
        offsets = torch.arange(-radius, radius + 1, device=device).view(1, -1)
        for side, center in (("leading", feats["first"]), ("trailing", feats["last"])):
            idx = (center.view(-1, 1) + offsets).clamp(0, tokens.shape[1] - 1)
            categorical[f"{side}_neg_col_or_r{radius}"] = rowwise_bitwise_or(neg_cols.gather(1, idx))
            categorical[f"{side}_pos_col_or_r{radius}"] = rowwise_bitwise_or(pos_cols.gather(1, idx))
    return binary, categorical


def top_feature_label_summary(
    acts: torch.Tensor,
    binary_labels: dict[str, torch.Tensor],
    categorical_labels: dict[str, torch.Tensor],
    *,
    top_examples: int,
    max_features: int,
) -> dict:
    acts = acts.detach().to(torch.float32)
    n, num_features = acts.shape
    k = min(max(1, top_examples), n)
    feature_mean = acts.mean(dim=0)
    feature_active = acts.gt(0).float().mean(dim=0)
    top_values, top_indices = torch.topk(acts, k=k, dim=0)
    del top_values

    rows = torch.arange(k, device=acts.device).view(k, 1).expand(k, num_features)
    del rows
    top_binary = []
    top_categorical = []
    best_by_feature = []
    for feat in range(num_features):
        idx = top_indices[:, feat]
        active_fraction = float(feature_active[feat].item())
        mean_activation = float(feature_mean[feat].item())
        best_score = -1.0
        best_record = None
        for name, labels in binary_labels.items():
            labels = labels.to(acts.device).to(torch.float32)
            base = float(labels.mean().item())
            precision = float(labels[idx].mean().item())
            score = abs(precision - base)
            record = {
                "feature": feat,
                "label": name,
                "kind": "binary",
                "precision_at_top": precision,
                "base_rate": base,
                "lift": precision / max(base, 1e-6),
                "score": score,
                "active_fraction": active_fraction,
                "mean_activation": mean_activation,
            }
            top_binary.append(record)
            if score > best_score:
                best_score = score
                best_record = record
        for name, labels in categorical_labels.items():
            labels_cpu = labels.detach().cpu().to(torch.long)
            top_vals = labels_cpu[idx.detach().cpu()]
            counter = Counter(int(x) for x in top_vals.tolist())
            majority, count = counter.most_common(1)[0]
            precision = count / k
            base = float((labels_cpu == majority).float().mean().item())
            score = precision - base
            record = {
                "feature": feat,
                "label": name,
                "kind": "categorical",
                "majority_value": int(majority),
                "precision_at_top": precision,
                "base_rate": base,
                "lift": precision / max(base, 1e-6),
                "score": score,
                "active_fraction": active_fraction,
                "mean_activation": mean_activation,
            }
            top_categorical.append(record)
            if score > best_score:
                best_score = score
                best_record = record
        if best_record is not None:
            if float(best_record["active_fraction"]) > 0.001 and float(best_record["mean_activation"]) > 0.0:
                best_by_feature.append(best_record)

    def key(item: dict) -> tuple[float, float]:
        return (float(item["score"]), float(item["active_fraction"]))

    top_binary = [
        item
        for item in top_binary
        if float(item["active_fraction"]) > 0.001 and float(item["mean_activation"]) > 0.0
    ]
    top_categorical = [
        item
        for item in top_categorical
        if float(item["active_fraction"]) > 0.001 and float(item["mean_activation"]) > 0.0
    ]
    top_binary = sorted(top_binary, key=key, reverse=True)[:max_features]
    top_categorical = sorted(top_categorical, key=key, reverse=True)[:max_features]
    best_by_feature = sorted(best_by_feature, key=key, reverse=True)[:max_features]
    selected = sorted({int(item["feature"]) for item in best_by_feature[:max_features]})
    by_descent = {}
    for label in ("descent_s1", "descent_s2", "descent_s3"):
        rows_for_label = [item for item in top_binary if item["label"] == label]
        by_descent[label] = rows_for_label[:8]
    return {
        "top_examples": k,
        "top_binary_labels": top_binary,
        "top_categorical_labels": top_categorical,
        "best_label_by_feature": best_by_feature,
        "selected_feature_ids": selected,
        "top_by_descent_label": by_descent,
    }


def metric_delta_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    metrics = metrics_from_logits(logits, labels)
    score = multilabel_logit_score(logits, labels).mean(dim=1)
    metrics["mean_score_against_true"] = float(score.mean().item())
    return metrics


@torch.no_grad()
def reconstruction_and_ablation_eval(
    model,
    batch: dict,
    *,
    spec: SiteSpec,
    sae: TopKSAE,
    selected_features: list[int],
    chunk_size: int,
) -> dict:
    tokens = batch["tokens"]
    labels = batch["label_bits"]
    base_logits = []
    for start in range(0, tokens.shape[0], chunk_size):
        base_logits.append(model(tokens[start : start + chunk_size]).detach().cpu())
    base_logits = torch.cat(base_logits, dim=0).to(tokens.device)
    rec_logits = logits_with_sae_reconstruction(model, tokens, spec=spec, sae=sae, chunk_size=chunk_size)
    out = {
        "original": metric_delta_from_logits(base_logits, labels),
        "sae_reconstruction": metric_delta_from_logits(rec_logits, labels),
    }
    if selected_features:
        feature_tensor = torch.tensor(selected_features, dtype=torch.long, device=tokens.device)
        ablated_logits = logits_with_sae_reconstruction(
            model,
            tokens,
            spec=spec,
            sae=sae,
            chunk_size=chunk_size,
            ablate_features=feature_tensor,
        )
        keep_logits = logits_with_sae_reconstruction(
            model,
            tokens,
            spec=spec,
            sae=sae,
            chunk_size=chunk_size,
            keep_features=feature_tensor,
        )
        out["ablate_selected_features"] = metric_delta_from_logits(ablated_logits, labels)
        out["keep_only_selected_features"] = metric_delta_from_logits(keep_logits, labels)
        out["selected_feature_count"] = len(selected_features)
        out["selected_features"] = selected_features
    return out


def valid_next_final_table() -> list[list[int]]:
    left_masks = LEFT_DESC_MASK.tolist()
    right_masks = RIGHT_DESC_MASK.tolist()
    table: list[list[int]] = []
    for prev_id in range(len(right_masks)):
        prev_right = right_masks[prev_id]
        table.append([idx for idx, left in enumerate(left_masks) if (left & prev_right) == left])
    return table


def bits_from_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(torch.long)
    return torch.stack([((mask >> idx) & 1) for idx in range(3)], dim=1).to(torch.float32)


@torch.no_grad()
def collect_prefix_fixed_pairs(
    loader: DataLoader,
    *,
    device: torch.device,
    length: int,
    absolute_depth: int,
    simple_mats: torch.Tensor,
    boundary_radius: int,
    num_pairs: int,
) -> dict:
    table = valid_next_final_table()
    right_masks = RIGHT_DESC_MASK.to(device)
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
            candidates = [idx for idx in table[prev_id] if int(right_masks[idx].item()) != current_mask]
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
    clean_tokens = apply_input_transform(clean_tokens, boundary_radius)
    corrupt_tokens = apply_input_transform(corrupt_tokens, boundary_radius)
    clean_masks = right_masks[clean_factor_ids[:, -1].to(torch.long)]
    corrupt_masks = right_masks[corrupt_factor_ids[:, -1].to(torch.long)]
    return {
        "clean_tokens": clean_tokens,
        "corrupt_tokens": corrupt_tokens,
        "clean_labels": bits_from_mask(clean_masks).to(device),
        "corrupt_labels": bits_from_mask(corrupt_masks).to(device),
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


@torch.no_grad()
def sae_feature_patch_logits(
    model,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    *,
    spec: SiteSpec,
    sae: TopKSAE,
    chunk_size: int,
    feature_ids: torch.Tensor | None,
) -> torch.Tensor:
    outs = []
    for start in range(0, clean_tokens.shape[0], chunk_size):
        clean = clean_tokens[start : start + chunk_size]
        corrupt = corrupt_tokens[start : start + chunk_size]
        _, clean_cache = run_with_cache(model, clean, names_filter={spec.hook})
        clean_x = select_site(clean_cache[spec.hook], clean, spec)
        clean_acts = sae.encode(clean_x)

        def hook(value: torch.Tensor, _: str) -> torch.Tensor:
            corrupt_x = select_site(value, corrupt, spec)
            corrupt_acts = sae.encode(corrupt_x)
            patched_acts = corrupt_acts.clone()
            if feature_ids is None:
                patched_acts = clean_acts.to(patched_acts.device)
            elif feature_ids.numel() > 0:
                ids = feature_ids.to(patched_acts.device)
                patched_acts[:, ids] = clean_acts[:, ids].to(patched_acts.device)
            recon = sae.decode(patched_acts)
            return patch_site(value, corrupt, spec, recon)

        logits, _ = run_with_cache(model, corrupt, hooks={spec.hook: hook}, names_filter=set())
        outs.append(logits.detach().cpu())
    return torch.cat(outs, dim=0).to(clean_tokens.device)


@torch.no_grad()
def prefix_feature_patch_eval(
    model,
    pairs: dict,
    *,
    spec: SiteSpec,
    sae: TopKSAE,
    selected_features: list[int],
    chunk_size: int,
) -> dict:
    clean_tokens = pairs["clean_tokens"]
    corrupt_tokens = pairs["corrupt_tokens"]
    clean_labels = pairs["clean_labels"]
    clean_logits = []
    corrupt_logits = []
    for start in range(0, clean_tokens.shape[0], chunk_size):
        clean_logits.append(model(clean_tokens[start : start + chunk_size]).detach().cpu())
        corrupt_logits.append(model(corrupt_tokens[start : start + chunk_size]).detach().cpu())
    clean_logits = torch.cat(clean_logits, dim=0).to(clean_tokens.device)
    corrupt_logits = torch.cat(corrupt_logits, dim=0).to(clean_tokens.device)
    clean_score = score_against_clean(clean_logits, clean_labels)
    corrupt_score = score_against_clean(corrupt_logits, clean_labels)

    out = {
        "matched_pairs": int(clean_tokens.shape[0]),
        "clean": metrics_from_logits(clean_logits, clean_labels),
        "corrupt_scored_as_clean": patch_metrics(corrupt_logits, clean_labels, clean_score, corrupt_score),
    }
    all_logits = sae_feature_patch_logits(
        model,
        clean_tokens,
        corrupt_tokens,
        spec=spec,
        sae=sae,
        chunk_size=chunk_size,
        feature_ids=None,
    )
    out["all_sae_features_from_clean"] = patch_metrics(all_logits, clean_labels, clean_score, corrupt_score)
    if selected_features:
        ids = torch.tensor(selected_features, dtype=torch.long, device=clean_tokens.device)
        selected_logits = sae_feature_patch_logits(
            model,
            clean_tokens,
            corrupt_tokens,
            spec=spec,
            sae=sae,
            chunk_size=chunk_size,
            feature_ids=ids,
        )
        out["selected_features_from_clean"] = patch_metrics(selected_logits, clean_labels, clean_score, corrupt_score)
        out["selected_feature_count"] = len(selected_features)
        out["selected_features"] = selected_features
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate SAEs on B4 boundary-only transformer activations.")
    parser.add_argument("--data-dir", default="interp/data/generated/b4_l25_p2_n16777216")
    parser.add_argument("--checkpoint", default="interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small/best_model.pt")
    parser.add_argument("--out-dir", default="interp/artifacts/b4_l25_zsign_boundary_r8_sae_suite")
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--sites", default="l1_resid_post_cls,final_hidden_cls,l1_attn_out_cls,l0_mlp_out_cls")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--train-examples", type=int, default=262_144)
    parser.add_argument("--eval-examples", type=int, default=32_768)
    parser.add_argument("--prefix-pairs", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--expansion", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--top-label-examples", type=int, default=256)
    parser.add_argument("--max-labeled-features", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-input-boundary-radius",
        type=int,
        default=-2,
        help="-2 means infer from checkpoint; -1 means no transform; nonnegative applies boundary-only radius.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    loaded = load_transformer_checkpoint(args.checkpoint, device)
    model = loaded.model.eval()
    checkpoint_radius = infer_boundary_radius_from_checkpoint(loaded.checkpoint)
    boundary_radius = checkpoint_radius if int(args.model_input_boundary_radius) == -2 else int(args.model_input_boundary_radius)
    shard_paths = discover_b4_shards(args.data_dir, num_shards=args.num_shards, allow_partial=args.allow_partial)
    data_config = infer_data_config(shard_paths)
    simple_mats = simple_mats_z(device)
    sites = parse_site_list(args.sites)
    d_in = int(model.config.d_model)
    n_features = int(args.expansion) * d_in

    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "out_dir": str(out_dir),
                "device": str(device),
                "sites": [site.key for site in sites],
                "boundary_radius": boundary_radius,
                "train_examples": args.train_examples,
                "eval_examples": args.eval_examples,
                "prefix_pairs": args.prefix_pairs,
                "d_in": d_in,
                "n_features": n_features,
                "top_k": args.top_k,
            },
            indent=2,
        ),
        flush=True,
    )

    eval_loader = make_loader(
        shard_paths,
        split="val",
        batch_size=args.batch_size,
        seed=args.seed + 1,
        epoch=0,
        shuffle=False,
        max_examples=args.eval_examples,
    )
    eval_batch = collect_eval_batch(
        eval_loader,
        device=device,
        length=data_config["length"],
        absolute_depth=data_config["absolute_depth"],
        simple_mats=simple_mats,
        boundary_radius=boundary_radius,
        examples=args.eval_examples,
    )
    prefix_loader = make_loader(
        shard_paths,
        split="test",
        batch_size=args.batch_size,
        seed=args.seed + 2,
        epoch=0,
        shuffle=False,
        max_examples=max(args.prefix_pairs * 16, args.batch_size),
    )
    pairs = collect_prefix_fixed_pairs(
        prefix_loader,
        device=device,
        length=data_config["length"],
        absolute_depth=data_config["absolute_depth"],
        simple_mats=simple_mats,
        boundary_radius=boundary_radius,
        num_pairs=args.prefix_pairs,
    )

    results = {
        "config": vars(args),
        "checkpoint_input_transform": loaded.checkpoint.get("input_transform", "unknown"),
        "effective_model_input_boundary_radius": boundary_radius,
        "data_config": data_config,
        "model_config": loaded.checkpoint["model_config"],
        "sites": {},
    }

    for site in sites:
        train_loader = make_loader(
            shard_paths,
            split="train",
            batch_size=args.batch_size,
            seed=args.seed,
            epoch=0,
            shuffle=True,
            max_examples=args.train_examples,
        )
        sae, history = train_sae_for_site(
            model,
            train_loader,
            spec=site,
            device=device,
            length=data_config["length"],
            absolute_depth=data_config["absolute_depth"],
            simple_mats=simple_mats,
            boundary_radius=boundary_radius,
            d_in=d_in,
            n_features=n_features,
            top_k=args.top_k,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            chunk_size=args.chunk_size,
        )
        sae_path = out_dir / f"{site.key}_sae.pt"
        torch.save(
            {
                "site": site.__dict__,
                "state_dict": sae.state_dict(),
                "d_in": d_in,
                "n_features": n_features,
                "top_k": args.top_k,
                "history": history,
            },
            sae_path,
        )

        eval_acts = collect_site_activation(model, eval_batch["tokens"], site, chunk_size=args.chunk_size)
        with torch.no_grad():
            _, eval_sparse = sae(eval_acts)
        binary_labels, categorical_labels = make_label_tables(eval_batch)
        labels = top_feature_label_summary(
            eval_sparse,
            binary_labels,
            categorical_labels,
            top_examples=args.top_label_examples,
            max_features=args.max_labeled_features,
        )
        selected = labels["selected_feature_ids"][: args.max_labeled_features]
        site_result = {
            "site": site.__dict__,
            "sae_path": str(sae_path),
            "sae_config": {
                "d_in": d_in,
                "n_features": n_features,
                "top_k": args.top_k,
                "expansion": args.expansion,
            },
            "train_history": history,
            "eval_reconstruction": sae_batch_summary(sae, eval_acts),
            "feature_labels": labels,
            "reconstruction_and_ablation": reconstruction_and_ablation_eval(
                model,
                eval_batch,
                spec=site,
                sae=sae,
                selected_features=selected,
                chunk_size=args.chunk_size,
            ),
            "prefix_feature_patching": prefix_feature_patch_eval(
                model,
                pairs,
                spec=site,
                sae=sae,
                selected_features=selected,
                chunk_size=args.chunk_size,
            ),
        }
        results["sites"][site.key] = site_result
        atomic_json_dump(results, out_dir / "results.json")
        print(
            f"site={site.key} eval_r2={site_result['eval_reconstruction']['r2']:.4f} "
            f"rec_exact={site_result['reconstruction_and_ablation']['sae_reconstruction']['exact_set_accuracy']:.4f} "
            f"patch_recovery={site_result['prefix_feature_patching']['all_sae_features_from_clean']['normalized_score_recovery']:.4f}",
            flush=True,
        )

    atomic_json_dump(results, out_dir / "results.json")


if __name__ == "__main__":
    main()
