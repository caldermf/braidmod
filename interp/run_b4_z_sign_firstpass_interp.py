#!/usr/bin/env python3
"""First-pass interp experiments for the B_4 Z[v] sign-token transformer."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.b4_data import discover_shards  # noqa: E402
from interp.b4_interp import (  # noqa: E402
    gather_relative_window,
    load_transformer_checkpoint,
    metrics_from_logits,
    run_with_cache,
    support_features,
    zero_except_windows,
    zero_windows,
)
from interp.b4_z_sign import B4FactorBatchIterable, factor_ids_to_z_sign_tokens, simple_mats_z  # noqa: E402
from interp.train_b4_transformer import atomic_json_dump, resolve_device, set_seed  # noqa: E402


def mask_from_bits(bits: torch.Tensor) -> torch.Tensor:
    bits = bits.to(torch.long)
    weights = torch.tensor([1, 2, 4], device=bits.device, dtype=torch.long).view(1, 3)
    return (bits * weights).sum(dim=1)


def bits_from_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(torch.long)
    return torch.stack([((mask >> idx) & 1) for idx in range(3)], dim=1).to(torch.float32)


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


@torch.no_grad()
def collect_examples(loader, *, device: torch.device, length: int, absolute_depth: int, min_examples: int) -> dict:
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
        pieces["tokens"].append(tokens.cpu())
        pieces["label_bits"].append(batch["label_bits"].cpu())
        pieces["descent_mask"].append(batch["descent_mask"].cpu())
        pieces["sample_id"].append(batch["sample_id"].cpu())
        pieces["final_factor_id"].append(batch["final_factor_id"].cpu())
        count += int(tokens.shape[0])
        if count >= min_examples:
            break
    out = {key: torch.cat(values, dim=0)[:min_examples] for key, values in pieces.items()}
    out["tokens"] = out["tokens"].to(device)
    out["label_bits"] = out["label_bits"].to(device)
    return out


@torch.no_grad()
def eval_tokens(model, tokens: torch.Tensor, labels: torch.Tensor, *, chunk_size: int) -> dict:
    logits = []
    for start in range(0, tokens.shape[0], chunk_size):
        logits.append(model(tokens[start : start + chunk_size]))
    return metrics_from_logits(torch.cat(logits, dim=0), labels)


@torch.no_grad()
def boundary_intervention_summary(model, batch: dict, radii: list[int], *, chunk_size: int) -> dict:
    tokens = batch["tokens"]
    labels = batch["label_bits"]
    summary = {"full": eval_tokens(model, tokens, labels, chunk_size=chunk_size)}
    for radius in radii:
        variants = {
            f"boundary_only_r{radius}": zero_except_windows(tokens, radius, leading=True, trailing=True),
            f"leading_only_r{radius}": zero_except_windows(tokens, radius, leading=True, trailing=False),
            f"trailing_only_r{radius}": zero_except_windows(tokens, radius, leading=False, trailing=True),
            f"drop_boundary_r{radius}": zero_windows(tokens, radius, leading=True, trailing=True),
        }
        for name, variant_tokens in variants.items():
            summary[name] = eval_tokens(model, variant_tokens, labels, chunk_size=chunk_size)
    return summary


def majority_lookup(train_keys: Iterable[tuple], train_masks: Iterable[int]) -> tuple[dict[tuple, int], int]:
    counts: dict[tuple, Counter] = defaultdict(Counter)
    global_counts: Counter = Counter()
    for key, mask in zip(train_keys, train_masks, strict=True):
        mask = int(mask)
        counts[key][mask] += 1
        global_counts[mask] += 1
    fallback = global_counts.most_common(1)[0][0]
    table = {key: counter.most_common(1)[0][0] for key, counter in counts.items()}
    return table, fallback


def lookup_metrics(table: dict[tuple, int], fallback: int, eval_keys: Iterable[tuple], eval_masks: torch.Tensor) -> dict:
    preds = []
    seen = 0
    for key in eval_keys:
        if key in table:
            seen += 1
        preds.append(table.get(key, fallback))
    pred_masks = torch.tensor(preds, dtype=torch.long)
    true_masks = eval_masks.cpu().to(torch.long)
    pred_bits = bits_from_mask(pred_masks)
    true_bits = bits_from_mask(true_masks)
    return {
        "coverage": seen / max(1, len(preds)),
        "exact_set_accuracy": float((pred_masks == true_masks).float().mean().item()),
        "bit_accuracy": float(pred_bits.eq(true_bits).float().mean().item()),
        "n": len(preds),
        "pred_mask_counts": torch.bincount(pred_masks, minlength=8).tolist(),
    }


def tensor_rows_as_tuples(x: torch.Tensor) -> list[tuple]:
    return [tuple(int(v) for v in row.tolist()) for row in x.cpu()]


def feature_keys(tokens: torch.Tensor, *, radius: int | None = None) -> dict[str, list[tuple]]:
    feats = support_features(tokens)
    base = {
        "min_degree": [(int(v),) for v in feats["first"].cpu().tolist()],
        "max_degree": [(int(v),) for v in feats["last"].cpu().tolist()],
        "support_width": [(int(v),) for v in feats["width"].cpu().tolist()],
        "leading_token": [(int(v),) for v in feats["leading_token"].cpu().tolist()],
        "trailing_token": [(int(v),) for v in feats["trailing_token"].cpu().tolist()],
    }
    base["boundary_tokens"] = [
        (int(a), int(b))
        for a, b in zip(feats["leading_token"].cpu().tolist(), feats["trailing_token"].cpu().tolist(), strict=True)
    ]
    if radius is not None:
        lead = gather_relative_window(tokens, feats["first"], radius)
        trail = gather_relative_window(tokens, feats["last"], radius)
        lead_rows = tensor_rows_as_tuples(lead)
        trail_rows = tensor_rows_as_tuples(trail)
        base[f"leading_window_r{radius}"] = lead_rows
        base[f"trailing_window_r{radius}"] = trail_rows
        base[f"both_windows_r{radius}"] = [
            lead_row + trail_row
            for lead_row, trail_row in zip(lead_rows, trail_rows, strict=True)
        ]
    return base


def raw_feature_lookup_summary(train_batch: dict, eval_batch: dict, radii: list[int]) -> dict:
    train_masks = train_batch["descent_mask"].cpu().to(torch.long)
    eval_masks = eval_batch["descent_mask"].cpu().to(torch.long)
    summary = {}
    all_train_keys = feature_keys(train_batch["tokens"], radius=None)
    all_eval_keys = feature_keys(eval_batch["tokens"], radius=None)
    for radius in radii:
        all_train_keys.update(feature_keys(train_batch["tokens"], radius=radius))
        all_eval_keys.update(feature_keys(eval_batch["tokens"], radius=radius))
    for name, keys in all_train_keys.items():
        table, fallback = majority_lookup(keys, train_masks.tolist())
        summary[name] = {
            "unique_train_keys": len(table),
            "fallback_mask": int(fallback),
            **lookup_metrics(table, fallback, all_eval_keys[name], eval_masks),
        }
    return summary


@torch.no_grad()
def attention_boundary_summary(model, batch: dict, *, chunk_size: int) -> dict:
    tokens = batch["tokens"]
    labels = batch["label_bits"]
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
    top_boundary = sorted(heads.items(), key=lambda item: item[1]["mean_cls_to_boundaries"], reverse=True)
    top_leading = sorted(heads.items(), key=lambda item: item[1]["mean_cls_to_leading"], reverse=True)
    top_trailing = sorted(heads.items(), key=lambda item: item[1]["mean_cls_to_trailing"], reverse=True)
    return {
        "n": n_total,
        "label_mask_counts": torch.bincount(mask_from_bits(labels), minlength=8).cpu().tolist(),
        "top_by_boundary_attention": [{"head": key, **value} for key, value in top_boundary[:12]],
        "top_by_leading_attention": [{"head": key, **value} for key, value in top_leading[:12]],
        "top_by_trailing_attention": [{"head": key, **value} for key, value in top_trailing[:12]],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run first-pass interp for B_4 Z-sign transformer.")
    parser.add_argument("--data-dir", default="interp/data/generated/b4_l25_p2_n16777216")
    parser.add_argument("--checkpoint", default="interp/artifacts/b4_l25_zsign_xfmr3_abs/best_model.pt")
    parser.add_argument("--out", default="interp/artifacts/b4_l25_zsign_firstpass_interp/results.json")
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-examples", type=int, default=8192)
    parser.add_argument("--train-lookup-examples", type=int, default=32768)
    parser.add_argument("--attn-examples", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--attn-chunk-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260602)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    loaded = load_transformer_checkpoint(args.checkpoint, device=device)
    model = loaded.model
    cfg = model.config
    shard_paths = discover_shards(args.data_dir, num_shards=args.num_shards)

    eval_loader = make_factor_loader(
        shard_paths,
        split="test",
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=False,
        max_examples=args.eval_examples,
    )
    train_lookup_loader = make_factor_loader(
        shard_paths,
        split="train",
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=True,
        max_examples=args.train_lookup_examples,
    )
    attn_loader = make_factor_loader(
        shard_paths,
        split="test",
        batch_size=args.batch_size,
        seed=args.seed + 1,
        shuffle=True,
        max_examples=args.attn_examples,
    )
    eval_batch = collect_examples(
        eval_loader,
        device=device,
        length=cfg.length,
        absolute_depth=cfg.absolute_depth,
        min_examples=args.eval_examples,
    )
    train_lookup_batch = collect_examples(
        train_lookup_loader,
        device=device,
        length=cfg.length,
        absolute_depth=cfg.absolute_depth,
        min_examples=args.train_lookup_examples,
    )
    attn_batch = collect_examples(
        attn_loader,
        device=device,
        length=cfg.length,
        absolute_depth=cfg.absolute_depth,
        min_examples=args.attn_examples,
    )

    result = {
        "config": vars(args),
        "checkpoint_model_config": cfg.to_dict(),
        "boundary_interventions": boundary_intervention_summary(
            model,
            eval_batch,
            radii=[0, 1, 2, 3, 5, 8],
            chunk_size=args.chunk_size,
        ),
        "raw_feature_lookup": raw_feature_lookup_summary(
            train_lookup_batch,
            eval_batch,
            radii=[0, 1, 2, 3, 5],
        ),
        "attention_boundary_summary": attention_boundary_summary(
            model,
            attn_batch,
            chunk_size=args.attn_chunk_size,
        ),
    }
    atomic_json_dump(result, Path(args.out))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
