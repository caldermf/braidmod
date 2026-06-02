#!/usr/bin/env python3
"""Matched-support boundary and activation patching for the B_4 transformer."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.b4_data import discover_shards  # noqa: E402
from interp.b4_interp import (  # noqa: E402
    load_transformer_checkpoint,
    metrics_from_logits,
    multilabel_logit_score,
    patch_positions_hook_from_cache,
    run_with_cache,
    support_features,
)
from interp.train_b4_transformer import atomic_json_dump, make_loader, resolve_device, set_seed  # noqa: E402


@torch.no_grad()
def collect_matched_pairs(loader, device: torch.device, max_examples: int, num_pairs: int) -> dict:
    groups: dict[tuple[int, int], dict[int, list[dict[str, torch.Tensor]]]] = defaultdict(lambda: defaultdict(list))
    scanned = 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        labels = batch["label_bits"].to(device)
        masks = batch["descent_mask"].to(device)
        final_factor = batch["final_factor_id"].to(device)
        sample_id = batch["sample_id"].to(device)
        feats = support_features(tokens)
        for idx in range(tokens.shape[0]):
            key = (int(feats["first"][idx].item()), int(feats["last"][idx].item()))
            mask = int(masks[idx].item())
            if len(groups[key][mask]) < 8:
                groups[key][mask].append(
                    {
                        "tokens": tokens[idx].detach().cpu(),
                        "label_bits": labels[idx].detach().cpu(),
                        "descent_mask": masks[idx].detach().cpu(),
                        "final_factor_id": final_factor[idx].detach().cpu(),
                        "sample_id": sample_id[idx].detach().cpu(),
                    }
                )
        scanned += int(tokens.shape[0])
        if scanned >= max_examples:
            break

    clean, corrupt = [], []
    for key in sorted(groups, key=lambda item: sum(len(v) for v in groups[item].values()), reverse=True):
        by_mask = groups[key]
        masks = [mask for mask, rows in by_mask.items() if rows]
        if len(masks) < 2:
            continue
        cursor = 0
        while len(clean) < num_pairs:
            mask_a = masks[cursor % len(masks)]
            mask_b = masks[(cursor + 1) % len(masks)]
            rows_a = by_mask[mask_a]
            rows_b = by_mask[mask_b]
            item_a = rows_a[(cursor // len(masks)) % len(rows_a)]
            item_b = rows_b[(cursor // len(masks)) % len(rows_b)]
            clean.append(item_a)
            corrupt.append(item_b)
            cursor += 1
            if cursor >= len(masks) * 8:
                break
        if len(clean) >= num_pairs:
            break

    if len(clean) < num_pairs:
        raise RuntimeError(f"Only found {len(clean)} matched pairs after scanning {scanned} examples")

    def stack(items: list[dict[str, torch.Tensor]], field: str) -> torch.Tensor:
        return torch.stack([item[field] for item in items], dim=0).to(device)

    out = {
        "clean_tokens": stack(clean, "tokens"),
        "corrupt_tokens": stack(corrupt, "tokens"),
        "clean_labels": stack(clean, "label_bits").to(torch.float32),
        "corrupt_labels": stack(corrupt, "label_bits").to(torch.float32),
        "clean_descent_mask": stack(clean, "descent_mask").to(torch.long),
        "corrupt_descent_mask": stack(corrupt, "descent_mask").to(torch.long),
        "clean_final_factor_id": stack(clean, "final_factor_id").to(torch.long),
        "corrupt_final_factor_id": stack(corrupt, "final_factor_id").to(torch.long),
        "clean_sample_id": stack(clean, "sample_id").to(torch.long),
        "corrupt_sample_id": stack(corrupt, "sample_id").to(torch.long),
        "matched_pairs": len(clean),
        "scanned_examples": scanned,
    }
    feats = support_features(out["clean_tokens"])
    out["matched_first_mode"] = int(feats["first"].mode().values.item())
    out["matched_last_mode"] = int(feats["last"].mode().values.item())
    return out


def window_token_indices(first: int, last: int, radius: int, depth: int, *, leading: bool, trailing: bool) -> list[int]:
    positions = set()
    centers = []
    if leading:
        centers.append(first)
    if trailing:
        centers.append(last)
    for center in centers:
        for pos in range(center - radius, center + radius + 1):
            if 0 <= pos < depth:
                positions.add(pos + 1)
    return sorted(positions)


def patch_input_tokens(clean_tokens: torch.Tensor, corrupt_tokens: torch.Tensor, token_indices: list[int]) -> torch.Tensor:
    patched = corrupt_tokens.clone()
    for token_idx in token_indices:
        if token_idx == 0:
            continue
        degree_idx = token_idx - 1
        patched[:, degree_idx] = clean_tokens[:, degree_idx]
    return patched


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
def token_patch_summary(model, pairs: dict, radii: list[int]) -> dict:
    clean_tokens = pairs["clean_tokens"]
    corrupt_tokens = pairs["corrupt_tokens"]
    clean_labels = pairs["clean_labels"]
    clean_logits = model(clean_tokens)
    corrupt_logits = model(corrupt_tokens)
    clean_score = score_against_clean(clean_logits, clean_labels)
    corrupt_score = score_against_clean(corrupt_logits, clean_labels)
    feats = support_features(clean_tokens)
    first = int(feats["first"].mode().values.item())
    last = int(feats["last"].mode().values.item())
    depth = clean_tokens.shape[1]

    out = {
        "clean": metrics_from_logits(clean_logits, clean_labels),
        "corrupt_scored_as_clean": patch_metrics(corrupt_logits, clean_labels, clean_score, corrupt_score),
        "variants": {},
    }
    for radius in radii:
        sets = {
            f"leading_r{radius}": window_token_indices(first, last, radius, depth, leading=True, trailing=False),
            f"trailing_r{radius}": window_token_indices(first, last, radius, depth, leading=False, trailing=True),
            f"both_boundaries_r{radius}": window_token_indices(first, last, radius, depth, leading=True, trailing=True),
        }
        interior = [idx for idx in range(1, depth + 1) if idx not in set(sets[f"both_boundaries_r{radius}"])]
        sets[f"interior_except_boundaries_r{radius}"] = interior
        for name, token_indices in sets.items():
            patched_tokens = patch_input_tokens(clean_tokens, corrupt_tokens, token_indices)
            logits = model(patched_tokens)
            out["variants"][name] = {
                "token_indices_count": len(token_indices),
                **patch_metrics(logits, clean_labels, clean_score, corrupt_score),
            }
    return out


@torch.no_grad()
def activation_patch_summary(model, pairs: dict, radii: list[int]) -> dict:
    clean_tokens = pairs["clean_tokens"]
    corrupt_tokens = pairs["corrupt_tokens"]
    clean_labels = pairs["clean_labels"]
    clean_logits, clean_cache = run_with_cache(model, clean_tokens)
    corrupt_logits, _ = run_with_cache(model, corrupt_tokens)
    clean_score = score_against_clean(clean_logits, clean_labels)
    corrupt_score = score_against_clean(corrupt_logits, clean_labels)
    feats = support_features(clean_tokens)
    first = int(feats["first"].mode().values.item())
    last = int(feats["last"].mode().values.item())
    depth = clean_tokens.shape[1]

    names = ["hook_resid_embed"]
    for layer in range(len(model.blocks)):
        names.extend(
            [
                f"blocks.{layer}.hook_resid_pre",
                f"blocks.{layer}.hook_attn_out",
                f"blocks.{layer}.hook_resid_mid",
                f"blocks.{layer}.hook_mlp_out",
                f"blocks.{layer}.hook_resid_post",
            ]
        )
    names.append("hook_final_hidden")

    variants = {}
    for radius in radii:
        position_sets = {
            f"cls": [0],
            f"leading_r{radius}": window_token_indices(first, last, radius, depth, leading=True, trailing=False),
            f"trailing_r{radius}": window_token_indices(first, last, radius, depth, leading=False, trailing=True),
            f"both_boundaries_r{radius}": window_token_indices(first, last, radius, depth, leading=True, trailing=True),
            f"cls_plus_boundaries_r{radius}": [0]
            + window_token_indices(first, last, radius, depth, leading=True, trailing=True),
        }
        for site in names:
            if site not in clean_cache:
                continue
            for set_name, token_indices in position_sets.items():
                logits, _ = run_with_cache(
                    model,
                    corrupt_tokens,
                    hooks={site: patch_positions_hook_from_cache(clean_cache, site, token_indices)},
                    names_filter=set(),
                )
                variants[f"{site}::{set_name}"] = {
                    "site": site,
                    "position_set": set_name,
                    "token_indices_count": len(token_indices),
                    **patch_metrics(logits, clean_labels, clean_score, corrupt_score),
                }
    flat = sorted(variants.values(), key=lambda item: item["normalized_score_recovery"], reverse=True)
    return {
        "clean": metrics_from_logits(clean_logits, clean_labels),
        "corrupt_scored_as_clean": patch_metrics(corrupt_logits, clean_labels, clean_score, corrupt_score),
        "variants": variants,
        "top_recoveries": flat[:24],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run B_4 matched boundary activation patching.")
    parser.add_argument("--data-dir", default="interp/data/generated/b4_l25_p2_n16777216")
    parser.add_argument("--checkpoint", default="interp/artifacts/b4_l25_p2_xfmr3_abs/best_model.pt")
    parser.add_argument("--out", default="interp/artifacts/b4_l25_p2_matched_boundary_patching/results.json")
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-scan-examples", type=int, default=524288)
    parser.add_argument("--matched-pairs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    shard_paths = discover_shards(args.data_dir, num_shards=args.num_shards)
    model = load_transformer_checkpoint(args.checkpoint, device=device).model
    loader = make_loader(
        shard_paths,
        split="test",
        batch_size=args.batch_size,
        seed=args.seed,
        epoch=0,
        shuffle=True,
        max_examples=args.max_scan_examples,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    pairs = collect_matched_pairs(loader, device, args.max_scan_examples, args.matched_pairs)
    results = {
        "data_dir": args.data_dir,
        "checkpoint": args.checkpoint,
        "matched_pairs": int(pairs["matched_pairs"]),
        "scanned_examples": int(pairs["scanned_examples"]),
        "matched_first_mode": int(pairs["matched_first_mode"]),
        "matched_last_mode": int(pairs["matched_last_mode"]),
        "clean_descent_mask_counts": torch.bincount(pairs["clean_descent_mask"], minlength=8).cpu().tolist(),
        "corrupt_descent_mask_counts": torch.bincount(pairs["corrupt_descent_mask"], minlength=8).cpu().tolist(),
        "token_patching": token_patch_summary(model, pairs, radii=[0, 1, 2, 3, 5, 8]),
        "activation_patching": activation_patch_summary(model, pairs, radii=[0, 1, 2, 3]),
    }
    atomic_json_dump(results, Path(args.out))
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
