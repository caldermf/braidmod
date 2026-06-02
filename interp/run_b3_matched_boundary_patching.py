#!/usr/bin/env python3
"""Matched-support boundary interventions and activation patching."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.b3_data import discover_shards  # noqa: E402
from interp.b3_interp import (  # noqa: E402
    activation_patch_token_sweep,
    binary_logit_score,
    load_transformer_checkpoint,
    patch_positions_hook_from_cache,
    run_with_cache,
    support_features,
)
from interp.train_b3_transformer import atomic_json_dump, make_loader, resolve_device, set_seed  # noqa: E402


def collect_matched_pairs(loader, device: torch.device, max_scan_examples: int, max_pairs: int) -> dict:
    groups: dict[tuple[int, int], dict[int, list[torch.Tensor]]] = defaultdict(lambda: {0: [], 1: []})
    scanned = 0
    for batch in loader:
        tokens = batch["tokens"]
        labels = batch["label"].to(torch.long)
        feats = support_features(tokens)
        for i in range(tokens.shape[0]):
            key = (int(feats["first"][i].item()), int(feats["last"][i].item()))
            label = int(labels[i].item())
            if len(groups[key][label]) < max_pairs:
                groups[key][label].append(tokens[i].clone())
        scanned += int(tokens.shape[0])
        best_found = max((min(len(v[0]), len(v[1])) for v in groups.values()), default=0)
        if best_found >= max_pairs or (max_scan_examples > 0 and scanned >= max_scan_examples):
            break

    best_key = None
    best_n = 0
    for key, sides in groups.items():
        n = min(len(sides[0]), len(sides[1]))
        if n > best_n:
            best_key = key
            best_n = n
    if best_key is None or best_n == 0:
        raise RuntimeError("No matched opposite-label pairs found")
    first, last = best_key
    sides = groups[best_key]
    n = min(best_n, max_pairs)
    label0 = torch.stack(sides[0][:n], dim=0)
    label1 = torch.stack(sides[1][:n], dim=0)
    return {
        "clean_tokens": torch.cat([label0, label1], dim=0).to(device),
        "corrupt_tokens": torch.cat([label1, label0], dim=0).to(device),
        "clean_labels": torch.cat([torch.zeros(n, dtype=torch.float32), torch.ones(n, dtype=torch.float32)], dim=0).to(device),
        "pair_meta": [{"first": first, "last": last, "label": 0}] * n
        + [{"first": first, "last": last, "label": 1}] * n,
        "matched_first": first,
        "matched_last": last,
        "matched_pairs": n,
        "scanned_examples": scanned,
    }


def swap_boundary_tokens(corrupt: torch.Tensor, clean: torch.Tensor, *, leading: bool, trailing: bool) -> torch.Tensor:
    clean_feats = support_features(clean)
    corrupt_feats = support_features(corrupt)
    if not bool(clean_feats["first"].eq(corrupt_feats["first"]).all().item()):
        raise ValueError("Matched pairs do not share first support degree")
    if not bool(clean_feats["last"].eq(corrupt_feats["last"]).all().item()):
        raise ValueError("Matched pairs do not share last support degree")
    out = corrupt.clone()
    rows = torch.arange(out.shape[0], device=out.device)
    if leading:
        idx = clean_feats["first"]
        out[rows, idx] = clean[rows, idx]
    if trailing:
        idx = clean_feats["last"]
        out[rows, idx] = clean[rows, idx]
    return out


def metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    labels = labels.to(device=logits.device, dtype=torch.float32)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
    preds = (logits >= 0).to(torch.float32)
    return {
        "loss": float(loss.item()),
        "accuracy": float((preds == labels).float().mean().item()),
        "mean_logit_score": float(binary_logit_score(logits, labels).mean().item()),
        "positive_rate": float(preds.mean().item()),
    }


@torch.no_grad()
def input_swap_summary(model, clean_tokens: torch.Tensor, corrupt_tokens: torch.Tensor, clean_labels: torch.Tensor) -> dict:
    variants = {
        "clean": clean_tokens,
        "corrupt": corrupt_tokens,
        "corrupt_with_clean_leading": swap_boundary_tokens(corrupt_tokens, clean_tokens, leading=True, trailing=False),
        "corrupt_with_clean_trailing": swap_boundary_tokens(corrupt_tokens, clean_tokens, leading=False, trailing=True),
        "corrupt_with_clean_both_boundaries": swap_boundary_tokens(corrupt_tokens, clean_tokens, leading=True, trailing=True),
    }
    return {name: metrics(model(tokens), clean_labels) for name, tokens in variants.items()}


@torch.no_grad()
def focused_patch_summary(model, clean_tokens: torch.Tensor, corrupt_tokens: torch.Tensor, clean_labels: torch.Tensor) -> dict:
    feats = support_features(clean_tokens)
    first = int(feats["first"].mode().values.item())
    last = int(feats["last"].mode().values.item())
    output = {}
    for site in ["hook_resid_post", "hook_attn_out", "hook_mlp_out"]:
        sweep = activation_patch_token_sweep(model, clean_tokens, corrupt_tokens, clean_labels.to(torch.long), site=site)
        recovery = sweep["recovery"]
        position_sets = {
            "cls": [0],
            "leading": [first + 1],
            "trailing": [last + 1],
            "both_boundaries": [first + 1, last + 1],
            "cls_plus_boundaries": [0, first + 1, last + 1],
        }
        position_set_recovery = {name: [] for name in position_sets}
        position_set_patched_score = {name: [] for name in position_sets}
        clean_score = sweep["clean_score"].to(device=clean_tokens.device)
        corrupt_score = sweep["corrupt_score"].to(device=clean_tokens.device)
        denom = (clean_score - corrupt_score).mean().clamp_min(1e-6)
        _, clean_cache = run_with_cache(model, clean_tokens)
        for layer in range(len(model.blocks)):
            hook_name = f"blocks.{layer}.{site}"
            for set_name, token_indices in position_sets.items():
                patched_logits, _ = run_with_cache(
                    model,
                    corrupt_tokens,
                    hooks={hook_name: patch_positions_hook_from_cache(clean_cache, hook_name, token_indices)},
                    names_filter=set(),
                )
                patched_score = binary_logit_score(patched_logits, clean_labels).mean()
                position_set_recovery[set_name].append(float(((patched_score - corrupt_score.mean()) / denom).item()))
                position_set_patched_score[set_name].append(float(patched_score.item()))
        output[site] = {
            "first_degree": first,
            "last_degree": last,
            "activation_token_indices": {
                "cls": 0,
                "leading": first + 1,
                "trailing": last + 1,
            },
            "cls_recovery_by_layer": recovery[:, 0].tolist(),
            "leading_recovery_by_layer": recovery[:, first + 1].tolist(),
            "trailing_recovery_by_layer": recovery[:, last + 1].tolist(),
            "position_set_recovery_by_layer": position_set_recovery,
            "position_set_patched_score_by_layer": position_set_patched_score,
            "max_recovery_by_layer": recovery.max(dim=1).values.tolist(),
            "argmax_token_by_layer": recovery.argmax(dim=1).tolist(),
            "mean_clean_score": float(sweep["clean_score"].mean().item()),
            "mean_corrupt_score": float(sweep["corrupt_score"].mean().item()),
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched-support B_3 boundary interventions.")
    parser.add_argument("--data-dir", default="interp/data/generated/b3_l25_p2_full")
    parser.add_argument("--checkpoint", default="interp/artifacts/b3_l25_p2_xfmr2_abs/best_model.pt")
    parser.add_argument("--out", default="interp/artifacts/b3_l25_p2_matched_boundary_patching/results.json")
    parser.add_argument("--length", type=int, default=25)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-scan-examples", type=int, default=1_048_576)
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    shard_paths = discover_shards(args.data_dir, num_shards=args.num_shards)
    loader = make_loader(
        shard_paths,
        split="test",
        length=args.length,
        batch_size=args.batch_size,
        seed=args.seed,
        epoch=0,
        shuffle=True,
        max_examples=args.max_scan_examples,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    pairs = collect_matched_pairs(loader, device, args.max_scan_examples, args.max_pairs)
    model = load_transformer_checkpoint(args.checkpoint, device=device).model
    with torch.no_grad():
        direct_logits = model(pairs["clean_tokens"])
        cached_logits, _ = run_with_cache(model, pairs["clean_tokens"])
    result = {
        "data_dir": args.data_dir,
        "checkpoint": args.checkpoint,
        "num_bidirectional_examples": int(pairs["clean_tokens"].shape[0]),
        "num_matched_pairs": int(pairs["matched_pairs"]),
        "matched_first": int(pairs["matched_first"]),
        "matched_last": int(pairs["matched_last"]),
        "scanned_examples": int(pairs["scanned_examples"]),
        "cache_forward_max_abs_diff": float((direct_logits - cached_logits).abs().max().item()),
        "input_swaps": input_swap_summary(model, pairs["clean_tokens"], pairs["corrupt_tokens"], pairs["clean_labels"]),
        "activation_patching": focused_patch_summary(
            model,
            pairs["clean_tokens"],
            pairs["corrupt_tokens"],
            pairs["clean_labels"],
        ),
        "pair_meta_sample": pairs["pair_meta"][:10],
    }
    atomic_json_dump(result, Path(args.out))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
