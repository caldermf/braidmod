#!/usr/bin/env python3
"""Targeted boundary-head circuit experiments for the B_4 transformer."""

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
    keep_heads_hook,
    load_transformer_checkpoint,
    metrics_from_logits,
    multilabel_logit_score,
    patch_head_positions_hook_from_cache,
    run_with_cache,
    support_features,
    zero_except_windows,
    zero_head_positions_hook,
    zero_hook,
)
from interp.run_b4_matched_boundary_patching import collect_matched_pairs, window_token_indices  # noqa: E402
from interp.train_b4_transformer import atomic_json_dump, make_loader, resolve_device, set_seed  # noqa: E402


@torch.no_grad()
def collect_examples(loader, device: torch.device, min_examples: int) -> dict:
    pieces: dict[str, list[torch.Tensor]] = defaultdict(list)
    count = 0
    for batch in loader:
        for key, value in batch.items():
            pieces[key].append(value)
        count += int(batch["label_bits"].shape[0])
        if count >= min_examples:
            break
    out = {key: torch.cat(values, dim=0)[:min_examples] for key, values in pieces.items()}
    out["tokens"] = out["tokens"].to(device)
    out["label_bits"] = out["label_bits"].to(device)
    return out


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
def eval_variant_metrics(model, batch: dict, boundary_radius: int) -> dict:
    tokens = batch["tokens"]
    labels = batch["label_bits"]
    boundary_tokens = zero_except_windows(tokens, boundary_radius, leading=True, trailing=True)
    variants = {
        "full": (tokens, {}),
        f"boundary_only_r{boundary_radius}": (boundary_tokens, {}),
    }
    for layer in range(len(model.blocks)):
        variants[f"full_zero_l{layer}_attn"] = (tokens, {f"blocks.{layer}.hook_attn_head_out": zero_hook})
        variants[f"full_zero_l{layer}_mlp"] = (tokens, {f"blocks.{layer}.hook_mlp_out": zero_hook})
        variants[f"boundary_r{boundary_radius}_zero_l{layer}_attn"] = (
            boundary_tokens,
            {f"blocks.{layer}.hook_attn_head_out": zero_hook},
        )
        variants[f"boundary_r{boundary_radius}_zero_l{layer}_mlp"] = (
            boundary_tokens,
            {f"blocks.{layer}.hook_mlp_out": zero_hook},
        )
        for head in range(model.blocks[layer].attn.num_heads):
            hook = zero_head_positions_hook(head_idx=head, token_indices=[0])
            variants[f"full_zero_l{layer}h{head}_cls"] = (tokens, {f"blocks.{layer}.hook_attn_head_out": hook})
            variants[f"boundary_r{boundary_radius}_zero_l{layer}h{head}_cls"] = (
                boundary_tokens,
                {f"blocks.{layer}.hook_attn_head_out": hook},
            )
    for layer in range(len(model.blocks)):
        for keep in ([0], [0, 1], [0, 1, 2]):
            if max(keep) >= model.blocks[layer].attn.num_heads:
                continue
            variants[f"full_keep_l{layer}h{''.join(map(str, keep))}_only_in_layer"] = (
                tokens,
                {f"blocks.{layer}.hook_attn_head_out": keep_heads_hook(keep)},
            )
            variants[f"boundary_r{boundary_radius}_keep_l{layer}h{''.join(map(str, keep))}_only_in_layer"] = (
                boundary_tokens,
                {f"blocks.{layer}.hook_attn_head_out": keep_heads_hook(keep)},
            )

    out = {}
    for name, (variant_tokens, hooks) in variants.items():
        if hooks:
            logits, _ = run_with_cache(model, variant_tokens, hooks=hooks, names_filter=set())
        else:
            logits = model(variant_tokens)
        out[name] = metrics_from_logits(logits, labels)
    return out


@torch.no_grad()
def matched_head_path_patch(model, pairs: dict, radii: list[int]) -> dict:
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

    variants = {}
    for layer in range(len(model.blocks)):
        prefix = f"blocks.{layer}"
        for head in range(model.blocks[layer].attn.num_heads):
            for hook_suffix in ["attn_head_out", "z"]:
                hook_name = f"{prefix}.hook_{hook_suffix}"
                logits, _ = run_with_cache(
                    model,
                    corrupt_tokens,
                    hooks={
                        hook_name: patch_head_positions_hook_from_cache(
                            clean_cache,
                            hook_name,
                            head_idx=head,
                            token_indices=[0],
                        )
                    },
                    names_filter=set(),
                )
                variants[f"l{layer}h{head}_{hook_suffix}_cls"] = {
                    "layer": layer,
                    "head": head,
                    "patch": f"{hook_suffix}_cls",
                    **patch_metrics(logits, clean_labels, clean_score, corrupt_score),
                }

            pattern_name = f"{prefix}.hook_pattern"
            logits, _ = run_with_cache(
                model,
                corrupt_tokens,
                hooks={
                    pattern_name: patch_head_positions_hook_from_cache(
                        clean_cache,
                        pattern_name,
                        head_idx=head,
                        token_indices=[0],
                    )
                },
                names_filter=set(),
            )
            variants[f"l{layer}h{head}_pattern_cls_row"] = {
                "layer": layer,
                "head": head,
                "patch": "pattern_cls_row",
                **patch_metrics(logits, clean_labels, clean_score, corrupt_score),
            }

            for radius in radii:
                source_positions = window_token_indices(first, last, radius, depth, leading=True, trailing=True)
                v_name = f"{prefix}.hook_v"
                logits, _ = run_with_cache(
                    model,
                    corrupt_tokens,
                    hooks={
                        v_name: patch_head_positions_hook_from_cache(
                            clean_cache,
                            v_name,
                            head_idx=head,
                            token_indices=source_positions,
                        )
                    },
                    names_filter=set(),
                )
                variants[f"l{layer}h{head}_v_boundaries_r{radius}"] = {
                    "layer": layer,
                    "head": head,
                    "patch": f"v_boundaries_r{radius}",
                    "source_positions_count": len(source_positions),
                    **patch_metrics(logits, clean_labels, clean_score, corrupt_score),
                }
                logits, _ = run_with_cache(
                    model,
                    corrupt_tokens,
                    hooks={
                        pattern_name: patch_head_positions_hook_from_cache(
                            clean_cache,
                            pattern_name,
                            head_idx=head,
                            token_indices=[0],
                        ),
                        v_name: patch_head_positions_hook_from_cache(
                            clean_cache,
                            v_name,
                            head_idx=head,
                            token_indices=source_positions,
                        ),
                    },
                    names_filter=set(),
                )
                variants[f"l{layer}h{head}_pattern_cls_plus_v_boundaries_r{radius}"] = {
                    "layer": layer,
                    "head": head,
                    "patch": f"pattern_cls_plus_v_boundaries_r{radius}",
                    "source_positions_count": len(source_positions),
                    **patch_metrics(logits, clean_labels, clean_score, corrupt_score),
                }

    top = sorted(variants.values(), key=lambda item: item["normalized_score_recovery"], reverse=True)
    return {
        "clean": metrics_from_logits(clean_logits, clean_labels),
        "corrupt_scored_as_clean": patch_metrics(corrupt_logits, clean_labels, clean_score, corrupt_score),
        "matched_first_mode": first,
        "matched_last_mode": last,
        "variants": variants,
        "top_recoveries": top[:32],
    }


@torch.no_grad()
def l2h0_attention_examples(model, batch: dict, examples: int) -> list[dict]:
    tokens = batch["tokens"][:examples]
    labels = batch["label_bits"][:examples]
    logits, cache = run_with_cache(model, tokens, names_filter={"blocks.2.hook_pattern"})
    pattern = cache["blocks.2.hook_pattern"][:, 0, 0]
    feats = support_features(tokens)
    out = []
    for idx in range(tokens.shape[0]):
        top_values, top_idx = torch.topk(pattern[idx], k=8)
        out.append(
            {
                "idx": idx,
                "true_bits": [int(x) for x in labels[idx].cpu().tolist()],
                "pred_bits": [int(x) for x in (logits[idx] >= 0).to(torch.long).cpu().tolist()],
                "first": int(feats["first"][idx].item()),
                "last": int(feats["last"][idx].item()),
                "top_token_indices": [int(x) for x in top_idx.cpu().tolist()],
                "top_degrees": [int(x) - 1 if int(x) > 0 else "CLS" for x in top_idx.cpu().tolist()],
                "top_attention": [float(x) for x in top_values.cpu().tolist()],
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run targeted B_4 boundary-head circuit experiments.")
    parser.add_argument("--data-dir", default="interp/data/generated/b4_l25_p2_n16777216")
    parser.add_argument("--checkpoint", default="interp/artifacts/b4_l25_p2_xfmr3_abs/best_model.pt")
    parser.add_argument("--out", default="interp/artifacts/b4_l25_p2_boundary_head_circuit/results.json")
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--max-scan-examples", type=int, default=524288)
    parser.add_argument("--matched-pairs", type=int, default=256)
    parser.add_argument("--boundary-radius", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    shard_paths = discover_shards(args.data_dir, num_shards=args.num_shards)
    model = load_transformer_checkpoint(args.checkpoint, device=device).model
    eval_loader = make_loader(
        shard_paths,
        split="test",
        batch_size=args.batch_size,
        seed=args.seed,
        epoch=0,
        shuffle=True,
        max_examples=args.examples,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    batch = collect_examples(eval_loader, device, args.examples)
    pair_loader = make_loader(
        shard_paths,
        split="test",
        batch_size=args.batch_size,
        seed=args.seed + 1234,
        epoch=0,
        shuffle=True,
        max_examples=args.max_scan_examples,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    pairs = collect_matched_pairs(pair_loader, device, args.max_scan_examples, args.matched_pairs)
    results = {
        "data_dir": args.data_dir,
        "checkpoint": args.checkpoint,
        "examples": int(batch["tokens"].shape[0]),
        "matched_pairs": int(pairs["matched_pairs"]),
        "variant_metrics": eval_variant_metrics(model, batch, args.boundary_radius),
        "matched_head_path_patching": matched_head_path_patch(model, pairs, radii=[0, 1, 3, 5, 8]),
        "l2h0_attention_examples": l2h0_attention_examples(model, batch, examples=min(16, args.examples)),
    }
    atomic_json_dump(results, Path(args.out))
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
