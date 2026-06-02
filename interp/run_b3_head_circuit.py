#!/usr/bin/env python3
"""Head-level circuit analysis for the B_3 Burau transformer."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.b3_data import discover_shards  # noqa: E402
from interp.b3_interp import (  # noqa: E402
    binary_logit_score,
    load_transformer_checkpoint,
    patch_head_positions_hook_from_cache,
    patch_head_set_positions_hook_from_cache,
    run_with_cache,
    support_features,
    zero_head_positions_hook,
    zero_head_set_positions_hook,
)
from interp.run_b3_matched_boundary_patching import collect_matched_pairs  # noqa: E402
from interp.train_b3_transformer import atomic_json_dump, make_loader, resolve_device, set_seed  # noqa: E402


def metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    labels = labels.to(device=logits.device, dtype=torch.float32)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
    preds = (logits >= 0).to(torch.float32)
    return {
        "loss": float(loss.item()),
        "accuracy": float((preds == labels).float().mean().item()),
        "mean_logit_score": float(binary_logit_score(logits, labels).mean().item()),
        "positive_rate": float(preds.mean().item()),
        "n": int(labels.numel()),
    }


@torch.no_grad()
def collect_examples(loader, device: torch.device, min_examples: int) -> dict:
    pieces: dict[str, list[torch.Tensor]] = defaultdict(list)
    count = 0
    for batch in loader:
        for key, value in batch.items():
            pieces[key].append(value)
        count += int(batch["label"].numel())
        if count >= min_examples:
            break
    out = {key: torch.cat(values, dim=0)[:min_examples] for key, values in pieces.items()}
    out["tokens"] = out["tokens"].to(device)
    out["label"] = out["label"].to(device)
    return out


def _rank_positions(attn: torch.Tensor) -> torch.Tensor:
    order = attn.argsort(dim=1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(attn.shape[1], device=attn.device).expand_as(order))
    return rank + 1


@torch.no_grad()
def attention_boundary_summary(model, tokens: torch.Tensor, labels: torch.Tensor) -> dict:
    logits, cache = run_with_cache(model, tokens)
    feats = support_features(tokens)
    rows = torch.arange(tokens.shape[0], device=tokens.device)
    first_pos = feats["first"] + 1
    last_pos = feats["last"] + 1
    support_mask = torch.cat([torch.zeros(tokens.shape[0], 1, dtype=torch.bool, device=tokens.device), feats["support"]], dim=1)
    out = {
        "metrics": metrics(logits, labels),
        "layers": {},
    }

    for layer in range(len(model.blocks)):
        pattern = cache[f"blocks.{layer}.hook_pattern"]
        cls_attn = pattern[:, :, 0, :]
        heads = {}
        for head in range(cls_attn.shape[1]):
            head_attn = cls_attn[:, head]
            rank = _rank_positions(head_attn)
            entropy = -(head_attn.clamp_min(1e-12) * head_attn.clamp_min(1e-12).log()).sum(dim=1)
            mean_attn = head_attn.mean(dim=0)
            top_values, top_idx = torch.topk(mean_attn, k=min(12, mean_attn.numel()))
            head_summary = {
                "mean_cls_to_leading": float(head_attn[rows, first_pos].mean().item()),
                "mean_cls_to_trailing": float(head_attn[rows, last_pos].mean().item()),
                "mean_cls_to_both_boundaries": float((head_attn[rows, first_pos] + head_attn[rows, last_pos]).mean().item()),
                "mean_cls_to_support": float((head_attn * support_mask).sum(dim=1).mean().item()),
                "mean_cls_to_cls": float(head_attn[:, 0].mean().item()),
                "mean_leading_rank": float(rank[rows, first_pos].float().mean().item()),
                "mean_trailing_rank": float(rank[rows, last_pos].float().mean().item()),
                "mean_entropy": float(entropy.mean().item()),
                "top_positions": top_idx.cpu().tolist(),
                "top_degrees": [int(i) - 1 if int(i) > 0 else "CLS" for i in top_idx.cpu().tolist()],
                "top_attention": top_values.cpu().tolist(),
                "by_label": {},
            }
            for label in [0, 1]:
                mask = labels.to(torch.long) == label
                if not bool(mask.any().item()):
                    continue
                n_label = int(mask.sum().item())
                head_summary["by_label"][str(label)] = {
                    "mean_cls_to_leading": float(
                        head_attn[mask][torch.arange(n_label, device=tokens.device), first_pos[mask]].mean().item()
                    ),
                    "mean_cls_to_trailing": float(
                        head_attn[mask][torch.arange(n_label, device=tokens.device), last_pos[mask]].mean().item()
                    ),
                    "mean_cls_to_support": float((head_attn[mask] * support_mask[mask]).sum(dim=1).mean().item()),
                    "n": n_label,
                }
            heads[str(head)] = head_summary
        out["layers"][str(layer)] = heads
    return out


@torch.no_grad()
def head_ablation_summary(model, tokens: torch.Tensor, labels: torch.Tensor) -> dict:
    base_logits, _ = run_with_cache(model, tokens)
    base_metrics = metrics(base_logits, labels)
    base_score = binary_logit_score(base_logits, labels).mean()
    layers = {}
    for layer in range(len(model.blocks)):
        name = f"blocks.{layer}.hook_attn_head_out"
        heads = {}
        for head in range(model.blocks[layer].attn.num_heads):
            head_results = {}
            for ablation_name, token_indices in {
                "all_destinations": None,
                "cls_destination": [0],
            }.items():
                logits, _ = run_with_cache(
                    model,
                    tokens,
                    hooks={name: zero_head_positions_hook(head_idx=head, token_indices=token_indices)},
                    names_filter=set(),
                )
                ablated_metrics = metrics(logits, labels)
                ablated_score = binary_logit_score(logits, labels).mean()
                head_results[ablation_name] = {
                    **ablated_metrics,
                    "logit_score_damage": float((base_score - ablated_score).item()),
                    "accuracy_damage": float(base_metrics["accuracy"] - ablated_metrics["accuracy"]),
                }
            heads[str(head)] = head_results
        layers[str(layer)] = heads
    return {
        "base": base_metrics,
        "layers": layers,
    }


def head_subsets(num_heads: int) -> list[tuple[int, ...]]:
    return [
        subset
        for size in range(1, num_heads + 1)
        for subset in combinations(range(num_heads), size)
    ]


@torch.no_grad()
def head_set_ablation_summary(model, tokens: torch.Tensor, labels: torch.Tensor) -> dict:
    base_logits, _ = run_with_cache(model, tokens)
    base_metrics = metrics(base_logits, labels)
    base_score = binary_logit_score(base_logits, labels).mean()
    layers = {}
    flat = []
    for layer in range(len(model.blocks)):
        name = f"blocks.{layer}.hook_attn_head_out"
        subsets = {}
        for subset in head_subsets(model.blocks[layer].attn.num_heads):
            key = ",".join(str(head) for head in subset)
            logits, _ = run_with_cache(
                model,
                tokens,
                hooks={name: zero_head_set_positions_hook(head_indices=list(subset), token_indices=[0])},
                names_filter=set(),
            )
            ablated_metrics = metrics(logits, labels)
            ablated_score = binary_logit_score(logits, labels).mean()
            entry = {
                **ablated_metrics,
                "heads": list(subset),
                "logit_score_damage": float((base_score - ablated_score).item()),
                "accuracy_damage": float(base_metrics["accuracy"] - ablated_metrics["accuracy"]),
            }
            subsets[key] = entry
            flat.append({"layer": layer, "head_set": key, **entry})
        layers[str(layer)] = subsets
    flat.sort(key=lambda item: item["logit_score_damage"], reverse=True)
    return {
        "base": base_metrics,
        "layers": layers,
        "top_logit_score_damage": flat[:16],
    }


@torch.no_grad()
def matched_head_patch_summary(model, pairs: dict) -> dict:
    clean_tokens = pairs["clean_tokens"]
    corrupt_tokens = pairs["corrupt_tokens"]
    clean_labels = pairs["clean_labels"]
    clean_logits, clean_cache = run_with_cache(model, clean_tokens)
    corrupt_logits, _ = run_with_cache(model, corrupt_tokens)
    clean_score = binary_logit_score(clean_logits, clean_labels)
    corrupt_score = binary_logit_score(corrupt_logits, clean_labels)
    denom = (clean_score - corrupt_score).mean().clamp_min(1e-6)

    feats = support_features(clean_tokens)
    first = int(feats["first"].mode().values.item())
    last = int(feats["last"].mode().values.item())
    position_sets = {
        "cls": [0],
        "leading": [first + 1],
        "trailing": [last + 1],
        "both_boundaries": [first + 1, last + 1],
        "cls_plus_boundaries": [0, first + 1, last + 1],
    }

    layers = {}
    flat = []
    for layer in range(len(model.blocks)):
        name = f"blocks.{layer}.hook_attn_head_out"
        heads = {}
        for head in range(model.blocks[layer].attn.num_heads):
            head_results = {}
            for set_name, token_indices in position_sets.items():
                patched_logits, _ = run_with_cache(
                    model,
                    corrupt_tokens,
                    hooks={
                        name: patch_head_positions_hook_from_cache(
                            clean_cache,
                            name,
                            head_idx=head,
                            token_indices=token_indices,
                        )
                    },
                    names_filter=set(),
                )
                patched_score = binary_logit_score(patched_logits, clean_labels).mean()
                recovery = (patched_score - corrupt_score.mean()) / denom
                entry = {
                    "patched_score": float(patched_score.item()),
                    "recovery": float(recovery.item()),
                }
                head_results[set_name] = entry
                flat.append(
                    {
                        "layer": layer,
                        "head": head,
                        "position_set": set_name,
                        **entry,
                    }
                )
            heads[str(head)] = head_results
        layers[str(layer)] = heads

    flat.sort(key=lambda item: item["recovery"], reverse=True)
    return {
        "num_bidirectional_examples": int(clean_tokens.shape[0]),
        "num_matched_pairs": int(pairs["matched_pairs"]),
        "matched_first": int(pairs["matched_first"]),
        "matched_last": int(pairs["matched_last"]),
        "activation_token_indices": {"cls": 0, "leading": first + 1, "trailing": last + 1},
        "clean": metrics(clean_logits, clean_labels),
        "corrupt_against_clean_labels": metrics(corrupt_logits, clean_labels),
        "mean_clean_score": float(clean_score.mean().item()),
        "mean_corrupt_score": float(corrupt_score.mean().item()),
        "layers": layers,
        "top_recoveries": flat[:16],
    }


@torch.no_grad()
def matched_head_set_patch_summary(model, pairs: dict) -> dict:
    clean_tokens = pairs["clean_tokens"]
    corrupt_tokens = pairs["corrupt_tokens"]
    clean_labels = pairs["clean_labels"]
    clean_logits, clean_cache = run_with_cache(model, clean_tokens)
    corrupt_logits, _ = run_with_cache(model, corrupt_tokens)
    clean_score = binary_logit_score(clean_logits, clean_labels)
    corrupt_score = binary_logit_score(corrupt_logits, clean_labels)
    denom = (clean_score - corrupt_score).mean().clamp_min(1e-6)

    feats = support_features(clean_tokens)
    first = int(feats["first"].mode().values.item())
    last = int(feats["last"].mode().values.item())
    position_sets = {
        "cls": [0],
        "both_boundaries": [first + 1, last + 1],
        "cls_plus_boundaries": [0, first + 1, last + 1],
    }

    layers = {}
    flat = []
    for layer in range(len(model.blocks)):
        name = f"blocks.{layer}.hook_attn_head_out"
        subsets = {}
        for subset in head_subsets(model.blocks[layer].attn.num_heads):
            key = ",".join(str(head) for head in subset)
            set_results = {}
            for set_name, token_indices in position_sets.items():
                patched_logits, _ = run_with_cache(
                    model,
                    corrupt_tokens,
                    hooks={
                        name: patch_head_set_positions_hook_from_cache(
                            clean_cache,
                            name,
                            head_indices=list(subset),
                            token_indices=token_indices,
                        )
                    },
                    names_filter=set(),
                )
                patched_score = binary_logit_score(patched_logits, clean_labels).mean()
                recovery = (patched_score - corrupt_score.mean()) / denom
                entry = {
                    "heads": list(subset),
                    "patched_score": float(patched_score.item()),
                    "recovery": float(recovery.item()),
                }
                set_results[set_name] = entry
                flat.append(
                    {
                        "layer": layer,
                        "head_set": key,
                        "position_set": set_name,
                        **entry,
                    }
                )
            subsets[key] = set_results
        layers[str(layer)] = subsets

    flat.sort(key=lambda item: item["recovery"], reverse=True)
    return {
        "num_bidirectional_examples": int(clean_tokens.shape[0]),
        "num_matched_pairs": int(pairs["matched_pairs"]),
        "matched_first": int(pairs["matched_first"]),
        "matched_last": int(pairs["matched_last"]),
        "activation_token_indices": {"cls": 0, "leading": first + 1, "trailing": last + 1},
        "mean_clean_score": float(clean_score.mean().item()),
        "mean_corrupt_score": float(corrupt_score.mean().item()),
        "layers": layers,
        "top_recoveries": flat[:24],
    }


@torch.no_grad()
def matched_attention_path_patch_summary(model, pairs: dict) -> dict:
    """Patch internals of attention heads to locate the causal subpath.

    On matched-support pairs, first/last support positions are shared across
    clean and corrupt examples. That lets us patch layer/head value vectors at
    the semantic source positions and compare them against patching the
    attention pattern row, z at CLS, and projected head output at CLS.
    """
    clean_tokens = pairs["clean_tokens"]
    corrupt_tokens = pairs["corrupt_tokens"]
    clean_labels = pairs["clean_labels"]
    clean_logits, clean_cache = run_with_cache(model, clean_tokens)
    corrupt_logits, _ = run_with_cache(model, corrupt_tokens)
    clean_score = binary_logit_score(clean_logits, clean_labels)
    corrupt_score = binary_logit_score(corrupt_logits, clean_labels)
    denom = (clean_score - corrupt_score).mean().clamp_min(1e-6)

    feats = support_features(clean_tokens)
    first = int(feats["first"].mode().values.item())
    last = int(feats["last"].mode().values.item())
    source_sets = {
        "all_value_sources": None,
        "leading_value_source": [first + 1],
        "trailing_value_source": [last + 1],
        "both_boundary_value_sources": [first + 1, last + 1],
    }
    destination_sets = {
        "all_destinations": None,
        "cls_dest": [0],
        "leading_dest": [first + 1],
        "trailing_dest": [last + 1],
        "both_boundary_dests": [first + 1, last + 1],
        "cls_plus_boundary_dests": [0, first + 1, last + 1],
    }

    def score_for_hooks(hooks: dict) -> dict:
        patched_logits, _ = run_with_cache(model, corrupt_tokens, hooks=hooks, names_filter=set())
        patched_score = binary_logit_score(patched_logits, clean_labels).mean()
        recovery = (patched_score - corrupt_score.mean()) / denom
        return {
            "patched_score": float(patched_score.item()),
            "recovery": float(recovery.item()),
        }

    layers = {}
    flat = []
    for layer in range(len(model.blocks)):
        layer_results = {}
        for head in range(model.blocks[layer].attn.num_heads):
            prefix = f"blocks.{layer}"
            head_results = {}

            for patch_name, token_indices in source_sets.items():
                hook_name = f"{prefix}.hook_v"
                entry = score_for_hooks(
                    {
                        hook_name: patch_head_positions_hook_from_cache(
                            clean_cache,
                            hook_name,
                            head_idx=head,
                            token_indices=token_indices,
                        )
                    }
                )
                head_results[f"v_{patch_name}"] = entry

            pattern_name = f"{prefix}.hook_pattern"
            for dest_name, token_indices in {
                "all_rows": None,
                "cls_row": [0],
            }.items():
                pattern_entry = score_for_hooks(
                    {
                        pattern_name: patch_head_positions_hook_from_cache(
                            clean_cache,
                            pattern_name,
                            head_idx=head,
                            token_indices=token_indices,
                        )
                    }
                )
                head_results[f"pattern_{dest_name}"] = pattern_entry

            for patch_name, token_indices in source_sets.items():
                v_name = f"{prefix}.hook_v"
                pattern_name = f"{prefix}.hook_pattern"
                entry = score_for_hooks(
                    {
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
                            token_indices=token_indices,
                        ),
                    }
                )
                head_results[f"pattern_cls_row_plus_v_{patch_name}"] = entry

            for hook_suffix in ["z", "attn_head_out"]:
                hook_name = f"{prefix}.hook_{hook_suffix}"
                for dest_name, token_indices in destination_sets.items():
                    entry = score_for_hooks(
                        {
                            hook_name: patch_head_positions_hook_from_cache(
                                clean_cache,
                                hook_name,
                                head_idx=head,
                                token_indices=token_indices,
                            )
                        }
                    )
                    head_results[f"{hook_suffix}_{dest_name}"] = entry

            layer_results[str(head)] = head_results
            for patch_name, entry in head_results.items():
                flat.append({"layer": layer, "head": head, "patch": patch_name, **entry})
        layers[str(layer)] = layer_results

    flat.sort(key=lambda item: item["recovery"], reverse=True)
    return {
        "num_bidirectional_examples": int(clean_tokens.shape[0]),
        "num_matched_pairs": int(pairs["matched_pairs"]),
        "matched_first": int(pairs["matched_first"]),
        "matched_last": int(pairs["matched_last"]),
        "activation_token_indices": {"cls": 0, "leading": first + 1, "trailing": last + 1},
        "mean_clean_score": float(clean_score.mean().item()),
        "mean_corrupt_score": float(corrupt_score.mean().item()),
        "layers": layers,
        "top_recoveries": flat[:24],
    }


@torch.no_grad()
def matched_head_destination_sweep_summary(model, pairs: dict) -> dict:
    """Patch one destination at a time for candidate attention heads."""
    clean_tokens = pairs["clean_tokens"]
    corrupt_tokens = pairs["corrupt_tokens"]
    clean_labels = pairs["clean_labels"]
    clean_logits, clean_cache = run_with_cache(model, clean_tokens)
    corrupt_logits, _ = run_with_cache(model, corrupt_tokens)
    clean_score = binary_logit_score(clean_logits, clean_labels)
    corrupt_score = binary_logit_score(corrupt_logits, clean_labels)
    denom = (clean_score - corrupt_score).mean().clamp_min(1e-6)

    feats = support_features(clean_tokens)
    first = int(feats["first"].mode().values.item())
    last = int(feats["last"].mode().values.item())
    seq_len = clean_tokens.shape[1] + 1
    candidates = [(0, 2), (1, 0), (1, 1), (1, 2)]
    results = {}
    for layer, head in candidates:
        if layer >= len(model.blocks) or head >= model.blocks[layer].attn.num_heads:
            continue
        hook_name = f"blocks.{layer}.hook_attn_head_out"
        all_dest_logits, _ = run_with_cache(
            model,
            corrupt_tokens,
            hooks={
                hook_name: patch_head_positions_hook_from_cache(
                    clean_cache,
                    hook_name,
                    head_idx=head,
                    token_indices=None,
                )
            },
            names_filter=set(),
        )
        all_dest_score = binary_logit_score(all_dest_logits, clean_labels).mean()
        all_dest_recovery = (all_dest_score - corrupt_score.mean()) / denom
        recovery = torch.empty(seq_len, device=clean_tokens.device)
        patched_score = torch.empty(seq_len, device=clean_tokens.device)
        for token_idx in range(seq_len):
            logits, _ = run_with_cache(
                model,
                corrupt_tokens,
                hooks={
                    hook_name: patch_head_positions_hook_from_cache(
                        clean_cache,
                        hook_name,
                        head_idx=head,
                        token_indices=[token_idx],
                    )
                },
                names_filter=set(),
            )
            score = binary_logit_score(logits, clean_labels).mean()
            patched_score[token_idx] = score
            recovery[token_idx] = (score - corrupt_score.mean()) / denom

        top_values, top_idx = torch.topk(recovery, k=min(16, seq_len))
        key = f"blocks.{layer}.head.{head}"
        results[key] = {
            "layer": layer,
            "head": head,
            "top_token_indices": top_idx.cpu().tolist(),
            "top_degrees": [int(i) - 1 if int(i) > 0 else "CLS" for i in top_idx.cpu().tolist()],
            "top_recovery": top_values.cpu().tolist(),
            "top_patched_score": patched_score[top_idx].cpu().tolist(),
            "recovery_at_cls": float(recovery[0].item()),
            "recovery_at_leading": float(recovery[first + 1].item()),
            "recovery_at_trailing": float(recovery[last + 1].item()),
            "recovery_at_both_boundary_sum": float((recovery[first + 1] + recovery[last + 1]).item()),
            "recovery_all_destinations_reference": float(all_dest_recovery.item()),
        }

    return {
        "num_bidirectional_examples": int(clean_tokens.shape[0]),
        "num_matched_pairs": int(pairs["matched_pairs"]),
        "matched_first": int(pairs["matched_first"]),
        "matched_last": int(pairs["matched_last"]),
        "activation_token_indices": {"cls": 0, "leading": first + 1, "trailing": last + 1},
        "mean_clean_score": float(clean_score.mean().item()),
        "mean_corrupt_score": float(corrupt_score.mean().item()),
        "candidate_heads": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run head-level circuit analysis for the B_3 transformer.")
    parser.add_argument("--data-dir", default="interp/data/generated/b3_l25_p2_full")
    parser.add_argument("--checkpoint", default="interp/artifacts/b3_l25_p2_xfmr2_abs/best_model.pt")
    parser.add_argument("--out", default="interp/artifacts/b3_l25_p2_head_circuit/results.json")
    parser.add_argument("--length", type=int, default=25)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--examples", type=int, default=8192)
    parser.add_argument("--max-scan-examples", type=int, default=1_048_576)
    parser.add_argument("--matched-pairs", type=int, default=256)
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
        length=args.length,
        batch_size=args.batch_size,
        seed=args.seed,
        epoch=0,
        shuffle=False,
        max_examples=args.examples,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    batch = collect_examples(eval_loader, device, args.examples)
    with torch.no_grad():
        direct_logits = model(batch["tokens"])
        cached_logits, _ = run_with_cache(model, batch["tokens"])

    matched_loader = make_loader(
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
    pairs = collect_matched_pairs(matched_loader, device, args.max_scan_examples, args.matched_pairs)

    result = {
        "data_dir": args.data_dir,
        "checkpoint": args.checkpoint,
        "examples": int(batch["label"].numel()),
        "cache_forward_max_abs_diff": float((direct_logits - cached_logits).abs().max().item()),
        "attention_boundary_summary": attention_boundary_summary(model, batch["tokens"], batch["label"]),
        "head_ablation_summary": head_ablation_summary(model, batch["tokens"], batch["label"]),
        "head_set_ablation_summary": head_set_ablation_summary(model, batch["tokens"], batch["label"]),
        "matched_head_patching": matched_head_patch_summary(model, pairs),
        "matched_head_set_patching": matched_head_set_patch_summary(model, pairs),
        "matched_attention_path_patching": matched_attention_path_patch_summary(model, pairs),
        "matched_head_destination_sweep": matched_head_destination_sweep_summary(model, pairs),
    }
    atomic_json_dump(result, Path(args.out))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
