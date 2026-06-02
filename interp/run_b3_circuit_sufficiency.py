#!/usr/bin/env python3
"""Evaluate candidate B_3 transformer subcircuits by targeted ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.b3_data import discover_shards  # noqa: E402
from interp.b3_interp import (  # noqa: E402
    HookFn,
    binary_logit_score,
    load_transformer_checkpoint,
    run_with_cache,
    zero_head_set_positions_hook,
)
from interp.train_b3_transformer import atomic_json_dump, make_loader, resolve_device, set_seed  # noqa: E402


def zero_hook(value: torch.Tensor, _: str) -> torch.Tensor:
    return torch.zeros_like(value)


def keep_heads_hook(keep: list[int]) -> HookFn:
    def hook(value: torch.Tensor, _: str) -> torch.Tensor:
        out = torch.zeros_like(value)
        out[:, keep] = value[:, keep]
        return out

    return hook


def metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    labels = labels.to(device=logits.device, dtype=torch.float32)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
    preds = (logits >= 0).to(torch.float32)
    return {
        "loss_sum": float(loss.item()) * int(labels.numel()),
        "correct": int((preds == labels).sum().item()),
        "n": int(labels.numel()),
        "score_sum": float(binary_logit_score(logits, labels).sum().item()),
        "positive_sum": int(preds.sum().item()),
    }


def finish(slot: dict) -> dict:
    n = max(1, int(slot["n"]))
    return {
        "loss": slot["loss_sum"] / n,
        "accuracy": slot["correct"] / n,
        "mean_logit_score": slot["score_sum"] / n,
        "positive_rate": slot["positive_sum"] / n,
        "n": n,
    }


def add_slot(total: dict, name: str, batch_metrics: dict) -> None:
    slot = total.setdefault(name, {"loss_sum": 0.0, "correct": 0, "n": 0, "score_sum": 0.0, "positive_sum": 0})
    for key, value in batch_metrics.items():
        slot[key] += value


def variant_hooks() -> dict[str, dict[str, HookFn]]:
    return {
        "full": {},
        "zero_l0_attn": {"blocks.0.hook_attn_head_out": zero_hook},
        "zero_l1_attn": {"blocks.1.hook_attn_head_out": zero_hook},
        "zero_l0_mlp": {"blocks.0.hook_mlp_out": zero_hook},
        "zero_l1_mlp": {"blocks.1.hook_mlp_out": zero_hook},
        "keep_l0h2_l1h0": {
            "blocks.0.hook_attn_head_out": keep_heads_hook([2]),
            "blocks.1.hook_attn_head_out": keep_heads_hook([0]),
        },
        "keep_l0h2_l1h01": {
            "blocks.0.hook_attn_head_out": keep_heads_hook([2]),
            "blocks.1.hook_attn_head_out": keep_heads_hook([0, 1]),
        },
        "keep_l0h2_l1h012": {
            "blocks.0.hook_attn_head_out": keep_heads_hook([2]),
            "blocks.1.hook_attn_head_out": keep_heads_hook([0, 1, 2]),
        },
        "keep_l1h012_zero_mlps": {
            "blocks.1.hook_attn_head_out": keep_heads_hook([0, 1, 2]),
            "blocks.0.hook_mlp_out": zero_hook,
            "blocks.1.hook_mlp_out": zero_hook,
        },
        "keep_l0h2_l1h012_zero_mlps": {
            "blocks.0.hook_attn_head_out": keep_heads_hook([2]),
            "blocks.1.hook_attn_head_out": keep_heads_hook([0, 1, 2]),
            "blocks.0.hook_mlp_out": zero_hook,
            "blocks.1.hook_mlp_out": zero_hook,
        },
        "keep_l1h0_only": {
            "blocks.0.hook_attn_head_out": zero_hook,
            "blocks.1.hook_attn_head_out": keep_heads_hook([0]),
        },
        "keep_l1h01_only": {
            "blocks.0.hook_attn_head_out": zero_hook,
            "blocks.1.hook_attn_head_out": keep_heads_hook([0, 1]),
        },
        "drop_l0h2": {
            "blocks.0.hook_attn_head_out": zero_head_set_positions_hook(head_indices=[2], token_indices=None),
        },
        "drop_l1h0": {
            "blocks.1.hook_attn_head_out": zero_head_set_positions_hook(head_indices=[0], token_indices=None),
        },
        "drop_l1h01": {
            "blocks.1.hook_attn_head_out": zero_head_set_positions_hook(head_indices=[0, 1], token_indices=None),
        },
        "drop_l1h012": {
            "blocks.1.hook_attn_head_out": zero_head_set_positions_hook(head_indices=[0, 1, 2], token_indices=None),
        },
    }


@torch.no_grad()
def evaluate_variants(model, loader, device: torch.device, max_examples: int) -> dict:
    variants = variant_hooks()
    totals: dict[str, dict] = {}
    seen = 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        labels = batch["label"].to(device)
        if max_examples > 0 and seen + labels.numel() > max_examples:
            take = max_examples - seen
            if take <= 0:
                break
            tokens = tokens[:take]
            labels = labels[:take]
        seen += int(labels.numel())

        direct_logits = model(tokens)
        cached_logits, _ = run_with_cache(model, tokens, names_filter=set())
        add_slot(totals, "full_direct", metrics(direct_logits, labels))
        add_slot(totals, "full_cached", metrics(cached_logits, labels))
        for name, hooks in variants.items():
            if name == "full":
                continue
            logits, _ = run_with_cache(model, tokens, hooks=hooks, names_filter=set())
            add_slot(totals, name, metrics(logits, labels))

        if max_examples > 0 and seen >= max_examples:
            break

    return {name: finish(slot) for name, slot in sorted(totals.items())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate candidate B_3 transformer subcircuits.")
    parser.add_argument("--data-dir", default="interp/data/generated/b3_l25_p2_full")
    parser.add_argument("--checkpoint", default="interp/artifacts/b3_l25_p2_xfmr2_abs/best_model.pt")
    parser.add_argument("--out", default="interp/artifacts/b3_l25_p2_circuit_sufficiency/results.json")
    parser.add_argument("--length", type=int, default=25)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--examples", type=int, default=131072)
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
        length=args.length,
        batch_size=args.batch_size,
        seed=args.seed,
        epoch=0,
        shuffle=False,
        max_examples=args.examples,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    results = {
        "data_dir": args.data_dir,
        "checkpoint": args.checkpoint,
        "examples": args.examples,
        "variants": evaluate_variants(model, loader, device=device, max_examples=args.examples),
    }
    atomic_json_dump(results, Path(args.out))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
