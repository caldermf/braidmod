#!/usr/bin/env python3
"""Boundary-token flip counterfactuals for B_3 models."""

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
from interp.b3_interp import flip_boundary_columns, load_transformer_checkpoint  # noqa: E402
from interp.b3_mlp import B3AbsoluteMLP, B3MLPConfig  # noqa: E402
from interp.train_b3_transformer import atomic_json_dump, make_loader, resolve_device, set_seed  # noqa: E402


def load_mlp_checkpoint(path: str | Path, device: torch.device | str) -> B3AbsoluteMLP:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = checkpoint["model_config"]
    model = B3AbsoluteMLP(
        B3MLPConfig(
            length=int(cfg["length"]),
            hidden_dim=int(cfg["hidden_dim"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    labels = labels.to(device=logits.device, dtype=torch.float32)
    preds = (logits >= 0).to(torch.float32)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
    return {
        "loss": float(loss.item()),
        "accuracy": float((preds == labels).float().mean().item()),
        "positive_rate": float(preds.mean().item()),
    }


@torch.no_grad()
def evaluate_model(model: nn.Module, loader, device: torch.device, max_examples: int) -> dict:
    totals: dict[str, dict[str, float]] = {}
    seen = 0

    def update(name: str, logits: torch.Tensor, labels: torch.Tensor, original_preds: torch.Tensor | None = None) -> None:
        m = metrics(logits, labels)
        preds = (logits >= 0).to(torch.float32)
        slot = totals.setdefault(name, {"loss_sum": 0.0, "correct": 0.0, "n": 0, "positive": 0.0, "changed": 0.0})
        n = int(labels.numel())
        slot["loss_sum"] += m["loss"] * n
        slot["correct"] += m["accuracy"] * n
        slot["positive"] += m["positive_rate"] * n
        if original_preds is not None:
            slot["changed"] += float((preds != original_preds).float().sum().item())
        slot["n"] += n

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
        flipped_labels = 1.0 - labels

        original_logits = model(tokens)
        original_preds = (original_logits >= 0).to(torch.float32)
        update("original_vs_true", original_logits, labels)

        variants = {
            "flip_leading_vs_flipped_label": flip_boundary_columns(tokens, leading=True, trailing=False),
            "flip_trailing_vs_flipped_label": flip_boundary_columns(tokens, leading=False, trailing=True),
            "flip_both_vs_flipped_label": flip_boundary_columns(tokens, leading=True, trailing=True),
        }
        for name, variant_tokens in variants.items():
            update(name, model(variant_tokens), flipped_labels, original_preds=original_preds)

        if max_examples > 0 and seen >= max_examples:
            break

    return {
        name: {
            "loss": slot["loss_sum"] / max(1, slot["n"]),
            "accuracy": slot["correct"] / max(1, slot["n"]),
            "positive_rate": slot["positive"] / max(1, slot["n"]),
            "prediction_change_rate": slot["changed"] / max(1, slot["n"]),
            "n": int(slot["n"]),
        }
        for name, slot in sorted(totals.items())
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run B_3 boundary-token flip counterfactuals.")
    parser.add_argument("--data-dir", default="interp/data/generated/b3_l25_p2_full")
    parser.add_argument("--transformer-checkpoint", default="interp/artifacts/b3_l25_p2_xfmr2_abs/best_model.pt")
    parser.add_argument("--mlp-checkpoint", default="interp/artifacts/b3_l25_p2_mlp1_abs_h128/best_model.pt")
    parser.add_argument("--out", default="interp/artifacts/b3_l25_p2_boundary_counterfactuals/results.json")
    parser.add_argument("--length", type=int, default=25)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--examples", type=int, default=1_048_576)
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
        shuffle=False,
        max_examples=args.examples,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    transformer = load_transformer_checkpoint(args.transformer_checkpoint, device=device).model
    mlp = load_mlp_checkpoint(args.mlp_checkpoint, device=device)
    result = {
        "data_dir": args.data_dir,
        "examples": args.examples,
        "label_convention": {"0": "{s_1}", "1": "{s_2}"},
        "counterfactual": "Flip the column of the unique nonzero boundary coefficient token; evaluate against flipped label.",
        "transformer": evaluate_model(transformer, loader, device, args.examples),
    }

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
    result["mlp"] = evaluate_model(mlp, loader, device, args.examples)
    atomic_json_dump(result, Path(args.out))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
