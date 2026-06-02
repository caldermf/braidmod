#!/usr/bin/env python3
"""Compose late-CLS unit-token probes with the exact B_3 boundary rule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.b3_data import discover_shards  # noqa: E402
from interp.b3_interp import load_transformer_checkpoint, run_with_cache, support_features  # noqa: E402
from interp.run_b3_semantic_probes import (  # noqa: E402
    UNIT_TOKEN_TO_CLASS,
    add_bias,
    collect_examples_from_all_shards,
    ridge_solve,
)
from interp.train_b3_transformer import atomic_json_dump, make_loader, resolve_device, set_seed  # noqa: E402


CLASS_TO_UNIT_TOKEN = torch.tensor([1, 2, 4, 8], dtype=torch.long)


def l1_resid_post_cls(model, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits, cache = run_with_cache(model, tokens, names_filter={"blocks.1.hook_resid_post"})
    return logits, cache["blocks.1.hook_resid_post"][:, 0]


def unit_targets(tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    feats = support_features(tokens)
    unit_map = UNIT_TOKEN_TO_CLASS.to(tokens.device)
    return {
        "leading": unit_map[feats["leading_token"].to(torch.long)],
        "trailing": unit_map[feats["trailing_token"].to(torch.long)],
    }


def label_from_leading_unit_class(unit_class: torch.Tensor) -> torch.Tensor:
    token = CLASS_TO_UNIT_TOKEN.to(unit_class.device)[unit_class.to(torch.long)]
    return ((token == 1) | (token == 4)).to(torch.long)


def label_from_trailing_unit_class(unit_class: torch.Tensor) -> torch.Tensor:
    token = CLASS_TO_UNIT_TOKEN.to(unit_class.device)[unit_class.to(torch.long)]
    return ((token == 2) | (token == 8)).to(torch.long)


def fit_multiclass_probe(x: torch.Tensor, y: torch.Tensor, ridge: float) -> dict[str, torch.Tensor]:
    x = x.to(torch.float32)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-4)
    x_norm = (x - mean) / std
    num_classes = int(y.max().item()) + 1
    y_onehot = torch.nn.functional.one_hot(y.to(torch.long), num_classes=num_classes).to(torch.float32)
    weights = ridge_solve(x_norm, y_onehot, ridge=ridge)
    return {"mean": mean, "std": std, "weights": weights}


def probe_scores(probe: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    x_norm = (x.to(torch.float32) - probe["mean"]) / probe["std"]
    return add_bias(x_norm) @ probe["weights"]


def multiclass_preds(probe: dict[str, torch.Tensor], x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scores = probe_scores(probe, x)
    pred = scores.argmax(dim=1)
    top2 = scores.topk(k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    return pred, margin


def slot() -> dict:
    return {
        "n": 0,
        "model_correct": 0,
        "actual_leading_rule_correct": 0,
        "actual_leading_rule_model_agree": 0,
        "actual_trailing_rule_correct": 0,
        "actual_trailing_rule_model_agree": 0,
        "leading_unit_correct": 0,
        "leading_rule_correct": 0,
        "leading_rule_model_agree": 0,
        "trailing_unit_correct": 0,
        "trailing_rule_correct": 0,
        "trailing_rule_model_agree": 0,
        "combined_rule_correct": 0,
        "combined_rule_model_agree": 0,
        "probe_rule_disagree": 0,
    }


def update_counts(
    total: dict,
    *,
    labels: torch.Tensor,
    model_pred: torch.Tensor,
    leading_target: torch.Tensor,
    trailing_target: torch.Tensor,
    leading_pred: torch.Tensor,
    leading_margin: torch.Tensor,
    trailing_pred: torch.Tensor,
    trailing_margin: torch.Tensor,
) -> None:
    labels = labels.to(torch.long)
    model_pred = model_pred.to(torch.long)
    actual_leading_label = label_from_leading_unit_class(leading_target)
    actual_trailing_label = label_from_trailing_unit_class(trailing_target)
    leading_label = label_from_leading_unit_class(leading_pred)
    trailing_label = label_from_trailing_unit_class(trailing_pred)
    disagree = leading_label != trailing_label
    combined_label = torch.where(leading_margin >= trailing_margin, leading_label, trailing_label)
    combined_label = torch.where(disagree, combined_label, leading_label)

    total["n"] += int(labels.numel())
    total["model_correct"] += int((model_pred == labels).sum().item())
    total["actual_leading_rule_correct"] += int((actual_leading_label == labels).sum().item())
    total["actual_leading_rule_model_agree"] += int((actual_leading_label == model_pred).sum().item())
    total["actual_trailing_rule_correct"] += int((actual_trailing_label == labels).sum().item())
    total["actual_trailing_rule_model_agree"] += int((actual_trailing_label == model_pred).sum().item())
    total["leading_unit_correct"] += int((leading_pred == leading_target).sum().item())
    total["leading_rule_correct"] += int((leading_label == labels).sum().item())
    total["leading_rule_model_agree"] += int((leading_label == model_pred).sum().item())
    total["trailing_unit_correct"] += int((trailing_pred == trailing_target).sum().item())
    total["trailing_rule_correct"] += int((trailing_label == labels).sum().item())
    total["trailing_rule_model_agree"] += int((trailing_label == model_pred).sum().item())
    total["combined_rule_correct"] += int((combined_label == labels).sum().item())
    total["combined_rule_model_agree"] += int((combined_label == model_pred).sum().item())
    total["probe_rule_disagree"] += int(disagree.sum().item())


def ratio(total: dict, key: str) -> float:
    return float(total[key] / max(1, total["n"]))


def finish(total: dict) -> dict:
    return {
        "eval_examples": int(total["n"]),
        "model": {
            "accuracy_vs_true": ratio(total, "model_correct"),
        },
        "actual_boundary_rules": {
            "leading_unit_column": {
                "accuracy_vs_true": ratio(total, "actual_leading_rule_correct"),
                "agreement_with_model_predictions": ratio(total, "actual_leading_rule_model_agree"),
            },
            "trailing_unit_column": {
                "accuracy_vs_true": ratio(total, "actual_trailing_rule_correct"),
                "agreement_with_model_predictions": ratio(total, "actual_trailing_rule_model_agree"),
            },
        },
        "probe_then_rule": {
            "leading_unit_decoder": {
                "unit_token_accuracy": ratio(total, "leading_unit_correct"),
                "rule_label_accuracy_vs_true": ratio(total, "leading_rule_correct"),
                "rule_label_agreement_with_model_predictions": ratio(total, "leading_rule_model_agree"),
            },
            "trailing_unit_decoder": {
                "unit_token_accuracy": ratio(total, "trailing_unit_correct"),
                "rule_label_accuracy_vs_true": ratio(total, "trailing_rule_correct"),
                "rule_label_agreement_with_model_predictions": ratio(total, "trailing_rule_model_agree"),
            },
            "margin_combined": {
                "rule_label_accuracy_vs_true": ratio(total, "combined_rule_correct"),
                "rule_label_agreement_with_model_predictions": ratio(total, "combined_rule_model_agree"),
                "leading_trailing_rule_disagreement_rate": ratio(total, "probe_rule_disagree"),
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a probe-derived algebraic B_3 circuit classifier.")
    parser.add_argument("--data-dir", default="interp/data/generated/b3_l25_p2_full")
    parser.add_argument("--checkpoint", default="interp/artifacts/b3_l25_p2_xfmr2_abs/best_model.pt")
    parser.add_argument("--out", default="interp/artifacts/b3_l25_p2_circuit_classifier/results.json")
    parser.add_argument("--length", type=int, default=25)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--train-examples", type=int, default=8192)
    parser.add_argument("--eval-examples", type=int, default=131072)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    shard_paths = discover_shards(args.data_dir, num_shards=args.num_shards)
    model = load_transformer_checkpoint(args.checkpoint, device=device).model

    train_batch = collect_examples_from_all_shards(
        shard_paths,
        split="train",
        length=args.length,
        total_examples=args.train_examples,
        seed=args.seed,
        device=device,
    )
    _, train_x = l1_resid_post_cls(model, train_batch["tokens"])
    train_targets = unit_targets(train_batch["tokens"])
    leading_probe = fit_multiclass_probe(train_x, train_targets["leading"], ridge=args.ridge)
    trailing_probe = fit_multiclass_probe(train_x, train_targets["trailing"], ridge=args.ridge)

    loader = make_loader(
        shard_paths,
        split="test",
        length=args.length,
        batch_size=args.batch_size,
        seed=args.seed,
        epoch=0,
        shuffle=False,
        max_examples=args.eval_examples,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    counts = slot()
    for batch in loader:
        tokens = batch["tokens"].to(device)
        labels = batch["label"].to(device)
        logits, x = l1_resid_post_cls(model, tokens)
        targets = unit_targets(tokens)
        leading_pred, leading_margin = multiclass_preds(leading_probe, x)
        trailing_pred, trailing_margin = multiclass_preds(trailing_probe, x)
        update_counts(
            counts,
            labels=labels,
            model_pred=(logits >= 0).to(torch.long),
            leading_target=targets["leading"],
            trailing_target=targets["trailing"],
            leading_pred=leading_pred,
            leading_margin=leading_margin,
            trailing_pred=trailing_pred,
            trailing_margin=trailing_margin,
        )

    results = {
        "data_dir": args.data_dir,
        "checkpoint": args.checkpoint,
        "representation": "blocks.1.hook_resid_post[:, 0] (late CLS)",
        "probe_type": "ridge multiclass unit-token decoder, then exact unit-column rule",
        "train_examples": args.train_examples,
        "ridge": args.ridge,
        "label_convention": {"0": "{s_1}", "1": "{s_2}"},
        "unit_token_order": [1, 2, 4, 8],
        **finish(counts),
    }
    atomic_json_dump(results, Path(args.out))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
