#!/usr/bin/env python3
"""Linear semantic probes for B_3 transformer activations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.b3_data import discover_shards, load_shard, matrix_bits_to_absolute_tokens, split_mask  # noqa: E402
from interp.b3_interp import load_transformer_checkpoint, run_with_cache, support_features  # noqa: E402
from interp.train_b3_transformer import atomic_json_dump, resolve_device, set_seed  # noqa: E402


UNIT_TOKEN_TO_CLASS = torch.full((16,), -1, dtype=torch.long)
for class_id, token in enumerate([1, 2, 4, 8]):
    UNIT_TOKEN_TO_CLASS[token] = class_id


@torch.no_grad()
def collect_examples_from_all_shards(
    shard_paths: list[Path],
    *,
    split: str,
    length: int,
    total_examples: int,
    seed: int,
    device: torch.device,
) -> dict:
    """Sample examples across all shards, avoiding first-factor/shard bias."""
    generator = torch.Generator().manual_seed(seed)
    per_shard = max(1, (total_examples + len(shard_paths) - 1) // len(shard_paths))
    pieces: dict[str, list[torch.Tensor]] = {
        "tokens": [],
        "label": [],
        "min_degree": [],
        "final_factor_id": [],
        "sample_id": [],
    }
    for path in shard_paths:
        payload = load_shard(path)
        meta = payload["metadata"]
        count = int(meta["sample_id_count"])
        start = int(meta["sample_id_start"])
        sample_ids = torch.arange(start, start + count, dtype=torch.long)
        rows = torch.nonzero(split_mask(sample_ids, split), as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        if rows.numel() > per_shard:
            rows = rows[torch.randperm(rows.numel(), generator=generator)[:per_shard]]
        matrix_bits = payload["matrix_bits"][rows]
        min_degree = payload["burau_min_degree"][rows]
        pieces["tokens"].append(matrix_bits_to_absolute_tokens(matrix_bits, min_degree, length=length))
        pieces["label"].append(payload["label"][rows].to(torch.float32))
        pieces["min_degree"].append(min_degree.to(torch.long))
        pieces["final_factor_id"].append(payload["final_factor_id"][rows].to(torch.long))
        pieces["sample_id"].append(sample_ids[rows])
        if sum(int(x.numel()) for x in pieces["label"]) >= total_examples:
            break

    out = {key: torch.cat(values, dim=0)[:total_examples] for key, values in pieces.items()}
    if out["label"].numel() < total_examples:
        raise RuntimeError(f"Only collected {out['label'].numel()} examples for split={split}")
    out["tokens"] = out["tokens"].to(device)
    out["label"] = out["label"].to(device)
    return out


def boundary_targets(tokens: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    feats = support_features(tokens)
    leading = feats["leading_token"].to(torch.long)
    trailing = feats["trailing_token"].to(torch.long)
    unit_map = UNIT_TOKEN_TO_CLASS.to(tokens.device)
    return {
        "label": labels.to(torch.long),
        "leading_col0": ((leading == 1) | (leading == 4)).to(torch.long),
        "trailing_col1": ((trailing == 2) | (trailing == 8)).to(torch.long),
        "leading_row1": ((leading == 4) | (leading == 8)).to(torch.long),
        "trailing_row1": ((trailing == 4) | (trailing == 8)).to(torch.long),
        "leading_unit_token": unit_map[leading],
        "trailing_unit_token": unit_map[trailing],
    }


@torch.no_grad()
def activation_representations(model, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    names_filter = {
        "hook_resid_embed",
        "blocks.0.hook_resid_post",
        "blocks.1.hook_resid_post",
        "hook_final_hidden",
        "blocks.0.hook_attn_head_out",
        "blocks.1.hook_z",
        "blocks.1.hook_attn_head_out",
    }
    _, cache = run_with_cache(model, tokens, names_filter=names_filter)
    feats = support_features(tokens)
    rows = torch.arange(tokens.shape[0], device=tokens.device)
    leading_pos = feats["first"] + 1
    trailing_pos = feats["last"] + 1
    support_mask = torch.cat([torch.zeros(tokens.shape[0], 1, dtype=torch.bool, device=tokens.device), feats["support"]], dim=1)
    support_weight = support_mask.to(torch.float32)
    support_weight = support_weight / support_weight.sum(dim=1, keepdim=True).clamp_min(1.0)

    reps = {
        "embed_cls": cache["hook_resid_embed"][:, 0],
        "l0_resid_post_cls": cache["blocks.0.hook_resid_post"][:, 0],
        "l0_resid_post_leading": cache["blocks.0.hook_resid_post"][rows, leading_pos],
        "l0_resid_post_trailing": cache["blocks.0.hook_resid_post"][rows, trailing_pos],
        "l1_resid_post_cls": cache["blocks.1.hook_resid_post"][:, 0],
        "final_hidden_cls": cache["hook_final_hidden"][:, 0],
        "l0h2_headout_cls": cache["blocks.0.hook_attn_head_out"][:, 2, 0],
        "l0h2_headout_support_mean": (cache["blocks.0.hook_attn_head_out"][:, 2] * support_weight.unsqueeze(-1)).sum(dim=1),
        "l1h0_z_cls": cache["blocks.1.hook_z"][:, 0, 0],
        "l1h0_headout_cls": cache["blocks.1.hook_attn_head_out"][:, 0, 0],
        "l1h1_headout_cls": cache["blocks.1.hook_attn_head_out"][:, 1, 0],
        "l1h2_headout_cls": cache["blocks.1.hook_attn_head_out"][:, 2, 0],
    }
    return reps


def normalize(train_x: torch.Tensor, eval_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-4)
    return (train_x - mean) / std, (eval_x - mean) / std


def add_bias(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)], dim=1)


def ridge_solve(x: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    x = add_bias(x.to(torch.float32))
    y = y.to(torch.float32)
    gram = x.T @ x
    reg = ridge * torch.eye(gram.shape[0], device=x.device, dtype=x.dtype)
    reg[-1, -1] = 0.0
    return torch.linalg.solve(gram + reg, x.T @ y)


def train_binary_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    ridge: float,
) -> dict:
    train_x, eval_x = normalize(train_x, eval_x)
    train_signed = train_y.to(torch.float32).mul(2).sub(1)
    weights = ridge_solve(train_x, train_signed, ridge=ridge)
    with torch.no_grad():
        train_score = add_bias(train_x.to(torch.float32)) @ weights
        eval_score = add_bias(eval_x.to(torch.float32)) @ weights
        train_pred = (train_score >= 0).to(torch.long)
        eval_pred = (eval_score >= 0).to(torch.long)
        train_acc = (train_pred == train_y).float().mean()
        eval_acc = (eval_pred == eval_y).float().mean()
        eval_signed = eval_y.to(torch.float32).mul(2).sub(1)
        eval_mse = (eval_score - eval_signed).pow(2).mean()
    return {
        "train_accuracy": float(train_acc.item()),
        "eval_accuracy": float(eval_acc.item()),
        "eval_mse": float(eval_mse.item()),
    }


def train_multiclass_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    ridge: float,
) -> dict:
    train_x, eval_x = normalize(train_x, eval_x)
    good_train = train_y >= 0
    good_eval = eval_y >= 0
    train_x = train_x[good_train]
    train_y = train_y[good_train]
    eval_x = eval_x[good_eval]
    eval_y = eval_y[good_eval]
    num_classes = int(max(train_y.max().item(), eval_y.max().item()) + 1)
    train_onehot = torch.nn.functional.one_hot(train_y, num_classes=num_classes).to(torch.float32)
    weights = ridge_solve(train_x, train_onehot, ridge=ridge)
    with torch.no_grad():
        train_score = add_bias(train_x.to(torch.float32)) @ weights
        eval_score = add_bias(eval_x.to(torch.float32)) @ weights
        train_pred = train_score.argmax(dim=1)
        eval_pred = eval_score.argmax(dim=1)
        eval_onehot = torch.nn.functional.one_hot(eval_y, num_classes=num_classes).to(torch.float32)
        eval_mse = (eval_score - eval_onehot).pow(2).mean()
    return {
        "train_accuracy": float((train_pred == train_y).float().mean().item()),
        "eval_accuracy": float((eval_pred == eval_y).float().mean().item()),
        "eval_mse": float(eval_mse.item()),
        "num_classes": num_classes,
        "eval_n": int(eval_y.numel()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear semantic probes on B_3 transformer activations.")
    parser.add_argument("--data-dir", default="interp/data/generated/b3_l25_p2_full")
    parser.add_argument("--checkpoint", default="interp/artifacts/b3_l25_p2_xfmr2_abs/best_model.pt")
    parser.add_argument("--out", default="interp/artifacts/b3_l25_p2_semantic_probes/results.json")
    parser.add_argument("--length", type=int, default=25)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--train-examples", type=int, default=8192)
    parser.add_argument("--eval-examples", type=int, default=8192)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


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
    eval_batch = collect_examples_from_all_shards(
        shard_paths,
        split="test",
        length=args.length,
        total_examples=args.eval_examples,
        seed=args.seed + 9999,
        device=device,
    )
    train_reps = activation_representations(model, train_batch["tokens"])
    eval_reps = activation_representations(model, eval_batch["tokens"])
    train_targets = boundary_targets(train_batch["tokens"], train_batch["label"])
    eval_targets = boundary_targets(eval_batch["tokens"], eval_batch["label"])

    binary_targets = ["label", "leading_col0", "trailing_col1", "leading_row1", "trailing_row1"]
    multiclass_targets = ["leading_unit_token", "trailing_unit_token"]
    results = {
        "data_dir": args.data_dir,
        "checkpoint": args.checkpoint,
        "train_examples": int(train_batch["label"].numel()),
        "eval_examples": int(eval_batch["label"].numel()),
        "probe_type": "ridge_linear_readout",
        "ridge": args.ridge,
        "representations": {},
    }
    for rep_name, train_x in train_reps.items():
        eval_x = eval_reps[rep_name]
        rep_result = {"dim": int(train_x.shape[1]), "targets": {}}
        for target_name in binary_targets:
            rep_result["targets"][target_name] = train_binary_probe(
                train_x,
                train_targets[target_name],
                eval_x,
                eval_targets[target_name],
                args.ridge,
            )
        for target_name in multiclass_targets:
            rep_result["targets"][target_name] = train_multiclass_probe(
                train_x,
                train_targets[target_name],
                eval_x,
                eval_targets[target_name],
                args.ridge,
            )
        results["representations"][rep_name] = rep_result

    atomic_json_dump(results, Path(args.out))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
