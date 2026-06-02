#!/usr/bin/env python3
"""First-pass mechanistic interpretability experiments for the B_4 transformer."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.b4_data import discover_shards  # noqa: E402
from interp.b4_interp import (  # noqa: E402
    gather_relative_window,
    keep_heads_hook,
    load_transformer_checkpoint,
    metrics_from_logits,
    multilabel_logit_score,
    run_with_cache,
    support_features,
    zero_except_windows,
    zero_head_positions_hook,
    zero_hook,
    zero_windows,
)
from interp.train_b4_transformer import atomic_json_dump, make_loader, resolve_device, set_seed  # noqa: E402


def bits_from_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(torch.long)
    return torch.stack([((mask >> idx) & 1) for idx in range(3)], dim=1).to(torch.float32)


def mask_from_bits(bits: torch.Tensor) -> torch.Tensor:
    bits = bits.to(torch.long)
    weights = torch.tensor([1, 2, 4], device=bits.device, dtype=torch.long).view(1, 3)
    return (bits * weights).sum(dim=1)


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


def finish_probe_multilabel(scores: torch.Tensor, labels: torch.Tensor) -> dict:
    labels = labels.to(device=scores.device, dtype=torch.float32)
    preds = (scores >= 0).to(torch.float32)
    return {
        "exact_set_accuracy": float(preds.eq(labels).all(dim=1).float().mean().item()),
        "bit_accuracy": float(preds.eq(labels).float().mean().item()),
        "per_label_accuracy": [float(x) for x in preds.eq(labels).float().mean(dim=0).tolist()],
        "pred_mask_counts": torch.bincount(mask_from_bits(preds), minlength=8).cpu().tolist(),
    }


def train_multilabel_probe(
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
    return {
        "train": finish_probe_multilabel(train_score, train_y),
        "eval": finish_probe_multilabel(eval_score, eval_y),
    }


def train_multiclass_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    ridge: float,
) -> dict:
    train_x, eval_x = normalize(train_x, eval_x)
    train_y = train_y.to(train_x.device)
    eval_y = eval_y.to(eval_x.device)
    good_train = train_y >= 0
    good_eval = eval_y >= 0
    train_x = train_x[good_train]
    train_y = train_y[good_train].to(torch.long)
    eval_x = eval_x[good_eval]
    eval_y = eval_y[good_eval].to(torch.long)
    num_classes = int(max(train_y.max().item(), eval_y.max().item()) + 1)
    train_onehot = torch.nn.functional.one_hot(train_y, num_classes=num_classes).to(device=train_x.device, dtype=torch.float32)
    weights = ridge_solve(train_x, train_onehot, ridge=ridge)
    with torch.no_grad():
        train_score = add_bias(train_x.to(torch.float32)) @ weights
        eval_score = add_bias(eval_x.to(torch.float32)) @ weights
        train_pred = train_score.argmax(dim=1)
        eval_pred = eval_score.argmax(dim=1)
    return {
        "train_accuracy": float((train_pred == train_y).float().mean().item()),
        "eval_accuracy": float((eval_pred == eval_y).float().mean().item()),
        "num_classes": num_classes,
        "eval_n": int(eval_y.numel()),
    }


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
        "pred_mask_counts": torch.bincount(pred_masks, minlength=8).tolist(),
        "n": len(preds),
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
    base["boundary_degrees_tokens"] = [
        (int(a), int(b), int(c), int(d))
        for a, b, c, d in zip(
            feats["first"].cpu().tolist(),
            feats["last"].cpu().tolist(),
            feats["leading_token"].cpu().tolist(),
            feats["trailing_token"].cpu().tolist(),
            strict=True,
        )
    ]
    if radius is not None:
        lead = gather_relative_window(tokens, feats["first"], radius)
        trail = gather_relative_window(tokens, feats["last"], radius)
        lead_rows = tensor_rows_as_tuples(lead)
        trail_rows = tensor_rows_as_tuples(trail)
        base[f"leading_window_r{radius}"] = [
            (int(pos),) + row for pos, row in zip(feats["first"].cpu().tolist(), lead_rows, strict=True)
        ]
        base[f"trailing_window_r{radius}"] = [
            (int(pos),) + row for pos, row in zip(feats["last"].cpu().tolist(), trail_rows, strict=True)
        ]
        base[f"both_windows_r{radius}"] = [
            (int(first), int(last)) + lead_row + trail_row
            for first, last, lead_row, trail_row in zip(
                feats["first"].cpu().tolist(),
                feats["last"].cpu().tolist(),
                lead_rows,
                trail_rows,
                strict=True,
            )
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
def activation_representations(model, tokens: torch.Tensor, chunk_size: int) -> dict[str, torch.Tensor]:
    names_filter = {"hook_resid_embed", "hook_final_hidden"}
    for layer in range(len(model.blocks)):
        names_filter.add(f"blocks.{layer}.hook_resid_post")
        names_filter.add(f"blocks.{layer}.hook_attn_head_out")

    pieces: dict[str, list[torch.Tensor]] = defaultdict(list)
    for start in range(0, tokens.shape[0], chunk_size):
        chunk = tokens[start : start + chunk_size]
        _, cache = run_with_cache(model, chunk, names_filter=names_filter)
        feats = support_features(chunk)
        rows = torch.arange(chunk.shape[0], device=chunk.device)
        leading_pos = feats["first"] + 1
        trailing_pos = feats["last"] + 1
        support_mask = torch.cat(
            [torch.zeros(chunk.shape[0], 1, dtype=torch.bool, device=chunk.device), feats["support"]],
            dim=1,
        )
        support_weight = support_mask.to(torch.float32)
        support_weight = support_weight / support_weight.sum(dim=1, keepdim=True).clamp_min(1.0)

        pieces["embed_cls"].append(cache["hook_resid_embed"][:, 0].float().cpu())
        pieces["final_hidden_cls"].append(cache["hook_final_hidden"][:, 0].float().cpu())
        for layer in range(len(model.blocks)):
            resid = cache[f"blocks.{layer}.hook_resid_post"]
            head_out = cache[f"blocks.{layer}.hook_attn_head_out"]
            pieces[f"l{layer}_resid_post_cls"].append(resid[:, 0].float().cpu())
            pieces[f"l{layer}_resid_post_leading"].append(resid[rows, leading_pos].float().cpu())
            pieces[f"l{layer}_resid_post_trailing"].append(resid[rows, trailing_pos].float().cpu())
            for head in range(head_out.shape[1]):
                pieces[f"l{layer}h{head}_headout_cls"].append(head_out[:, head, 0].float().cpu())
                support_mean = (head_out[:, head] * support_weight.unsqueeze(-1)).sum(dim=1)
                pieces[f"l{layer}h{head}_headout_support_mean"].append(support_mean.float().cpu())
    return {name: torch.cat(values, dim=0).to(tokens.device) for name, values in pieces.items()}


def semantic_probe_summary(model, train_batch: dict, eval_batch: dict, ridge: float, chunk_size: int) -> dict:
    train_reps = activation_representations(model, train_batch["tokens"], chunk_size)
    eval_reps = activation_representations(model, eval_batch["tokens"], chunk_size)

    train_targets = {
        "descent_bits": train_batch["label_bits"].to(device=train_batch["tokens"].device, dtype=torch.float32),
        "descent_mask": train_batch["descent_mask"].to(device=train_batch["tokens"].device, dtype=torch.long),
        "final_factor_id": train_batch["final_factor_id"].to(device=train_batch["tokens"].device, dtype=torch.long),
    }
    eval_targets = {
        "descent_bits": eval_batch["label_bits"].to(device=eval_batch["tokens"].device, dtype=torch.float32),
        "descent_mask": eval_batch["descent_mask"].to(device=eval_batch["tokens"].device, dtype=torch.long),
        "final_factor_id": eval_batch["final_factor_id"].to(device=eval_batch["tokens"].device, dtype=torch.long),
    }
    for batch_name, batch, targets in [
        ("train", train_batch, train_targets),
        ("eval", eval_batch, eval_targets),
    ]:
        feats = support_features(batch["tokens"])
        targets["leading_token"] = feats["leading_token"].to(torch.long)
        targets["trailing_token"] = feats["trailing_token"].to(torch.long)
        targets["min_degree"] = feats["first"].to(torch.long)
        targets["max_degree"] = feats["last"].to(torch.long)
        del batch_name

    results = {
        "ridge": ridge,
        "train_examples": int(train_batch["label_bits"].shape[0]),
        "eval_examples": int(eval_batch["label_bits"].shape[0]),
        "representations": {},
    }
    for rep_name, train_x in train_reps.items():
        eval_x = eval_reps[rep_name]
        rep_result = {"dim": int(train_x.shape[1]), "targets": {}}
        rep_result["targets"]["descent_bits"] = train_multilabel_probe(
            train_x,
            train_targets["descent_bits"],
            eval_x,
            eval_targets["descent_bits"],
            ridge,
        )
        for target_name in ["descent_mask", "final_factor_id", "leading_token", "trailing_token", "min_degree", "max_degree"]:
            rep_result["targets"][target_name] = train_multiclass_probe(
                train_x,
                train_targets[target_name],
                eval_x,
                eval_targets[target_name],
                ridge,
            )
        results["representations"][rep_name] = rep_result
    return results


@torch.no_grad()
def boundary_intervention_summary(model, batch: dict, radii: list[int]) -> dict:
    tokens = batch["tokens"]
    labels = batch["label_bits"]
    variants = {"full": tokens}
    for radius in radii:
        variants[f"boundary_only_r{radius}"] = zero_except_windows(tokens, radius, leading=True, trailing=True)
        variants[f"leading_only_r{radius}"] = zero_except_windows(tokens, radius, leading=True, trailing=False)
        variants[f"trailing_only_r{radius}"] = zero_except_windows(tokens, radius, leading=False, trailing=True)
        variants[f"drop_boundary_r{radius}"] = zero_windows(tokens, radius, leading=True, trailing=True)
    out = {}
    for name, variant_tokens in variants.items():
        out[name] = metrics_from_logits(model(variant_tokens), labels)
    return out


@torch.no_grad()
def component_ablation_summary(model, batch: dict) -> dict:
    tokens = batch["tokens"]
    labels = batch["label_bits"]
    direct_logits = model(tokens)
    cached_logits, _ = run_with_cache(model, tokens, names_filter=set())
    variants: dict[str, dict] = {
        "full_direct": metrics_from_logits(direct_logits, labels),
        "full_cached": metrics_from_logits(cached_logits, labels),
    }
    for layer in range(len(model.blocks)):
        variants[f"zero_l{layer}_attn"] = metrics_from_logits(
            run_with_cache(model, tokens, hooks={f"blocks.{layer}.hook_attn_head_out": zero_hook}, names_filter=set())[0],
            labels,
        )
        variants[f"zero_l{layer}_mlp"] = metrics_from_logits(
            run_with_cache(model, tokens, hooks={f"blocks.{layer}.hook_mlp_out": zero_hook}, names_filter=set())[0],
            labels,
        )
        for head in range(model.blocks[layer].attn.num_heads):
            for suffix, token_indices in {"all_dest": None, "cls_dest": [0]}.items():
                hook = zero_head_positions_hook(head_idx=head, token_indices=token_indices)
                logits, _ = run_with_cache(
                    model,
                    tokens,
                    hooks={f"blocks.{layer}.hook_attn_head_out": hook},
                    names_filter=set(),
                )
                variants[f"zero_l{layer}h{head}_{suffix}"] = metrics_from_logits(logits, labels)
    return {
        "cache_forward_max_abs_diff": float((direct_logits - cached_logits).abs().max().item()),
        "variants": variants,
    }


@torch.no_grad()
def attention_boundary_summary(model, batch: dict, chunk_size: int) -> dict:
    tokens = batch["tokens"]
    labels = batch["label_bits"]
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts = 0
    for start in range(0, tokens.shape[0], chunk_size):
        chunk = tokens[start : start + chunk_size]
        _, cache = run_with_cache(
            model,
            chunk,
            names_filter={f"blocks.{layer}.hook_pattern" for layer in range(len(model.blocks))},
        )
        feats = support_features(chunk)
        rows = torch.arange(chunk.shape[0], device=chunk.device)
        first_pos = feats["first"] + 1
        last_pos = feats["last"] + 1
        support_mask = torch.cat(
            [torch.zeros(chunk.shape[0], 1, dtype=torch.bool, device=chunk.device), feats["support"]],
            dim=1,
        )
        for layer in range(len(model.blocks)):
            pattern = cache[f"blocks.{layer}.hook_pattern"]
            cls_attn = pattern[:, :, 0, :]
            for head in range(cls_attn.shape[1]):
                key = f"l{layer}h{head}"
                head_attn = cls_attn[:, head]
                entropy = -(head_attn.clamp_min(1e-12) * head_attn.clamp_min(1e-12).log()).sum(dim=1)
                totals[key]["mean_cls_to_leading"] += float(head_attn[rows, first_pos].sum().item())
                totals[key]["mean_cls_to_trailing"] += float(head_attn[rows, last_pos].sum().item())
                totals[key]["mean_cls_to_boundaries"] += float((head_attn[rows, first_pos] + head_attn[rows, last_pos]).sum().item())
                totals[key]["mean_cls_to_support"] += float((head_attn * support_mask).sum(dim=1).sum().item())
                totals[key]["mean_cls_to_cls"] += float(head_attn[:, 0].sum().item())
                totals[key]["mean_entropy"] += float(entropy.sum().item())
        counts += int(chunk.shape[0])

    heads = {}
    for key, values in totals.items():
        heads[key] = {name: value / max(1, counts) for name, value in values.items()}
    top_by_boundary = sorted(heads.items(), key=lambda item: item[1]["mean_cls_to_boundaries"], reverse=True)
    top_by_support = sorted(heads.items(), key=lambda item: item[1]["mean_cls_to_support"], reverse=True)
    return {
        "examples": int(tokens.shape[0]),
        "label_mean": [float(x) for x in labels.float().mean(dim=0).cpu().tolist()],
        "heads": heads,
        "top_by_boundary_attention": [{"head": k, **v} for k, v in top_by_boundary[:12]],
        "top_by_support_attention": [{"head": k, **v} for k, v in top_by_support[:12]],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run first-pass B_4 transformer interp experiments.")
    parser.add_argument("--data-dir", default="interp/data/generated/b4_l25_p2_n16777216")
    parser.add_argument("--checkpoint", default="interp/artifacts/b4_l25_p2_xfmr3_abs/best_model.pt")
    parser.add_argument("--out", default="interp/artifacts/b4_l25_p2_firstpass_interp/results.json")
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--probe-train-examples", type=int, default=8192)
    parser.add_argument("--probe-eval-examples", type=int, default=8192)
    parser.add_argument("--eval-examples", type=int, default=8192)
    parser.add_argument("--attn-examples", type=int, default=2048)
    parser.add_argument("--rep-chunk-size", type=int, default=1024)
    parser.add_argument("--attn-chunk-size", type=int, default=512)
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

    train_loader = make_loader(
        shard_paths,
        split="train",
        batch_size=args.batch_size,
        seed=args.seed,
        epoch=0,
        shuffle=True,
        max_examples=args.probe_train_examples,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    eval_loader = make_loader(
        shard_paths,
        split="test",
        batch_size=args.batch_size,
        seed=args.seed + 10_000,
        epoch=0,
        shuffle=True,
        max_examples=max(args.probe_eval_examples, args.eval_examples, args.attn_examples),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    train_batch = collect_examples(train_loader, device, args.probe_train_examples)
    eval_all = collect_examples(eval_loader, device, max(args.probe_eval_examples, args.eval_examples, args.attn_examples))
    probe_eval = {key: value[: args.probe_eval_examples] for key, value in eval_all.items()}
    eval_batch = {key: value[: args.eval_examples] for key, value in eval_all.items()}
    attn_batch = {key: value[: args.attn_examples] for key, value in eval_all.items()}

    with torch.no_grad():
        direct_metrics = metrics_from_logits(model(eval_batch["tokens"]), eval_batch["label_bits"])

    results = {
        "data_dir": args.data_dir,
        "checkpoint": args.checkpoint,
        "model_config": model.config.to_dict(),
        "direct_metrics": direct_metrics,
        "raw_feature_lookup": raw_feature_lookup_summary(train_batch, probe_eval, radii=[0, 1, 2, 3, 5]),
        "semantic_probes": semantic_probe_summary(
            model,
            train_batch,
            probe_eval,
            ridge=args.ridge,
            chunk_size=args.rep_chunk_size,
        ),
        "boundary_interventions": boundary_intervention_summary(model, eval_batch, radii=[0, 1, 2, 3, 5, 8]),
        "component_ablations": component_ablation_summary(model, eval_batch),
        "attention_boundary_summary": attention_boundary_summary(model, attn_batch, chunk_size=args.attn_chunk_size),
    }
    atomic_json_dump(results, Path(args.out))
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
