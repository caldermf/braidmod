#!/usr/bin/env python3
"""Run first-pass mechanistic experiments for the B_3 transformer."""

from __future__ import annotations

import argparse
import json
import math
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
    boundary_feature_keys,
    load_transformer_checkpoint,
    run_with_cache,
    zero_except_windows,
    zero_windows,
)
from interp.train_b3_transformer import atomic_json_dump, make_loader, resolve_device, set_seed  # noqa: E402


def to_jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    return obj


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    labels = labels.to(device=logits.device, dtype=torch.float32)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
    preds = (logits >= 0).to(torch.float32)
    return {
        "loss": float(loss.item()),
        "accuracy": float((preds == labels).float().mean().item()),
        "n": int(labels.numel()),
    }


@torch.no_grad()
def collect_first_batch(loader, device: torch.device, min_examples: int) -> dict:
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


@torch.no_grad()
def evaluate_transforms(model, loader, device: torch.device, max_examples: int, radii: list[int]) -> dict:
    totals: dict[str, dict[str, float]] = {}
    seen = 0

    def update(name: str, logits: torch.Tensor, labels: torch.Tensor) -> None:
        m = metrics_from_logits(logits.detach(), labels)
        slot = totals.setdefault(name, {"loss_sum": 0.0, "correct": 0.0, "n": 0})
        slot["loss_sum"] += m["loss"] * m["n"]
        slot["correct"] += m["accuracy"] * m["n"]
        slot["n"] += m["n"]

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

        update("original", model(tokens), labels)
        for radius in radii:
            transforms = {
                f"keep_leading_r{radius}": zero_except_windows(tokens, radius, leading=True, trailing=False),
                f"keep_trailing_r{radius}": zero_except_windows(tokens, radius, leading=False, trailing=True),
                f"keep_both_r{radius}": zero_except_windows(tokens, radius, leading=True, trailing=True),
                f"zero_leading_r{radius}": zero_windows(tokens, radius, leading=True, trailing=False),
                f"zero_trailing_r{radius}": zero_windows(tokens, radius, leading=False, trailing=True),
                f"zero_both_r{radius}": zero_windows(tokens, radius, leading=True, trailing=True),
            }
            for name, transformed in transforms.items():
                update(name, model(transformed), labels)

        if max_examples > 0 and seen >= max_examples:
            break

    return {
        name: {
            "loss": slot["loss_sum"] / max(1, slot["n"]),
            "accuracy": slot["correct"] / max(1, slot["n"]),
            "n": int(slot["n"]),
        }
        for name, slot in sorted(totals.items())
    }


def update_feature_counts(counts: dict[str, dict[int, list[int]]], tokens: torch.Tensor, labels: torch.Tensor, radius: int) -> None:
    keys = boundary_feature_keys(tokens.cpu(), radius=radius)
    labels_list = labels.cpu().to(torch.long).tolist()
    for feature_name, feature_keys in keys.items():
        table = counts.setdefault(feature_name, {})
        for key, label in zip(feature_keys.to(torch.long).tolist(), labels_list):
            slot = table.setdefault(int(key), [0, 0])
            slot[int(label)] += 1


def evaluate_feature_counts(
    counts: dict[str, dict[int, list[int]]],
    tokens: torch.Tensor,
    labels: torch.Tensor,
    radius: int,
    global_majority: int,
) -> dict[str, dict[str, float]]:
    keys = boundary_feature_keys(tokens.cpu(), radius=radius)
    labels_list = labels.cpu().to(torch.long).tolist()
    out: dict[str, dict[str, float]] = {}
    for feature_name, feature_keys in keys.items():
        table = counts[feature_name]
        correct = 0
        known = 0
        for key, label in zip(feature_keys.to(torch.long).tolist(), labels_list):
            slot = table.get(int(key))
            if slot is None:
                pred = global_majority
            else:
                known += 1
                pred = 1 if slot[1] >= slot[0] else 0
            correct += int(pred == int(label))
        out[feature_name] = {
            "correct": correct,
            "n": len(labels_list),
            "known": known,
        }
    return out


def run_feature_lookup_baselines(train_loader, eval_loader, train_examples: int, eval_examples: int, radius: int) -> dict:
    counts: dict[str, dict[int, list[int]]] = {}
    label_counts = [0, 0]
    seen = 0
    for batch in train_loader:
        tokens = batch["tokens"]
        labels = batch["label"].to(torch.long)
        if train_examples > 0 and seen + labels.numel() > train_examples:
            take = train_examples - seen
            if take <= 0:
                break
            tokens = tokens[:take]
            labels = labels[:take]
        label_counts[0] += int((labels == 0).sum().item())
        label_counts[1] += int((labels == 1).sum().item())
        update_feature_counts(counts, tokens, labels, radius)
        seen += int(labels.numel())
        if train_examples > 0 and seen >= train_examples:
            break

    global_majority = 1 if label_counts[1] >= label_counts[0] else 0
    totals: dict[str, dict[str, float]] = {}
    eval_seen = 0
    for batch in eval_loader:
        tokens = batch["tokens"]
        labels = batch["label"].to(torch.long)
        if eval_examples > 0 and eval_seen + labels.numel() > eval_examples:
            take = eval_examples - eval_seen
            if take <= 0:
                break
            tokens = tokens[:take]
            labels = labels[:take]
        batch_metrics = evaluate_feature_counts(counts, tokens, labels, radius, global_majority)
        for name, m in batch_metrics.items():
            slot = totals.setdefault(name, {"correct": 0, "n": 0, "known": 0})
            slot["correct"] += int(m["correct"])
            slot["n"] += int(m["n"])
            slot["known"] += int(m["known"])
        eval_seen += int(labels.numel())
        if eval_examples > 0 and eval_seen >= eval_examples:
            break

    return {
        name: {
            "accuracy": slot["correct"] / max(1, slot["n"]),
            "coverage": slot["known"] / max(1, slot["n"]),
            "n": int(slot["n"]),
            "num_train_keys": len(counts[name]),
        }
        for name, slot in sorted(totals.items())
    }


@torch.no_grad()
def attention_summary(model, batch: dict) -> dict:
    logits, cache = run_with_cache(model, batch["tokens"])
    labels = batch["label"].to(torch.long)
    summary = {"batch_accuracy": metrics_from_logits(logits, batch["label"])["accuracy"], "layers": {}}
    for layer in range(len(model.blocks)):
        pattern = cache[f"blocks.{layer}.hook_pattern"]
        cls_to_degree = pattern[:, :, 0, 1:]
        layer_summary = {}
        for label in [0, 1]:
            mask = labels == label
            if not bool(mask.any().item()):
                continue
            mean_attn = cls_to_degree[mask].mean(dim=0)
            heads = {}
            for head in range(mean_attn.shape[0]):
                values, idx = torch.topk(mean_attn[head], k=min(10, mean_attn.shape[1]))
                heads[str(head)] = {
                    "top_degrees": idx.cpu().tolist(),
                    "top_attention": values.cpu().tolist(),
                }
            layer_summary[f"label_{label}"] = heads
        summary["layers"][str(layer)] = layer_summary
    return summary


@torch.no_grad()
def patching_summary(model, batch: dict, num_pairs: int, site: str) -> dict:
    labels = batch["label"].to(torch.long)
    idx0 = torch.nonzero(labels == 0, as_tuple=False).flatten()
    idx1 = torch.nonzero(labels == 1, as_tuple=False).flatten()
    n = min(num_pairs, idx0.numel(), idx1.numel())
    if n == 0:
        return {"error": "batch does not contain both labels"}
    idx0 = idx0[:n]
    idx1 = idx1[:n]
    clean_tokens = torch.cat([batch["tokens"][idx0], batch["tokens"][idx1]], dim=0)
    corrupt_tokens = torch.cat([batch["tokens"][idx1], batch["tokens"][idx0]], dim=0)
    clean_labels = torch.cat([labels[idx0], labels[idx1]], dim=0).to(batch["tokens"].device)
    result = activation_patch_token_sweep(
        model,
        clean_tokens,
        corrupt_tokens,
        clean_labels,
        site=site,
    )
    recovery = result["recovery"]
    top = {}
    for layer_i, layer in enumerate(result["layers"].tolist()):
        vals, idx = torch.topk(recovery[layer_i], k=min(10, recovery.shape[1]))
        top[str(layer)] = {
            "top_token_indices": idx.tolist(),
            "top_recovery": vals.tolist(),
        }
    return {
        "site": site,
        "num_clean_corrupt_pairs": int(2 * n),
        "mean_clean_score": float(result["clean_score"].mean().item()),
        "mean_corrupt_score": float(result["corrupt_score"].mean().item()),
        "top_recovery_by_layer": top,
        "recovery_matrix": result["recovery"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run B_3 transformer interp experiments.")
    parser.add_argument("--data-dir", default="interp/data/generated/b3_l25_p2_full")
    parser.add_argument("--checkpoint", default="interp/artifacts/b3_l25_p2_xfmr2_abs/best_model.pt")
    parser.add_argument("--out-dir", default="interp/artifacts/b3_l25_p2_interp_firstpass")
    parser.add_argument("--length", type=int, default=25)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--feature-radius", type=int, default=2)
    parser.add_argument("--feature-train-examples", type=int, default=2_097_152)
    parser.add_argument("--feature-eval-examples", type=int, default=671_088)
    parser.add_argument("--transform-eval-examples", type=int, default=262_144)
    parser.add_argument("--cache-check-examples", type=int, default=512)
    parser.add_argument("--patch-pairs", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    shard_paths = discover_shards(args.data_dir, num_shards=args.num_shards)
    loaded = load_transformer_checkpoint(args.checkpoint, device=device)
    model = loaded.model

    val_loader = make_loader(
        shard_paths,
        split="val",
        length=args.length,
        batch_size=args.batch_size,
        seed=args.seed,
        epoch=0,
        shuffle=False,
        max_examples=max(args.transform_eval_examples, args.cache_check_examples, 2 * args.patch_pairs),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    first_batch = collect_first_batch(val_loader, device, max(args.cache_check_examples, 2 * args.patch_pairs))

    with torch.no_grad():
        direct_logits = model(first_batch["tokens"][: args.cache_check_examples])
        cached_logits, _ = run_with_cache(model, first_batch["tokens"][: args.cache_check_examples])
        cache_max_abs_diff = float((direct_logits - cached_logits).abs().max().item())

    transform_loader = make_loader(
        shard_paths,
        split="test",
        length=args.length,
        batch_size=args.batch_size,
        seed=args.seed,
        epoch=0,
        shuffle=False,
        max_examples=args.transform_eval_examples,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    transform_results = evaluate_transforms(
        model,
        transform_loader,
        device=device,
        max_examples=args.transform_eval_examples,
        radii=[0, 1, 2],
    )

    train_loader = make_loader(
        shard_paths,
        split="train",
        length=args.length,
        batch_size=args.batch_size,
        seed=args.seed,
        epoch=0,
        shuffle=True,
        max_examples=args.feature_train_examples,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    eval_loader = make_loader(
        shard_paths,
        split="test",
        length=args.length,
        batch_size=args.batch_size,
        seed=args.seed,
        epoch=0,
        shuffle=False,
        max_examples=args.feature_eval_examples,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    feature_results = run_feature_lookup_baselines(
        train_loader,
        eval_loader,
        train_examples=args.feature_train_examples,
        eval_examples=args.feature_eval_examples,
        radius=args.feature_radius,
    )

    attn_results = attention_summary(model, first_batch)
    patch_results = {
        "resid_post": patching_summary(model, first_batch, args.patch_pairs, "hook_resid_post"),
        "attn_out": patching_summary(model, first_batch, args.patch_pairs, "hook_attn_out"),
        "mlp_out": patching_summary(model, first_batch, args.patch_pairs, "hook_mlp_out"),
    }

    tensor_payload = {
        "patch_resid_post_recovery": patch_results["resid_post"].pop("recovery_matrix", None),
        "patch_attn_out_recovery": patch_results["attn_out"].pop("recovery_matrix", None),
        "patch_mlp_out_recovery": patch_results["mlp_out"].pop("recovery_matrix", None),
    }
    torch.save(tensor_payload, out_dir / "tensors.pt")

    results = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "cache_forward_max_abs_diff": cache_max_abs_diff,
        "transform_counterfactuals": transform_results,
        "feature_lookup_baselines": feature_results,
        "attention_summary": attn_results,
        "activation_patching": patch_results,
    }
    atomic_json_dump(to_jsonable(results), out_dir / "results.json")
    print(json.dumps(to_jsonable(results), indent=2)[:20000])
    print(f"results={out_dir / 'results.json'}")
    print(f"tensors={out_dir / 'tensors.pt'}")


if __name__ == "__main__":
    main()
