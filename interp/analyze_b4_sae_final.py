#!/usr/bin/env python3
"""Final sparse-feature analysis for the B4 Z-sign boundary transformer.

This is a post-training analysis runner.  It assumes the transformer and SAE
checkpoints already exist, then asks the questions that matter for the repo
story:

1. Which SAE features are algebraically labeled and individually causal?
2. Do analogous sparse features recur across independent model seeds?
3. Can a tiny classifier using only selected SAE features recover the model?
4. Which attention-head paths feed the late sparse descent features?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.b4_interp import metrics_from_logits, multilabel_logit_score, run_with_cache, support_features  # noqa: E402
from interp.b4_z_sign import discover_b4_shards, simple_mats_z  # noqa: E402
from interp.run_b4_sae_experiments import (  # noqa: E402
    TopKSAE,
    collect_eval_batch,
    collect_prefix_fixed_pairs,
    collect_site_activation,
    infer_boundary_radius_from_checkpoint,
    infer_data_config,
    make_label_tables,
    make_loader,
    parse_site_list,
    patch_metrics,
    patch_site,
    score_against_clean,
    select_site,
)
from interp.stress_b4_sae_controls import (  # noqa: E402
    ablation_metrics_for_features,
    load_sae,
    model_logits,
    ordered_features,
    patch_recovery_for_features,
)
from interp.train_b4_transformer import atomic_json_dump, resolve_device, set_seed  # noqa: E402


@dataclass(frozen=True)
class RunSpec:
    name: str
    checkpoint: Path
    sae_dir: Path
    results_json: Path
    stress_json: Path


@dataclass
class LoadedRun:
    spec: RunSpec
    model: nn.Module
    checkpoint: dict
    sae_results: dict
    stress_results: dict
    boundary_radius: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final B4 SAE feature analysis.")
    parser.add_argument("--data-dir", default="interp/data/generated/b4_l25_p2_n16777216")
    parser.add_argument("--out-dir", default="interp/artifacts/b4_l25_zsign_boundary_r8_sae_final")
    parser.add_argument("--seed42-checkpoint", default="interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small/best_model.pt")
    parser.add_argument("--seed42-sae-dir", default="interp/artifacts/b4_l25_zsign_boundary_r8_sae_seed42_v2")
    parser.add_argument("--seed7-checkpoint", default="interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small_seed7/best_model.pt")
    parser.add_argument("--seed7-sae-dir", default="interp/artifacts/b4_l25_zsign_boundary_r8_sae_seed7_v2")
    parser.add_argument("--sites", default="final_hidden_cls+l1_resid_post_cls+l1_attn_out_cls")
    parser.add_argument("--atlas-sites", default="final_hidden_cls+l1_resid_post_cls")
    parser.add_argument("--classifier-sites", default="final_hidden_cls+l1_resid_post_cls")
    parser.add_argument("--path-sites", default="final_hidden_cls+l1_resid_post_cls")
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-examples", type=int, default=8192)
    parser.add_argument("--train-examples", type=int, default=32768)
    parser.add_argument("--prefix-pairs", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-atlas-features", type=int, default=32)
    parser.add_argument("--classifier-steps", type=int, default=500)
    parser.add_argument("--classifier-lr", type=float, default=0.05)
    parser.add_argument("--random-classifier-trials", type=int, default=10)
    parser.add_argument("--top-examples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-input-boundary-radius",
        type=int,
        default=-2,
        help="-2 means infer from each checkpoint; nonnegative overrides.",
    )
    return parser.parse_args()


def load_run(spec: RunSpec, device: torch.device, override_radius: int) -> LoadedRun:
    from interp.b4_interp import load_transformer_checkpoint

    loaded = load_transformer_checkpoint(spec.checkpoint, device)
    radius = infer_boundary_radius_from_checkpoint(loaded.checkpoint)
    if override_radius != -2:
        radius = override_radius
    return LoadedRun(
        spec=spec,
        model=loaded.model.eval(),
        checkpoint=loaded.checkpoint,
        sae_results=json.load(open(spec.results_json)),
        stress_results=json.load(open(spec.stress_json)) if spec.stress_json.exists() else {},
        boundary_radius=radius,
    )


def run_specs(args: argparse.Namespace) -> list[RunSpec]:
    seed42_dir = Path(args.seed42_sae_dir)
    seed7_dir = Path(args.seed7_sae_dir)
    return [
        RunSpec(
            name="seed42",
            checkpoint=Path(args.seed42_checkpoint),
            sae_dir=seed42_dir,
            results_json=seed42_dir / "results.json",
            stress_json=seed42_dir / "stress_controls.json",
        ),
        RunSpec(
            name="seed7",
            checkpoint=Path(args.seed7_checkpoint),
            sae_dir=seed7_dir,
            results_json=seed7_dir / "results.json",
            stress_json=seed7_dir / "stress_controls.json",
        ),
    ]


def mask_from_bits(bits: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([1, 2, 4], dtype=torch.long, device=bits.device).view(1, 3)
    return (bits.to(torch.long) * weights).sum(dim=1)


def feature_strategy_map(site_result: dict, max_features: int) -> dict[str, list[int]]:
    return {
        "descent_label": ordered_features(site_result, "descent_label")[:max_features],
        "binary": ordered_features(site_result, "binary")[:max_features],
        "best_label": ordered_features(site_result, "best_label")[:max_features],
    }


def union_preserving_order(groups: Iterable[list[int]]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for group in groups:
        for value in group:
            value = int(value)
            if value not in seen:
                seen.add(value)
                out.append(value)
    return out


def records_for_feature(site_result: dict, feature: int) -> dict:
    labels = site_result["feature_labels"]
    binary = [row for row in labels.get("top_binary_labels", []) if int(row["feature"]) == int(feature)]
    categorical = [row for row in labels.get("top_categorical_labels", []) if int(row["feature"]) == int(feature)]
    best = [row for row in labels.get("best_label_by_feature", []) if int(row["feature"]) == int(feature)]
    return {
        "best_label": best[0] if best else None,
        "top_binary": binary[:5],
        "top_categorical": categorical[:5],
    }


def binary_stats(values: torch.Tensor, labels: torch.Tensor, top_k: int) -> dict:
    values = values.detach().to(torch.float32)
    labels = labels.detach().to(torch.float32)
    n = int(values.numel())
    k = min(max(1, top_k), n)
    idx = torch.topk(values, k=k).indices
    active = values > 0
    label_sum = labels.sum().clamp_min(1.0)
    value_std = values.std(unbiased=False).clamp_min(1e-8)
    label_std = labels.std(unbiased=False).clamp_min(1e-8)
    corr = ((values - values.mean()) * (labels - labels.mean())).mean() / (value_std * label_std)
    active_count = int(active.sum().item())
    active_precision = float(labels[active].mean().item()) if active_count else 0.0
    active_recall = float(labels[active].sum().div(label_sum).item()) if active_count else 0.0
    return {
        "base_rate": float(labels.mean().item()),
        "precision_at_top": float(labels[idx].mean().item()),
        "active_precision": active_precision,
        "active_recall": active_recall,
        "active_fraction": active_count / max(1, n),
        "mean_when_label_true": float(values[labels > 0.5].mean().item()) if bool((labels > 0.5).any()) else 0.0,
        "mean_when_label_false": float(values[labels <= 0.5].mean().item()) if bool((labels <= 0.5).any()) else 0.0,
        "activation_label_corr": float(corr.item()),
    }


def categorical_stats(values: torch.Tensor, labels: torch.Tensor, top_k: int) -> dict:
    values = values.detach().to(torch.float32)
    labels = labels.detach().to(torch.long)
    n = int(values.numel())
    k = min(max(1, top_k), n)
    top_idx = torch.topk(values, k=k).indices
    top_vals = labels[top_idx]
    active_vals = labels[values > 0]

    def majority(xs: torch.Tensor) -> tuple[int, float]:
        if xs.numel() == 0:
            return -1, 0.0
        bincount = torch.bincount(xs.cpu(), minlength=int(labels.max().item()) + 1)
        item = int(bincount.argmax().item())
        return item, float((xs.cpu() == item).float().mean().item())

    top_value, top_precision = majority(top_vals)
    active_value, active_precision = majority(active_vals)
    base = float((labels.cpu() == top_value).float().mean().item()) if top_value >= 0 else 0.0
    return {
        "top_majority_value": top_value,
        "top_precision": top_precision,
        "top_base_rate": base,
        "active_majority_value": active_value,
        "active_precision": active_precision,
    }


def top_example_rows(batch: dict, sparse: torch.Tensor, feature: int, count: int) -> list[dict]:
    values = sparse[:, feature].detach().to(torch.float32)
    k = min(count, int(values.numel()))
    top_values, top_idx = torch.topk(values, k=k)
    feats = support_features(batch["tokens"])
    rows = []
    masks = mask_from_bits(batch["label_bits"])
    for rank, (value, idx) in enumerate(zip(top_values.cpu().tolist(), top_idx.cpu().tolist(), strict=True), start=1):
        rows.append(
            {
                "rank": rank,
                "activation": float(value),
                "sample_id": int(batch["sample_id"][idx].item()),
                "final_factor_id": int(batch["final_factor_id"][idx].item()),
                "descent_mask": int(masks[idx].item()),
                "label_bits": [int(x) for x in batch["label_bits"][idx].cpu().to(torch.long).tolist()],
                "first_degree_index": int(feats["first"][idx].item()),
                "last_degree_index": int(feats["last"][idx].item()),
                "support_width": int(feats["width"][idx].item()),
                "leading_token": int(feats["leading_token"][idx].item()),
                "trailing_token": int(feats["trailing_token"][idx].item()),
            }
        )
    return rows


@torch.no_grad()
def collect_sparse(model, tokens: torch.Tensor, spec, sae: TopKSAE, chunk_size: int) -> torch.Tensor:
    acts = collect_site_activation(model, tokens, spec, chunk_size=chunk_size)
    _, sparse = sae(acts)
    return sparse.detach()


@torch.no_grad()
def feature_atlas_for_run(
    run: LoadedRun,
    eval_batch: dict,
    pairs: dict,
    sites: list,
    *,
    max_features: int,
    top_examples: int,
    chunk_size: int,
) -> dict:
    out = {}
    binary_labels, categorical_labels = make_label_tables(eval_batch)
    for spec in sites:
        site_result = run.sae_results["sites"][spec.key]
        strategies = feature_strategy_map(site_result, max_features)
        selected = union_preserving_order([strategies["descent_label"], strategies["binary"], strategies["best_label"]])
        sae = load_sae(run.spec.sae_dir / f"{spec.key}_sae.pt", eval_batch["tokens"].device)
        sparse = collect_sparse(run.model, eval_batch["tokens"], spec, sae, chunk_size)
        clean_logits = model_logits(run.model, pairs["clean_tokens"], chunk_size)
        corrupt_logits = model_logits(run.model, pairs["corrupt_tokens"], chunk_size)
        clean_score = score_against_clean(clean_logits, pairs["clean_labels"])
        corrupt_score = score_against_clean(corrupt_logits, pairs["clean_labels"])

        feature_rows = []
        for feature in selected[:max_features]:
            values = sparse[:, feature]
            memberships = [name for name, ids in strategies.items() if int(feature) in {int(x) for x in ids}]
            descent_stats = {
                name: binary_stats(values, labels, top_k=256)
                for name, labels in binary_labels.items()
            }
            categorical = {
                name: categorical_stats(values, labels, top_k=256)
                for name, labels in categorical_labels.items()
            }
            patch = patch_recovery_for_features(
                run.model,
                pairs,
                spec=spec,
                sae=sae,
                feature_ids=[int(feature)],
                clean_score=clean_score,
                corrupt_score=corrupt_score,
                chunk_size=chunk_size,
            )
            ablate = ablation_metrics_for_features(
                run.model,
                eval_batch,
                spec=spec,
                sae=sae,
                feature_ids=[int(feature)],
                chunk_size=chunk_size,
            )
            direct_logit_direction = None
            if spec.key == "final_hidden_cls":
                decoder_col = sae.decoder.weight[:, int(feature)].detach()
                direct = torch.matmul(run.model.head.weight.detach(), decoder_col)
                direct_logit_direction = [float(x) for x in direct.cpu().tolist()]
            feature_rows.append(
                {
                    "feature": int(feature),
                    "strategy_membership": memberships,
                    "stored_label_records": records_for_feature(site_result, int(feature)),
                    "active_fraction": float((values > 0).float().mean().item()),
                    "mean_activation": float(values.to(torch.float32).mean().item()),
                    "descent_stats": descent_stats,
                    "categorical_stats": categorical,
                    "individual_patch_recovery": patch,
                    "individual_ablation_metrics": ablate,
                    "direct_logit_direction": direct_logit_direction,
                    "top_examples": top_example_rows(eval_batch, sparse, int(feature), top_examples),
                }
            )

        out[spec.key] = {
            "strategies": strategies,
            "selected_features": selected[:max_features],
            "stress_controls": run.stress_results.get("sites", {}).get(spec.key, {}),
            "features": feature_rows,
        }
    return out


def standardize_columns(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)


@torch.no_grad()
def cross_seed_matches(
    runs: list[LoadedRun],
    eval_batch: dict,
    sites: list,
    *,
    max_features: int,
    chunk_size: int,
    top_matches: int = 5,
) -> dict:
    if len(runs) != 2:
        return {}
    run_a, run_b = runs
    out = {}
    for spec in sites:
        sae_a = load_sae(run_a.spec.sae_dir / f"{spec.key}_sae.pt", eval_batch["tokens"].device)
        sae_b = load_sae(run_b.spec.sae_dir / f"{spec.key}_sae.pt", eval_batch["tokens"].device)
        sparse_a = collect_sparse(run_a.model, eval_batch["tokens"], spec, sae_a, chunk_size)
        sparse_b = collect_sparse(run_b.model, eval_batch["tokens"], spec, sae_b, chunk_size)
        site_a = run_a.sae_results["sites"][spec.key]
        site_b = run_b.sae_results["sites"][spec.key]
        strategies_a = feature_strategy_map(site_a, max_features)
        strategies_b = feature_strategy_map(site_b, max_features)
        selected_a = union_preserving_order([strategies_a["descent_label"], strategies_a["binary"], strategies_a["best_label"]])
        selected_b_set = set(union_preserving_order([strategies_b["descent_label"], strategies_b["binary"], strategies_b["best_label"]]))
        x = standardize_columns(sparse_a[:, selected_a])
        y = standardize_columns(sparse_b)
        corr = torch.matmul(x.T, y) / max(1, x.shape[0])
        matches = []
        for row, feature_a in enumerate(selected_a):
            scores = corr[row].abs()
            values, indices = torch.topk(scores, k=min(top_matches, scores.numel()))
            matches.append(
                {
                    f"{run_a.spec.name}_feature": int(feature_a),
                    f"{run_a.spec.name}_records": records_for_feature(site_a, int(feature_a)),
                    "matches": [
                        {
                            f"{run_b.spec.name}_feature": int(idx.item()),
                            "activation_corr": float(corr[row, idx].item()),
                            "abs_activation_corr": float(value.item()),
                            f"{run_b.spec.name}_is_selected": int(idx.item()) in selected_b_set,
                            f"{run_b.spec.name}_records": records_for_feature(site_b, int(idx.item())),
                        }
                        for value, idx in zip(values.cpu(), indices.cpu(), strict=True)
                    ],
                }
            )
        out[spec.key] = {
            "source": run_a.spec.name,
            "target": run_b.spec.name,
            "source_selected_features": selected_a,
            "target_selected_features": sorted(selected_b_set),
            "matches": matches,
        }
    return out


def classifier_metrics(logits: torch.Tensor, labels: torch.Tensor, teacher_logits: torch.Tensor) -> dict:
    metrics = metrics_from_logits(logits, labels)
    preds = logits.ge(0).to(torch.float32)
    teacher_preds = teacher_logits.ge(0).to(torch.float32)
    metrics["agreement_exact_with_transformer"] = float(preds.eq(teacher_preds).all(dim=1).float().mean().item())
    metrics["agreement_bit_with_transformer"] = float(preds.eq(teacher_preds).float().mean().item())
    return metrics


def train_linear_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    steps: int,
    lr: float,
    seed: int,
) -> dict:
    if x_train.shape[1] == 0:
        return {"error": "no_features"}
    torch.manual_seed(seed)
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_eval = (x_eval - mean) / std
    probe = nn.Linear(x_train.shape[1], y_train.shape[1]).to(x_train.device)
    nn.init.zeros_(probe.weight)
    nn.init.zeros_(probe.bias)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        logits = probe(x_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        eval_logits = probe(x_eval)
        train_logits = probe(x_train)
    return {
        "train": metrics_from_logits(train_logits, y_train),
        "eval": classifier_metrics(eval_logits, y_eval, teacher_logits),
        "weight_norm": float(probe.weight.detach().norm().item()),
    }


def mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "trials": 0}
    x = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        "trials": len(values),
    }


@torch.no_grad()
def active_feature_ids(sparse: torch.Tensor) -> list[int]:
    return torch.nonzero(sparse.gt(0).any(dim=0), as_tuple=False).flatten().cpu().tolist()


def sparse_feature_classifiers(
    run: LoadedRun,
    train_batch: dict,
    eval_batch: dict,
    sites: list,
    *,
    max_features: int,
    classifier_steps: int,
    classifier_lr: float,
    random_trials: int,
    chunk_size: int,
    seed: int,
) -> dict:
    out = {}
    teacher_logits = model_logits(run.model, eval_batch["tokens"], chunk_size)
    teacher_metrics = metrics_from_logits(teacher_logits, eval_batch["label_bits"])
    generator = torch.Generator(device=eval_batch["tokens"].device).manual_seed(seed)
    for spec in sites:
        site_result = run.sae_results["sites"][spec.key]
        strategies = feature_strategy_map(site_result, max_features)
        sae = load_sae(run.spec.sae_dir / f"{spec.key}_sae.pt", eval_batch["tokens"].device)
        train_sparse = collect_sparse(run.model, train_batch["tokens"], spec, sae, chunk_size)
        eval_sparse = collect_sparse(run.model, eval_batch["tokens"], spec, sae, chunk_size)
        eligible = active_feature_ids(eval_sparse)
        site_out = {"teacher_metrics": teacher_metrics, "strategies": {}, "random_controls": {}}
        for strategy, features in strategies.items():
            features = [int(x) for x in features[:max_features]]
            if not features:
                continue
            ids = torch.tensor(features, dtype=torch.long, device=eval_sparse.device)
            result = train_linear_probe(
                train_sparse[:, ids],
                train_batch["label_bits"],
                eval_sparse[:, ids],
                eval_batch["label_bits"],
                teacher_logits,
                steps=classifier_steps,
                lr=classifier_lr,
                seed=seed + len(features),
            )
            result["features"] = features
            result["feature_count"] = len(features)
            site_out["strategies"][strategy] = result

            random_exact = []
            random_bit = []
            random_agree = []
            count = min(len(features), len(eligible))
            for trial in range(random_trials):
                perm = torch.randperm(len(eligible), generator=generator, device=eval_sparse.device)[:count]
                random_features = [eligible[int(i)] for i in perm.cpu().tolist()]
                rid = torch.tensor(random_features, dtype=torch.long, device=eval_sparse.device)
                random_result = train_linear_probe(
                    train_sparse[:, rid],
                    train_batch["label_bits"],
                    eval_sparse[:, rid],
                    eval_batch["label_bits"],
                    teacher_logits,
                    steps=classifier_steps,
                    lr=classifier_lr,
                    seed=seed + 1000 + trial,
                )
                random_exact.append(float(random_result["eval"]["exact_set_accuracy"]))
                random_bit.append(float(random_result["eval"]["bit_accuracy"]))
                random_agree.append(float(random_result["eval"]["agreement_exact_with_transformer"]))
            site_out["random_controls"][strategy] = {
                "feature_count": count,
                "exact_set_accuracy": mean_std(random_exact),
                "bit_accuracy": mean_std(random_bit),
                "agreement_exact_with_transformer": mean_std(random_agree),
            }
        out[spec.key] = site_out
    return out


def selected_sparse_from_cache(cache: dict, tokens: torch.Tensor, spec, sae: TopKSAE, features: list[int]) -> torch.Tensor:
    x = select_site(cache[spec.hook], tokens, spec)
    sparse = sae.encode(x)
    ids = torch.tensor(features, dtype=torch.long, device=tokens.device)
    return sparse[:, ids]


def activation_recovery(candidate: torch.Tensor, clean: torch.Tensor, corrupt: torch.Tensor) -> float:
    denom = (clean - corrupt).pow(2).mean().clamp_min(1e-8)
    mse = (candidate - clean).pow(2).mean()
    return float((1.0 - mse / denom).item())


@torch.no_grad()
def path_patch_for_site(
    run: LoadedRun,
    pairs: dict,
    spec,
    *,
    features: list[int],
    chunk_size: int,
) -> dict:
    if not features:
        return {"error": "no_features"}
    sae = load_sae(run.spec.sae_dir / f"{spec.key}_sae.pt", pairs["clean_tokens"].device)
    clean_logits, clean_cache = run_with_cache(
        run.model,
        pairs["clean_tokens"],
        names_filter={spec.hook} | {f"blocks.{layer}.hook_attn_head_out" for layer in range(len(run.model.blocks))},
    )
    corrupt_logits, corrupt_cache = run_with_cache(run.model, pairs["corrupt_tokens"], names_filter={spec.hook})
    clean_score = score_against_clean(clean_logits, pairs["clean_labels"])
    corrupt_score = score_against_clean(corrupt_logits, pairs["clean_labels"])
    clean_sparse = selected_sparse_from_cache(clean_cache, pairs["clean_tokens"], spec, sae, features)
    corrupt_sparse = selected_sparse_from_cache(corrupt_cache, pairs["corrupt_tokens"], spec, sae, features)

    variants = {}

    def eval_hooked(name: str, hooks: dict) -> None:
        logits, cache = run_with_cache(run.model, pairs["corrupt_tokens"], hooks=hooks, names_filter={spec.hook})
        sparse = selected_sparse_from_cache(cache, pairs["corrupt_tokens"], spec, sae, features)
        row = patch_metrics(logits, pairs["clean_labels"], clean_score, corrupt_score)
        row["selected_feature_activation_recovery"] = activation_recovery(sparse, clean_sparse, corrupt_sparse)
        variants[name] = row

    for layer, block in enumerate(run.model.blocks):
        hook_name = f"blocks.{layer}.hook_attn_head_out"
        num_heads = block.attn.num_heads
        for head in range(num_heads):
            def make_head_hook(layer_name: str, head_idx: int):
                def hook(value: torch.Tensor, _: str) -> torch.Tensor:
                    patched = value.clone()
                    src = clean_cache[layer_name].to(device=value.device, dtype=value.dtype)
                    patched[:, head_idx, 0] = src[:, head_idx, 0]
                    return patched

                return hook

            eval_hooked(f"patch_l{layer}h{head}_cls_head_out", {hook_name: make_head_hook(hook_name, head)})

        def make_all_heads_hook(layer_name: str):
            def hook(value: torch.Tensor, _: str) -> torch.Tensor:
                patched = value.clone()
                src = clean_cache[layer_name].to(device=value.device, dtype=value.dtype)
                patched[:, :, 0] = src[:, :, 0]
                return patched

            return hook

        eval_hooked(f"patch_l{layer}_all_heads_cls_head_out", {hook_name: make_all_heads_hook(hook_name)})

    top_by_activation = sorted(
        [{"variant": key, **value} for key, value in variants.items()],
        key=lambda item: item["selected_feature_activation_recovery"],
        reverse=True,
    )
    top_by_logit = sorted(
        [{"variant": key, **value} for key, value in variants.items()],
        key=lambda item: item["normalized_score_recovery"],
        reverse=True,
    )
    return {
        "features": features,
        "feature_count": len(features),
        "clean": metrics_from_logits(clean_logits, pairs["clean_labels"]),
        "corrupt_scored_as_clean": patch_metrics(corrupt_logits, pairs["clean_labels"], clean_score, corrupt_score),
        "selected_feature_clean_corrupt_mse": float((clean_sparse - corrupt_sparse).pow(2).mean().item()),
        "variants": variants,
        "top_by_selected_feature_activation_recovery": top_by_activation[:16],
        "top_by_logit_recovery": top_by_logit[:16],
    }


def path_patch_analysis(
    run: LoadedRun,
    pairs: dict,
    sites: list,
    *,
    max_features: int,
    chunk_size: int,
) -> dict:
    out = {}
    for spec in sites:
        site_result = run.sae_results["sites"][spec.key]
        strategies = feature_strategy_map(site_result, max_features)
        features = union_preserving_order([strategies["descent_label"], strategies["binary"]])[:max_features]
        out[spec.key] = path_patch_for_site(run, pairs, spec, features=features, chunk_size=chunk_size)
    return out


def fmt(x: float | int | None, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, int):
        return str(x)
    if math.isnan(float(x)):
        return "nan"
    return f"{float(x):.{digits}f}"


def top_classifier_rows(results: dict) -> list[dict]:
    rows = []
    for seed, seed_result in results["sparse_feature_classifiers"].items():
        for site, site_result in seed_result.items():
            for strategy, strategy_result in site_result["strategies"].items():
                rand = site_result["random_controls"].get(strategy, {})
                rows.append(
                    {
                        "seed": seed,
                        "site": site,
                        "strategy": strategy,
                        "features": strategy_result["feature_count"],
                        "exact": strategy_result["eval"]["exact_set_accuracy"],
                        "bit": strategy_result["eval"]["bit_accuracy"],
                        "agreement": strategy_result["eval"]["agreement_exact_with_transformer"],
                        "random_exact": rand.get("exact_set_accuracy", {}).get("mean"),
                        "random_bit": rand.get("bit_accuracy", {}).get("mean"),
                    }
                )
    return sorted(rows, key=lambda row: row["exact"], reverse=True)


def write_markdown_summary(results: dict, path: Path) -> None:
    lines = [
        "# B4 SAE Final Analysis",
        "",
        "This file is generated by `interp/analyze_b4_sae_final.py`. It summarizes the post-training SAE analysis for the B4 Z-sign boundary transformer.",
        "",
        "## Sparse Feature Classifiers",
        "",
        "| Seed | Site | Feature set | k | Exact | Bit | Agreement with transformer | Random exact | Random bit |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_classifier_rows(results):
        lines.append(
            "| {seed} | `{site}` | `{strategy}` | {features} | {exact} | {bit} | {agreement} | {random_exact} | {random_bit} |".format(
                seed=row["seed"],
                site=row["site"],
                strategy=row["strategy"],
                features=row["features"],
                exact=fmt(row["exact"]),
                bit=fmt(row["bit"]),
                agreement=fmt(row["agreement"]),
                random_exact=fmt(row["random_exact"]),
                random_bit=fmt(row["random_bit"]),
            )
        )
    lines.extend(["", "## Path Patching", ""])
    for seed, seed_result in results["path_patching"].items():
        lines.extend([f"### {seed}", ""])
        for site, site_result in seed_result.items():
            lines.extend(
                [
                    f"`{site}` selected features: `{site_result.get('feature_count', 0)}`",
                    "",
                    "| Variant | Feature activation recovery | Logit recovery | Exact | Bit |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for row in site_result.get("top_by_selected_feature_activation_recovery", [])[:8]:
                lines.append(
                    "| `{}` | {} | {} | {} | {} |".format(
                        row["variant"],
                        fmt(row.get("selected_feature_activation_recovery")),
                        fmt(row.get("normalized_score_recovery")),
                        fmt(row.get("exact_set_accuracy")),
                        fmt(row.get("bit_accuracy")),
                    )
                )
            lines.append("")
    lines.extend(["## Cross-Seed Matches", ""])
    for site, site_result in results["cross_seed_matches"].items():
        lines.extend(
            [
                f"### `{site}`",
                "",
                "| Seed42 feature | Best seed7 feature | Corr | Seed7 selected? |",
                "|---:|---:|---:|---|",
            ]
        )
        for row in site_result["matches"][:16]:
            best = row["matches"][0]
            lines.append(
                "| {} | {} | {} | {} |".format(
                    row["seed42_feature"],
                    best["seed7_feature"],
                    fmt(best["activation_corr"]),
                    "yes" if best["seed7_is_selected"] else "no",
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    shard_paths = discover_b4_shards(args.data_dir, num_shards=args.num_shards, allow_partial=False)
    data_config = infer_data_config(shard_paths)
    simple_mats = simple_mats_z(device)
    sites = parse_site_list(args.sites)
    atlas_sites = parse_site_list(args.atlas_sites)
    classifier_sites = parse_site_list(args.classifier_sites)
    path_sites = parse_site_list(args.path_sites)
    runs = [load_run(spec, device, int(args.model_input_boundary_radius)) for spec in run_specs(args)]
    boundary_radii = sorted({run.boundary_radius for run in runs})
    if len(boundary_radii) != 1:
        raise ValueError(f"Expected both runs to use the same boundary radius, got {boundary_radii}")
    boundary_radius = boundary_radii[0]

    eval_loader = make_loader(
        shard_paths,
        split="val",
        batch_size=args.batch_size,
        seed=args.seed + 1,
        epoch=0,
        shuffle=False,
        max_examples=args.eval_examples,
    )
    train_loader = make_loader(
        shard_paths,
        split="train",
        batch_size=args.batch_size,
        seed=args.seed + 2,
        epoch=0,
        shuffle=True,
        max_examples=args.train_examples,
    )
    pair_loader = make_loader(
        shard_paths,
        split="test",
        batch_size=args.batch_size,
        seed=args.seed + 3,
        epoch=0,
        shuffle=False,
        max_examples=max(args.prefix_pairs * 16, args.batch_size),
    )
    eval_batch = collect_eval_batch(
        eval_loader,
        device=device,
        length=data_config["length"],
        absolute_depth=data_config["absolute_depth"],
        simple_mats=simple_mats,
        boundary_radius=boundary_radius,
        examples=args.eval_examples,
    )
    train_batch = collect_eval_batch(
        train_loader,
        device=device,
        length=data_config["length"],
        absolute_depth=data_config["absolute_depth"],
        simple_mats=simple_mats,
        boundary_radius=boundary_radius,
        examples=args.train_examples,
    )
    pairs = collect_prefix_fixed_pairs(
        pair_loader,
        device=device,
        length=data_config["length"],
        absolute_depth=data_config["absolute_depth"],
        simple_mats=simple_mats,
        boundary_radius=boundary_radius,
        num_pairs=args.prefix_pairs,
    )

    results = {
        "config": vars(args),
        "data_config": data_config,
        "effective_model_input_boundary_radius": boundary_radius,
        "run_configs": {
            run.spec.name: {
                "checkpoint": str(run.spec.checkpoint),
                "sae_dir": str(run.spec.sae_dir),
                "model_config": run.checkpoint["model_config"],
            }
            for run in runs
        },
        "feature_atlas": {},
        "cross_seed_matches": {},
        "sparse_feature_classifiers": {},
        "path_patching": {},
    }

    for run in runs:
        print(f"feature_atlas {run.spec.name}", flush=True)
        results["feature_atlas"][run.spec.name] = feature_atlas_for_run(
            run,
            eval_batch,
            pairs,
            atlas_sites,
            max_features=args.max_atlas_features,
            top_examples=args.top_examples,
            chunk_size=args.chunk_size,
        )
        atomic_json_dump(results, out_dir / "results.json")

    print("cross_seed_matches", flush=True)
    results["cross_seed_matches"] = cross_seed_matches(
        runs,
        eval_batch,
        sites,
        max_features=args.max_atlas_features,
        chunk_size=args.chunk_size,
    )
    atomic_json_dump(results, out_dir / "results.json")

    for run in runs:
        print(f"sparse_feature_classifiers {run.spec.name}", flush=True)
        results["sparse_feature_classifiers"][run.spec.name] = sparse_feature_classifiers(
            run,
            train_batch,
            eval_batch,
            classifier_sites,
            max_features=args.max_atlas_features,
            classifier_steps=args.classifier_steps,
            classifier_lr=args.classifier_lr,
            random_trials=args.random_classifier_trials,
            chunk_size=args.chunk_size,
            seed=args.seed + (42 if run.spec.name == "seed42" else 7),
        )
        atomic_json_dump(results, out_dir / "results.json")

    for run in runs:
        print(f"path_patching {run.spec.name}", flush=True)
        results["path_patching"][run.spec.name] = path_patch_analysis(
            run,
            pairs,
            path_sites,
            max_features=args.max_atlas_features,
            chunk_size=args.chunk_size,
        )
        atomic_json_dump(results, out_dir / "results.json")

    write_markdown_summary(results, out_dir / "SUMMARY.md")
    atomic_json_dump(results, out_dir / "results.json")
    print(json.dumps({"out_dir": str(out_dir), "summary": str(out_dir / "SUMMARY.md")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
