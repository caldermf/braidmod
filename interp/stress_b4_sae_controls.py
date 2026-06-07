#!/usr/bin/env python3
"""Stress tests for B4 SAE feature-patching claims.

This script is deliberately skeptical.  The all-feature SAE patch is expected
to work whenever the SAE reconstructs the activation well; by itself, that is
mostly a compressed activation patch.  The nontrivial question is whether small
algebraically labeled feature subsets beat random controls and causally recover
the clean counterfactual decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.b4_interp import load_transformer_checkpoint, metrics_from_logits, run_with_cache  # noqa: E402
from interp.b4_z_sign import discover_b4_shards, simple_mats_z  # noqa: E402
from interp.run_b4_sae_experiments import (  # noqa: E402
    TopKSAE,
    collect_eval_batch,
    collect_prefix_fixed_pairs,
    collect_site_activation,
    infer_boundary_radius_from_checkpoint,
    infer_data_config,
    logits_with_sae_reconstruction,
    make_label_tables,
    make_loader,
    parse_site_list,
    patch_metrics,
    patch_site,
    sae_feature_patch_logits,
    score_against_clean,
    select_site,
)
from interp.train_b4_transformer import atomic_json_dump, resolve_device, set_seed  # noqa: E402


def load_sae(path: Path, device: torch.device) -> TopKSAE:
    payload = torch.load(path, map_location=device, weights_only=False)
    sae = TopKSAE(
        d_in=int(payload["d_in"]),
        n_features=int(payload["n_features"]),
        top_k=int(payload["top_k"]),
    ).to(device)
    sae.load_state_dict(payload["state_dict"])
    sae.eval()
    return sae


def ordered_features(site_result: dict, strategy: str) -> list[int]:
    if strategy == "best_label":
        rows = site_result["feature_labels"]["best_label_by_feature"]
    elif strategy == "descent_label":
        grouped = site_result["feature_labels"]["top_by_descent_label"]
        rows = []
        for key in ("descent_s1", "descent_s2", "descent_s3"):
            rows.extend(grouped.get(key, []))
        rows = sorted(rows, key=lambda item: float(item["score"]), reverse=True)
    elif strategy == "categorical":
        rows = site_result["feature_labels"]["top_categorical_labels"]
    elif strategy == "binary":
        rows = site_result["feature_labels"]["top_binary_labels"]
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")
    out = []
    seen = set()
    for row in rows:
        feature = int(row["feature"])
        if feature not in seen:
            seen.add(feature)
            out.append(feature)
    return out


@torch.no_grad()
def model_logits(model, tokens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    pieces = []
    for start in range(0, tokens.shape[0], chunk_size):
        pieces.append(model(tokens[start : start + chunk_size]).detach().cpu())
    return torch.cat(pieces, dim=0).to(tokens.device)


@torch.no_grad()
def sae_patch_logits_permuted(
    model,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    *,
    spec,
    sae: TopKSAE,
    chunk_size: int,
    feature_ids: torch.Tensor | None,
    generator: torch.Generator,
) -> torch.Tensor:
    outs = []
    for start in range(0, clean_tokens.shape[0], chunk_size):
        clean = clean_tokens[start : start + chunk_size]
        corrupt = corrupt_tokens[start : start + chunk_size]
        _, clean_cache = run_with_cache(model, clean, names_filter={spec.hook})
        clean_x = select_site(clean_cache[spec.hook], clean, spec)
        clean_acts = sae.encode(clean_x)
        perm = torch.randperm(clean_acts.shape[0], generator=generator, device=clean_acts.device)
        clean_acts = clean_acts[perm]

        def hook(value: torch.Tensor, _: str) -> torch.Tensor:
            corrupt_x = select_site(value, corrupt, spec)
            corrupt_acts = sae.encode(corrupt_x)
            patched_acts = corrupt_acts.clone()
            if feature_ids is None:
                patched_acts = clean_acts.to(patched_acts.device)
            elif feature_ids.numel() > 0:
                ids = feature_ids.to(patched_acts.device)
                patched_acts[:, ids] = clean_acts[:, ids].to(patched_acts.device)
            recon = sae.decode(patched_acts)
            return patch_site(value, corrupt, spec, recon)

        logits, _ = run_with_cache(model, corrupt, hooks={spec.hook: hook}, names_filter=set())
        outs.append(logits.detach().cpu())
    return torch.cat(outs, dim=0).to(clean_tokens.device)


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
def patch_recovery_for_features(
    model,
    pairs: dict,
    *,
    spec,
    sae: TopKSAE,
    feature_ids: list[int] | None,
    clean_score: torch.Tensor,
    corrupt_score: torch.Tensor,
    chunk_size: int,
) -> dict:
    tensor_ids = None if feature_ids is None else torch.tensor(feature_ids, dtype=torch.long, device=pairs["clean_tokens"].device)
    logits = sae_feature_patch_logits(
        model,
        pairs["clean_tokens"],
        pairs["corrupt_tokens"],
        spec=spec,
        sae=sae,
        chunk_size=chunk_size,
        feature_ids=tensor_ids,
    )
    return patch_metrics(logits, pairs["clean_labels"], clean_score, corrupt_score)


@torch.no_grad()
def ablation_metrics_for_features(
    model,
    eval_batch: dict,
    *,
    spec,
    sae: TopKSAE,
    feature_ids: list[int],
    chunk_size: int,
) -> dict:
    ids = torch.tensor(feature_ids, dtype=torch.long, device=eval_batch["tokens"].device)
    logits = logits_with_sae_reconstruction(
        model,
        eval_batch["tokens"],
        spec=spec,
        sae=sae,
        chunk_size=chunk_size,
        ablate_features=ids,
    )
    return metrics_from_logits(logits, eval_batch["label_bits"])


def parse_counts(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress-test B4 SAE feature claims.")
    parser.add_argument("--data-dir", default="interp/data/generated/b4_l25_p2_n16777216")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sae-dir", required=True)
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sites", default="l1_resid_post_cls+final_hidden_cls+l1_attn_out_cls")
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-examples", type=int, default=8192)
    parser.add_argument("--prefix-pairs", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--feature-counts", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--random-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-input-boundary-radius", type=int, default=-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    set_seed(args.seed)
    device = resolve_device(args.device)
    loaded = load_transformer_checkpoint(args.checkpoint, device)
    model = loaded.model.eval()
    checkpoint_radius = infer_boundary_radius_from_checkpoint(loaded.checkpoint)
    boundary_radius = checkpoint_radius if int(args.model_input_boundary_radius) == -2 else int(args.model_input_boundary_radius)
    shard_paths = discover_b4_shards(args.data_dir, num_shards=args.num_shards, allow_partial=False)
    data_config = infer_data_config(shard_paths)
    simple_mats = simple_mats_z(device)
    sites = parse_site_list(args.sites)
    counts = parse_counts(args.feature_counts)
    sae_results = json.load(open(args.results_json))
    generator = torch.Generator(device=device).manual_seed(args.seed + 17)

    eval_loader = make_loader(
        shard_paths,
        split="val",
        batch_size=args.batch_size,
        seed=args.seed + 1,
        epoch=0,
        shuffle=False,
        max_examples=args.eval_examples,
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
    prefix_loader = make_loader(
        shard_paths,
        split="test",
        batch_size=args.batch_size,
        seed=args.seed + 2,
        epoch=0,
        shuffle=False,
        max_examples=max(args.prefix_pairs * 16, args.batch_size),
    )
    pairs = collect_prefix_fixed_pairs(
        prefix_loader,
        device=device,
        length=data_config["length"],
        absolute_depth=data_config["absolute_depth"],
        simple_mats=simple_mats,
        boundary_radius=boundary_radius,
        num_pairs=args.prefix_pairs,
    )
    clean_logits = model_logits(model, pairs["clean_tokens"], args.chunk_size)
    corrupt_logits = model_logits(model, pairs["corrupt_tokens"], args.chunk_size)
    clean_score = score_against_clean(clean_logits, pairs["clean_labels"])
    corrupt_score = score_against_clean(corrupt_logits, pairs["clean_labels"])

    out = {
        "config": vars(args),
        "effective_model_input_boundary_radius": boundary_radius,
        "clean": metrics_from_logits(clean_logits, pairs["clean_labels"]),
        "corrupt_scored_as_clean": patch_metrics(corrupt_logits, pairs["clean_labels"], clean_score, corrupt_score),
        "sites": {},
    }

    for spec in sites:
        site_result = sae_results["sites"][spec.key]
        sae = load_sae(Path(args.sae_dir) / f"{spec.key}_sae.pt", device)
        eval_acts = collect_site_activation(model, eval_batch["tokens"], spec, chunk_size=args.chunk_size)
        _, eval_sparse = sae(eval_acts)
        active = eval_sparse.gt(0).any(dim=0)
        eligible = torch.nonzero(active, as_tuple=False).flatten().tolist()
        if not eligible:
            eligible = list(range(sae.n_features))

        strategies = {
            "best_label": ordered_features(site_result, "best_label"),
            "descent_label": ordered_features(site_result, "descent_label"),
            "categorical": ordered_features(site_result, "categorical"),
            "binary": ordered_features(site_result, "binary"),
        }
        all_patch = patch_recovery_for_features(
            model,
            pairs,
            spec=spec,
            sae=sae,
            feature_ids=None,
            clean_score=clean_score,
            corrupt_score=corrupt_score,
            chunk_size=args.chunk_size,
        )
        permuted_logits = sae_patch_logits_permuted(
            model,
            pairs["clean_tokens"],
            pairs["corrupt_tokens"],
            spec=spec,
            sae=sae,
            chunk_size=args.chunk_size,
            feature_ids=None,
            generator=generator,
        )
        all_permuted = patch_metrics(permuted_logits, pairs["clean_labels"], clean_score, corrupt_score)

        curves = {}
        for strategy, ordered in strategies.items():
            rows = []
            for count in counts:
                features = ordered[:count]
                if not features:
                    continue
                metrics = patch_recovery_for_features(
                    model,
                    pairs,
                    spec=spec,
                    sae=sae,
                    feature_ids=features,
                    clean_score=clean_score,
                    corrupt_score=corrupt_score,
                    chunk_size=args.chunk_size,
                )
                ablate = ablation_metrics_for_features(
                    model,
                    eval_batch,
                    spec=spec,
                    sae=sae,
                    feature_ids=features,
                    chunk_size=args.chunk_size,
                )
                rows.append(
                    {
                        "count": len(features),
                        "patch_recovery": metrics["normalized_score_recovery"],
                        "patch_exact": metrics["exact_set_accuracy"],
                        "patch_bit": metrics["bit_accuracy"],
                        "ablate_exact": ablate["exact_set_accuracy"],
                        "ablate_bit": ablate["bit_accuracy"],
                    }
                )
            curves[strategy] = rows

        random_rows = []
        for count in counts:
            count = min(count, len(eligible))
            recoveries = []
            exacts = []
            ablate_exacts = []
            for _ in range(args.random_trials):
                perm = torch.randperm(len(eligible), generator=generator, device=device)[:count].cpu().tolist()
                features = [eligible[idx] for idx in perm]
                metrics = patch_recovery_for_features(
                    model,
                    pairs,
                    spec=spec,
                    sae=sae,
                    feature_ids=features,
                    clean_score=clean_score,
                    corrupt_score=corrupt_score,
                    chunk_size=args.chunk_size,
                )
                ablate = ablation_metrics_for_features(
                    model,
                    eval_batch,
                    spec=spec,
                    sae=sae,
                    feature_ids=features,
                    chunk_size=args.chunk_size,
                )
                recoveries.append(metrics["normalized_score_recovery"])
                exacts.append(metrics["exact_set_accuracy"])
                ablate_exacts.append(ablate["exact_set_accuracy"])
            random_rows.append(
                {
                    "count": count,
                    "patch_recovery": mean_std(recoveries),
                    "patch_exact": mean_std(exacts),
                    "ablate_exact": mean_std(ablate_exacts),
                }
            )

        out["sites"][spec.key] = {
            "all_features_patch": all_patch,
            "all_features_patch_permuted_clean_rows": all_permuted,
            "eligible_active_features": len(eligible),
            "strategy_curves": curves,
            "random_active_feature_controls": random_rows,
        }
        atomic_json_dump(out, out_path)
        print(
            f"site={spec.key} all={all_patch['normalized_score_recovery']:.3f} "
            f"perm={all_permuted['normalized_score_recovery']:.3f}",
            flush=True,
        )

    atomic_json_dump(out, out_path)


if __name__ == "__main__":
    main()
