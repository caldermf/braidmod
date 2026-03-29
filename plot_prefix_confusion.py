#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from braid_data import positive_word_to_garside_normal_form
from predict_garside_mlp import PERMUTATIONS_S4, build_model, confusion_score_from_logits, resolve_device
from track_confusion_prefix import prefix_braid_word, tensor_with_optional_truncation


PERM_TO_ID = {perm: idx for idx, perm in enumerate(PERMUTATIONS_S4)}


def parse_artin_word(text: str) -> List[int]:
    cleaned = text.replace("\n", " ").replace("\t", " ")
    parts = [part.strip() for part in cleaned.split(",")]
    word = [int(part) for part in parts if part]
    if not word:
        raise ValueError("Artin word must contain at least one generator")
    if any(gen <= 0 for gen in word):
        raise ValueError("Only positive Artin words are supported")
    return word


def load_artin_word(path: str) -> List[int]:
    return parse_artin_word(Path(path).read_text(encoding="utf-8"))


def load_gnf(path: str):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GNF JSON must be an object")
    if "d" not in payload or "gnf_factors" not in payload:
        raise ValueError("GNF JSON must contain keys 'd' and 'gnf_factors'")
    d = int(payload["d"])
    factors = [tuple(int(v) for v in factor) for factor in payload["gnf_factors"]]
    if not factors:
        raise ValueError("GNF JSON must contain at least one factor")
    return d, factors, payload


def build_progression(
    checkpoint_path: str,
    device_arg: str,
    d: int,
    factors: Sequence[Tuple[int, int, int, int]],
    topk: int,
    truncate_overflow: bool,
):
    device = resolve_device(device_arg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint, device)

    expected_d = int(checkpoint["D"])
    p = int(checkpoint["p"])
    perm_classes = checkpoint.get("perm_classes")

    progression = []
    with torch.no_grad():
        for k in range(1, len(factors) + 1):
            word = prefix_braid_word(d, factors, k)
            tensor, min_degree, overflow_terms = tensor_with_optional_truncation(
                word, p=p, depth=expected_d, truncate_overflow=truncate_overflow
            )
            x = torch.tensor(tensor, dtype=torch.long)
            min_degree_tensor = torch.tensor([min_degree], dtype=torch.float32, device=device)
            garside_length_tensor = torch.tensor([k], dtype=torch.float32, device=device)
            factor_logits, _ = model(
                x.unsqueeze(0).to(device),
                min_degree=min_degree_tensor,
                garside_length=garside_length_tensor,
            )
            entropy_confusion = confusion_score_from_logits(factor_logits)[0]
            target_id = PERM_TO_ID[tuple(factors[k - 1])]
            target_tensor = torch.tensor([target_id], dtype=torch.long, device=device)
            target_cross_entropy = F.cross_entropy(factor_logits, target_tensor)
            probs = torch.softmax(factor_logits[0], dim=-1).cpu()

            top_probs, top_ids = torch.topk(probs, k=min(topk, probs.shape[0]))
            progression.append(
                {
                    "prefix_len": k,
                    "factor_added": list(factors[k - 1]),
                    "confusion_score": float(entropy_confusion.item()),
                    "entropy_confusion_score": float(entropy_confusion.item()),
                    "confidence_score": float(1.0 - entropy_confusion.item()),
                    "target_cross_entropy": float(target_cross_entropy.item()),
                    "ground_truth_class_id": int(target_id),
                    "burau_min_degree": int(min_degree),
                    "truncated_terms": int(overflow_terms),
                    "top_predictions": [
                        {
                            "class_id": int(class_id),
                            "perm": perm_classes[int(class_id)],
                            "prob": float(prob),
                        }
                        for class_id, prob in zip(top_ids.tolist(), top_probs.tolist())
                    ],
                }
            )

    return {
        "checkpoint": checkpoint_path,
        "device": str(device),
        "D": expected_d,
        "p": p,
        "metrics": {
            "confusion_score": "Normalized entropy of the factor logits",
            "entropy_confusion_score": "Normalized entropy of the factor logits",
            "target_cross_entropy": "Cross-entropy loss against the actual final factor of the current prefix",
        },
        "d": d,
        "truncate_overflow": bool(truncate_overflow),
        "num_factors": len(factors),
        "factors": [list(factor) for factor in factors],
        "progression": progression,
    }


def save_plot(
    progression: Sequence[dict],
    out_path: str,
    title: str,
    metric_key: str = "confusion_score",
    y_label: str = "Confusion",
):
    prefix_lens = [item["prefix_len"] for item in progression]
    metric_values = [item[metric_key] for item in progression]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(prefix_lens, metric_values, marker="o", markersize=3, linewidth=1.8, color="#1f4e79")
    ax.set_xlabel("Garside Prefix Length")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xlim(1, max(prefix_lens))
    if metric_key in {"confusion_score", "entropy_confusion_score"}:
        ax.set_ylim(0.0, 1.0)
    else:
        ymax = max(metric_values) if metric_values else 1.0
        ax.set_ylim(0.0, max(1.0, 1.05 * ymax))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Factorize a positive Artin word, save its Garside normal form, and plot model confusion by Garside prefix."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--word-path", help="Path to a comma-separated positive Artin word")
    parser.add_argument("--gnf-path", help="Path to an existing Garside normal form JSON")
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, cuda:0, ...")
    parser.add_argument("--topk", type=int, default=3, help="Number of top permutation predictions to store")
    parser.add_argument("--truncate-overflow", action="store_true", help="Ignore Burau terms outside [0, D-1]")
    parser.add_argument("--gnf-out", required=True, help="Output path for the saved Garside normal form JSON")
    parser.add_argument("--json-out", required=True, help="Output path for the prefix confusion JSON")
    parser.add_argument("--plot-out", required=True, help="Output path for the confusion plot PNG")
    parser.add_argument("--title", default="Model Confusion Along Garside Prefixes", help="Plot title")
    args = parser.parse_args()

    if bool(args.word_path) == bool(args.gnf_path):
        raise ValueError("Provide exactly one of --word-path or --gnf-path")

    if args.word_path:
        artin_word = load_artin_word(args.word_path)
        d, factors = positive_word_to_garside_normal_form(artin_word, n=4)
        gnf_payload = {
            "source_word_path": args.word_path,
            "artin_word": artin_word,
            "d": d,
            "num_factors": len(factors),
            "gnf_factors": [list(factor) for factor in factors],
        }
    else:
        d, factors, gnf_payload = load_gnf(args.gnf_path)
        gnf_payload = dict(gnf_payload)
        gnf_payload["num_factors"] = len(factors)

    Path(args.gnf_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.gnf_out).write_text(json.dumps(gnf_payload, indent=2), encoding="utf-8")

    result = build_progression(
        checkpoint_path=args.checkpoint,
        device_arg=args.device,
        d=d,
        factors=factors,
        topk=args.topk,
        truncate_overflow=args.truncate_overflow,
    )
    if args.word_path:
        result["source_word_path"] = args.word_path
        result["artin_word"] = artin_word
    else:
        result["source_gnf_path"] = args.gnf_path
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    Path(args.plot_out).parent.mkdir(parents=True, exist_ok=True)
    save_plot(result["progression"], args.plot_out, args.title)

    print(json.dumps(
        {
            "gnf_out": args.gnf_out,
            "json_out": args.json_out,
            "plot_out": args.plot_out,
            "num_factors": len(factors),
            "d": d,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
