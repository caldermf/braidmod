#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import torch

from garside_models import PERMUTATIONS_S4, build_model_from_config


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_tensor(args, expected_d: int):
    if args.dataset_path:
        records = json.loads(Path(args.dataset_path).read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("--dataset-path must point to a JSON list of records")
        if args.index < 0 or args.index >= len(records):
            raise IndexError(f"--index out of range (got {args.index}, dataset size {len(records)})")
        if "burau_tensor" not in records[args.index]:
            raise ValueError("Selected record does not contain 'burau_tensor'")
        tensor = records[args.index]["burau_tensor"]
        min_degree = int(records[args.index].get("burau_min_degree", 0))
        if "gnf_factors" in records[args.index]:
            garside_length = len(records[args.index]["gnf_factors"])
        else:
            garside_length = int(records[args.index].get("garside_length", records[args.index].get("gnf_length", 0)))
    else:
        payload = json.loads(Path(args.tensor_path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "burau_tensor" not in payload:
                raise ValueError("Tensor JSON dict must contain key 'burau_tensor'")
            tensor = payload["burau_tensor"]
            min_degree = int(payload.get("burau_min_degree", 0))
            if "gnf_factors" in payload:
                garside_length = len(payload["gnf_factors"])
            else:
                garside_length = int(payload.get("garside_length", payload.get("gnf_length", 0)))
        else:
            tensor = payload
            min_degree = 0
            garside_length = 0

    x = torch.tensor(tensor, dtype=torch.long)
    if x.ndim != 3 or tuple(x.shape[1:]) != (3, 3):
        raise ValueError(f"Expected tensor shape [D, 3, 3], got {tuple(x.shape)}")
    if x.shape[0] != expected_d:
        raise ValueError(f"Checkpoint expects D={expected_d}, got D={x.shape[0]}")
    if args.garside_length is not None:
        garside_length = int(args.garside_length)
    return x, min_degree, garside_length


def build_model(checkpoint: dict, device: torch.device):
    config = checkpoint.get("config", {})
    p = int(checkpoint["p"])
    d = int(checkpoint["D"])
    matrix_size = int(checkpoint.get("matrix_size", config.get("matrix_size", 3)))
    model = build_model_from_config(config, p=p, D=d, matrix_size=matrix_size).to(device)
    state = checkpoint.get("model_state")
    if state is None:
        raise ValueError("Checkpoint missing 'model_state'")
    model.load_state_dict(state)
    model.eval()
    return model


def confusion_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Normalized entropy confusion score in [0, 1].
    0 => very certain, 1 => maximally confused (uniform over classes).
    Expects logits shape [B, C] or [C].
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2:
        raise ValueError(f"Expected logits shape [B, C] or [C], got {tuple(logits.shape)}")
    n_classes = logits.shape[-1]
    if n_classes <= 1:
        raise ValueError("Need at least 2 classes for confusion score")

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy / math.log(float(n_classes))


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained Garside model checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, cuda:0, ...")
    parser.add_argument("--topk", type=int, default=5, help="Number of top permutation predictions to show")
    parser.add_argument("--desc-threshold", type=float, default=0.5, help="Threshold for right-descent predictions")
    parser.add_argument("--dataset-path", help="Dataset JSON path; uses --index record")
    parser.add_argument("--index", type=int, default=0, help="Record index when using --dataset-path")
    parser.add_argument("--tensor-path", help="JSON file containing a [D,3,3] tensor or {\"burau_tensor\": ...}")
    parser.add_argument("--garside-length", type=int, help="Optional override for the Garside length input")
    args = parser.parse_args()

    if bool(args.dataset_path) == bool(args.tensor_path):
        raise ValueError("Provide exactly one of --dataset-path or --tensor-path")

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(checkpoint, device)
    config = checkpoint.get("config", {})

    expected_d = int(checkpoint["D"])
    p = int(checkpoint["p"])
    x, min_degree, garside_length = load_tensor(args, expected_d=expected_d)
    if x.min().item() < 0 or x.max().item() >= p:
        raise ValueError(f"Input values must be in [0, {p - 1}]")
    if bool(config.get("use_garside_length", False)) and garside_length <= 0:
        raise ValueError(
            "Checkpoint expects a positive Garside length input; provide a dataset record with "
            "'gnf_factors'/'garside_length' or pass --garside-length."
        )

    with torch.no_grad():
        min_degree_tensor = torch.tensor([min_degree], dtype=torch.float32, device=device)
        garside_length_tensor = torch.tensor([garside_length], dtype=torch.float32, device=device)
        factor_logits, desc_logits = model(
            x.unsqueeze(0).to(device),
            min_degree=min_degree_tensor,
            garside_length=garside_length_tensor,
        )
        confusion = confusion_score_from_logits(factor_logits)[0]
        probs = torch.softmax(factor_logits[0], dim=-1).cpu()
        topk = min(args.topk, probs.shape[0])
        top_probs, top_ids = torch.topk(probs, k=topk)

        perm_classes = checkpoint.get("perm_classes")
        if perm_classes is None:
            perm_classes = [list(p_) for p_ in PERMUTATIONS_S4]

        top_predictions = []
        for class_id, prob in zip(top_ids.tolist(), top_probs.tolist()):
            top_predictions.append(
                {
                    "class_id": class_id,
                    "perm": perm_classes[class_id],
                    "prob": round(float(prob), 6),
                }
            )

        result = {
            "checkpoint": args.checkpoint,
            "device": str(device),
            "D": expected_d,
            "p": p,
            "burau_min_degree": min_degree,
            "garside_length": garside_length,
            "confusion_score": round(float(confusion.item()), 6),
            "confidence_score": round(float(1.0 - confusion.item()), 6),
            "top_predictions": top_predictions,
        }

        if desc_logits is not None:
            desc_probs = torch.sigmoid(desc_logits[0]).cpu().tolist()
            result["right_descent_probs"] = [round(float(v), 6) for v in desc_probs]
            result["right_descent_pred"] = [
                idx for idx, v in enumerate(desc_probs) if float(v) >= args.desc_threshold
            ]

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
