#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from braid_data import GarsideFactor, GNF, burau_mod_p_polynomial_matrix, burau_mod_p_tensor
from predict_garside_mlp import build_model, confusion_score_from_logits, resolve_device


def _parse_factors_json(text: str) -> List[Tuple[int, int, int, int]]:
    payload = json.loads(text)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Factors must be a non-empty JSON list")

    factors = []
    for idx, factor in enumerate(payload):
        if not isinstance(factor, list) or len(factor) != 4:
            raise ValueError(f"Factor at index {idx} must be a list of length 4")
        factors.append(tuple(int(v) for v in factor))
    return factors


def load_input_factors(args) -> List[Tuple[int, int, int, int]]:
    if bool(args.factors_json) == bool(args.factors_path):
        raise ValueError("Provide exactly one of --factors-json or --factors-path")

    if args.factors_json:
        return _parse_factors_json(args.factors_json)

    payload = json.loads(Path(args.factors_path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not payload:
            raise ValueError("--factors-path points to an empty list")
        if all(isinstance(x, list) and len(x) == 4 for x in payload):
            pass
        else:
            if args.index < 0 or args.index >= len(payload):
                raise IndexError(f"--index out of range (got {args.index}, dataset size {len(payload)})")
            record = payload[args.index]
            if not isinstance(record, dict) or "gnf_factors" not in record:
                raise ValueError("Dataset record must contain key 'gnf_factors'")
            payload = record["gnf_factors"]
    elif isinstance(payload, dict):
        if "gnf_factors" not in payload:
            raise ValueError("Factors JSON dict must contain key 'gnf_factors'")
        payload = payload["gnf_factors"]
    else:
        raise ValueError("--factors-path must contain a JSON list or dict")
    return _parse_factors_json(json.dumps(payload))


def prefix_braid_word(d: int, factors: Sequence[Tuple[int, int, int, int]], k: int) -> List[int]:
    n = 4
    delta = GarsideFactor(GNF.delta_perm(n)).artin_factors()
    word: List[int] = []

    if d >= 0:
        for _ in range(d):
            word.extend([idx + 1 for idx in delta])
    else:
        inv_delta = [-(idx + 1) for idx in reversed(delta)]
        for _ in range(-d):
            word.extend(inv_delta)

    for factor in factors[:k]:
        artin = GarsideFactor(factor).artin_factors()
        word.extend([idx + 1 for idx in artin])

    return word


def tensor_with_optional_truncation(word: Sequence[int], p: int, depth: int, truncate_overflow: bool):
    if not truncate_overflow:
        return burau_mod_p_tensor(word, p=p, D=depth, n=4), 0

    poly_mat = burau_mod_p_polynomial_matrix(word, p=p, n=4)
    tensor = [[[0 for _ in range(3)] for _ in range(3)] for _ in range(depth)]
    overflow_terms = 0
    for i in range(3):
        for j in range(3):
            for exp, coeff in poly_mat[i][j].items():
                if exp < 0 or exp >= depth:
                    overflow_terms += 1
                    continue
                tensor[exp][i][j] = coeff % p
    return tensor, overflow_terms


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Track confusion score as Garside factors are appended. "
            "Factors are permutations in S4, e.g. [[1,0,2,3],[1,2,0,3]]."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, cuda:0, ...")
    parser.add_argument("--d", type=int, default=0, help="Garside Delta exponent d")
    parser.add_argument("--topk", type=int, default=3, help="Top-k predictions to show at each prefix")
    parser.add_argument("--index", type=int, default=0, help="Dataset record index when --factors-path is a list of records")
    parser.add_argument("--factors-json", help='JSON string of factors, e.g. "[[1,0,2,3],[1,2,0,3]]"')
    parser.add_argument("--factors-path", help="Path to JSON file containing list or {'gnf_factors': [...]} ")
    parser.add_argument(
        "--truncate-overflow",
        action="store_true",
        help="If set, ignore Burau polynomial terms with exponent outside [0, D-1] instead of failing",
    )
    args = parser.parse_args()

    factors = load_input_factors(args)

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(checkpoint, device)

    expected_d = int(checkpoint["D"])
    p = int(checkpoint["p"])
    perm_classes = checkpoint.get("perm_classes")

    if perm_classes is None:
        from train_garside_mlp import PERMUTATIONS_S4

        perm_classes = [list(p_) for p_ in PERMUTATIONS_S4]

    progression = []
    with torch.no_grad():
        for k in range(1, len(factors) + 1):
            word = prefix_braid_word(args.d, factors, k)
            tensor, overflow_terms = tensor_with_optional_truncation(
                word, p=p, depth=expected_d, truncate_overflow=args.truncate_overflow
            )
            x = torch.tensor(tensor, dtype=torch.long)

            factor_logits, _ = model(x.unsqueeze(0).to(device))
            confusion = confusion_score_from_logits(factor_logits)[0]
            probs = torch.softmax(factor_logits[0], dim=-1).cpu()

            topk = min(args.topk, probs.shape[0])
            top_probs, top_ids = torch.topk(probs, k=topk)
            top_predictions = [
                {
                    "class_id": class_id,
                    "perm": perm_classes[class_id],
                    "prob": round(float(prob), 6),
                }
                for class_id, prob in zip(top_ids.tolist(), top_probs.tolist())
            ]

            progression.append(
                {
                    "prefix_len": k,
                    "factor_added": list(factors[k - 1]),
                    "confusion_score": round(float(confusion.item()), 6),
                    "confidence_score": round(float(1.0 - confusion.item()), 6),
                    "truncated_terms": int(overflow_terms),
                    "top_predictions": top_predictions,
                }
            )

    result = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "D": expected_d,
        "p": p,
        "d": args.d,
        "truncate_overflow": bool(args.truncate_overflow),
        "num_factors": len(factors),
        "factors": [list(f) for f in factors],
        "progression": progression,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
