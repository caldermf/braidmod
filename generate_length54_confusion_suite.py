#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from braid_data import DataSetBuilder, GNF
from plot_prefix_confusion import build_progression, save_plot


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sample_random_gnf_via_backtracking(builder: DataSetBuilder, length: int) -> GNF:
    memo: Dict[Tuple[int, Tuple[int, ...]], bool] = {}

    def build_suffix(pos: int, required_left: Tuple[int, ...]) -> Optional[List]:
        key = (pos, required_left)
        if key in memo and not memo[key]:
            return None

        candidates = builder._valid_factor_candidates(
            required_left_subset=set(required_left),
            exclude_delta=(pos == 0),
            exclude_identity=(pos == length - 1),
        )
        builder.rng.shuffle(candidates)

        if pos == 0:
            if not candidates:
                memo[key] = False
                return None
            return [candidates[0]]

        for candidate in candidates:
            left_req = tuple(sorted(candidate.left_descent()))
            prefix = build_suffix(pos - 1, left_req)
            if prefix is not None:
                return prefix + [candidate]

        memo[key] = False
        return None

    factors = build_suffix(length - 1, tuple())
    if factors is None:
        raise RuntimeError(f"Failed to sample valid GNF of length {length} via backtracking")
    return GNF(0, factors)


def emit_case(checkpoint: str, device: str, out_dir: Path, name: str, gnf_payload: dict, topk: int):
    gnf_path = out_dir / f"{name}_gnf.json"
    json_path = out_dir / f"{name}_confusion.json"
    entropy_png_path = out_dir / f"{name}_entropy_confusion.png"
    xent_png_path = out_dir / f"{name}_target_cross_entropy.png"

    write_json(gnf_path, gnf_payload)
    result = build_progression(
        checkpoint_path=checkpoint,
        device_arg=device,
        d=int(gnf_payload["d"]),
        factors=[tuple(factor) for factor in gnf_payload["gnf_factors"]],
        topk=topk,
        truncate_overflow=False,
    )
    if "source_word_path" in gnf_payload:
        result["source_word_path"] = gnf_payload["source_word_path"]
    if "artin_word" in gnf_payload:
        result["artin_word"] = gnf_payload["artin_word"]
    write_json(json_path, result)
    title_base = name.replace("_", " ").title()
    save_plot(
        result["progression"],
        str(entropy_png_path),
        f"{title_base} Entropy Confusion",
        metric_key="entropy_confusion_score",
        y_label="Entropy Confusion",
    )
    save_plot(
        result["progression"],
        str(xent_png_path),
        f"{title_base} Target Cross-Entropy",
        metric_key="target_cross_entropy",
        y_label="Target Cross-Entropy",
    )


def main():
    parser = argparse.ArgumentParser(description="Generate the GWY kernel element plus five random length-54 confusion plots.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, ...")
    parser.add_argument("--geordie-gnf-path", required=True, help="Path to the saved GWY kernel-element GNF JSON")
    parser.add_argument("--out-dir", required=True, help="Directory for the six charts and JSON outputs")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random GNF generation")
    parser.add_argument("--topk", type=int, default=3, help="Top-k predictions to store in JSON outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    geordie_payload = json.loads(Path(args.geordie_gnf_path).read_text(encoding="utf-8"))
    emit_case(args.checkpoint, args.device, out_dir, "geordie_kernel", geordie_payload, args.topk)

    builder = DataSetBuilder(p=5, D=96, n=4, d_range=(0, 0), seed=args.seed)
    manifest = []
    manifest.append({"name": "geordie_kernel", "gnf_path": str(out_dir / "geordie_kernel_gnf.json")})

    for idx in range(1, 6):
        gnf = sample_random_gnf_via_backtracking(builder, 54)
        name = f"random_{idx:02d}"
        gnf_payload = {
            "generator": "DataSetBuilder backtracking sampler",
            "seed": args.seed,
            "sample_index": idx,
            "d": int(gnf.d),
            "num_factors": len(gnf.factors),
            "gnf_factors": [list(f.perm) for f in gnf.factors],
        }
        emit_case(args.checkpoint, args.device, out_dir, name, gnf_payload, args.topk)
        manifest.append({"name": name, "gnf_path": str(out_dir / f"{name}_gnf.json")})

    write_json(
        out_dir / "manifest.json",
        {"cases": manifest, "seed": args.seed, "length": 54, "generator": "DataSetBuilder backtracking sampler"},
    )


if __name__ == "__main__":
    main()
