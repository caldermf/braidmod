#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from braid_data import DataSetBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Burau/Garside supervised dataset. "
            "Each record contains a projectively normalized Burau tensor, "
            "the final Garside factor, and its right descent set."
        )
    )
    parser.add_argument("--output-path", required=True, help="Path to the output JSON file")
    parser.add_argument("--num-samples", type=int, required=True, help="Number of samples to generate")
    parser.add_argument("--length-min", type=int, required=True, help="Minimum Garside length")
    parser.add_argument("--length-max", type=int, required=True, help="Maximum Garside length")
    parser.add_argument("--p", type=int, default=5, help="Modulus p for the Burau representation")
    parser.add_argument("--D", type=int, required=True, help="Tensor depth D")
    parser.add_argument("--d-min", type=int, default=0, help="Minimum Delta power")
    parser.add_argument("--d-max", type=int, default=0, help="Maximum Delta power")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print a progress message every N generated samples",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.length_min <= 0 or args.length_max <= 0:
        raise ValueError("Lengths must be positive")
    if args.length_min > args.length_max:
        raise ValueError("--length-min must be <= --length-max")

    builder = DataSetBuilder(
        p=args.p,
        D=args.D,
        d_range=(args.d_min, args.d_max),
        seed=args.seed,
    )

    records = []
    for idx in range(args.num_samples):
        length = builder.rng.randint(args.length_min, args.length_max)
        records.append(builder.sample(length))
        if args.progress_every > 0 and (idx + 1) % args.progress_every == 0:
            print(f"generated={idx + 1}")

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_path": str(out_path),
                "num_samples": len(records),
                "length_min": args.length_min,
                "length_max": args.length_max,
                "p": args.p,
                "D": args.D,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
