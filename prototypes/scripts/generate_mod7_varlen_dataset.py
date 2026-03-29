import argparse
import json
import random
import time
from itertools import permutations
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from braid_data import GNF, GarsideFactor, burau_mod_p_projective_tensor_from_gnf


class FastGNFSampler:
    def __init__(self, n: int, d_range: tuple[int, int], seed: int):
        if n != 4:
            raise ValueError("Only n=4 is supported by the current Burau tensor pipeline.")
        d_min, d_max = d_range
        if d_min > d_max:
            raise ValueError("d_range must satisfy min <= max")

        self.n = n
        self.d_range = (d_min, d_max)
        self.rng = random.Random(seed)

        self.perms = list(permutations(range(n)))
        self.factors = [GarsideFactor(p) for p in self.perms]
        self.identity_idx = self.perms.index(GNF.identity_perm(n))
        self.delta_idx = self.perms.index(GNF.delta_perm(n))

        left = [f.left_descent() for f in self.factors]
        right = [f.right_descent() for f in self.factors]
        self.next_idx = [
            [j for j in range(len(self.perms)) if right[i].issuperset(left[j])]
            for i in range(len(self.perms))
        ]
        self._suffix_counts_cache: dict[int, list[list[int]]] = {}

    def _get_suffix_counts(self, L: int) -> list[list[int]]:
        if L in self._suffix_counts_cache:
            return self._suffix_counts_cache[L]
        if L <= 0:
            raise ValueError("L must be positive")

        P = len(self.perms)
        counts = [[0] * P for _ in range(L)]
        for i in range(P):
            counts[L - 1][i] = 0 if i == self.identity_idx else 1

        for pos in range(L - 2, -1, -1):
            nxt = counts[pos + 1]
            cur = counts[pos]
            for i in range(P):
                total = 0
                for j in self.next_idx[i]:
                    total += nxt[j]
                cur[i] = total

        self._suffix_counts_cache[L] = counts
        return counts

    def _weighted_choice(self, items: list[int], weights: list[int]) -> int:
        total = sum(weights)
        if total <= 0:
            raise RuntimeError("No valid weighted choices available")
        r = self.rng.randrange(total)
        acc = 0
        for item, w in zip(items, weights):
            acc += w
            if r < acc:
                return item
        return items[-1]

    def random_gnf(self, L: int) -> GNF:
        counts = self._get_suffix_counts(L)

        first_candidates = [i for i in range(len(self.perms)) if i != self.delta_idx and counts[0][i] > 0]
        first_weights = [counts[0][i] for i in first_candidates]
        first = self._weighted_choice(first_candidates, first_weights)

        seq = [first]
        for pos in range(0, L - 1):
            candidates = []
            weights = []
            for j in self.next_idx[seq[-1]]:
                w = counts[pos + 1][j]
                if w > 0:
                    candidates.append(j)
                    weights.append(w)
            seq.append(self._weighted_choice(candidates, weights))

        d = self.rng.randint(self.d_range[0], self.d_range[1])
        factor_perms = [self.perms[i] for i in seq]
        return GNF(d, factor_perms)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a random Garside/Burau dataset with mixed braid lengths. "
            "Defaults match mod-7, N=200000, lengths in [5,100]."
        )
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to output JSON dataset file. Defaults to a name derived from the resolved tensor depth D.",
    )
    parser.add_argument("--num-samples", type=int, default=200000, help="Number of records to generate.")
    parser.add_argument("--length-min", type=int, default=5, help="Minimum Garside length (inclusive).")
    parser.add_argument("--length-max", type=int, default=100, help="Maximum Garside length (inclusive).")
    parser.add_argument("--p", type=int, default=7, help="Modulus p for Burau tensor entries.")
    parser.add_argument(
        "--D",
        type=int,
        default=None,
        help=(
            "Tensor depth D for the projectively normalized burau_tensor (shape D x 3 x 3). "
            "Defaults to a conservative 4*(length_max + max(0, d_max))+1."
        ),
    )
    parser.add_argument("--n", type=int, default=4, help="Braid index n (currently only n=4 is supported).")
    parser.add_argument("--d-min", type=int, default=0, help="Minimum Delta exponent d (inclusive).")
    parser.add_argument("--d-max", type=int, default=0, help="Maximum Delta exponent d (inclusive).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--length-mode",
        choices=("iid_uniform", "balanced_uniform"),
        default="iid_uniform",
        help=(
            "How to sample lengths in [length_min, length_max]. "
            "'iid_uniform' draws each sample independently; "
            "'balanced_uniform' precomputes a shuffled schedule with per-length counts differing by at most 1."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=1000, help="Progress print frequency.")
    parser.add_argument(
        "--heartbeat-secs",
        type=float,
        default=10.0,
        help="Emit time-based heartbeat progress every N seconds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.length_min <= 0:
        raise ValueError("--length-min must be positive")
    if args.length_min > args.length_max:
        raise ValueError("--length-min must be <= --length-max")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if args.heartbeat_secs <= 0:
        raise ValueError("--heartbeat-secs must be positive")

    tensor_depth = args.D if args.D is not None else 4 * (args.length_max + max(0, args.d_max)) + 1
    output_path = args.output_path
    if output_path is None:
        output_path = (
            f"data/burau_gnf_L{args.length_min}to{args.length_max}_p{args.p}_D{tensor_depth}_N{args.num_samples}.json"
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sampler = FastGNFSampler(n=args.n, d_range=(args.d_min, args.d_max), seed=args.seed)
    length_rng = random.Random(args.seed + 1)
    lengths = list(range(args.length_min, args.length_max + 1))

    if args.length_mode == "balanced_uniform":
        base = args.num_samples // len(lengths)
        remainder = args.num_samples % len(lengths)
        length_schedule = []
        for idx, L in enumerate(lengths):
            copies = base + (1 if idx < remainder else 0)
            length_schedule.extend([L] * copies)
        length_rng.shuffle(length_schedule)
    else:
        length_schedule = None

    started = time.time()
    last_heartbeat = started
    depth_overflows = 0

    with out_path.open("w", encoding="utf-8") as f:
        f.write("[\n")
        for idx in range(args.num_samples):
            retries_for_depth = 0
            while True:
                if length_schedule is None:
                    L = length_rng.randint(args.length_min, args.length_max)
                else:
                    L = length_schedule[idx]

                now = time.time()
                if now - last_heartbeat >= args.heartbeat_secs:
                    elapsed = now - started
                    rate = idx / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[heartbeat] accepted={idx}/{args.num_samples} "
                        f"current_L={L} depth_overflows={depth_overflows} "
                        f"retries_for_current={retries_for_depth} "
                        f"elapsed={elapsed:.1f}s rate={rate:.2f} samples/s"
                    )
                    last_heartbeat = now

                gnf = sampler.random_gnf(L)
                try:
                    tensor, min_degree = burau_mod_p_projective_tensor_from_gnf(gnf, p=args.p, D=tensor_depth)
                    break
                except ValueError as err:
                    msg = str(err)
                    if "Tensor depth D=" in msg:
                        depth_overflows += 1
                        retries_for_depth += 1
                        # Resample; this keeps fixed tensor depth while avoiding hard failure.
                        continue
                    raise

            final_factor = gnf.factors[-1]
            rec = {
                "burau_tensor": tensor,
                "burau_min_degree": min_degree,
                "final_factor_perm": list(final_factor.perm),
                "final_factor_right_descent": sorted(final_factor.right_descent()),
                "gnf_d": gnf.d,
                "gnf_factors": [list(x.perm) for x in gnf.factors],
            }

            if idx > 0:
                f.write(",\n")
            json.dump(rec, f, separators=(",", ":"))

            if (idx + 1) % args.progress_every == 0 or idx + 1 == args.num_samples:
                elapsed = time.time() - started
                rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{idx + 1}/{args.num_samples}] L={L} depth_overflows={depth_overflows} "
                    f"elapsed={elapsed:.1f}s rate={rate:.2f} samples/s"
                )

        f.write("\n]\n")

    elapsed = time.time() - started
    print(f"Wrote {args.num_samples} records to {out_path}")
    print(f"Total time: {elapsed:.1f}s ({args.num_samples / elapsed:.2f} samples/s)")


if __name__ == "__main__":
    main()
