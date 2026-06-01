#!/usr/bin/env python3
"""Generate compact exhaustive B_3 Burau/descent dataset shards.

The first production target is B_3, Garside length 25, p=2.  We enumerate all
proper-simple normal forms exactly once by sample id:

    2 bits for the first factor, then one transition bit for each suffix.

The generated matrix is the reduced Burau image over F_2[v, v^-1], stored as
four 51-bit int64 bitsets after projective degree normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from braid_data import (  # noqa: E402
    GNF,
    GarsideFactor,
    burau_mod_p_polynomial_matrix,
    gnf_to_braid_word,
)


PROPER_FACTOR_PERMS = [
    (0, 2, 1),  # s_2, right descent {s_2}
    (1, 0, 2),  # s_1, right descent {s_1}
    (1, 2, 0),  # s_1 s_2, right descent {s_2}
    (2, 0, 1),  # s_2 s_1, right descent {s_1}
]

RIGHT_DESC_LABEL = torch.tensor([1, 0, 1, 0], dtype=torch.uint8)
NEXT_FACTOR = torch.tensor(
    [
        [0, 3],  # R(s_2) can be followed by left descent {s_2}
        [1, 2],  # R(s_1) can be followed by left descent {s_1}
        [0, 3],
        [1, 2],
    ],
    dtype=torch.long,
)

SIMPLE_MATS = torch.zeros(4, 4, 2, 2, dtype=torch.bool)
SIMPLE_MATS[0, 0, 0, 0] = True
SIMPLE_MATS[0, 1, 1, 0] = True
SIMPLE_MATS[0, 2, 1, 1] = True
SIMPLE_MATS[1, 2, 0, 0] = True
SIMPLE_MATS[1, 1, 0, 1] = True
SIMPLE_MATS[1, 0, 1, 1] = True
SIMPLE_MATS[2, 3, 0, 1] = True
SIMPLE_MATS[2, 1, 1, 0] = True
SIMPLE_MATS[2, 2, 1, 1] = True
SIMPLE_MATS[3, 2, 0, 0] = True
SIMPLE_MATS[3, 1, 0, 1] = True
SIMPLE_MATS[3, 3, 1, 0] = True


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def total_examples(length: int) -> int:
    if length <= 0:
        raise ValueError("length must be positive")
    return 4 * (1 << (length - 1))


def decode_factor_ids(sample_ids: torch.Tensor, length: int, device: torch.device) -> torch.Tensor:
    if sample_ids.dtype != torch.long:
        sample_ids = sample_ids.to(torch.long)
    next_factor = NEXT_FACTOR.to(device=device)
    out = torch.empty(sample_ids.shape[0], length, dtype=torch.long, device=device)
    current = torch.bitwise_right_shift(sample_ids, length - 1) & 3
    out[:, 0] = current
    for pos in range(1, length):
        bit = torch.bitwise_right_shift(sample_ids, length - 1 - pos) & 1
        current = next_factor[current, bit]
        out[:, pos] = current
    return out


def right_multiply_dense(mat: torch.Tensor, factor_ids: torch.Tensor, simple_mats: torch.Tensor) -> torch.Tensor:
    batch_size, depth = mat.shape[:2]
    coeff = simple_mats[factor_ids]
    out = torch.zeros_like(mat)
    for shift in range(simple_mats.shape[1]):
        if shift >= depth:
            break
        for k in range(2):
            src = mat[:, : depth - shift, :, k]
            for j in range(2):
                active = coeff[:, shift, k, j]
                if bool(active.any().item()):
                    out[:, shift:, :, j] ^= src & active.view(batch_size, 1, 1)
    return out


def dense_burau_for_factor_ids(factor_ids: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = factor_ids.device
    batch_size = factor_ids.shape[0]
    abs_depth = 3 * length + 1
    mat = torch.zeros(batch_size, abs_depth, 2, 2, dtype=torch.bool, device=device)
    mat[:, 0, 0, 0] = True
    mat[:, 0, 1, 1] = True
    simple_mats = SIMPLE_MATS.to(device=device)
    for pos in range(length):
        mat = right_multiply_dense(mat, factor_ids[:, pos], simple_mats)

    occupied = mat.any(dim=(-1, -2))
    min_degree = occupied.to(torch.long).argmax(dim=1)
    max_degree = abs_depth - 1 - occupied.flip(dims=[1]).to(torch.long).argmax(dim=1)
    width = max_degree - min_degree + 1
    expected_width = 2 * length + 1
    if not bool(width.eq(expected_width).all().item()):
        bad = torch.nonzero(width.ne(expected_width), as_tuple=False).flatten()[:10].tolist()
        raise RuntimeError(f"Unexpected projective width; expected {expected_width}, bad rows {bad}")

    rel = torch.arange(expected_width, device=device).view(1, expected_width)
    idx = min_degree.view(-1, 1) + rel
    batch_idx = torch.arange(batch_size, device=device).view(-1, 1)
    normalized = mat[batch_idx, idx]
    return normalized, min_degree.to(torch.int16)


def pack_normalized_matrix(normalized: torch.Tensor) -> torch.Tensor:
    depth = normalized.shape[1]
    if depth > 62:
        raise ValueError("int64 bit packing requires depth <= 62")
    weights = (1 << torch.arange(depth, dtype=torch.long, device=normalized.device)).view(1, depth)
    entries = [
        (normalized[:, :, 0, 0].to(torch.long) * weights).sum(dim=1),
        (normalized[:, :, 0, 1].to(torch.long) * weights).sum(dim=1),
        (normalized[:, :, 1, 0].to(torch.long) * weights).sum(dim=1),
        (normalized[:, :, 1, 1].to(torch.long) * weights).sum(dim=1),
    ]
    return torch.stack(entries, dim=1).to(torch.long)


def generate_batch(sample_ids: torch.Tensor, length: int, device: torch.device) -> dict[str, torch.Tensor]:
    sample_ids = sample_ids.to(device=device, dtype=torch.long)
    factor_ids = decode_factor_ids(sample_ids, length=length, device=device)
    normalized, min_degree = dense_burau_for_factor_ids(factor_ids, length=length)
    matrix_bits = pack_normalized_matrix(normalized)
    final_factor_id = factor_ids[:, -1].to(torch.uint8)
    labels = RIGHT_DESC_LABEL.to(device=device)[final_factor_id.long()]
    return {
        "matrix_bits": matrix_bits.cpu(),
        "burau_min_degree": min_degree.cpu(),
        "final_factor_id": final_factor_id.cpu(),
        "label": labels.cpu(),
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def slow_reference(sample_id: int, length: int) -> dict[str, object]:
    factor_ids = decode_factor_ids(torch.tensor([sample_id], dtype=torch.long), length, torch.device("cpu"))[0].tolist()
    factors = [PROPER_FACTOR_PERMS[idx] for idx in factor_ids]
    gnf = GNF(0, factors)
    mat = burau_mod_p_polynomial_matrix(gnf_to_braid_word(gnf), p=2, n=3)
    exps = [exp for row in mat for entry in row for exp in entry]
    min_degree = min(exps)
    max_degree = max(exps)
    width = max_degree - min_degree + 1
    bits = []
    for i in range(2):
        for j in range(2):
            value = 0
            for exp, coeff in mat[i][j].items():
                if coeff % 2:
                    value |= 1 << (exp - min_degree)
            bits.append(value)
    label = int(RIGHT_DESC_LABEL[factor_ids[-1]].item())
    return {
        "matrix_bits": bits,
        "burau_min_degree": min_degree,
        "final_factor_id": factor_ids[-1],
        "label": label,
        "width": width,
    }


def run_spot_checks(shard: dict[str, torch.Tensor], start: int, count: int, length: int, num_checks: int, seed: int) -> None:
    if num_checks <= 0:
        return
    rng = random.Random(seed)
    offsets = sorted(rng.sample(range(count), k=min(num_checks, count)))
    for offset in offsets:
        ref = slow_reference(start + offset, length)
        got_bits = shard["matrix_bits"][offset].tolist()
        if got_bits != ref["matrix_bits"]:
            raise AssertionError(f"matrix_bits mismatch at sample_id={start + offset}: {got_bits} != {ref['matrix_bits']}")
        if int(shard["burau_min_degree"][offset].item()) != ref["burau_min_degree"]:
            raise AssertionError(f"burau_min_degree mismatch at sample_id={start + offset}")
        if int(shard["final_factor_id"][offset].item()) != ref["final_factor_id"]:
            raise AssertionError(f"final_factor_id mismatch at sample_id={start + offset}")
        if int(shard["label"][offset].item()) != ref["label"]:
            raise AssertionError(f"label mismatch at sample_id={start + offset}")
        if ref["width"] != 2 * length + 1:
            raise AssertionError(f"reference width mismatch at sample_id={start + offset}: {ref['width']}")


def iter_ranges(start: int, count: int, batch_size: int) -> Iterable[tuple[int, int]]:
    end = start + count
    cursor = start
    while cursor < end:
        next_cursor = min(cursor + batch_size, end)
        yield cursor, next_cursor - cursor
        cursor = next_cursor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate compact B_3 Burau/descent dataset shards.")
    parser.add_argument("--out-dir", default="interp/data/generated/b3_l25_p2_full")
    parser.add_argument("--length", type=int, default=25)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--spot-check", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.length <= 0:
        raise ValueError("--length must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must lie in [0, --num-shards)")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    total = total_examples(args.length)
    shard_size = math.ceil(total / args.num_shards)
    start = args.shard_index * shard_size
    count = min(shard_size, total - start)
    if count <= 0:
        raise ValueError("Shard is empty; reduce --num-shards or choose a lower --shard-index")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{args.shard_index:04d}_of_{args.num_shards:04d}.pt"
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} already exists; pass --overwrite to replace it")

    device = resolve_device(args.device)
    matrix_bits = torch.empty(count, 4, dtype=torch.long)
    min_degrees = torch.empty(count, dtype=torch.int16)
    final_factor_ids = torch.empty(count, dtype=torch.uint8)
    labels = torch.empty(count, dtype=torch.uint8)

    offset = 0
    for batch_start, batch_count in iter_ranges(start, count, args.batch_size):
        sample_ids = torch.arange(batch_start, batch_start + batch_count, dtype=torch.long)
        batch = generate_batch(sample_ids, length=args.length, device=device)
        matrix_bits[offset : offset + batch_count] = batch["matrix_bits"]
        min_degrees[offset : offset + batch_count] = batch["burau_min_degree"]
        final_factor_ids[offset : offset + batch_count] = batch["final_factor_id"]
        labels[offset : offset + batch_count] = batch["label"]
        offset += batch_count

    shard = {
        "matrix_bits": matrix_bits,
        "burau_min_degree": min_degrees,
        "final_factor_id": final_factor_ids,
        "label": labels,
    }
    run_spot_checks(
        shard=shard,
        start=start,
        count=count,
        length=args.length,
        num_checks=args.spot_check,
        seed=args.seed + args.shard_index,
    )

    metadata = {
        "group": "B_3",
        "length": args.length,
        "p": 2,
        "D": 2 * args.length + 1,
        "matrix_size": 2,
        "sample_id_start": start,
        "sample_id_count": count,
        "total_examples": total,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "factor_perms": [list(x) for x in PROPER_FACTOR_PERMS],
        "label_convention": {"0": "{s_1}", "1": "{s_2}"},
        "device": str(device),
        "spot_checks": min(args.spot_check, count),
        "checksums": {name: tensor_sha256(value) for name, value in shard.items()},
    }
    payload = dict(shard)
    payload["metadata"] = metadata
    tmp_path = out_path.with_name(f"{out_path.name}.tmp.{args.shard_index}")
    torch.save(payload, tmp_path)
    tmp_path.replace(out_path)
    print(json.dumps({"out_path": str(out_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
