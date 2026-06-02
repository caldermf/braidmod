#!/usr/bin/env python3
"""Generate random B_4 Burau/descent dataset shards.

The B_3 corpus was small enough to enumerate.  In B_4, length-25 normal forms
are far too numerous, so this generator samples a large reproducible corpus of
valid positive Garside normal forms.  The model input is the reduced Burau
matrix over F_2[v], stored as packed absolute-degree coefficient bitsets for a
3 x 3 matrix.  Factor ids are stored only for auditing and analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from itertools import permutations
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from braid_data import GNF, GarsideFactor, burau_mod_p_polynomial_matrix, gnf_to_braid_word  # noqa: E402


N = 4
P = 2
MATRIX_SIZE = N - 1
BITS_PER_CHUNK = 62


def _desc_mask(desc: set[int]) -> int:
    mask = 0
    for idx in desc:
        mask |= 1 << idx
    return mask


IDENTITY_PERM = GNF.identity_perm(N)
DELTA_PERM = GNF.delta_perm(N)
PROPER_FACTOR_PERMS = [
    perm for perm in permutations(range(N)) if perm not in (IDENTITY_PERM, DELTA_PERM)
]
LEFT_DESC_MASK = torch.tensor(
    [_desc_mask(GarsideFactor(perm).left_descent()) for perm in PROPER_FACTOR_PERMS],
    dtype=torch.long,
)
RIGHT_DESC_MASK = torch.tensor(
    [_desc_mask(GarsideFactor(perm).right_descent()) for perm in PROPER_FACTOR_PERMS],
    dtype=torch.long,
)


def _build_transition_candidates() -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[list[int]] = []
    counts: list[int] = []
    for required_left_mask in range(1 << (N - 1)):
        valid = [
            factor_id
            for factor_id, right_mask in enumerate(RIGHT_DESC_MASK.tolist())
            if (right_mask & required_left_mask) == required_left_mask
        ]
        if not valid:
            # The all-three left-descent mask belongs only to Delta, which is
            # excluded from the proper-simple factors. Keep a padded dummy row
            # so the table is rectangular; sampling below will fail if this
            # supposedly unreachable mask is ever requested.
            valid = [0]
            counts.append(0)
            rows.append(valid)
            continue
        rows.append(valid)
        counts.append(len(valid))
    max_count = max(counts)
    table = torch.zeros(len(rows), max_count, dtype=torch.long)
    for row_idx, valid in enumerate(rows):
        table[row_idx, : len(valid)] = torch.tensor(valid, dtype=torch.long)
    return table, torch.tensor(counts, dtype=torch.long)


PREV_FACTOR_CANDIDATES, PREV_FACTOR_COUNTS = _build_transition_candidates()


def _simple_matrix_for_perm(perm: tuple[int, ...]) -> dict[tuple[int, int, int], int]:
    word = [idx + 1 for idx in GarsideFactor(perm).artin_factors()]
    mat = burau_mod_p_polynomial_matrix(word, p=P, n=N)
    out: dict[tuple[int, int, int], int] = {}
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            for exp, coeff in mat[i][j].items():
                coeff_mod = coeff % P
                if coeff_mod:
                    if exp < 0:
                        raise RuntimeError("Positive simple factor unexpectedly produced a negative degree")
                    out[(exp, i, j)] = coeff_mod
    return out


def _build_simple_mats() -> torch.Tensor:
    sparse = [_simple_matrix_for_perm(perm) for perm in PROPER_FACTOR_PERMS]
    max_degree = max((exp for mat in sparse for exp, _, _ in mat), default=0)
    mats = torch.zeros(
        len(PROPER_FACTOR_PERMS),
        max_degree + 1,
        MATRIX_SIZE,
        MATRIX_SIZE,
        dtype=torch.bool,
    )
    for factor_id, mat in enumerate(sparse):
        for (exp, i, j), coeff in mat.items():
            if coeff % P:
                mats[factor_id, exp, i, j] = True
    return mats


SIMPLE_MATS = _build_simple_mats()


def absolute_depth_for_length(length: int) -> int:
    return (SIMPLE_MATS.shape[1] - 1) * int(length) + 1


def num_chunks_for_depth(depth: int) -> int:
    return (int(depth) + BITS_PER_CHUNK - 1) // BITS_PER_CHUNK


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    if device.type == "cuda":
        generator = torch.Generator(device=device)
    else:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def sample_factor_ids(batch_size: int, length: int, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    if length <= 0:
        raise ValueError("length must be positive")
    factor_count = len(PROPER_FACTOR_PERMS)
    out = torch.empty(batch_size, length, dtype=torch.long, device=device)
    current = torch.randint(0, factor_count, (batch_size,), device=device, generator=generator)
    out[:, length - 1] = current

    left_masks = LEFT_DESC_MASK.to(device=device)
    candidates = PREV_FACTOR_CANDIDATES.to(device=device)
    counts = PREV_FACTOR_COUNTS.to(device=device)
    for pos in range(length - 2, -1, -1):
        required = left_masks[current]
        count = counts[required]
        if bool((count == 0).any().item()):
            raise RuntimeError("Encountered an unreachable left-descent mask while sampling")
        rank = (torch.rand(batch_size, device=device, generator=generator) * count.to(torch.float32)).to(torch.long)
        current = candidates[required, rank]
        out[:, pos] = current
    return out


def right_multiply_dense(mat: torch.Tensor, factor_ids: torch.Tensor, simple_mats: torch.Tensor) -> torch.Tensor:
    batch_size, depth = mat.shape[:2]
    coeff = simple_mats[factor_ids]
    out = torch.zeros_like(mat)
    for shift in range(simple_mats.shape[1]):
        if shift >= depth:
            break
        for k in range(MATRIX_SIZE):
            src = mat[:, : depth - shift, :, k]
            for j in range(MATRIX_SIZE):
                active = coeff[:, shift, k, j]
                if bool(active.any().item()):
                    out[:, shift:, :, j] ^= src & active.view(batch_size, 1, 1)
    return out


def dense_burau_for_factor_ids(factor_ids: torch.Tensor, length: int, absolute_depth: int) -> torch.Tensor:
    device = factor_ids.device
    batch_size = factor_ids.shape[0]
    mat = torch.zeros(batch_size, absolute_depth, MATRIX_SIZE, MATRIX_SIZE, dtype=torch.bool, device=device)
    for i in range(MATRIX_SIZE):
        mat[:, 0, i, i] = True
    simple_mats = SIMPLE_MATS.to(device=device)
    for pos in range(length):
        mat = right_multiply_dense(mat, factor_ids[:, pos], simple_mats)
    return mat


def degree_bounds(mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    occupied = mat.any(dim=(-1, -2))
    any_occupied = occupied.any(dim=1)
    if not bool(any_occupied.all().item()):
        raise RuntimeError("Encountered an all-zero Burau matrix")
    min_degree = occupied.to(torch.long).argmax(dim=1)
    max_degree = mat.shape[1] - 1 - occupied.flip(dims=[1]).to(torch.long).argmax(dim=1)
    return min_degree.to(torch.int16), max_degree.to(torch.int16)


def pack_absolute_matrix(mat: torch.Tensor, absolute_depth: int) -> torch.Tensor:
    chunks = num_chunks_for_depth(absolute_depth)
    packed = torch.empty(mat.shape[0], MATRIX_SIZE * MATRIX_SIZE, chunks, dtype=torch.long, device=mat.device)
    entry_idx = 0
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            entry_bits = mat[:, :, i, j].to(torch.long)
            for chunk_idx in range(chunks):
                start = chunk_idx * BITS_PER_CHUNK
                width = min(BITS_PER_CHUNK, absolute_depth - start)
                weights = (1 << torch.arange(width, dtype=torch.long, device=mat.device)).view(1, width)
                packed[:, entry_idx, chunk_idx] = (entry_bits[:, start : start + width] * weights).sum(dim=1)
            entry_idx += 1
    return packed


def label_bits_from_final_factor(final_factor_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    masks = RIGHT_DESC_MASK.to(device=final_factor_id.device)[final_factor_id.long()]
    bits = torch.stack([((masks >> idx) & 1) for idx in range(N - 1)], dim=1).to(torch.uint8)
    return masks.to(torch.uint8), bits


def generate_batch(
    *,
    batch_size: int,
    length: int,
    absolute_depth: int,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    factor_ids = sample_factor_ids(batch_size, length=length, device=device, generator=generator)
    mat = dense_burau_for_factor_ids(factor_ids, length=length, absolute_depth=absolute_depth)
    min_degree, max_degree = degree_bounds(mat)
    matrix_bits = pack_absolute_matrix(mat, absolute_depth=absolute_depth)
    final_factor_id = factor_ids[:, -1].to(torch.uint8)
    descent_mask, label_bits = label_bits_from_final_factor(final_factor_id)
    return {
        "matrix_bits": matrix_bits.cpu(),
        "burau_min_degree": min_degree.cpu(),
        "burau_max_degree": max_degree.cpu(),
        "factor_ids": factor_ids.to(torch.uint8).cpu(),
        "final_factor_id": final_factor_id.cpu(),
        "descent_mask": descent_mask.cpu(),
        "label_bits": label_bits.cpu(),
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _pack_python_poly_matrix(poly_mat: list[list[dict[int, int]]], absolute_depth: int) -> list[list[int]]:
    chunks = num_chunks_for_depth(absolute_depth)
    packed = [[0 for _ in range(chunks)] for _ in range(MATRIX_SIZE * MATRIX_SIZE)]
    entry_idx = 0
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            for exp, coeff in poly_mat[i][j].items():
                if coeff % P:
                    if exp < 0 or exp >= absolute_depth:
                        raise RuntimeError(f"Degree {exp} outside absolute depth {absolute_depth}")
                    chunk_idx = exp // BITS_PER_CHUNK
                    bit_idx = exp % BITS_PER_CHUNK
                    packed[entry_idx][chunk_idx] |= 1 << bit_idx
            entry_idx += 1
    return packed


def slow_reference(factor_ids: list[int], absolute_depth: int) -> dict[str, object]:
    perms = [PROPER_FACTOR_PERMS[idx] for idx in factor_ids]
    gnf = GNF(0, perms)
    poly_mat = burau_mod_p_polynomial_matrix(gnf_to_braid_word(gnf), p=P, n=N)
    exps = [exp for row in poly_mat for entry in row for exp, coeff in entry.items() if coeff % P]
    if not exps:
        raise RuntimeError("Reference Burau matrix is zero")
    final_factor_id = factor_ids[-1]
    descent_mask = int(RIGHT_DESC_MASK[final_factor_id].item())
    label_bits = [(descent_mask >> idx) & 1 for idx in range(N - 1)]
    return {
        "matrix_bits": _pack_python_poly_matrix(poly_mat, absolute_depth=absolute_depth),
        "burau_min_degree": min(exps),
        "burau_max_degree": max(exps),
        "final_factor_id": final_factor_id,
        "descent_mask": descent_mask,
        "label_bits": label_bits,
    }


def run_spot_checks(
    shard: dict[str, torch.Tensor],
    *,
    count: int,
    absolute_depth: int,
    num_checks: int,
    seed: int,
) -> None:
    if num_checks <= 0:
        return
    rng = random.Random(seed)
    offsets = sorted(rng.sample(range(count), k=min(num_checks, count)))
    for offset in offsets:
        factor_ids = shard["factor_ids"][offset].tolist()
        ref = slow_reference(factor_ids, absolute_depth=absolute_depth)
        if shard["matrix_bits"][offset].tolist() != ref["matrix_bits"]:
            raise AssertionError(f"matrix_bits mismatch at row offset={offset}")
        for key in ("burau_min_degree", "burau_max_degree", "final_factor_id", "descent_mask"):
            if int(shard[key][offset].item()) != ref[key]:
                raise AssertionError(f"{key} mismatch at row offset={offset}")
        if shard["label_bits"][offset].tolist() != ref["label_bits"]:
            raise AssertionError(f"label_bits mismatch at row offset={offset}")


def iter_counts(count: int, batch_size: int) -> Iterable[int]:
    emitted = 0
    while emitted < count:
        take = min(batch_size, count - emitted)
        emitted += take
        yield take


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate random compact B_4 Burau/descent dataset shards.")
    parser.add_argument("--out-dir", default="interp/data/generated/b4_l25_p2_n16777216")
    parser.add_argument("--length", type=int, default=25)
    parser.add_argument("--num-samples", type=int, default=16_777_216)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--spot-check", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.length <= 0:
        raise ValueError("--length must be positive")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must lie in [0, --num-shards)")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    shard_size = math.ceil(args.num_samples / args.num_shards)
    start = args.shard_index * shard_size
    count = min(shard_size, args.num_samples - start)
    if count <= 0:
        raise ValueError("Shard is empty; reduce --num-shards or choose a lower --shard-index")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{args.shard_index:04d}_of_{args.num_shards:04d}.pt"
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} already exists; pass --overwrite to replace it")

    device = resolve_device(args.device)
    generator = make_generator(device, seed=args.seed + 1_000_003 * args.shard_index)
    absolute_depth = absolute_depth_for_length(args.length)
    num_chunks = num_chunks_for_depth(absolute_depth)

    matrix_bits = torch.empty(count, MATRIX_SIZE * MATRIX_SIZE, num_chunks, dtype=torch.long)
    min_degrees = torch.empty(count, dtype=torch.int16)
    max_degrees = torch.empty(count, dtype=torch.int16)
    factor_ids = torch.empty(count, args.length, dtype=torch.uint8)
    final_factor_ids = torch.empty(count, dtype=torch.uint8)
    descent_masks = torch.empty(count, dtype=torch.uint8)
    label_bits = torch.empty(count, N - 1, dtype=torch.uint8)

    offset = 0
    for batch_count in iter_counts(count, args.batch_size):
        batch = generate_batch(
            batch_size=batch_count,
            length=args.length,
            absolute_depth=absolute_depth,
            device=device,
            generator=generator,
        )
        matrix_bits[offset : offset + batch_count] = batch["matrix_bits"]
        min_degrees[offset : offset + batch_count] = batch["burau_min_degree"]
        max_degrees[offset : offset + batch_count] = batch["burau_max_degree"]
        factor_ids[offset : offset + batch_count] = batch["factor_ids"]
        final_factor_ids[offset : offset + batch_count] = batch["final_factor_id"]
        descent_masks[offset : offset + batch_count] = batch["descent_mask"]
        label_bits[offset : offset + batch_count] = batch["label_bits"]
        offset += batch_count

    shard = {
        "matrix_bits": matrix_bits,
        "burau_min_degree": min_degrees,
        "burau_max_degree": max_degrees,
        "factor_ids": factor_ids,
        "final_factor_id": final_factor_ids,
        "descent_mask": descent_masks,
        "label_bits": label_bits,
    }
    run_spot_checks(
        shard,
        count=count,
        absolute_depth=absolute_depth,
        num_checks=args.spot_check,
        seed=args.seed + args.shard_index,
    )

    metadata = {
        "group": "B_4",
        "length": args.length,
        "p": P,
        "absolute_depth": absolute_depth,
        "bits_per_chunk": BITS_PER_CHUNK,
        "num_chunks": num_chunks,
        "matrix_size": MATRIX_SIZE,
        "num_factor_ids": len(PROPER_FACTOR_PERMS),
        "sample_id_start": start,
        "sample_id_count": count,
        "total_examples": args.num_samples,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "factor_perms": [list(x) for x in PROPER_FACTOR_PERMS],
        "left_descent_masks": LEFT_DESC_MASK.tolist(),
        "right_descent_masks": RIGHT_DESC_MASK.tolist(),
        "label_convention": {
            "label_bits[:,0]": "s_1 in final right descent set",
            "label_bits[:,1]": "s_2 in final right descent set",
            "label_bits[:,2]": "s_3 in final right descent set",
        },
        "sampling": "final proper simple factor uniform; preceding factors sampled uniformly from proper simples satisfying R(prev) superset L(next)",
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
