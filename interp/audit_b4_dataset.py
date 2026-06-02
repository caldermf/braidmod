#!/usr/bin/env python3
"""Audit generated B_4 Burau/descent dataset shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.generate_b4_dataset import (  # noqa: E402
    BITS_PER_CHUNK,
    LEFT_DESC_MASK,
    MATRIX_SIZE,
    RIGHT_DESC_MASK,
    slow_reference,
)


def tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def load_shard(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "matrix_bits",
        "burau_min_degree",
        "burau_max_degree",
        "factor_ids",
        "final_factor_id",
        "descent_mask",
        "label_bits",
        "metadata",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path} missing keys: {sorted(missing)}")
    return payload


def check_transitions(payload: dict, path: Path) -> None:
    factor_ids = payload["factor_ids"].long()
    left_masks = LEFT_DESC_MASK
    right_masks = RIGHT_DESC_MASK
    for pos in range(factor_ids.shape[1] - 1):
        left = factor_ids[:, pos]
        right = factor_ids[:, pos + 1]
        ok = (right_masks[left] & left_masks[right]) == left_masks[right]
        if not bool(ok.all().item()):
            bad = int(torch.nonzero(~ok, as_tuple=False)[0].item())
            raise AssertionError(f"{path}: invalid transition at row={bad}, pos={pos}")


def check_labels(payload: dict, path: Path) -> None:
    final_factor_id = payload["final_factor_id"].long()
    expected_mask = RIGHT_DESC_MASK[final_factor_id].to(torch.uint8)
    if not bool(payload["descent_mask"].eq(expected_mask).all().item()):
        raise AssertionError(f"{path}: descent_mask mismatch")
    expected_bits = torch.stack([((expected_mask.long() >> idx) & 1) for idx in range(3)], dim=1).to(torch.uint8)
    if not bool(payload["label_bits"].eq(expected_bits).all().item()):
        raise AssertionError(f"{path}: label_bits mismatch")


def check_spot(payload: dict, path: Path, checks_per_shard: int, seed: int) -> None:
    if checks_per_shard <= 0:
        return
    meta = payload["metadata"]
    count = int(meta["sample_id_count"])
    absolute_depth = int(meta["absolute_depth"])
    rng = random.Random(seed + int(meta["shard_index"]))
    offsets = sorted(rng.sample(range(count), k=min(checks_per_shard, count)))
    for offset in offsets:
        factor_ids = payload["factor_ids"][offset].tolist()
        ref = slow_reference(factor_ids, absolute_depth=absolute_depth)
        if payload["matrix_bits"][offset].tolist() != ref["matrix_bits"]:
            raise AssertionError(f"{path}: matrix_bits mismatch at row offset={offset}")
        for key in ("burau_min_degree", "burau_max_degree", "final_factor_id", "descent_mask"):
            if int(payload[key][offset].item()) != ref[key]:
                raise AssertionError(f"{path}: {key} mismatch at row offset={offset}")
        if payload["label_bits"][offset].tolist() != ref["label_bits"]:
            raise AssertionError(f"{path}: label_bits mismatch at row offset={offset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit compact random B_4 dataset shards.")
    parser.add_argument("--data-dir", default="interp/data/generated/b4_l25_p2_n16777216")
    parser.add_argument("--length", type=int, default=25)
    parser.add_argument("--num-samples", type=int, default=16_777_216)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--spot-check-per-shard", type=int, default=4)
    parser.add_argument("--seed", type=int, default=98765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    shard_paths = sorted(data_dir.glob(f"shard_*_of_{args.num_shards:04d}.pt"))
    if len(shard_paths) != args.num_shards:
        raise FileNotFoundError(f"Expected {args.num_shards} shards, found {len(shard_paths)} in {data_dir}")

    total_count = 0
    label_counts = torch.zeros(3, 2, dtype=torch.long)
    descent_mask_counts = torch.zeros(8, dtype=torch.long)
    factor_counts = torch.zeros(len(RIGHT_DESC_MASK), dtype=torch.long)
    min_degree_min = None
    min_degree_max = None
    max_degree_min = None
    max_degree_max = None
    seen_starts = []

    for path in shard_paths:
        payload = load_shard(path)
        meta = payload["metadata"]
        count = int(meta["sample_id_count"])
        absolute_depth = int(meta["absolute_depth"])
        num_chunks = int(meta["num_chunks"])
        if int(meta["length"]) != args.length:
            raise AssertionError(f"{path}: metadata length mismatch")
        if int(meta["bits_per_chunk"]) != BITS_PER_CHUNK:
            raise AssertionError(f"{path}: bits_per_chunk mismatch")
        if payload["matrix_bits"].shape != (count, MATRIX_SIZE * MATRIX_SIZE, num_chunks):
            raise AssertionError(f"{path}: matrix_bits shape mismatch")
        if payload["factor_ids"].shape != (count, args.length):
            raise AssertionError(f"{path}: factor_ids shape mismatch")
        for key in ("burau_min_degree", "burau_max_degree", "final_factor_id", "descent_mask"):
            if payload[key].shape != (count,):
                raise AssertionError(f"{path}: {key} shape mismatch")
        if payload["label_bits"].shape != (count, 3):
            raise AssertionError(f"{path}: label_bits shape mismatch")

        last_width = absolute_depth - BITS_PER_CHUNK * (num_chunks - 1)
        high_mask = ~((1 << last_width) - 1)
        if bool((payload["matrix_bits"][:, :, -1] & high_mask).any().item()):
            raise AssertionError(f"{path}: matrix bitsets contain bits above absolute_depth={absolute_depth}")
        for name, expected in meta.get("checksums", {}).items():
            actual = tensor_sha256(payload[name])
            if actual != expected:
                raise AssertionError(f"{path}: checksum mismatch for {name}")

        check_transitions(payload, path)
        check_labels(payload, path)
        check_spot(payload, path, args.spot_check_per_shard, args.seed)

        total_count += count
        for idx in range(3):
            label_counts[idx] += torch.bincount(payload["label_bits"][:, idx].long(), minlength=2)
        descent_mask_counts += torch.bincount(payload["descent_mask"].long(), minlength=8)
        factor_counts += torch.bincount(payload["final_factor_id"].long(), minlength=len(RIGHT_DESC_MASK))
        min_d = payload["burau_min_degree"].long()
        max_d = payload["burau_max_degree"].long()
        min_degree_min = int(min_d.min().item()) if min_degree_min is None else min(min_degree_min, int(min_d.min().item()))
        min_degree_max = int(min_d.max().item()) if min_degree_max is None else max(min_degree_max, int(min_d.max().item()))
        max_degree_min = int(max_d.min().item()) if max_degree_min is None else min(max_degree_min, int(max_d.min().item()))
        max_degree_max = int(max_d.max().item()) if max_degree_max is None else max(max_degree_max, int(max_d.max().item()))
        seen_starts.append((int(meta["sample_id_start"]), count))

    seen_starts.sort()
    cursor = 0
    for start, count in seen_starts:
        if start != cursor:
            raise AssertionError(f"Shard coverage gap or overlap: expected start {cursor}, got {start}")
        cursor += count
    if total_count != args.num_samples or cursor != args.num_samples:
        raise AssertionError(f"Expected {args.num_samples} examples, saw total_count={total_count}, cursor={cursor}")

    print(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "num_shards": len(shard_paths),
                "total_count": total_count,
                "label_counts_by_generator": label_counts.tolist(),
                "descent_mask_counts": descent_mask_counts.tolist(),
                "final_factor_counts": factor_counts.tolist(),
                "min_degree_range": [min_degree_min, min_degree_max],
                "max_degree_range": [max_degree_min, max_degree_max],
                "spot_check_per_shard": args.spot_check_per_shard,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
