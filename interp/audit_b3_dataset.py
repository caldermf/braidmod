#!/usr/bin/env python3
"""Audit generated B_3 Burau/descent dataset shards."""

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

from interp.generate_b3_dataset import slow_reference, total_examples  # noqa: E402


def tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def load_shard(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"matrix_bits", "burau_min_degree", "final_factor_id", "label", "metadata"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path} missing keys: {sorted(missing)}")
    return payload


def check_spot(payload: dict, path: Path, checks_per_shard: int, seed: int) -> None:
    if checks_per_shard <= 0:
        return
    meta = payload["metadata"]
    count = int(meta["sample_id_count"])
    start = int(meta["sample_id_start"])
    length = int(meta["length"])
    rng = random.Random(seed + int(meta["shard_index"]))
    offsets = sorted(rng.sample(range(count), k=min(checks_per_shard, count)))
    for offset in offsets:
        ref = slow_reference(start + offset, length)
        got_bits = payload["matrix_bits"][offset].tolist()
        if got_bits != ref["matrix_bits"]:
            raise AssertionError(f"{path}: matrix_bits mismatch at sample_id={start + offset}")
        if int(payload["burau_min_degree"][offset].item()) != ref["burau_min_degree"]:
            raise AssertionError(f"{path}: burau_min_degree mismatch at sample_id={start + offset}")
        if int(payload["final_factor_id"][offset].item()) != ref["final_factor_id"]:
            raise AssertionError(f"{path}: final_factor_id mismatch at sample_id={start + offset}")
        if int(payload["label"][offset].item()) != ref["label"]:
            raise AssertionError(f"{path}: label mismatch at sample_id={start + offset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit compact B_3 dataset shards.")
    parser.add_argument("--data-dir", default="interp/data/generated/b3_l25_p2_full")
    parser.add_argument("--length", type=int, default=25)
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

    expected_total = total_examples(args.length)
    expected_D = 2 * args.length + 1
    total_count = 0
    label_counts = torch.zeros(2, dtype=torch.long)
    factor_counts = torch.zeros(4, dtype=torch.long)
    seen_starts = []

    high_mask = ~((1 << expected_D) - 1)
    for path in shard_paths:
        payload = load_shard(path)
        meta = payload["metadata"]
        if int(meta["length"]) != args.length or int(meta["D"]) != expected_D:
            raise AssertionError(f"{path}: metadata length/D mismatch")
        count = int(meta["sample_id_count"])
        if payload["matrix_bits"].shape != (count, 4):
            raise AssertionError(f"{path}: matrix_bits shape mismatch")
        for key in ("burau_min_degree", "final_factor_id", "label"):
            if payload[key].shape != (count,):
                raise AssertionError(f"{path}: {key} shape mismatch")
        if bool((payload["matrix_bits"] & high_mask).any().item()):
            raise AssertionError(f"{path}: matrix bitsets contain bits above D={expected_D}")
        for name, expected in meta.get("checksums", {}).items():
            actual = tensor_sha256(payload[name])
            if actual != expected:
                raise AssertionError(f"{path}: checksum mismatch for {name}")
        check_spot(payload, path, args.spot_check_per_shard, args.seed)
        total_count += count
        label_counts += torch.bincount(payload["label"].long(), minlength=2)
        factor_counts += torch.bincount(payload["final_factor_id"].long(), minlength=4)
        seen_starts.append((int(meta["sample_id_start"]), count))

    seen_starts.sort()
    cursor = 0
    for start, count in seen_starts:
        if start != cursor:
            raise AssertionError(f"Shard coverage gap or overlap: expected start {cursor}, got {start}")
        cursor += count
    if total_count != expected_total or cursor != expected_total:
        raise AssertionError(f"Expected {expected_total} examples, saw total_count={total_count}, cursor={cursor}")
    expected_per_label = expected_total // 2
    if label_counts.tolist() != [expected_per_label, expected_per_label]:
        raise AssertionError(f"Unexpected label counts: {label_counts.tolist()}")

    print(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "num_shards": len(shard_paths),
                "total_count": total_count,
                "label_counts": label_counts.tolist(),
                "factor_counts": factor_counts.tolist(),
                "spot_check_per_shard": args.spot_check_per_shard,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
