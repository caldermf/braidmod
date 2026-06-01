"""Data loading and tokenization for the B_3 Burau/descent corpus."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import IterableDataset, get_worker_info


SPLIT_MODULUS = 10_000
SPLIT_RANGES = {
    "train": (0, 9_800),
    "val": (9_800, 9_900),
    "test": (9_900, 10_000),
}


def absolute_depth_for_length(length: int) -> int:
    return 3 * int(length) + 1


def relative_depth_for_length(length: int) -> int:
    return 2 * int(length) + 1


def discover_shards(data_dir: str | Path, num_shards: int | None = None, allow_partial: bool = False) -> list[Path]:
    data_dir = Path(data_dir)
    suffix = f"_of_{num_shards:04d}.pt" if num_shards is not None else ".pt"
    paths = sorted(path for path in data_dir.glob("shard_*_of_*.pt") if path.name.endswith(suffix))
    if num_shards is not None and len(paths) != num_shards and not allow_partial:
        raise FileNotFoundError(f"Expected {num_shards} shards in {data_dir}, found {len(paths)}")
    if not paths:
        raise FileNotFoundError(f"No shard files found in {data_dir}")
    return paths


def split_bucket(sample_ids: torch.Tensor) -> torch.Tensor:
    sample_ids = sample_ids.to(torch.long)
    return (sample_ids * 1_103_515_245 + 12_345).remainder(SPLIT_MODULUS)


def split_mask(sample_ids: torch.Tensor, split: str) -> torch.Tensor:
    if split not in SPLIT_RANGES:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(SPLIT_RANGES)}")
    lo, hi = SPLIT_RANGES[split]
    bucket = split_bucket(sample_ids)
    return (bucket >= lo) & (bucket < hi)


def matrix_bits_to_absolute_tokens(
    matrix_bits: torch.Tensor,
    min_degree: torch.Tensor,
    *,
    length: int,
    absolute_depth: int | None = None,
) -> torch.Tensor:
    """Convert packed normalized 2x2 bitsets to absolute-degree 16-way tokens."""
    if matrix_bits.ndim != 2 or matrix_bits.shape[1] != 4:
        raise ValueError(f"matrix_bits must have shape [B, 4], got {tuple(matrix_bits.shape)}")
    relative_depth = relative_depth_for_length(length)
    absolute_depth = absolute_depth_for_length(length) if absolute_depth is None else int(absolute_depth)
    device = matrix_bits.device
    batch_size = matrix_bits.shape[0]

    rel = torch.arange(relative_depth, device=device, dtype=torch.long)
    matrix_bits = matrix_bits.to(torch.long)
    rel_tokens = torch.zeros(batch_size, relative_depth, dtype=torch.long, device=device)
    for entry_idx in range(4):
        entry_bits = torch.bitwise_right_shift(matrix_bits[:, entry_idx].unsqueeze(1), rel) & 1
        rel_tokens |= entry_bits << entry_idx

    min_degree = min_degree.to(device=device, dtype=torch.long)
    abs_idx = min_degree.unsqueeze(1) + rel.unsqueeze(0)
    if bool((abs_idx < 0).any().item()) or bool((abs_idx >= absolute_depth).any().item()):
        bad_min = int(min_degree.min().item())
        bad_max = int(min_degree.max().item())
        raise ValueError(
            f"Absolute degree out of range for depth {absolute_depth}; min_degree range {bad_min}..{bad_max}"
        )

    tokens = torch.zeros(batch_size, absolute_depth, dtype=torch.long, device=device)
    tokens.scatter_(dim=1, index=abs_idx, src=rel_tokens)
    return tokens


def load_shard(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


class B3ShardBatchIterable(IterableDataset):
    """Yield already-batched tensors from bit-packed B_3 shard files."""

    def __init__(
        self,
        shard_paths: Iterable[str | Path],
        *,
        split: str,
        length: int,
        batch_size: int,
        seed: int,
        epoch: int = 0,
        shuffle_shards: bool = False,
        shuffle_rows: bool = False,
        max_examples: int = 0,
    ):
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if split not in SPLIT_RANGES:
            raise ValueError(f"Unknown split {split!r}; expected one of {sorted(SPLIT_RANGES)}")
        self.shard_paths = [Path(path) for path in shard_paths]
        self.split = split
        self.length = int(length)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_rows = bool(shuffle_rows)
        self.max_examples = int(max_examples)

    def _worker_paths(self) -> list[Path]:
        paths = list(self.shard_paths)
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        if self.shuffle_shards:
            rng = random.Random(self.seed + 10_000 * self.epoch + worker_id)
            rng.shuffle(paths)
        return paths[worker_id::num_workers]

    def __iter__(self):
        emitted = 0
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        torch_rng = torch.Generator().manual_seed(self.seed + 1_000_003 * self.epoch + 97 * worker_id)

        for path in self._worker_paths():
            payload = load_shard(path)
            meta = payload["metadata"]
            count = int(meta["sample_id_count"])
            start = int(meta["sample_id_start"])
            sample_ids = torch.arange(start, start + count, dtype=torch.long)
            rows = torch.nonzero(split_mask(sample_ids, self.split), as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            if self.shuffle_rows:
                rows = rows[torch.randperm(rows.numel(), generator=torch_rng)]

            cursor = 0
            while cursor < rows.numel():
                if self.max_examples > 0 and emitted >= self.max_examples:
                    return
                take = min(self.batch_size, rows.numel() - cursor)
                if self.max_examples > 0:
                    take = min(take, self.max_examples - emitted)
                batch_rows = rows[cursor : cursor + take]
                cursor += take
                emitted += take

                matrix_bits = payload["matrix_bits"][batch_rows]
                min_degree = payload["burau_min_degree"][batch_rows]
                yield {
                    "tokens": matrix_bits_to_absolute_tokens(
                        matrix_bits,
                        min_degree,
                        length=self.length,
                    ),
                    "label": payload["label"][batch_rows].to(torch.float32),
                    "min_degree": min_degree.to(torch.long),
                    "final_factor_id": payload["final_factor_id"][batch_rows].to(torch.long),
                    "sample_id": sample_ids[batch_rows],
                }
