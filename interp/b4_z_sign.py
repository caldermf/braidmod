"""B_4 integer-Burau sign-token data utilities.

Each coefficient slice is a 3x3 sign matrix over {-1, 0, +1}, encoded as a
base-3 token with digit convention 0=zero, 1=negative, 2=positive.  The
vocabulary size is therefore 3^9 = 19683.
"""

from __future__ import annotations

import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import IterableDataset, get_worker_info

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from braid_data import GarsideFactor, burau_polynomial_matrix  # noqa: E402
from interp.b4_data import SPLIT_RANGES, discover_shards, load_shard, split_mask  # noqa: E402
from interp.generate_b4_dataset import MATRIX_SIZE, PROPER_FACTOR_PERMS, absolute_depth_for_length  # noqa: E402


SIGN_VOCAB_SIZE = 3**9
SIGN_DIGIT_CONVENTION = {"0": "zero", "1": "negative", "2": "positive"}


@lru_cache(maxsize=1)
def simple_mats_z_cpu() -> torch.Tensor:
    sparse = []
    max_degree = 0
    for perm in PROPER_FACTOR_PERMS:
        word = [idx + 1 for idx in GarsideFactor(perm).artin_factors()]
        mat = burau_polynomial_matrix(word, n=4)
        terms = []
        for i in range(MATRIX_SIZE):
            for j in range(MATRIX_SIZE):
                for exp, coeff in mat[i][j].items():
                    if exp < 0:
                        raise RuntimeError("positive simple factor produced a negative degree")
                    if coeff:
                        terms.append((int(exp), i, j, int(coeff)))
                        max_degree = max(max_degree, int(exp))
        sparse.append(terms)

    out = torch.zeros(len(PROPER_FACTOR_PERMS), max_degree + 1, MATRIX_SIZE, MATRIX_SIZE, dtype=torch.long)
    for factor_id, terms in enumerate(sparse):
        for exp, i, j, coeff in terms:
            out[factor_id, exp, i, j] = coeff
    return out


def simple_mats_z(device: torch.device | str) -> torch.Tensor:
    return simple_mats_z_cpu().to(device=device)


class B4FactorBatchIterable(IterableDataset):
    """Yield batched factor ids and labels from generated B_4 shard files."""

    def __init__(
        self,
        shard_paths: Iterable[str | Path],
        *,
        split: str,
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

                yield {
                    "factor_ids": payload["factor_ids"][batch_rows].to(torch.long),
                    "label_bits": payload["label_bits"][batch_rows].to(torch.float32),
                    "descent_mask": payload["descent_mask"][batch_rows].to(torch.long),
                    "final_factor_id": payload["final_factor_id"][batch_rows].to(torch.long),
                    "sample_id": sample_ids[batch_rows],
                }


def dense_burau_z_for_factor_ids(
    factor_ids: torch.Tensor,
    *,
    length: int,
    absolute_depth: int,
    simple_mats: torch.Tensor,
) -> torch.Tensor:
    device = factor_ids.device
    batch_size = factor_ids.shape[0]
    mat = torch.zeros(batch_size, absolute_depth, MATRIX_SIZE, MATRIX_SIZE, dtype=torch.long, device=device)
    for i in range(MATRIX_SIZE):
        mat[:, 0, i, i] = 1

    for pos in range(length):
        coeff = simple_mats[factor_ids[:, pos].to(torch.long)]
        out = torch.zeros_like(mat)
        for shift in range(simple_mats.shape[1]):
            if shift >= absolute_depth:
                break
            for k in range(MATRIX_SIZE):
                src = mat[:, : absolute_depth - shift, :, k]
                for j in range(MATRIX_SIZE):
                    active = coeff[:, shift, k, j]
                    if bool(active.ne(0).any().item()):
                        out[:, shift:, :, j] += src * active.view(batch_size, 1, 1)
        mat = out
    return mat


def sign_tokens_from_dense(mat: torch.Tensor) -> torch.Tensor:
    signs = torch.zeros_like(mat, dtype=torch.long)
    signs = torch.where(mat.lt(0), torch.ones_like(signs), signs)
    signs = torch.where(mat.gt(0), torch.full_like(signs, 2), signs)
    digits = signs.view(signs.shape[0], signs.shape[1], MATRIX_SIZE * MATRIX_SIZE)
    weights = torch.tensor([3**i for i in range(MATRIX_SIZE * MATRIX_SIZE)], dtype=torch.long, device=mat.device)
    return (digits * weights.view(1, 1, -1)).sum(dim=-1)


def factor_ids_to_z_sign_tokens(
    factor_ids: torch.Tensor,
    *,
    length: int,
    absolute_depth: int | None = None,
    simple_mats: torch.Tensor | None = None,
) -> torch.Tensor:
    if absolute_depth is None:
        absolute_depth = absolute_depth_for_length(length)
    if simple_mats is None:
        simple_mats = simple_mats_z(factor_ids.device)
    mat = dense_burau_z_for_factor_ids(
        factor_ids,
        length=length,
        absolute_depth=absolute_depth,
        simple_mats=simple_mats,
    )
    return sign_tokens_from_dense(mat)


def discover_b4_shards(data_dir: str | Path, num_shards: int | None = None, allow_partial: bool = False) -> list[Path]:
    return discover_shards(data_dir, num_shards=num_shards, allow_partial=allow_partial)
