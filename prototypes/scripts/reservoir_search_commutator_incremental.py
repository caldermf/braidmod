#!/usr/bin/env python3
"""
Incremental commutator reservoir search.

This version propagates both Burau(b) and Burau(b^{-1}) through the frontier so
that commutator scoring uses tensor matrix products instead of recomputing the
full Burau polynomial image of [a, b] from scratch for every candidate.
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from braid_data import GarsideFactor, burau_mod_p_projective_tensor
from reservoir_search_braidmod import (
    LENGTH_DTYPE,
    MATRIX_DTYPE,
    PERMUTATIONS_S4,
    SCORE_DTYPE,
    WORD_DTYPE,
    _build_expansion_indices,
    _build_garside_tables,
    _monomial_target_match_batch,
    _right_multiply_simple_batch,
    compute_projlen_batch,
    set_seed,
)
from reservoir_search_commutator import (
    SignedWordLeftNormalForm,
    invert_signed_word,
    parse_signed_word,
    resolve_device,
    scalar_identity_metadata,
)


@dataclass
class IncrementalCommutatorConfig:
    p: int
    max_length: int
    search_D: int
    bucket_size: int
    num_buckets: int
    use_best: int
    bootstrap_length: int
    seed: int
    device: str
    expansion_chunk_size: int
    score_chunk_size: int
    bucket_mode: str
    topk_save: int
    save_kernel_hits: int
    reference_braid_word: Tuple[int, ...]
    out_json: Optional[str]


@dataclass
class IncrementalFrontierBatch:
    tensors: torch.Tensor
    min_degrees: torch.Tensor
    inv_tensors: torch.Tensor
    inv_min_degrees: torch.Tensor
    words: torch.Tensor
    lengths: torch.Tensor
    last_factor_ids: torch.Tensor
    scores: Optional[torch.Tensor] = None
    bucket_ids: Optional[torch.Tensor] = None

    @property
    def size(self) -> int:
        return int(self.tensors.shape[0])


def _build_initial_frontier(config: IncrementalCommutatorConfig, device: torch.device) -> IncrementalFrontierBatch:
    tensors = torch.zeros(1, config.search_D, 3, 3, dtype=MATRIX_DTYPE, device=device)
    tensors[0, 0, 0, 0] = 1
    tensors[0, 0, 1, 1] = 1
    tensors[0, 0, 2, 2] = 1
    inv_tensors = tensors.clone()
    min_degrees = torch.zeros(1, dtype=LENGTH_DTYPE, device=device)
    inv_min_degrees = torch.zeros(1, dtype=LENGTH_DTYPE, device=device)
    words = torch.zeros(1, config.max_length, dtype=WORD_DTYPE, device=device)
    lengths = torch.zeros(1, dtype=LENGTH_DTYPE, device=device)
    last_factor_ids = torch.full((1,), -1, dtype=torch.int16, device=device)
    return IncrementalFrontierBatch(
        tensors=tensors,
        min_degrees=min_degrees,
        inv_tensors=inv_tensors,
        inv_min_degrees=inv_min_degrees,
        words=words,
        lengths=lengths,
        last_factor_ids=last_factor_ids,
    )


def _move_frontier_batch(frontier: IncrementalFrontierBatch, device: torch.device) -> IncrementalFrontierBatch:
    return IncrementalFrontierBatch(
        tensors=frontier.tensors.to(device=device),
        min_degrees=frontier.min_degrees.to(device=device),
        inv_tensors=frontier.inv_tensors.to(device=device),
        inv_min_degrees=frontier.inv_min_degrees.to(device=device),
        words=frontier.words.to(device=device),
        lengths=frontier.lengths.to(device=device),
        last_factor_ids=frontier.last_factor_ids.to(device=device),
        scores=frontier.scores.to(device=device) if frontier.scores is not None else None,
        bucket_ids=frontier.bucket_ids.to(device=device) if frontier.bucket_ids is not None else None,
    )


def _trim_tensor_depth(tensors: torch.Tensor) -> torch.Tensor:
    degree_has_nonzero = tensors.ne(0).any(dim=(-1, -2))
    has_nonzero = degree_has_nonzero.any(dim=-1)
    if not bool(has_nonzero.any().item()):
        return tensors[:, :1]
    widths = tensors.shape[1] - degree_has_nonzero.flip(dims=[1]).int().argmax(dim=1)
    widths = torch.where(has_nonzero, widths, torch.ones_like(widths))
    return tensors[:, : int(widths.max().item())]


def _normalize_projective_batch(out: torch.Tensor, base_min_degrees: torch.Tensor, p: int) -> Tuple[torch.Tensor, torch.Tensor]:
    out = torch.round(out).to(torch.int32)
    out %= p
    out = out.to(MATRIX_DTYPE)

    degree_has_nonzero = out.ne(0).any(dim=(-1, -2))
    has_nonzero = degree_has_nonzero.any(dim=-1)
    min_rel = degree_has_nonzero.int().argmax(dim=-1).to(base_min_degrees.dtype)
    min_rel = torch.where(has_nonzero, min_rel, torch.zeros_like(min_rel))

    normalized = torch.zeros_like(out)
    for shift in torch.unique(min_rel).tolist():
        shift = int(shift)
        mask = min_rel == shift
        if shift == 0:
            normalized[mask] = out[mask]
        else:
            normalized[mask, : out.shape[1] - shift] = out[mask, shift:]

    return normalized, base_min_degrees + min_rel


def _multiply_projective_batch(
    lhs_tensors: torch.Tensor,
    lhs_min_degrees: torch.Tensor,
    rhs_tensors: torch.Tensor,
    rhs_min_degrees: torch.Tensor,
    p: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if lhs_tensors.shape[0] == 1 and rhs_tensors.shape[0] > 1:
        lhs_tensors = lhs_tensors.expand(rhs_tensors.shape[0], -1, -1, -1)
        lhs_min_degrees = lhs_min_degrees.expand(rhs_tensors.shape[0])
    elif rhs_tensors.shape[0] == 1 and lhs_tensors.shape[0] > 1:
        rhs_tensors = rhs_tensors.expand(lhs_tensors.shape[0], -1, -1, -1)
        rhs_min_degrees = rhs_min_degrees.expand(lhs_tensors.shape[0])
    elif lhs_tensors.shape[0] != rhs_tensors.shape[0]:
        raise ValueError("Batch sizes must match unless one side has batch size 1")

    lhs_tensors = _trim_tensor_depth(lhs_tensors)
    rhs_tensors = _trim_tensor_depth(rhs_tensors)

    batch_size, lhs_depth = lhs_tensors.shape[:2]
    rhs_depth = rhs_tensors.shape[1]
    out = torch.zeros(batch_size, lhs_depth + rhs_depth - 1, 3, 3, dtype=torch.float32, device=lhs_tensors.device)
    lhs_float = lhs_tensors.to(torch.float32)
    rhs_float = rhs_tensors.to(torch.float32)

    for shift in range(rhs_depth):
        coeff = rhs_float[:, shift]
        if not torch.any(coeff):
            continue
        out[:, shift : shift + lhs_depth] += torch.einsum("ndik,nkj->ndij", lhs_float, coeff)

    base_min = lhs_min_degrees.to(torch.int32) + rhs_min_degrees.to(torch.int32)
    return _normalize_projective_batch(out, base_min.to(LENGTH_DTYPE), p=p)


def _left_multiply_simple_inverse_batch(
    parent_tensors: torch.Tensor,
    parent_min_degrees: torch.Tensor,
    suffix_ids: torch.Tensor,
    inverse_simple_tensors: torch.Tensor,
    inverse_simple_min_degrees: torch.Tensor,
    p: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_items, depth = parent_tensors.shape[:2]
    factor_depth = int(inverse_simple_tensors.shape[1])
    out = torch.zeros(num_items, depth, 3, 3, dtype=torch.float32, device=parent_tensors.device)
    parent_float = parent_tensors.to(torch.float32)

    for shift in range(factor_depth):
        if shift >= depth:
            break
        coeff = inverse_simple_tensors[suffix_ids, shift].to(torch.float32)
        if not torch.any(coeff):
            continue
        src = parent_float[:, : depth - shift]
        out[:, shift:] += torch.einsum("nij,ndjk->ndik", coeff, src)

    base_min = parent_min_degrees + inverse_simple_min_degrees[suffix_ids]
    return _normalize_projective_batch(out, base_min.to(LENGTH_DTYPE), p=p)


class IncrementalBuckets:
    def __init__(self, bucket_size: int, device: torch.device):
        self.bucket_size = int(bucket_size)
        self.device = device
        self.data: Dict[int, Tuple[torch.Tensor, ...]] = {}

    def add_chunk(
        self,
        tensors: torch.Tensor,
        min_degrees: torch.Tensor,
        inv_tensors: torch.Tensor,
        inv_min_degrees: torch.Tensor,
        words: torch.Tensor,
        lengths: torch.Tensor,
        last_factor_ids: torch.Tensor,
        raw_scores: torch.Tensor,
        bucket_ids: torch.Tensor,
        disable_cap: bool = False,
    ) -> None:
        if tensors.shape[0] == 0:
            return

        priorities = torch.rand(tensors.shape[0], device=self.device, dtype=SCORE_DTYPE)
        for bucket_id in torch.unique(bucket_ids).tolist():
            mask = bucket_ids == bucket_id
            new_tensors = tensors[mask].to(MATRIX_DTYPE)
            new_min_degrees = min_degrees[mask].to(LENGTH_DTYPE)
            new_inv_tensors = inv_tensors[mask].to(MATRIX_DTYPE)
            new_inv_min_degrees = inv_min_degrees[mask].to(LENGTH_DTYPE)
            new_words = words[mask].to(WORD_DTYPE)
            new_lengths = lengths[mask].to(LENGTH_DTYPE)
            new_last = last_factor_ids[mask].to(torch.int16)
            new_scores = raw_scores[mask].to(SCORE_DTYPE)
            new_priorities = priorities[mask]

            if bucket_id in self.data:
                old = self.data[bucket_id]
                merged = tuple(torch.cat([old_i, new_i], dim=0) for old_i, new_i in zip(old, (
                    new_tensors,
                    new_min_degrees,
                    new_inv_tensors,
                    new_inv_min_degrees,
                    new_words,
                    new_lengths,
                    new_last,
                    new_scores,
                    new_priorities,
                )))
            else:
                merged = (
                    new_tensors,
                    new_min_degrees,
                    new_inv_tensors,
                    new_inv_min_degrees,
                    new_words,
                    new_lengths,
                    new_last,
                    new_scores,
                    new_priorities,
                )

            if (not disable_cap) and merged[0].shape[0] > self.bucket_size:
                _, topk_idx = torch.topk(merged[-1], self.bucket_size, largest=False)
                merged = tuple(item[topk_idx] for item in merged)

            self.data[bucket_id] = merged

    def materialize(self, out_device: Optional[torch.device] = None) -> IncrementalFrontierBatch:
        if not self.data:
            raise RuntimeError("No bucket data to materialize")
        if out_device is None:
            out_device = self.device
        ordered = sorted(self.data.items(), key=lambda item: item[0])
        tensors = torch.cat([entry[1][0].to(device=out_device) for entry in ordered], dim=0)
        min_degrees = torch.cat([entry[1][1].to(device=out_device) for entry in ordered], dim=0)
        inv_tensors = torch.cat([entry[1][2].to(device=out_device) for entry in ordered], dim=0)
        inv_min_degrees = torch.cat([entry[1][3].to(device=out_device) for entry in ordered], dim=0)
        words = torch.cat([entry[1][4].to(device=out_device) for entry in ordered], dim=0)
        lengths = torch.cat([entry[1][5].to(device=out_device) for entry in ordered], dim=0)
        last_factor_ids = torch.cat([entry[1][6].to(device=out_device) for entry in ordered], dim=0)
        scores = torch.cat([entry[1][7].to(device=out_device) for entry in ordered], dim=0)
        bucket_ids = torch.cat(
            [torch.full((entry[1][0].shape[0],), entry[0], dtype=torch.long, device=out_device) for entry in ordered],
            dim=0,
        )
        return IncrementalFrontierBatch(
            tensors=tensors,
            min_degrees=min_degrees,
            inv_tensors=inv_tensors,
            inv_min_degrees=inv_min_degrees,
            words=words,
            lengths=lengths,
            last_factor_ids=last_factor_ids,
            scores=scores,
            bucket_ids=bucket_ids,
        )

    def bucket_counts(self) -> Dict[int, int]:
        return {bucket_id: int(data[0].shape[0]) for bucket_id, data in sorted(self.data.items())}


class ReservoirSearchCommutatorIncremental:
    def __init__(self, config: IncrementalCommutatorConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.tables = _build_garside_tables(config.p, self.device)
        self.frontier = _build_initial_frontier(config, self.device)
        self.level_summaries: List[dict] = []
        self.kernel_hits: List[dict] = []
        self.word_normalizer = SignedWordLeftNormalForm(n=4)
        self.reference_word = tuple(config.reference_braid_word)
        self.reference_inverse_word = tuple(invert_signed_word(self.reference_word))
        self.factor_artin_words = {
            idx: [int(gen) + 1 for gen in GarsideFactor(perm).artin_factors()]
            for idx, perm in enumerate(PERMUTATIONS_S4)
        }
        self.identity_coeff = torch.eye(3, dtype=MATRIX_DTYPE, device=self.device)
        self.reference_tensors, self.reference_min_degrees = self._word_to_batch_tensor(self.reference_word)
        self.reference_inverse_tensors, self.reference_inverse_min_degrees = self._word_to_batch_tensor(self.reference_inverse_word)
        self.inverse_simple_tensors, self.inverse_simple_min_degrees = self._build_inverse_simple_tables()

    def _word_to_batch_tensor(self, word: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        depth = max(1, 4 * len(word) + 1)
        tensor, min_degree = burau_mod_p_projective_tensor(word, p=self.config.p, D=depth, n=4)
        t = torch.tensor(tensor, dtype=MATRIX_DTYPE, device=self.device).unsqueeze(0)
        m = torch.tensor([int(min_degree)], dtype=LENGTH_DTYPE, device=self.device)
        return t, m

    def _build_inverse_simple_tables(self) -> Tuple[torch.Tensor, torch.Tensor]:
        tensors: List[torch.Tensor] = []
        min_degrees: List[int] = []
        for perm in PERMUTATIONS_S4:
            word = [int(gen) + 1 for gen in GarsideFactor(perm).artin_factors()]
            inv_word = invert_signed_word(word)
            depth = max(1, 4 * len(inv_word) + 1)
            tensor, min_degree = burau_mod_p_projective_tensor(inv_word, p=self.config.p, D=depth, n=4)
            tensors.append(torch.tensor(tensor, dtype=MATRIX_DTYPE, device=self.device))
            min_degrees.append(int(min_degree))
        max_depth = max(t.shape[0] for t in tensors)
        padded = torch.zeros(len(tensors), max_depth, 3, 3, dtype=MATRIX_DTYPE, device=self.device)
        for idx, tensor in enumerate(tensors):
            padded[idx, : tensor.shape[0]] = tensor
        mins = torch.tensor(min_degrees, dtype=LENGTH_DTYPE, device=self.device)
        return padded, mins

    def _factor_ids_to_artin_word(self, factor_ids: Sequence[int]) -> List[int]:
        word: List[int] = []
        for factor_id in factor_ids:
            word.extend(self.factor_artin_words[int(factor_id)])
        return word

    def _is_trivial_commutator(self, factor_ids: Sequence[int]) -> bool:
        b_word = self._factor_ids_to_artin_word(factor_ids)
        commutator_word = (
            list(self.reference_word)
            + b_word
            + list(self.reference_inverse_word)
            + invert_signed_word(b_word)
        )
        delta_power, factors = self.word_normalizer.normalize(commutator_word)
        return delta_power == 0 and len(factors) == 0

    def _commutator_stats_for_factor_ids(self, factor_ids: Sequence[int]) -> dict:
        b_word = self._factor_ids_to_artin_word(factor_ids)
        commutator_word = (
            list(self.reference_word)
            + b_word
            + list(self.reference_inverse_word)
            + invert_signed_word(b_word)
        )
        depth = max(1, 4 * len(commutator_word) + 1)
        tensor, min_degree = burau_mod_p_projective_tensor(commutator_word, p=self.config.p, D=depth, n=4)
        comm_tensor = torch.tensor(tensor, dtype=MATRIX_DTYPE)
        projlen = int(compute_projlen_batch(comm_tensor.unsqueeze(0)).item())
        checked_triviality = False
        is_trivial = False
        commutator_delta_power = None
        commutator_gnf_factors = None
        if projlen == 1:
            checked_triviality = True
            delta_power, factors = self.word_normalizer.normalize(commutator_word)
            commutator_delta_power = int(delta_power)
            commutator_gnf_factors = [list(perm) for perm in factors]
            is_trivial = delta_power == 0 and len(factors) == 0
        kernel_hit = scalar_identity_metadata(
            [[{k: int(v) for k, v in entry.items()} for entry in row] for row in []],
            p=self.config.p,
        ) if False else None
        if projlen == 1:
            identity_match = _monomial_target_match_batch(
                comm_tensor.unsqueeze(0),
                torch.tensor([int(min_degree)], dtype=LENGTH_DTYPE),
                target_coeff=torch.eye(3, dtype=MATRIX_DTYPE),
                target_degree=0,
                p=self.config.p,
            )
            if bool(identity_match[0][0].item()):
                kernel_hit = {
                    "kernel_type": "identity",
                    "support_degree": int(identity_match[1][0].item()),
                    "scalar_degree": int(identity_match[3][0].item()),
                    "scalar_coeff_mod_p": int(identity_match[2][0].item()),
                }
        return {
            "is_trivial": is_trivial,
            "checked_triviality": checked_triviality,
            "projlen": projlen,
            "min_degree": int(min_degree),
            "b_artin_word": [int(x) for x in b_word],
            "kernel_hit": kernel_hit,
            "commutator_delta_power": commutator_delta_power,
            "commutator_gnf_factors": commutator_gnf_factors,
        }

    def _bucket_id_for_score(self, projlen: int, garside_length: int) -> int:
        if self.config.bucket_mode == "exact":
            return int(projlen)
        denom = max(1.0, 3.0 * float(garside_length))
        normalized = max(0.0, min(1.0, (float(projlen) - 1.0) / denom))
        bucket = int(math.floor(normalized * float(self.config.num_buckets)))
        return max(0, min(self.config.num_buckets - 1, bucket))

    def _select_best(self, frontier: IncrementalFrontierBatch) -> IncrementalFrontierBatch:
        if frontier.scores is None:
            return frontier
        select_device = frontier.scores.device
        total = frontier.size
        if self.config.use_best <= 0 or total <= self.config.use_best:
            return frontier
        if frontier.bucket_ids is None:
            _, idx = torch.topk(frontier.scores, self.config.use_best, largest=False)
        else:
            selected_idx_parts: List[torch.Tensor] = []
            total_selected = 0
            for bucket_id in torch.sort(torch.unique(frontier.bucket_ids)).values.tolist():
                bucket_idx = torch.where(frontier.bucket_ids == bucket_id)[0]
                bucket_count = int(bucket_idx.shape[0])
                remaining = self.config.use_best - total_selected
                if remaining <= 0:
                    break
                if bucket_count <= remaining:
                    selected_idx_parts.append(bucket_idx)
                    total_selected += bucket_count
                    continue
                choice = torch.randperm(bucket_count, device=select_device)[:remaining]
                selected_idx_parts.append(bucket_idx[choice])
                total_selected += remaining
                break
            idx = torch.cat(selected_idx_parts, dim=0)
        return IncrementalFrontierBatch(
            tensors=frontier.tensors[idx],
            min_degrees=frontier.min_degrees[idx],
            inv_tensors=frontier.inv_tensors[idx],
            inv_min_degrees=frontier.inv_min_degrees[idx],
            words=frontier.words[idx],
            lengths=frontier.lengths[idx],
            last_factor_ids=frontier.last_factor_ids[idx],
            scores=frontier.scores[idx],
            bucket_ids=frontier.bucket_ids[idx] if frontier.bucket_ids is not None else None,
        )

    def _serialize_candidate(self, frontier: IncrementalFrontierBatch, idx: int) -> dict:
        length = int(frontier.lengths[idx].item())
        factor_ids = [int(x) for x in frontier.words[idx, :length].tolist()]
        stats = self._commutator_stats_for_factor_ids(factor_ids)
        result = {
            "score": float(frontier.scores[idx].item()) if frontier.scores is not None else float(stats["projlen"]),
            "score_label": "commutator_projlen",
            "garside_length": length,
            "burau_min_degree": int(frontier.min_degrees[idx].item()),
            "factor_ids": factor_ids,
            "gnf_factors": [list(PERMUTATIONS_S4[factor_id]) for factor_id in factor_ids],
            "b_artin_word": stats["b_artin_word"],
            "commutator_projlen": stats["projlen"],
            "commutator_min_degree": stats["min_degree"],
            "checked_triviality": bool(stats["checked_triviality"]),
        }
        if stats["commutator_delta_power"] is not None:
            result["commutator_delta_power"] = stats["commutator_delta_power"]
        if stats["commutator_gnf_factors"] is not None:
            result["commutator_gnf_factors"] = stats["commutator_gnf_factors"]
        if frontier.bucket_ids is not None:
            result["bucket_id"] = int(frontier.bucket_ids[idx].item())
        return result

    def _record_kernel_hit(self, factor_ids: Sequence[int], garside_length: int, score: float, hit: dict) -> None:
        if len(self.kernel_hits) >= self.config.save_kernel_hits:
            return
        self.kernel_hits.append(
            {
                "score": float(score),
                "score_label": "commutator_projlen",
                "kernel_type": hit["kernel_type"],
                "support_degree": int(hit["support_degree"]),
                "scalar_degree": int(hit["scalar_degree"]),
                "scalar_coeff_mod_p": int(hit["scalar_coeff_mod_p"]),
                "garside_length": int(garside_length),
                "factor_ids": [int(x) for x in factor_ids],
                "gnf_factors": [list(PERMUTATIONS_S4[int(factor_id)]) for factor_id in factor_ids],
                "reference_braid_word": [int(x) for x in self.reference_word],
            }
        )

    def _score_subchunk(
        self,
        child_tensors: torch.Tensor,
        child_min_degrees: torch.Tensor,
        child_inv_tensors: torch.Tensor,
        child_inv_min_degrees: torch.Tensor,
        child_words: torch.Tensor,
        child_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        tmp_tensors, tmp_min = _multiply_projective_batch(
            self.reference_tensors,
            self.reference_min_degrees,
            child_tensors,
            child_min_degrees,
            p=self.config.p,
        )
        tmp2_tensors, tmp2_min = _multiply_projective_batch(
            tmp_tensors,
            tmp_min,
            self.reference_inverse_tensors,
            self.reference_inverse_min_degrees,
            p=self.config.p,
        )
        comm_tensors, comm_min = _multiply_projective_batch(
            tmp2_tensors,
            tmp2_min,
            child_inv_tensors,
            child_inv_min_degrees,
            p=self.config.p,
        )
        projlens = compute_projlen_batch(comm_tensors)

        keep_mask = torch.ones(projlens.shape[0], dtype=torch.bool, device=child_tensors.device)
        trivial_discarded = 0
        projlen1_idx = torch.where(projlens == 1)[0]
        if projlen1_idx.numel() > 0:
            words_cpu = child_words[projlen1_idx].to(device="cpu")
            lengths_cpu = child_lengths[projlen1_idx].to(device="cpu")
            for local_idx, global_idx in enumerate(projlen1_idx.tolist()):
                length = int(lengths_cpu[local_idx].item())
                factor_ids = [int(x) for x in words_cpu[local_idx, :length].tolist()]
                if self._is_trivial_commutator(factor_ids):
                    keep_mask[global_idx] = False
                    trivial_discarded += 1

        kept_idx = torch.where(keep_mask)[0]
        if kept_idx.numel() == 0:
            return (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=SCORE_DTYPE),
                torch.empty(0, dtype=torch.long),
                trivial_discarded,
                0,
            )

        kept_projlens = projlens[kept_idx]
        kept_bucket_ids = torch.tensor(
            [
                self._bucket_id_for_score(int(kept_projlens[i].item()), int(child_lengths[kept_idx[i]].item()))
                for i in range(kept_idx.shape[0])
            ],
            dtype=torch.long,
            device=child_tensors.device,
        )

        kernel_hits_found = 0
        kept_projlen1_idx = torch.where(kept_projlens == 1)[0]
        if kept_projlen1_idx.numel() > 0:
            subset_idx = kept_idx[kept_projlen1_idx]
            identity_match = _monomial_target_match_batch(
                comm_tensors[subset_idx],
                comm_min[subset_idx],
                target_coeff=self.identity_coeff,
                target_degree=0,
                p=self.config.p,
            )
            identity_mask, support_deg, coeffs, scalar_deg = identity_match
            if bool(identity_mask.any().item()):
                subset_words_cpu = child_words[subset_idx].to(device="cpu")
                subset_lengths_cpu = child_lengths[subset_idx].to(device="cpu")
                for local_idx in torch.where(identity_mask)[0].tolist():
                    kernel_hits_found += 1
                    length = int(subset_lengths_cpu[local_idx].item())
                    factor_ids = [int(x) for x in subset_words_cpu[local_idx, :length].tolist()]
                    self._record_kernel_hit(
                        factor_ids,
                        garside_length=length,
                        score=1.0,
                        hit={
                            "kernel_type": "identity",
                            "support_degree": int(support_deg[local_idx].item()),
                            "scalar_degree": int(scalar_deg[local_idx].item()),
                            "scalar_coeff_mod_p": int(coeffs[local_idx].item()),
                        },
                    )

        return kept_idx, kept_projlens.to(SCORE_DTYPE), kept_bucket_ids, trivial_discarded, kernel_hits_found

    def process_level(self, level: int) -> bool:
        level_start = time.time()
        is_bootstrap = level <= self.config.bootstrap_length
        braid_indices, suffix_ids = _build_expansion_indices(
            self.frontier.last_factor_ids,
            level=level,
            tables=self.tables,
            device=self.device,
        )
        num_candidates = int(suffix_ids.shape[0])
        if num_candidates == 0:
            print(f"Level {level}: no valid expansions remain")
            return False

        print(f"\nLevel {level} ({'BOOTSTRAP' if is_bootstrap else 'SAMPLING'})")
        print(f"  parents={self.frontier.size} candidates={num_candidates}")
        print(f"  reference_word={list(self.reference_word)}")
        print("  score_type=commutator_projlen_incremental")

        buckets = IncrementalBuckets(bucket_size=self.config.bucket_size, device=self.device)
        incoming_bucket_counts: Dict[int, int] = {}
        trivial_discard_count = 0
        nontrivial_candidate_count = 0
        kernel_hit_count = 0
        score_min = None
        score_max = None
        score_sum = 0.0

        matmul_time = 0.0
        score_time = 0.0
        bucket_time = 0.0

        for start in range(0, num_candidates, self.config.expansion_chunk_size):
            end = min(start + self.config.expansion_chunk_size, num_candidates)
            chunk_braid_idx = braid_indices[start:end]
            chunk_suffix_ids = suffix_ids[start:end]

            parent_tensors = self.frontier.tensors[chunk_braid_idx]
            parent_min_degrees = self.frontier.min_degrees[chunk_braid_idx]
            parent_inv_tensors = self.frontier.inv_tensors[chunk_braid_idx]
            parent_inv_min_degrees = self.frontier.inv_min_degrees[chunk_braid_idx]
            parent_words = self.frontier.words[chunk_braid_idx]
            parent_lengths = self.frontier.lengths[chunk_braid_idx]

            t0 = time.time()
            child_tensors, child_min_degrees = _right_multiply_simple_batch(
                parent_tensors=parent_tensors,
                parent_min_degrees=parent_min_degrees,
                suffix_ids=chunk_suffix_ids,
                simple_shift_mats=self.tables.simple_shift_mats,
                p=self.config.p,
            )
            child_inv_tensors, child_inv_min_degrees = _left_multiply_simple_inverse_batch(
                parent_tensors=parent_inv_tensors,
                parent_min_degrees=parent_inv_min_degrees,
                suffix_ids=chunk_suffix_ids.to(torch.long),
                inverse_simple_tensors=self.inverse_simple_tensors,
                inverse_simple_min_degrees=self.inverse_simple_min_degrees,
                p=self.config.p,
            )
            matmul_time += time.time() - t0

            child_words = parent_words.clone()
            row_idx = torch.arange(child_words.shape[0], device=self.device)
            child_words[row_idx, parent_lengths.long()] = chunk_suffix_ids.to(WORD_DTYPE)
            child_lengths = parent_lengths + 1
            child_last = chunk_suffix_ids.to(torch.int16)

            t0 = time.time()
            kept_idx_parts: List[torch.Tensor] = []
            raw_score_parts: List[torch.Tensor] = []
            bucket_id_parts: List[torch.Tensor] = []
            for score_start in range(0, child_tensors.shape[0], self.config.score_chunk_size):
                score_end = min(score_start + self.config.score_chunk_size, child_tensors.shape[0])
                sub_kept_idx, sub_scores, sub_buckets, sub_trivial, sub_hits = self._score_subchunk(
                    child_tensors[score_start:score_end],
                    child_min_degrees[score_start:score_end],
                    child_inv_tensors[score_start:score_end],
                    child_inv_min_degrees[score_start:score_end],
                    child_words[score_start:score_end],
                    child_lengths[score_start:score_end],
                )
                trivial_discard_count += sub_trivial
                kernel_hit_count += sub_hits
                if sub_kept_idx.numel() == 0:
                    continue
                kept_idx_parts.append(sub_kept_idx + score_start)
                raw_score_parts.append(sub_scores)
                bucket_id_parts.append(sub_buckets)

            if not kept_idx_parts:
                score_time += time.time() - t0
                continue

            kept_idx = torch.cat(kept_idx_parts, dim=0)
            raw_scores = torch.cat(raw_score_parts, dim=0)
            bucket_ids = torch.cat(bucket_id_parts, dim=0)
            nontrivial_candidate_count += int(kept_idx.numel())
            score_time += time.time() - t0

            child_tensors = child_tensors[kept_idx]
            child_min_degrees = child_min_degrees[kept_idx]
            child_inv_tensors = child_inv_tensors[kept_idx]
            child_inv_min_degrees = child_inv_min_degrees[kept_idx]
            child_words = child_words[kept_idx]
            child_lengths = child_lengths[kept_idx]
            child_last = child_last[kept_idx]

            score_min_val = float(raw_scores.min().item())
            score_max_val = float(raw_scores.max().item())
            score_min = score_min_val if score_min is None else min(score_min, score_min_val)
            score_max = score_max_val if score_max is None else max(score_max, score_max_val)
            score_sum += float(raw_scores.sum().item())

            unique_buckets, counts = torch.unique(bucket_ids, return_counts=True)
            for bucket_id, count in zip(unique_buckets.tolist(), counts.tolist()):
                incoming_bucket_counts[int(bucket_id)] = incoming_bucket_counts.get(int(bucket_id), 0) + int(count)

            t0 = time.time()
            buckets.add_chunk(
                tensors=child_tensors,
                min_degrees=child_min_degrees,
                inv_tensors=child_inv_tensors,
                inv_min_degrees=child_inv_min_degrees,
                words=child_words,
                lengths=child_lengths,
                last_factor_ids=child_last,
                raw_scores=raw_scores,
                bucket_ids=bucket_ids,
                disable_cap=is_bootstrap,
            )
            bucket_time += time.time() - t0

        if nontrivial_candidate_count == 0:
            print(
                f"  discarded_trivial={trivial_discard_count} nontrivial=0 "
                "no survivors remain after projlen-1 triviality filtering"
            )
            return False

        materialized = buckets.materialize(out_device=torch.device("cpu"))
        selected = materialized if is_bootstrap else self._select_best(materialized)
        self.frontier = _move_frontier_batch(selected, self.device)
        level_time = time.time() - level_start
        score_mean = score_sum / float(nontrivial_candidate_count)

        best_candidate = None
        if materialized.size > 0:
            best_idx = int(torch.argmin(materialized.scores).item())
            best_candidate = self._serialize_candidate(materialized, best_idx)

        summary = {
            "level": level,
            "num_parents": self.frontier.size,
            "num_candidates": num_candidates,
            "num_trivial_commutators_discarded": trivial_discard_count,
            "num_nontrivial_candidates": nontrivial_candidate_count,
            "num_kept_after_reservoir": materialized.size,
            "num_selected_for_next_level": selected.size,
            "nonempty_buckets": len(buckets.data),
            "incoming_bucket_counts": incoming_bucket_counts,
            "kept_bucket_counts": buckets.bucket_counts(),
            "score_min": score_min,
            "score_mean": score_mean,
            "score_max": score_max,
            "kernel_hits_found": kernel_hit_count,
            "kernel_hit_type_counts": {"identity": kernel_hit_count},
            "best_candidate": best_candidate,
            "timing_sec": {
                "matmul": round(matmul_time, 4),
                "score": round(score_time, 4),
                "reservoir": round(bucket_time, 4),
                "total": round(level_time, 4),
            },
        }
        self.level_summaries.append(summary)

        print(
            f"  discarded_trivial={trivial_discard_count} nontrivial={nontrivial_candidate_count} "
            f"kept={materialized.size} selected={selected.size} buckets={len(buckets.data)} "
            f"score=[{score_min:.4f}, {score_mean:.4f}, {score_max:.4f}] kernel_hits={kernel_hit_count}"
        )
        print(
            f"  timing: matmul={matmul_time:.2f}s score={score_time:.2f}s "
            f"reservoir={bucket_time:.2f}s total={level_time:.2f}s"
        )
        if best_candidate is not None:
            print(
                f"  best score={best_candidate['score']:.6f} "
                f"last_factor={best_candidate['gnf_factors'][-1]}"
            )
        return True

    def run(self) -> dict:
        total_start = time.time()
        completed_levels = 0
        try:
            for level in range(1, self.config.max_length + 1):
                if not self.process_level(level):
                    break
                completed_levels = level
        finally:
            total_time = time.time() - total_start

        final_frontier = self.frontier
        topk = min(self.config.topk_save, final_frontier.size)
        if final_frontier.scores is not None:
            _, top_idx = torch.topk(final_frontier.scores, topk, largest=False)
        else:
            top_idx = torch.arange(topk, device=self.device)
        best_candidates = [self._serialize_candidate(final_frontier, int(idx.item())) for idx in top_idx]
        return {
            "config": asdict(self.config),
            "device_resolved": str(self.device),
            "reference_braid_word": [int(x) for x in self.reference_word],
            "reference_inverse_word": [int(x) for x in self.reference_inverse_word],
            "completed_levels": completed_levels,
            "level_summaries": self.level_summaries,
            "best_candidates": best_candidates,
            "kernel_hits": self.kernel_hits,
            "kernel_hit_type_counts": {"identity": len(self.kernel_hits)},
            "total_time_sec": round(total_time, 4),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Incremental reservoir search that propagates Burau(b) and Burau(b^{-1}) "
            "and scores by projlen(Burau([a,b]))."
        )
    )
    parser.add_argument("--reference-braid-word", required=True, help="Signed Artin word for the fixed braid a")
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--search-D", type=int, help="Projective support window for propagating b and b^{-1}")
    parser.add_argument("--bucket-size", type=int, default=4000)
    parser.add_argument("--num-buckets", type=int, default=32)
    parser.add_argument("--use-best", type=int, default=0)
    parser.add_argument("--bootstrap-length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--expansion-chunk-size", type=int, default=50000)
    parser.add_argument("--score-chunk-size", type=int, default=512)
    parser.add_argument("--bucket-mode", choices=("exact", "score"), default="exact")
    parser.add_argument("--topk-save", type=int, default=50)
    parser.add_argument("--save-kernel-hits", type=int, default=100)
    parser.add_argument("--out-json")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    if args.bucket_size <= 0:
        raise ValueError("--bucket-size must be positive")
    if args.num_buckets <= 0:
        raise ValueError("--num-buckets must be positive")
    if args.bootstrap_length < 0:
        raise ValueError("--bootstrap-length must be non-negative")
    if args.expansion_chunk_size <= 0:
        raise ValueError("--expansion-chunk-size must be positive")
    if args.score_chunk_size <= 0:
        raise ValueError("--score-chunk-size must be positive")

    config = IncrementalCommutatorConfig(
        p=args.p,
        max_length=args.max_length,
        search_D=args.search_D if args.search_D is not None else 4 * args.max_length + 1,
        bucket_size=args.bucket_size,
        num_buckets=args.num_buckets,
        use_best=args.use_best,
        bootstrap_length=args.bootstrap_length,
        seed=args.seed,
        device=args.device,
        expansion_chunk_size=args.expansion_chunk_size,
        score_chunk_size=args.score_chunk_size,
        bucket_mode=args.bucket_mode,
        topk_save=args.topk_save,
        save_kernel_hits=args.save_kernel_hits,
        reference_braid_word=parse_signed_word(args.reference_braid_word),
        out_json=args.out_json,
    )

    set_seed(config.seed)
    search = ReservoirSearchCommutatorIncremental(config)
    result = search.run()
    if config.out_json:
        out_path = Path(config.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
