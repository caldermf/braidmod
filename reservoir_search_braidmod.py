#!/usr/bin/env python3
import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from braid_data import GNF, GarsideFactor, burau_mod_p_polynomial_matrix
from predict_garside_mlp import build_model, confusion_score_from_logits, resolve_device


PERMUTATIONS_S4 = list(permutations(range(4)))
PERM_TO_ID = {perm: idx for idx, perm in enumerate(PERMUTATIONS_S4)}

MATRIX_DTYPE = torch.int16
WORD_DTYPE = torch.int16
LENGTH_DTYPE = torch.int32
SCORE_DTYPE = torch.float32
TARGET_XENT_WINDOW = 5
MAXIMIZING_SCORE_TYPES = {"target_xent_maximize", "target_xent_max_maximize"}
SWITCHING_SCORE_TYPES = {"projlen_then_target_xent_maximize"}
FRONTIER_SCORE_TYPES = {"frontier_target_xent"}


@dataclass
class SearchConfig:
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
    score_type: str
    score_switch_length: int
    projlen_bucket_mode: str
    checkpoint: Optional[str]
    projlen_weight: float
    confusion_weight: float
    topk_save: int
    save_kernel_hits: int
    out_json: Optional[str]


@dataclass
class ScoreBatch:
    raw_scores: torch.Tensor
    bucket_scores: torch.Tensor


@dataclass
class FrontierBatch:
    tensors: torch.Tensor
    min_degrees: torch.Tensor
    words: torch.Tensor
    lengths: torch.Tensor
    last_factor_ids: torch.Tensor
    xent_history: Optional[torch.Tensor] = None
    xent_max: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
    bucket_ids: Optional[torch.Tensor] = None

    @property
    def size(self) -> int:
        return int(self.tensors.shape[0])


@dataclass
class GarsideTables:
    factor_perms: torch.Tensor
    first_factor_ids: torch.Tensor
    valid_suffixes: torch.Tensor
    num_valid_suffixes: torch.Tensor
    simple_shift_mats: torch.Tensor
    max_shift: int
    identity_id: int
    delta_id: int
    delta_degree: int
    delta_coeff: torch.Tensor


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_garside_tables(p: int, device: torch.device) -> GarsideTables:
    factor_perms = torch.tensor(PERMUTATIONS_S4, dtype=torch.int16)
    identity_id = PERM_TO_ID[GNF.identity_perm(4)]
    delta_id = PERM_TO_ID[GNF.delta_perm(4)]

    left_desc = []
    right_desc = []
    simple_poly = []
    max_shift = 0

    for perm in PERMUTATIONS_S4:
        factor = GarsideFactor(perm)
        left_desc.append(factor.left_descent())
        right_desc.append(factor.right_descent())

        word = [idx + 1 for idx in factor.artin_factors()]
        poly = burau_mod_p_polynomial_matrix(word, p=p, n=4)
        simple_poly.append(poly)
        for i in range(3):
            for j in range(3):
                for exp in poly[i][j].keys():
                    if exp < 0:
                        raise ValueError("Expected non-negative exponents for positive simple factors")
                    max_shift = max(max_shift, exp)

    simple_shift_mats = torch.zeros(len(PERMUTATIONS_S4), max_shift + 1, 3, 3, dtype=MATRIX_DTYPE)
    for factor_id, poly in enumerate(simple_poly):
        for i in range(3):
            for j in range(3):
                for exp, coeff in poly[i][j].items():
                    simple_shift_mats[factor_id, exp, i, j] = int(coeff)

    delta_support = simple_shift_mats[delta_id].ne(0).any(dim=(-1, -2))
    delta_degrees = torch.where(delta_support)[0]
    if delta_degrees.numel() != 1:
        raise ValueError("Expected Delta to have monomial Burau image in the corrected normalization")
    delta_degree = int(delta_degrees.item())
    delta_coeff = simple_shift_mats[delta_id, delta_degree].clone()

    valid_suffix_lists: List[List[int]] = []
    for factor_id in range(len(PERMUTATIONS_S4)):
        valid = []
        for suffix_id in range(len(PERMUTATIONS_S4)):
            if suffix_id == identity_id:
                continue
            if right_desc[factor_id].issuperset(left_desc[suffix_id]):
                valid.append(suffix_id)
        valid_suffix_lists.append(valid)

    max_valid = max(len(x) for x in valid_suffix_lists)
    valid_suffixes = torch.full((len(PERMUTATIONS_S4), max_valid), -1, dtype=torch.int16)
    num_valid_suffixes = torch.zeros(len(PERMUTATIONS_S4), dtype=torch.int32)
    for factor_id, suffixes in enumerate(valid_suffix_lists):
        num_valid_suffixes[factor_id] = len(suffixes)
        if suffixes:
            valid_suffixes[factor_id, : len(suffixes)] = torch.tensor(suffixes, dtype=torch.int16)

    first_factor_ids = torch.tensor(
        [idx for idx in range(len(PERMUTATIONS_S4)) if idx not in (identity_id, delta_id)],
        dtype=torch.int16,
    )

    return GarsideTables(
        factor_perms=factor_perms.to(device),
        first_factor_ids=first_factor_ids.to(device),
        valid_suffixes=valid_suffixes.to(device),
        num_valid_suffixes=num_valid_suffixes.to(device),
        simple_shift_mats=simple_shift_mats.to(device),
        max_shift=max_shift,
        identity_id=identity_id,
        delta_id=delta_id,
        delta_degree=delta_degree,
        delta_coeff=delta_coeff.to(device),
    )


def _build_initial_frontier(config: SearchConfig, device: torch.device) -> FrontierBatch:
    tensors = torch.zeros(1, config.search_D, 3, 3, dtype=MATRIX_DTYPE, device=device)
    tensors[0, 0, 0, 0] = 1
    tensors[0, 0, 1, 1] = 1
    tensors[0, 0, 2, 2] = 1
    min_degrees = torch.zeros(1, dtype=LENGTH_DTYPE, device=device)
    words = torch.zeros(1, config.max_length, dtype=WORD_DTYPE, device=device)
    lengths = torch.zeros(1, dtype=LENGTH_DTYPE, device=device)
    last_factor_ids = torch.full((1,), -1, dtype=torch.int16, device=device)
    xent_history = torch.zeros(1, TARGET_XENT_WINDOW, dtype=SCORE_DTYPE, device=device)
    xent_max = torch.zeros(1, dtype=SCORE_DTYPE, device=device)
    return FrontierBatch(
        tensors=tensors,
        min_degrees=min_degrees,
        words=words,
        lengths=lengths,
        last_factor_ids=last_factor_ids,
        xent_history=xent_history,
        xent_max=xent_max,
    )


def _build_expansion_indices(
    last_factor_ids: torch.Tensor,
    level: int,
    tables: GarsideTables,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_parents = int(last_factor_ids.shape[0])
    if level == 1:
        num_suffixes = int(tables.first_factor_ids.shape[0])
        if num_suffixes == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )
        braid_indices = torch.repeat_interleave(torch.arange(num_parents, device=device), num_suffixes)
        suffix_ids = tables.first_factor_ids.repeat(num_parents).to(torch.long)
        return braid_indices, suffix_ids

    parent_ids = last_factor_ids.to(torch.long)
    suffix_counts = tables.num_valid_suffixes[parent_ids]
    total_expansions = int(suffix_counts.sum().item())
    if total_expansions == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    braid_indices = torch.repeat_interleave(torch.arange(num_parents, device=device), suffix_counts)
    cumsum = torch.cumsum(suffix_counts, dim=0)
    starts = cumsum - suffix_counts
    global_positions = torch.arange(total_expansions, device=device)
    local_suffix_indices = global_positions - starts[braid_indices]
    suffix_ids = tables.valid_suffixes[parent_ids[braid_indices], local_suffix_indices].to(torch.long)
    return braid_indices, suffix_ids


def _right_multiply_simple_batch(
    parent_tensors: torch.Tensor,
    parent_min_degrees: torch.Tensor,
    suffix_ids: torch.Tensor,
    simple_shift_mats: torch.Tensor,
    p: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_items, depth, _, _ = parent_tensors.shape
    out = torch.zeros(num_items, depth, 3, 3, dtype=torch.float32, device=parent_tensors.device)
    parent_float = parent_tensors.to(torch.float32)
    max_shift = int(simple_shift_mats.shape[1]) - 1

    for shift in range(max_shift + 1):
        if shift >= depth:
            break
        coeff = simple_shift_mats[suffix_ids, shift].to(torch.float32)
        if not torch.any(coeff):
            continue
        src = parent_float[:, : depth - shift]
        out[:, shift:] += torch.einsum("ndik,nkj->ndij", src, coeff)

    out = torch.round(out).to(torch.int32)
    out %= p
    out = out.to(MATRIX_DTYPE)

    degree_has_nonzero = out.ne(0).any(dim=(-1, -2))
    has_nonzero = degree_has_nonzero.any(dim=-1)
    min_rel = degree_has_nonzero.int().argmax(dim=-1).to(parent_min_degrees.dtype)
    min_rel = torch.where(has_nonzero, min_rel, torch.zeros_like(min_rel))

    normalized = torch.zeros_like(out)
    for shift in torch.unique(min_rel).tolist():
        shift = int(shift)
        mask = min_rel == shift
        if shift == 0:
            normalized[mask] = out[mask]
        else:
            normalized[mask, : depth - shift] = out[mask, shift:]

    child_min_degrees = parent_min_degrees + min_rel
    return normalized, child_min_degrees


def compute_projlen_batch(tensors: torch.Tensor) -> torch.Tensor:
    degree_has_nonzero = tensors.ne(0).any(dim=(-1, -2))
    has_nonzero = degree_has_nonzero.any(dim=-1)
    min_degrees = degree_has_nonzero.int().argmax(dim=-1)
    max_degrees = tensors.shape[1] - 1 - degree_has_nonzero.flip(dims=[-1]).int().argmax(dim=-1)
    projlens = max_degrees - min_degrees + 1
    return torch.where(has_nonzero, projlens, torch.zeros_like(projlens)).to(torch.int32)


def update_xent_history(parent_history: torch.Tensor, current_xent: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if parent_history.ndim != 2 or parent_history.shape[1] != TARGET_XENT_WINDOW:
        raise ValueError("parent_history must have shape [N, TARGET_XENT_WINDOW]")
    if current_xent.ndim != 1:
        raise ValueError("current_xent must have shape [N]")
    if lengths.ndim != 1:
        raise ValueError("lengths must have shape [N]")

    child_history = torch.roll(parent_history, shifts=-1, dims=1)
    child_history[:, -1] = current_xent.to(SCORE_DTYPE)

    counts = torch.clamp(lengths.to(torch.long), min=1, max=TARGET_XENT_WINDOW)
    valid_mask = (
        torch.arange(TARGET_XENT_WINDOW, device=parent_history.device)
        .unsqueeze(0)
        >= (TARGET_XENT_WINDOW - counts).unsqueeze(1)
    )
    averaged = (child_history * valid_mask.to(SCORE_DTYPE)).sum(dim=1) / counts.to(SCORE_DTYPE)
    return child_history, averaged


def is_maximizing_score_type(score_type: str) -> bool:
    return score_type in MAXIMIZING_SCORE_TYPES


def is_switching_score_type(score_type: str) -> bool:
    return score_type in SWITCHING_SCORE_TYPES


def is_frontier_score_type(score_type: str) -> bool:
    return score_type in FRONTIER_SCORE_TYPES


def maximize_score_transform(scores: torch.Tensor) -> ScoreBatch:
    # Preserve the search engine's lower-is-better convention internally.
    raw_scores = -scores.to(SCORE_DTYPE)
    bucket_scores = 1.0 / (1.0 + torch.clamp(scores.to(SCORE_DTYPE), min=0.0))
    return ScoreBatch(raw_scores=raw_scores, bucket_scores=bucket_scores)


def _monomial_target_match_batch(
    tensors: torch.Tensor,
    min_degrees: torch.Tensor,
    target_coeff: torch.Tensor,
    target_degree: int,
    p: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_items = int(tensors.shape[0])
    if num_items == 0:
        empty_mask = torch.empty(0, dtype=torch.bool, device=tensors.device)
        empty_long = torch.empty(0, dtype=torch.long, device=tensors.device)
        return empty_mask, empty_long, empty_long, empty_long

    degree_has_nonzero = tensors.ne(0).any(dim=(-1, -2))
    support_counts = degree_has_nonzero.sum(dim=-1)
    monomial_mask = support_counts.eq(1)
    support_offsets = degree_has_nonzero.int().argmax(dim=-1)
    batch_idx = torch.arange(num_items, device=tensors.device)
    coeff_slices = tensors[batch_idx, support_offsets].to(torch.long) % p

    target_coeff_long = target_coeff.to(torch.long) % p
    target_nz = torch.nonzero(target_coeff_long, as_tuple=False)
    if target_nz.shape[0] == 0:
        raise ValueError("Target coefficient matrix must be nonzero")
    ref_i = int(target_nz[0, 0].item())
    ref_j = int(target_nz[0, 1].item())
    ref_coeff = int(target_coeff_long[ref_i, ref_j].item())
    inv_ref = pow(ref_coeff, -1, p)

    scalar_coeffs = (coeff_slices[:, ref_i, ref_j] * inv_ref) % p
    expected = (scalar_coeffs[:, None, None] * target_coeff_long.unsqueeze(0)) % p
    match_mask = monomial_mask & coeff_slices.eq(expected).all(dim=(-1, -2))
    support_degrees = min_degrees.to(torch.long) + support_offsets.to(torch.long)
    scalar_degrees = support_degrees - int(target_degree)
    return match_mask, support_degrees, scalar_coeffs, scalar_degrees


class BaseScoreFunction:
    def __init__(self, device: torch.device):
        self.device = device

    def score_batch(
        self,
        tensors: torch.Tensor,
        min_degrees: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        target_factor_ids: Optional[torch.Tensor] = None,
    ) -> ScoreBatch:
        raise NotImplementedError

    def bucketize(self, bucket_scores: torch.Tensor, num_buckets: int) -> torch.Tensor:
        if num_buckets <= 0:
            raise ValueError("num_buckets must be positive")
        num_items = int(bucket_scores.shape[0])
        if num_items == 0:
            return torch.empty(0, dtype=torch.long, device=bucket_scores.device)

        clamped = torch.clamp(bucket_scores, 0.0, 1.0)
        bucket_ids = torch.floor(clamped * float(num_buckets)).to(torch.long)
        return torch.clamp(bucket_ids, 0, num_buckets - 1)


class ProjlenScoreFunction(BaseScoreFunction):
    def __init__(self, device: torch.device, search_D: int, bucket_mode: str):
        super().__init__(device)
        self.search_D = search_D
        if bucket_mode not in ("exact", "score"):
            raise ValueError("bucket_mode must be 'exact' or 'score'")
        self.bucket_mode = bucket_mode

    def score_batch(
        self,
        tensors: torch.Tensor,
        min_degrees: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        target_factor_ids: Optional[torch.Tensor] = None,
    ) -> ScoreBatch:
        del min_degrees
        del target_factor_ids
        if lengths is None:
            raise ValueError("ProjlenScoreFunction requires lengths")
        projlens = compute_projlen_batch(tensors)
        denom = torch.clamp(3 * lengths.to(SCORE_DTYPE), min=1.0)
        bucket_scores = (projlens.to(SCORE_DTYPE) - 1.0) / denom
        return ScoreBatch(raw_scores=projlens.to(SCORE_DTYPE), bucket_scores=bucket_scores)

    def bucketize(self, bucket_scores: torch.Tensor, num_buckets: int) -> torch.Tensor:
        if self.bucket_mode == "score":
            return super().bucketize(bucket_scores, num_buckets)
        del num_buckets
        denom = max(1, self.search_D - 1)
        return torch.round(bucket_scores * float(denom) + 1.0).to(torch.long)


class ModelConfusionScoreFunction(BaseScoreFunction):
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        score_chunk_size: int,
        expected_p: int,
        metric_type: str = "entropy",
    ):
        super().__init__(device)
        self.score_chunk_size = score_chunk_size
        if metric_type not in ("entropy", "target_xent"):
            raise ValueError("metric_type must be 'entropy' or 'target_xent'")
        self.metric_type = metric_type
        checkpoint = torch.load(checkpoint_path, map_location=device)
        checkpoint_p = int(checkpoint["p"])
        if checkpoint_p != int(expected_p):
            raise ValueError(
                f"Checkpoint was trained with p={checkpoint_p}, but the search is configured with p={expected_p}"
            )
        self.model = build_model(checkpoint, device)
        self.model_D = int(checkpoint["D"])

    def score_batch(
        self,
        tensors: torch.Tensor,
        min_degrees: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        target_factor_ids: Optional[torch.Tensor] = None,
    ) -> ScoreBatch:
        del lengths
        if self.metric_type == "target_xent" and target_factor_ids is None:
            raise ValueError("target_factor_ids are required for target_xent scoring")
        out = torch.empty(tensors.shape[0], dtype=SCORE_DTYPE, device=self.device)
        num_classes = len(PERMUTATIONS_S4)
        xent_norm = float(math.log(float(num_classes)))
        with torch.no_grad():
            for start in range(0, tensors.shape[0], self.score_chunk_size):
                end = min(start + self.score_chunk_size, tensors.shape[0])
                chunk = tensors[start:end]
                chunk_min_degrees = None if min_degrees is None else min_degrees[start:end]
                chunk_targets = None if target_factor_ids is None else target_factor_ids[start:end]
                if chunk.shape[1] >= self.model_D:
                    x = chunk[:, : self.model_D].to(dtype=torch.long)
                else:
                    x = torch.zeros(
                        chunk.shape[0],
                        self.model_D,
                        3,
                        3,
                        dtype=torch.long,
                        device=self.device,
                    )
                    x[:, : chunk.shape[1]] = chunk.to(dtype=torch.long)
                if chunk_min_degrees is None:
                    chunk_min_degrees = torch.zeros(chunk.shape[0], dtype=torch.float32, device=self.device)
                else:
                    chunk_min_degrees = chunk_min_degrees.to(dtype=torch.float32, device=self.device)
                factor_logits, _ = self.model(x, min_degree=chunk_min_degrees)
                if self.metric_type == "entropy":
                    out[start:end] = confusion_score_from_logits(factor_logits).to(SCORE_DTYPE)
                else:
                    chunk_targets = chunk_targets.to(dtype=torch.long, device=self.device)
                    xent = F.cross_entropy(factor_logits, chunk_targets, reduction="none")
                    out[start:end] = (xent / xent_norm).to(SCORE_DTYPE)
        return ScoreBatch(raw_scores=out, bucket_scores=out)


class HybridScoreFunction(BaseScoreFunction):
    def __init__(
        self,
        device: torch.device,
        p: int,
        search_D: int,
        checkpoint_path: str,
        score_chunk_size: int,
        projlen_weight: float,
        confusion_weight: float,
        confusion_metric_type: str = "entropy",
    ):
        super().__init__(device)
        self.projlen = ProjlenScoreFunction(device=device, search_D=search_D, bucket_mode="exact")
        self.confusion = ModelConfusionScoreFunction(
            checkpoint_path=checkpoint_path,
            device=device,
            score_chunk_size=score_chunk_size,
            expected_p=p,
            metric_type=confusion_metric_type,
        )
        self.projlen_weight = float(projlen_weight)
        self.confusion_weight = float(confusion_weight)
        self.total_weight = self.projlen_weight + self.confusion_weight
        if self.total_weight <= 0:
            raise ValueError("projlen_weight + confusion_weight must be positive")

    def score_batch(
        self,
        tensors: torch.Tensor,
        min_degrees: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        target_factor_ids: Optional[torch.Tensor] = None,
    ) -> ScoreBatch:
        proj = self.projlen.score_batch(
            tensors, min_degrees=min_degrees, lengths=lengths, target_factor_ids=target_factor_ids
        )
        conf = self.confusion.score_batch(
            tensors, min_degrees=min_degrees, lengths=lengths, target_factor_ids=target_factor_ids
        )
        combined = (
            self.projlen_weight * proj.bucket_scores + self.confusion_weight * conf.bucket_scores
        ) / self.total_weight
        return ScoreBatch(raw_scores=combined, bucket_scores=combined)


class FrontierDistanceScoreFunction(BaseScoreFunction):
    def __init__(self, device: torch.device, projlen_weight: float, confusion_weight: float):
        super().__init__(device)
        self.projlen_weight = float(projlen_weight)
        self.confusion_weight = float(confusion_weight)
        self.total_weight = self.projlen_weight + self.confusion_weight
        if self.total_weight <= 0:
            raise ValueError("projlen_weight + confusion_weight must be positive")

    def score_batch(
        self,
        tensors: torch.Tensor,
        min_degrees: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        target_factor_ids: Optional[torch.Tensor] = None,
    ) -> ScoreBatch:
        del tensors
        del min_degrees
        del lengths
        del target_factor_ids
        raise RuntimeError("FrontierDistanceScoreFunction is level-relative and must be used via process_level")


def _update_best_xent_by_projlen(
    best_xent_by_projlen: Dict[int, float],
    projlens: torch.Tensor,
    averaged_xent: torch.Tensor,
) -> None:
    unique_projlen = torch.unique(projlens.to(torch.int32))
    for projlen in unique_projlen.tolist():
        mask = projlens == int(projlen)
        best_for_projlen = float(averaged_xent[mask].max().item())
        prev = best_xent_by_projlen.get(int(projlen))
        if prev is None or best_for_projlen > prev:
            best_xent_by_projlen[int(projlen)] = best_for_projlen


def _build_projlen_xent_frontier(
    best_xent_by_projlen: Dict[int, float],
    level: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    if not best_xent_by_projlen:
        raise ValueError("Need at least one candidate to build a frontier")

    denom = max(1.0, 3.0 * float(level))
    frontier_proj_bad: List[float] = []
    frontier_x_bad: List[float] = []
    frontier_summary: List[dict] = []
    best_so_far = float("-inf")
    for projlen in sorted(best_xent_by_projlen):
        avg5_xent = max(0.0, min(1.0, float(best_xent_by_projlen[projlen])))
        if avg5_xent <= best_so_far:
            continue
        best_so_far = avg5_xent
        proj_bad = max(0.0, min(1.0, (float(projlen) - 1.0) / denom))
        x_bad = 1.0 - avg5_xent
        frontier_proj_bad.append(proj_bad)
        frontier_x_bad.append(x_bad)
        frontier_summary.append(
            {
                "projlen": int(projlen),
                "normalized_projlen": proj_bad,
                "avg5_target_xent": avg5_xent,
                "target_xent_badness": x_bad,
            }
        )

    return (
        torch.tensor(frontier_proj_bad, dtype=SCORE_DTYPE, device=device),
        torch.tensor(frontier_x_bad, dtype=SCORE_DTYPE, device=device),
        frontier_summary,
    )


def _frontier_distance_score_batch(
    projlens: torch.Tensor,
    averaged_xent: torch.Tensor,
    level: int,
    frontier_proj_bad: torch.Tensor,
    frontier_x_bad: torch.Tensor,
    projlen_weight: float,
    confusion_weight: float,
) -> ScoreBatch:
    total_weight = float(projlen_weight + confusion_weight)
    if total_weight <= 0:
        raise ValueError("projlen_weight + confusion_weight must be positive")
    if frontier_proj_bad.numel() == 0 or frontier_x_bad.numel() == 0:
        raise ValueError("Frontier tensors must be non-empty")

    denom = max(1.0, 3.0 * float(level))
    proj_bad = torch.clamp((projlens.to(SCORE_DTYPE) - 1.0) / denom, 0.0, 1.0)
    x_bad = torch.clamp(1.0 - averaged_xent.to(SCORE_DTYPE), 0.0, 1.0)

    delta_proj = proj_bad.unsqueeze(1) - frontier_proj_bad.unsqueeze(0)
    delta_x = x_bad.unsqueeze(1) - frontier_x_bad.unsqueeze(0)
    dominates = (delta_proj >= -1e-7) & (delta_x >= -1e-7)
    weighted_sq = (
        float(projlen_weight) * torch.clamp(delta_proj, min=0.0).square()
        + float(confusion_weight) * torch.clamp(delta_x, min=0.0).square()
    ) / total_weight
    distances = torch.sqrt(weighted_sq)
    inf = torch.full_like(distances, float("inf"))
    frontier_distance = torch.where(dominates, distances, inf).min(dim=1).values

    utopia_distance = torch.sqrt(
        (
            float(projlen_weight) * proj_bad.square()
            + float(confusion_weight) * x_bad.square()
        ) / total_weight
    )
    raw_scores = frontier_distance + 1e-3 * utopia_distance
    bucket_scores = torch.clamp(raw_scores, 0.0, 1.0)
    return ScoreBatch(raw_scores=raw_scores, bucket_scores=bucket_scores)


class GPUBuckets:
    def __init__(self, bucket_size: int, device: torch.device):
        self.bucket_size = bucket_size
        self.device = device
        self.data: Dict[int, Tuple[torch.Tensor, ...]] = {}

    def add_chunk(
        self,
        tensors: torch.Tensor,
        min_degrees: torch.Tensor,
        words: torch.Tensor,
        lengths: torch.Tensor,
        last_factor_ids: torch.Tensor,
        xent_history: torch.Tensor,
        xent_max: torch.Tensor,
        raw_scores: torch.Tensor,
        bucket_ids: torch.Tensor,
        disable_cap: bool = False,
    ) -> None:
        if tensors.shape[0] == 0:
            return

        priorities = torch.rand(tensors.shape[0], device=self.device, dtype=SCORE_DTYPE)
        unique_ids = torch.unique(bucket_ids)
        for bucket_id in unique_ids.tolist():
            mask = bucket_ids == bucket_id
            new_tensors = tensors[mask].to(MATRIX_DTYPE)
            new_min_degrees = min_degrees[mask].to(LENGTH_DTYPE)
            new_words = words[mask].to(WORD_DTYPE)
            new_lengths = lengths[mask].to(LENGTH_DTYPE)
            new_last = last_factor_ids[mask].to(torch.int16)
            new_xent_history = xent_history[mask].to(SCORE_DTYPE)
            new_xent_max = xent_max[mask].to(SCORE_DTYPE)
            new_scores = raw_scores[mask].to(SCORE_DTYPE)
            new_priorities = priorities[mask]

            if bucket_id in self.data:
                (
                    old_tensors,
                    old_min_degrees,
                    old_words,
                    old_lengths,
                    old_last,
                    old_xent_history,
                    old_xent_max,
                    old_scores,
                    old_priorities,
                ) = self.data[bucket_id]
                merged_tensors = torch.cat([old_tensors, new_tensors], dim=0)
                merged_min_degrees = torch.cat([old_min_degrees, new_min_degrees], dim=0)
                merged_words = torch.cat([old_words, new_words], dim=0)
                merged_lengths = torch.cat([old_lengths, new_lengths], dim=0)
                merged_last = torch.cat([old_last, new_last], dim=0)
                merged_xent_history = torch.cat([old_xent_history, new_xent_history], dim=0)
                merged_xent_max = torch.cat([old_xent_max, new_xent_max], dim=0)
                merged_scores = torch.cat([old_scores, new_scores], dim=0)
                merged_priorities = torch.cat([old_priorities, new_priorities], dim=0)
            else:
                merged_tensors = new_tensors
                merged_min_degrees = new_min_degrees
                merged_words = new_words
                merged_lengths = new_lengths
                merged_last = new_last
                merged_xent_history = new_xent_history
                merged_xent_max = new_xent_max
                merged_scores = new_scores
                merged_priorities = new_priorities

            if (not disable_cap) and merged_tensors.shape[0] > self.bucket_size:
                _, topk_idx = torch.topk(merged_priorities, self.bucket_size, largest=False)
                merged_tensors = merged_tensors[topk_idx]
                merged_min_degrees = merged_min_degrees[topk_idx]
                merged_words = merged_words[topk_idx]
                merged_lengths = merged_lengths[topk_idx]
                merged_last = merged_last[topk_idx]
                merged_xent_history = merged_xent_history[topk_idx]
                merged_xent_max = merged_xent_max[topk_idx]
                merged_scores = merged_scores[topk_idx]
                merged_priorities = merged_priorities[topk_idx]

            self.data[bucket_id] = (
                merged_tensors,
                merged_min_degrees,
                merged_words,
                merged_lengths,
                merged_last,
                merged_xent_history,
                merged_xent_max,
                merged_scores,
                merged_priorities,
            )

    def materialize(self) -> FrontierBatch:
        if not self.data:
            raise RuntimeError("No bucket data to materialize")

        ordered = sorted(self.data.items(), key=lambda item: item[0])
        tensors = torch.cat([entry[1][0] for entry in ordered], dim=0)
        min_degrees = torch.cat([entry[1][1] for entry in ordered], dim=0)
        words = torch.cat([entry[1][2] for entry in ordered], dim=0)
        lengths = torch.cat([entry[1][3] for entry in ordered], dim=0)
        last_factor_ids = torch.cat([entry[1][4] for entry in ordered], dim=0)
        xent_history = torch.cat([entry[1][5] for entry in ordered], dim=0)
        xent_max = torch.cat([entry[1][6] for entry in ordered], dim=0)
        scores = torch.cat([entry[1][7] for entry in ordered], dim=0)
        bucket_ids = torch.cat(
            [
                torch.full((entry[1][0].shape[0],), entry[0], dtype=torch.long, device=self.device)
                for entry in ordered
            ],
            dim=0,
        )
        return FrontierBatch(
            tensors=tensors,
            min_degrees=min_degrees,
            words=words,
            lengths=lengths,
            last_factor_ids=last_factor_ids,
            xent_history=xent_history,
            xent_max=xent_max,
            scores=scores,
            bucket_ids=bucket_ids,
        )

    def bucket_counts(self) -> Dict[int, int]:
        return {bucket_id: int(data[0].shape[0]) for bucket_id, data in sorted(self.data.items())}

    def total_count(self) -> int:
        return sum(int(data[0].shape[0]) for data in self.data.values())


class ReservoirSearchBraidmod:
    def __init__(self, config: SearchConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.tables = _build_garside_tables(config.p, self.device)
        self.scorer = self._build_scorer()
        self.projlen_scorer: Optional[ProjlenScoreFunction] = None
        self.target_xent_scorer: Optional[ModelConfusionScoreFunction] = None
        if is_switching_score_type(config.score_type) or is_frontier_score_type(config.score_type):
            self.projlen_scorer = ProjlenScoreFunction(
                device=self.device,
                search_D=self.config.search_D,
                bucket_mode=self.config.projlen_bucket_mode,
            )
            if not self.config.checkpoint:
                raise ValueError("--checkpoint is required for switching or frontier score types")
            self.target_xent_scorer = ModelConfusionScoreFunction(
                checkpoint_path=self.config.checkpoint,
                device=self.device,
                score_chunk_size=self.config.score_chunk_size,
                expected_p=self.config.p,
                metric_type="target_xent",
            )
        self.frontier = _build_initial_frontier(config, self.device)
        self.level_summaries: List[dict] = []
        self.kernel_hits: List[dict] = []
        self.identity_coeff = torch.eye(3, dtype=MATRIX_DTYPE, device=self.device)

    def _build_scorer(self) -> BaseScoreFunction:
        if self.config.score_type == "projlen_then_target_xent_maximize":
            return ProjlenScoreFunction(
                device=self.device,
                search_D=self.config.search_D,
                bucket_mode=self.config.projlen_bucket_mode,
            )
        if self.config.score_type == "projlen":
            return ProjlenScoreFunction(
                device=self.device,
                search_D=self.config.search_D,
                bucket_mode=self.config.projlen_bucket_mode,
            )
        if self.config.score_type == "confusion":
            if not self.config.checkpoint:
                raise ValueError("--checkpoint is required for score-type=confusion")
            return ModelConfusionScoreFunction(
                checkpoint_path=self.config.checkpoint,
                device=self.device,
                score_chunk_size=self.config.score_chunk_size,
                expected_p=self.config.p,
                metric_type="entropy",
            )
        if self.config.score_type == "target_xent":
            if not self.config.checkpoint:
                raise ValueError("--checkpoint is required for score-type=target_xent")
            return ModelConfusionScoreFunction(
                checkpoint_path=self.config.checkpoint,
                device=self.device,
                score_chunk_size=self.config.score_chunk_size,
                expected_p=self.config.p,
                metric_type="target_xent",
            )
        if self.config.score_type == "target_xent_maximize":
            if not self.config.checkpoint:
                raise ValueError("--checkpoint is required for score-type=target_xent_maximize")
            return ModelConfusionScoreFunction(
                checkpoint_path=self.config.checkpoint,
                device=self.device,
                score_chunk_size=self.config.score_chunk_size,
                expected_p=self.config.p,
                metric_type="target_xent",
            )
        if self.config.score_type == "target_xent_max":
            if not self.config.checkpoint:
                raise ValueError("--checkpoint is required for score-type=target_xent_max")
            return ModelConfusionScoreFunction(
                checkpoint_path=self.config.checkpoint,
                device=self.device,
                score_chunk_size=self.config.score_chunk_size,
                expected_p=self.config.p,
                metric_type="target_xent",
            )
        if self.config.score_type == "target_xent_max_maximize":
            if not self.config.checkpoint:
                raise ValueError("--checkpoint is required for score-type=target_xent_max_maximize")
            return ModelConfusionScoreFunction(
                checkpoint_path=self.config.checkpoint,
                device=self.device,
                score_chunk_size=self.config.score_chunk_size,
                expected_p=self.config.p,
                metric_type="target_xent",
            )
        if self.config.score_type == "hybrid":
            if not self.config.checkpoint:
                raise ValueError("--checkpoint is required for score-type=hybrid")
            return HybridScoreFunction(
                device=self.device,
                p=self.config.p,
                search_D=self.config.search_D,
                checkpoint_path=self.config.checkpoint,
                score_chunk_size=self.config.score_chunk_size,
                projlen_weight=self.config.projlen_weight,
                confusion_weight=self.config.confusion_weight,
                confusion_metric_type="entropy",
            )
        if self.config.score_type == "hybrid_target_xent":
            if not self.config.checkpoint:
                raise ValueError("--checkpoint is required for score-type=hybrid_target_xent")
            return HybridScoreFunction(
                device=self.device,
                p=self.config.p,
                search_D=self.config.search_D,
                checkpoint_path=self.config.checkpoint,
                score_chunk_size=self.config.score_chunk_size,
                projlen_weight=self.config.projlen_weight,
                confusion_weight=self.config.confusion_weight,
                confusion_metric_type="target_xent",
            )
        if self.config.score_type == "frontier_target_xent":
            if not self.config.checkpoint:
                raise ValueError("--checkpoint is required for score-type=frontier_target_xent")
            return FrontierDistanceScoreFunction(
                device=self.device,
                projlen_weight=self.config.projlen_weight,
                confusion_weight=self.config.confusion_weight,
            )
        raise ValueError(f"Unsupported score type: {self.config.score_type}")

    def _score_type_for_level(self, level: int) -> str:
        if self.config.score_type == "projlen_then_target_xent_maximize":
            if level <= self.config.score_switch_length:
                return "projlen"
            return "target_xent_maximize"
        return self.config.score_type

    def _build_frontier_target_xent_level(
        self,
        level: int,
        braid_indices: torch.Tensor,
        suffix_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
        if self.projlen_scorer is None or self.target_xent_scorer is None:
            raise RuntimeError("frontier_target_xent requires both projlen and target_xent scorers")

        best_xent_by_projlen: Dict[int, float] = {}
        num_candidates = int(suffix_ids.shape[0])
        for start in range(0, num_candidates, self.config.expansion_chunk_size):
            end = min(start + self.config.expansion_chunk_size, num_candidates)
            chunk_braid_idx = braid_indices[start:end]
            chunk_suffix_ids = suffix_ids[start:end]

            parent_tensors = self.frontier.tensors[chunk_braid_idx]
            parent_min_degrees = self.frontier.min_degrees[chunk_braid_idx]
            parent_lengths = self.frontier.lengths[chunk_braid_idx]
            parent_xent_history = self.frontier.xent_history[chunk_braid_idx]

            child_tensors, child_min_degrees = _right_multiply_simple_batch(
                parent_tensors=parent_tensors,
                parent_min_degrees=parent_min_degrees,
                suffix_ids=chunk_suffix_ids,
                simple_shift_mats=self.tables.simple_shift_mats,
                p=self.config.p,
            )
            child_lengths = parent_lengths + 1
            child_last = chunk_suffix_ids.to(torch.int16)

            projlen_score_batch = self.projlen_scorer.score_batch(
                child_tensors,
                min_degrees=child_min_degrees,
                lengths=child_lengths,
                target_factor_ids=child_last,
            )
            target_xent_score_batch = self.target_xent_scorer.score_batch(
                child_tensors,
                min_degrees=child_min_degrees,
                lengths=child_lengths,
                target_factor_ids=child_last,
            )
            _, averaged_scores = update_xent_history(
                parent_xent_history,
                target_xent_score_batch.raw_scores,
                child_lengths,
            )
            _update_best_xent_by_projlen(
                best_xent_by_projlen,
                projlen_score_batch.raw_scores.to(torch.int32),
                averaged_scores,
            )

        return _build_projlen_xent_frontier(best_xent_by_projlen, level=level, device=self.device)

    def _select_best(self, frontier: FrontierBatch) -> FrontierBatch:
        if frontier.scores is None:
            return frontier

        total = frontier.size
        if self.config.use_best <= 0 or total <= self.config.use_best:
            return frontier

        if frontier.bucket_ids is None:
            _, idx = torch.topk(frontier.scores, self.config.use_best, largest=False)
            return FrontierBatch(
                tensors=frontier.tensors[idx],
                min_degrees=frontier.min_degrees[idx],
                words=frontier.words[idx],
                lengths=frontier.lengths[idx],
                last_factor_ids=frontier.last_factor_ids[idx],
                xent_history=frontier.xent_history[idx] if frontier.xent_history is not None else None,
                xent_max=frontier.xent_max[idx] if frontier.xent_max is not None else None,
                scores=frontier.scores[idx],
                bucket_ids=None,
            )

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

            choice = torch.randperm(bucket_count, device=self.device)[:remaining]
            selected_idx_parts.append(bucket_idx[choice])
            total_selected += remaining
            break

        idx = torch.cat(selected_idx_parts, dim=0)
        return FrontierBatch(
            tensors=frontier.tensors[idx],
            min_degrees=frontier.min_degrees[idx],
            words=frontier.words[idx],
            lengths=frontier.lengths[idx],
            last_factor_ids=frontier.last_factor_ids[idx],
            xent_history=frontier.xent_history[idx] if frontier.xent_history is not None else None,
            xent_max=frontier.xent_max[idx] if frontier.xent_max is not None else None,
            scores=frontier.scores[idx],
            bucket_ids=frontier.bucket_ids[idx] if frontier.bucket_ids is not None else None,
        )

    def _serialize_candidate(self, frontier: FrontierBatch, idx: int, score_type: Optional[str] = None) -> dict:
        length = int(frontier.lengths[idx].item())
        factor_ids = frontier.words[idx, :length].tolist()
        factors = [list(PERMUTATIONS_S4[factor_id]) for factor_id in factor_ids]
        score = float(frontier.scores[idx].item()) if frontier.scores is not None else None
        score_type = self.config.score_type if score_type is None else score_type
        if score is not None and is_maximizing_score_type(score_type):
            score = -score
        result = {
            "score": score,
            "garside_length": length,
            "burau_min_degree": int(frontier.min_degrees[idx].item()),
            "factor_ids": [int(x) for x in factor_ids],
            "gnf_factors": factors,
        }
        if frontier.bucket_ids is not None:
            result["bucket_id"] = int(frontier.bucket_ids[idx].item())
        return result

    def _record_kernel_hits(
        self,
        words: torch.Tensor,
        lengths: torch.Tensor,
        scores: torch.Tensor,
        score_type: str,
        identity_match: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        delta_match: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[int, int]:
        identity_mask, identity_support, identity_coeffs, identity_scalar_degrees = identity_match
        delta_mask, delta_support, delta_coeffs, delta_scalar_degrees = delta_match

        identity_count = int(identity_mask.sum().item())
        delta_count = int(delta_mask.sum().item())
        hit_mask = identity_mask | delta_mask
        hit_indices = torch.where(hit_mask)[0]
        if hit_indices.numel() == 0:
            return identity_count, delta_count

        remaining_slots = self.config.save_kernel_hits - len(self.kernel_hits)
        if remaining_slots <= 0:
            return identity_count, delta_count

        for idx in hit_indices[:remaining_slots].tolist():
            length = int(lengths[idx].item())
            factor_ids = words[idx, :length].tolist()
            score = float(scores[idx].item())
            if is_maximizing_score_type(score_type):
                score = -score
            if bool(identity_mask[idx].item()):
                kernel_type = "identity"
                delta_power = 0
                support_degree = int(identity_support[idx].item())
                scalar_degree = int(identity_scalar_degrees[idx].item())
                scalar_coeff = int(identity_coeffs[idx].item())
            else:
                kernel_type = "delta"
                delta_power = 1
                support_degree = int(delta_support[idx].item())
                scalar_degree = int(delta_scalar_degrees[idx].item())
                scalar_coeff = int(delta_coeffs[idx].item())

            self.kernel_hits.append(
                {
                    "score": score,
                    "kernel_type": kernel_type,
                    "delta_power": delta_power,
                    "support_degree": support_degree,
                    "scalar_degree": scalar_degree,
                    "scalar_coeff_mod_p": scalar_coeff,
                    "garside_length": length,
                    "factor_ids": [int(x) for x in factor_ids],
                    "gnf_factors": [list(PERMUTATIONS_S4[factor_id]) for factor_id in factor_ids],
                }
            )
        return identity_count, delta_count

    def process_level(self, level: int) -> bool:
        level_start = time.time()
        is_bootstrap = level <= self.config.bootstrap_length
        active_score_type = self._score_type_for_level(level)
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

        mode = "BOOTSTRAP" if is_bootstrap else "SAMPLING"
        print(f"\nLevel {level} ({mode})")
        print(f"  parents={self.frontier.size} candidates={num_candidates}")
        print(f"  score_type={active_score_type}")

        buckets = GPUBuckets(bucket_size=self.config.bucket_size, device=self.device)
        incoming_bucket_counts: Dict[int, int] = {}
        identity_hit_count = 0
        delta_hit_count = 0
        score_min = None
        score_max = None
        score_sum = 0.0

        matmul_time = 0.0
        score_time = 0.0
        bucket_time = 0.0
        frontier_time = 0.0
        frontier_summary: Optional[List[dict]] = None
        frontier_proj_bad = None
        frontier_x_bad = None
        if self.config.score_type == "frontier_target_xent":
            t0 = time.time()
            frontier_proj_bad, frontier_x_bad, frontier_summary = self._build_frontier_target_xent_level(
                level=level,
                braid_indices=braid_indices,
                suffix_ids=suffix_ids,
            )
            frontier_time = time.time() - t0
            print(f"  frontier_points={len(frontier_summary)}")

        for start in range(0, num_candidates, self.config.expansion_chunk_size):
            end = min(start + self.config.expansion_chunk_size, num_candidates)
            chunk_braid_idx = braid_indices[start:end]
            chunk_suffix_ids = suffix_ids[start:end]

            parent_tensors = self.frontier.tensors[chunk_braid_idx]
            parent_min_degrees = self.frontier.min_degrees[chunk_braid_idx]
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
            matmul_time += time.time() - t0

            child_words = parent_words.clone()
            row_idx = torch.arange(child_words.shape[0], device=self.device)
            child_words[row_idx, parent_lengths.long()] = chunk_suffix_ids.to(WORD_DTYPE)
            child_lengths = parent_lengths + 1
            child_last = chunk_suffix_ids.to(torch.int16)
            parent_xent_history = self.frontier.xent_history[chunk_braid_idx]
            parent_xent_max = self.frontier.xent_max[chunk_braid_idx]

            t0 = time.time()
            child_xent_history = parent_xent_history
            child_xent_max = parent_xent_max
            bucketizer = self.scorer
            if self.config.score_type == "frontier_target_xent":
                if (
                    self.projlen_scorer is None
                    or self.target_xent_scorer is None
                    or frontier_proj_bad is None
                    or frontier_x_bad is None
                ):
                    raise RuntimeError("frontier_target_xent requires initialized scorers and frontier tensors")
                projlen_score_batch = self.projlen_scorer.score_batch(
                    child_tensors,
                    min_degrees=child_min_degrees,
                    lengths=child_lengths,
                    target_factor_ids=child_last,
                )
                target_xent_score_batch = self.target_xent_scorer.score_batch(
                    child_tensors,
                    min_degrees=child_min_degrees,
                    lengths=child_lengths,
                    target_factor_ids=child_last,
                )
                child_xent_history, averaged_scores = update_xent_history(
                    parent_xent_history,
                    target_xent_score_batch.raw_scores,
                    child_lengths,
                )
                score_batch = _frontier_distance_score_batch(
                    projlens=projlen_score_batch.raw_scores,
                    averaged_xent=averaged_scores,
                    level=level,
                    frontier_proj_bad=frontier_proj_bad,
                    frontier_x_bad=frontier_x_bad,
                    projlen_weight=self.config.projlen_weight,
                    confusion_weight=self.config.confusion_weight,
                )
            elif self.config.score_type == "projlen_then_target_xent_maximize":
                if self.projlen_scorer is None or self.target_xent_scorer is None:
                    raise RuntimeError("Switching score policy requires both projlen and target_xent scorers")
                projlen_score_batch = self.projlen_scorer.score_batch(
                    child_tensors,
                    min_degrees=child_min_degrees,
                    lengths=child_lengths,
                    target_factor_ids=child_last,
                )
                target_xent_score_batch = self.target_xent_scorer.score_batch(
                    child_tensors,
                    min_degrees=child_min_degrees,
                    lengths=child_lengths,
                    target_factor_ids=child_last,
                )
                child_xent_history, averaged_scores = update_xent_history(
                    parent_xent_history,
                    target_xent_score_batch.raw_scores,
                    child_lengths,
                )
                if level <= self.config.score_switch_length:
                    score_batch = projlen_score_batch
                    bucketizer = self.projlen_scorer
                else:
                    score_batch = maximize_score_transform(averaged_scores)
                    bucketizer = self.target_xent_scorer
            else:
                score_batch = self.scorer.score_batch(
                    child_tensors,
                    min_degrees=child_min_degrees,
                    lengths=child_lengths,
                    target_factor_ids=child_last,
                )
                if self.config.score_type == "target_xent":
                    child_xent_history, averaged_scores = update_xent_history(
                        parent_xent_history,
                        score_batch.raw_scores,
                        child_lengths,
                    )
                    score_batch = ScoreBatch(raw_scores=averaged_scores, bucket_scores=averaged_scores)
                elif self.config.score_type == "target_xent_maximize":
                    child_xent_history, averaged_scores = update_xent_history(
                        parent_xent_history,
                        score_batch.raw_scores,
                        child_lengths,
                    )
                    score_batch = maximize_score_transform(averaged_scores)
                elif self.config.score_type == "target_xent_max":
                    child_xent_max = torch.maximum(parent_xent_max.to(SCORE_DTYPE), score_batch.raw_scores.to(SCORE_DTYPE))
                    score_batch = ScoreBatch(raw_scores=child_xent_max, bucket_scores=child_xent_max)
                elif self.config.score_type == "target_xent_max_maximize":
                    child_xent_max = torch.maximum(parent_xent_max.to(SCORE_DTYPE), score_batch.raw_scores.to(SCORE_DTYPE))
                    score_batch = maximize_score_transform(child_xent_max)
            bucket_ids = bucketizer.bucketize(score_batch.bucket_scores, self.config.num_buckets)
            score_time += time.time() - t0

            identity_match = _monomial_target_match_batch(
                child_tensors,
                min_degrees=child_min_degrees,
                target_coeff=self.identity_coeff,
                target_degree=0,
                p=self.config.p,
            )
            delta_match = _monomial_target_match_batch(
                child_tensors,
                min_degrees=child_min_degrees,
                target_coeff=self.tables.delta_coeff,
                target_degree=self.tables.delta_degree,
                p=self.config.p,
            )
            new_identity_hits, new_delta_hits = self._record_kernel_hits(
                child_words,
                child_lengths,
                score_batch.raw_scores,
                active_score_type,
                identity_match=identity_match,
                delta_match=delta_match,
            )
            identity_hit_count += new_identity_hits
            delta_hit_count += new_delta_hits

            score_min_val = float(score_batch.raw_scores.min().item())
            score_max_val = float(score_batch.raw_scores.max().item())
            if is_maximizing_score_type(active_score_type):
                score_min_val, score_max_val = -score_max_val, -score_min_val
            score_min = score_min_val if score_min is None else min(score_min, score_min_val)
            score_max = score_max_val if score_max is None else max(score_max, score_max_val)
            score_sum_chunk = float(score_batch.raw_scores.sum().item())
            if is_maximizing_score_type(active_score_type):
                score_sum_chunk = -score_sum_chunk
            score_sum += score_sum_chunk

            unique_buckets, counts = torch.unique(bucket_ids, return_counts=True)
            for bucket_id, count in zip(unique_buckets.tolist(), counts.tolist()):
                incoming_bucket_counts[bucket_id] = incoming_bucket_counts.get(bucket_id, 0) + int(count)

            t0 = time.time()
            buckets.add_chunk(
                tensors=child_tensors,
                min_degrees=child_min_degrees,
                words=child_words,
                lengths=child_lengths,
                last_factor_ids=child_last,
                xent_history=child_xent_history,
                xent_max=child_xent_max,
                raw_scores=score_batch.raw_scores,
                bucket_ids=bucket_ids,
                disable_cap=is_bootstrap,
            )
            bucket_time += time.time() - t0

        materialized = buckets.materialize()
        selected = materialized if is_bootstrap else self._select_best(materialized)
        level_time = time.time() - level_start

        score_mean = score_sum / float(num_candidates)
        best_idx = int(torch.argmin(materialized.scores).item())
        best_candidate = self._serialize_candidate(materialized, best_idx, score_type=active_score_type)

        summary = {
            "level": level,
            "num_parents": self.frontier.size,
            "num_candidates": num_candidates,
            "num_kept_after_reservoir": materialized.size,
            "num_selected_for_next_level": selected.size,
            "nonempty_buckets": len(buckets.data),
            "incoming_bucket_counts": incoming_bucket_counts,
            "kept_bucket_counts": buckets.bucket_counts(),
            "score_min": score_min,
            "score_mean": score_mean,
            "score_max": score_max,
            "kernel_hits_found": identity_hit_count + delta_hit_count,
            "kernel_hit_type_counts": {
                "identity": identity_hit_count,
                "delta": delta_hit_count,
            },
            "best_candidate": best_candidate,
            "timing_sec": {
                "matmul": round(matmul_time, 4),
                "frontier": round(frontier_time, 4),
                "score": round(score_time, 4),
                "reservoir": round(bucket_time, 4),
                "total": round(level_time, 4),
            },
        }
        if frontier_summary is not None:
            summary["frontier_points"] = frontier_summary
        self.level_summaries.append(summary)
        self.frontier = selected

        print(
            f"  kept={materialized.size} selected={selected.size} buckets={len(buckets.data)} "
            f"score=[{score_min:.4f}, {score_mean:.4f}, {score_max:.4f}] "
            f"kernel_hits={identity_hit_count + delta_hit_count} "
            f"(identity={identity_hit_count} delta={delta_hit_count})"
        )
        print(
            f"  timing: matmul={matmul_time:.2f}s frontier={frontier_time:.2f}s score={score_time:.2f}s "
            f"reservoir={bucket_time:.2f}s total={level_time:.2f}s"
        )
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

        final_score_type = self._score_type_for_level(completed_levels) if completed_levels > 0 else self.config.score_type
        best_candidates = [
            self._serialize_candidate(final_frontier, int(idx.item()), score_type=final_score_type) for idx in top_idx
        ]
        kernel_hit_type_counts = {
            "identity": sum(1 for hit in self.kernel_hits if hit["kernel_type"] == "identity"),
            "delta": sum(1 for hit in self.kernel_hits if hit["kernel_type"] == "delta"),
        }
        result = {
            "config": asdict(self.config),
            "device_resolved": str(self.device),
            "completed_levels": completed_levels,
            "level_summaries": self.level_summaries,
            "best_candidates": best_candidates,
            "kernel_hits": self.kernel_hits,
            "kernel_hit_type_counts": kernel_hit_type_counts,
            "total_time_sec": round(total_time, 4),
        }
        return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reservoir-sampled Garside search in braidmod conventions. "
            "Each level expands positive GNF braids by one simple suffix, buckets by a discretized score, "
            "keeps a uniform reservoir in each bucket, then advances only the best survivors."
        )
    )
    parser.add_argument("--p", type=int, default=5, help="Modulus p for Burau arithmetic")
    parser.add_argument("--max-length", type=int, required=True, help="Maximum Garside length to search")
    parser.add_argument(
        "--search-D",
        type=int,
        help="Projective support window for the search tensors; default is 4*max_length+1",
    )
    parser.add_argument("--bucket-size", type=int, default=4000, help="Reservoir size per score bucket")
    parser.add_argument("--num-buckets", type=int, default=32, help="Number of score buckets")
    parser.add_argument("--use-best", type=int, default=0, help="How many best survivors to expand next; 0 means all")
    parser.add_argument(
        "--bootstrap-length",
        type=int,
        default=5,
        help="Number of initial levels to expand without use-best pruning or bucket caps",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reservoir priorities")
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, cuda:0, ...")
    parser.add_argument(
        "--expansion-chunk-size",
        type=int,
        default=50000,
        help="How many candidate children to expand per chunk",
    )
    parser.add_argument(
        "--score-chunk-size",
        type=int,
        default=8192,
        help="How many tensors to score per model-forward chunk",
    )
    parser.add_argument(
        "--score-type",
        choices=("projlen", "projlen_then_target_xent_maximize", "confusion", "target_xent", "target_xent_maximize", "target_xent_max", "target_xent_max_maximize", "hybrid", "hybrid_target_xent", "frontier_target_xent"),
        default="projlen",
        help="Scoring function to minimize",
    )
    parser.add_argument(
        "--score-switch-length",
        type=int,
        default=30,
        help="For switching score policies, use the first score through this Garside length before switching.",
    )
    parser.add_argument(
        "--projlen-bucket-mode",
        choices=("exact", "score"),
        default="exact",
        help="For score-type=projlen, bucket either by exact projlen or by generic fixed value buckets.",
    )
    parser.add_argument("--checkpoint", help="Model checkpoint for confusion, target_xent, target_xent_max, hybrid, hybrid_target_xent, or frontier_target_xent scoring")
    parser.add_argument("--projlen-weight", type=float, default=1.0, help="Weight on normalized projlen for hybrid or frontier_target_xent scoring")
    parser.add_argument("--confusion-weight", type=float, default=1.0, help="Weight on model score for hybrid or frontier_target_xent scoring")
    parser.add_argument("--topk-save", type=int, default=50, help="How many best final candidates to save")
    parser.add_argument(
        "--save-kernel-hits",
        type=int,
        default=100,
        help="Maximum number of kernel hits (identity or Delta power) to keep in the JSON output",
    )
    parser.add_argument(
        "--save-scalar-hits",
        dest="save_kernel_hits",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--out-json", help="Optional JSON path for the search result")
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
    if args.score_switch_length < 0:
        raise ValueError("--score-switch-length must be non-negative")

    search_D = args.search_D if args.search_D is not None else 4 * args.max_length + 1
    config = SearchConfig(
        p=args.p,
        max_length=args.max_length,
        search_D=search_D,
        bucket_size=args.bucket_size,
        num_buckets=args.num_buckets,
        use_best=args.use_best,
        bootstrap_length=args.bootstrap_length,
        seed=args.seed,
        device=args.device,
        expansion_chunk_size=args.expansion_chunk_size,
        score_chunk_size=args.score_chunk_size,
        score_type=args.score_type,
        score_switch_length=args.score_switch_length,
        projlen_bucket_mode=args.projlen_bucket_mode,
        checkpoint=args.checkpoint,
        projlen_weight=args.projlen_weight,
        confusion_weight=args.confusion_weight,
        topk_save=args.topk_save,
        save_kernel_hits=args.save_kernel_hits,
        out_json=args.out_json,
    )

    set_seed(config.seed)
    search = ReservoirSearchBraidmod(config)
    result = search.run()

    if config.out_json:
        out_path = Path(config.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
