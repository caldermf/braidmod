#!/usr/bin/env python3
"""
Separate reservoir search for low-projlen Burau commutators.

This intentionally lives outside ``reservoir_search_braidmod.py``. It grows a
positive Garside-normal-form braid ``b`` level by level, but scores each
candidate by the projective support width of

    Burau([a, b]) = Burau(a b a^{-1} b^{-1})

for a fixed input braid ``a`` supplied as a signed Artin word.
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from braid_data import GarsideFactor, _simple_braid_tables, burau_mod_p_polynomial_matrix
from reservoir_search_braidmod import (
    LENGTH_DTYPE,
    PERMUTATIONS_S4,
    SCORE_DTYPE,
    WORD_DTYPE,
    FrontierBatch,
    GPUBuckets,
    _build_expansion_indices,
    _build_garside_tables,
    _build_initial_frontier,
    _move_frontier_batch,
    _right_multiply_simple_batch,
    set_seed,
)


@dataclass
class CommutatorSearchConfig:
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


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def parse_signed_word(spec: str) -> Tuple[int, ...]:
    text = spec.replace(",", " ").strip()
    if not text:
        raise ValueError("--reference-braid-word must not be empty")
    out: List[int] = []
    for part in text.split():
        value = int(part)
        if value == 0:
            raise ValueError("Generator index 0 is invalid")
        if abs(value) > 3:
            raise ValueError("This search currently expects B_4 generators in -3..-1 or 1..3")
        out.append(value)
    return tuple(out)


def invert_signed_word(word: Sequence[int]) -> List[int]:
    return [-int(g) for g in reversed(word)]


def poly_matrix_degree_bounds(poly_mat: Sequence[Sequence[Dict[int, int]]]) -> Tuple[int, int]:
    exponents: List[int] = []
    for row in poly_mat:
        for entry in row:
            exponents.extend(entry.keys())
    if not exponents:
        return 0, 0
    return min(exponents), max(exponents)


def poly_matrix_projlen(poly_mat: Sequence[Sequence[Dict[int, int]]]) -> Tuple[int, int]:
    has_nonzero = any(bool(entry) for row in poly_mat for entry in row)
    if not has_nonzero:
        return 0, 0
    min_exp, max_exp = poly_matrix_degree_bounds(poly_mat)
    return max_exp - min_exp + 1, min_exp


def scalar_identity_metadata(poly_mat: Sequence[Sequence[Dict[int, int]]], p: int) -> Optional[dict]:
    del p
    diagonal_entry = None
    for i in range(3):
        for j in range(3):
            entry = poly_mat[i][j]
            if i == j:
                if len(entry) != 1:
                    return None
                if diagonal_entry is None:
                    diagonal_entry = entry
                elif entry != diagonal_entry:
                    return None
            elif entry:
                return None

    if diagonal_entry is None:
        return None
    (scalar_degree, scalar_coeff), = diagonal_entry.items()
    return {
        "kernel_type": "identity",
        "support_degree": int(scalar_degree),
        "scalar_degree": int(scalar_degree),
        "scalar_coeff_mod_p": int(scalar_coeff),
    }


class SignedWordLeftNormalForm:
    def __init__(self, n: int = 4):
        if n != 4:
            raise ValueError("This normalizer currently expects B_4")
        self.tables = _simple_braid_tables(n)
        self.tau = self.tables["tau"]
        self.pair_table = self.tables["pair_table"]
        self.generator_to_perm = self.tables["generator_to_perm"]
        self.left_complements = self._build_left_complements()

    def _build_left_complements(self) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
        complements: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
        for gen, perm in self.generator_to_perm.items():
            del gen
            matches = []
            for candidate in self.tables["simple_words"]:
                if candidate == self.tables["delta"]:
                    continue
                pair_d, pair_factors = self.pair_table[(candidate, perm)]
                if pair_d == 1 and tuple(pair_factors) == ():
                    matches.append(candidate)
            if len(matches) != 1:
                raise RuntimeError(f"Expected a unique left complement for generator perm {perm}, got {matches}")
            complements[perm] = matches[0]
        return complements

    def _append_simple(self, delta_power: int, factors: List[Tuple[int, ...]], perm: Tuple[int, ...]) -> Tuple[int, List[Tuple[int, ...]]]:
        factors = list(factors)
        factors.append(perm)

        changed = True
        while changed:
            changed = False
            for idx in range(len(factors) - 2, -1, -1):
                left = factors[idx]
                right = factors[idx + 1]
                pair_d, pair_factors = self.pair_table[(left, right)]
                if pair_d == 0 and list(pair_factors) == [left, right]:
                    continue

                prefix = factors[:idx]
                suffix = factors[idx + 2 :]
                if pair_d:
                    delta_power += pair_d
                    prefix = [self.tau[p] for p in prefix]
                factors = prefix + list(pair_factors) + suffix
                changed = True
                break

        return delta_power, factors

    def normalize(self, word: Sequence[int]) -> Tuple[int, Tuple[Tuple[int, ...], ...]]:
        delta_power = 0
        factors: List[Tuple[int, ...]] = []

        for g in word:
            if g == 0:
                raise ValueError("Generator index 0 is invalid")
            perm = self.generator_to_perm[abs(int(g))]
            if g > 0:
                delta_power, factors = self._append_simple(delta_power, factors, perm)
                continue

            delta_power -= 1
            factors = [self.tau[p] for p in factors]
            delta_power, factors = self._append_simple(delta_power, factors, self.left_complements[perm])

        return delta_power, tuple(factors)


class ReservoirSearchCommutator:
    def __init__(self, config: CommutatorSearchConfig):
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

    def _factor_ids_to_artin_word(self, factor_ids: Sequence[int]) -> List[int]:
        word: List[int] = []
        for factor_id in factor_ids:
            word.extend(self.factor_artin_words[int(factor_id)])
        return word

    def _commutator_stats_for_factor_ids(self, factor_ids: Sequence[int]) -> dict:
        b_word = self._factor_ids_to_artin_word(factor_ids)
        commutator_word = (
            list(self.reference_word)
            + b_word
            + list(self.reference_inverse_word)
            + invert_signed_word(b_word)
        )
        poly_mat = burau_mod_p_polynomial_matrix(commutator_word, p=self.config.p, n=4)
        projlen, min_degree = poly_matrix_projlen(poly_mat)
        kernel_hit = scalar_identity_metadata(poly_mat, p=self.config.p)
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
            if is_trivial:
                return {
                    "is_trivial": True,
                    "checked_triviality": checked_triviality,
                    "projlen": None,
                    "min_degree": None,
                    "b_artin_word": [int(x) for x in b_word],
                    "commutator_artin_word": [int(x) for x in commutator_word],
                    "kernel_hit": None,
                    "commutator_delta_power": commutator_delta_power,
                    "commutator_gnf_factors": [],
                }

        return {
            "is_trivial": False,
            "checked_triviality": checked_triviality,
            "projlen": int(projlen),
            "min_degree": int(min_degree),
            "b_artin_word": [int(x) for x in b_word],
            "commutator_artin_word": [int(x) for x in commutator_word],
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

    def _select_best(self, frontier: FrontierBatch) -> FrontierBatch:
        if frontier.scores is None:
            return frontier
        select_device = frontier.scores.device

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

            choice = torch.randperm(bucket_count, device=select_device)[:remaining]
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

    def _serialize_candidate(self, frontier: FrontierBatch, idx: int) -> dict:
        length = int(frontier.lengths[idx].item())
        factor_ids = [int(x) for x in frontier.words[idx, :length].tolist()]
        factors = [list(PERMUTATIONS_S4[factor_id]) for factor_id in factor_ids]
        stats = self._commutator_stats_for_factor_ids(factor_ids)
        score = float(frontier.scores[idx].item()) if frontier.scores is not None else float(stats["projlen"])
        result = {
            "score": score,
            "score_label": "commutator_projlen",
            "garside_length": length,
            "burau_min_degree": int(frontier.min_degrees[idx].item()),
            "factor_ids": factor_ids,
            "gnf_factors": factors,
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

    def _score_chunk(
        self,
        words_cpu: torch.Tensor,
        lengths_cpu: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        count = int(lengths_cpu.shape[0])
        kept_positions: List[int] = []
        raw_scores: List[float] = []
        bucket_ids: List[int] = []
        trivial_discarded = 0
        kernel_hits_found = 0

        for idx in range(count):
            length = int(lengths_cpu[idx].item())
            factor_ids = [int(x) for x in words_cpu[idx, :length].tolist()]
            stats = self._commutator_stats_for_factor_ids(factor_ids)
            if stats["is_trivial"]:
                trivial_discarded += 1
                continue
            score = float(stats["projlen"])
            kept_positions.append(idx)
            raw_scores.append(score)
            bucket_ids.append(self._bucket_id_for_score(stats["projlen"], garside_length=length))
            if stats["kernel_hit"] is not None:
                kernel_hits_found += 1
                self._record_kernel_hit(factor_ids, garside_length=length, score=score, hit=stats["kernel_hit"])

        return (
            torch.tensor(kept_positions, dtype=torch.long),
            torch.tensor(raw_scores, dtype=SCORE_DTYPE),
            torch.tensor(bucket_ids, dtype=torch.long),
            trivial_discarded,
            kernel_hits_found,
        )

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

        mode = "BOOTSTRAP" if is_bootstrap else "SAMPLING"
        print(f"\nLevel {level} ({mode})")
        print(f"  parents={self.frontier.size} candidates={num_candidates}")
        print(f"  reference_word={list(self.reference_word)}")
        print("  score_type=commutator_projlen")

        buckets = GPUBuckets(bucket_size=self.config.bucket_size, device=self.device)
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
            parent_words = self.frontier.words[chunk_braid_idx]
            parent_lengths = self.frontier.lengths[chunk_braid_idx]
            parent_xent_history = self.frontier.xent_history[chunk_braid_idx]
            parent_xent_max = self.frontier.xent_max[chunk_braid_idx]

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

            t0 = time.time()
            chunk_words_cpu = child_words.to(device="cpu")
            chunk_lengths_cpu = child_lengths.to(device="cpu")
            kept_idx_parts: List[torch.Tensor] = []
            raw_score_parts: List[torch.Tensor] = []
            bucket_id_parts: List[torch.Tensor] = []
            for score_start in range(0, chunk_words_cpu.shape[0], self.config.score_chunk_size):
                score_end = min(score_start + self.config.score_chunk_size, chunk_words_cpu.shape[0])
                sub_kept_idx, sub_scores, sub_buckets, sub_trivial, sub_hits = self._score_chunk(
                    chunk_words_cpu[score_start:score_end],
                    chunk_lengths_cpu[score_start:score_end],
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
            raw_scores_cpu = torch.cat(raw_score_parts, dim=0)
            bucket_ids_cpu = torch.cat(bucket_id_parts, dim=0)
            nontrivial_candidate_count += int(kept_idx.numel())

            kept_idx_device = kept_idx.to(device=self.device)
            child_tensors = child_tensors[kept_idx_device]
            child_min_degrees = child_min_degrees[kept_idx_device]
            child_words = child_words[kept_idx_device]
            child_lengths = child_lengths[kept_idx_device]
            child_last = child_last[kept_idx_device]
            parent_xent_history = parent_xent_history[kept_idx_device]
            parent_xent_max = parent_xent_max[kept_idx_device]

            raw_scores = raw_scores_cpu.to(device=self.device)
            bucket_ids = bucket_ids_cpu.to(device=self.device)
            score_time += time.time() - t0

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
                words=child_words,
                lengths=child_lengths,
                last_factor_ids=child_last,
                xent_history=parent_xent_history,
                xent_max=parent_xent_max,
                raw_scores=raw_scores,
                bucket_ids=bucket_ids,
                disable_cap=is_bootstrap,
            )
            bucket_time += time.time() - t0

        if nontrivial_candidate_count == 0:
            level_time = time.time() - level_start
            summary = {
                "level": level,
                "num_parents": self.frontier.size,
                "num_candidates": num_candidates,
                "num_trivial_commutators_discarded": trivial_discard_count,
                "num_nontrivial_candidates": 0,
                "num_kept_after_reservoir": 0,
                "num_selected_for_next_level": 0,
                "nonempty_buckets": 0,
                "incoming_bucket_counts": {},
                "kept_bucket_counts": {},
                "score_min": None,
                "score_mean": None,
                "score_max": None,
                "kernel_hits_found": 0,
                "kernel_hit_type_counts": {
                    "identity": 0,
                },
                "best_candidate": None,
                "timing_sec": {
                    "matmul": round(matmul_time, 4),
                    "score": round(score_time, 4),
                    "reservoir": round(bucket_time, 4),
                    "total": round(level_time, 4),
                },
            }
            self.level_summaries.append(summary)
            print(
                f"  discarded_trivial={trivial_discard_count} nontrivial=0 "
                "no survivors remain after exact triviality filtering"
            )
            return False

        materialized = buckets.materialize(out_device=torch.device("cpu"))
        selected = materialized if is_bootstrap else self._select_best(materialized)
        next_frontier = _move_frontier_batch(selected, self.device)
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
            "kernel_hit_type_counts": {
                "identity": kernel_hit_count,
            },
            "best_candidate": best_candidate,
            "timing_sec": {
                "matmul": round(matmul_time, 4),
                "score": round(score_time, 4),
                "reservoir": round(bucket_time, 4),
                "total": round(level_time, 4),
            },
        }
        self.level_summaries.append(summary)
        self.frontier = next_frontier

        print(
            f"  discarded_trivial={trivial_discard_count} nontrivial={nontrivial_candidate_count} "
            f"kept={materialized.size} selected={selected.size} buckets={len(buckets.data)} "
            f"score=[{score_min:.4f}, {score_mean:.4f}, {score_max:.4f}] "
            f"kernel_hits={kernel_hit_count}"
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
        else:
            print("  no survivors remain")
        return materialized.size > 0

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
        result = {
            "config": asdict(self.config),
            "device_resolved": str(self.device),
            "reference_braid_word": [int(x) for x in self.reference_word],
            "reference_inverse_word": [int(x) for x in self.reference_inverse_word],
            "completed_levels": completed_levels,
            "level_summaries": self.level_summaries,
            "best_candidates": best_candidates,
            "kernel_hits": self.kernel_hits,
            "kernel_hit_type_counts": {
                "identity": len(self.kernel_hits),
            },
            "total_time_sec": round(total_time, 4),
        }
        return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Separate reservoir search that grows positive GNF braids b and scores them by "
            "projlen(Burau([a,b])) for a fixed input braid a."
        )
    )
    parser.add_argument("--reference-braid-word", required=True, help="Signed Artin word for the fixed braid a, e.g. '1' or '1,-2,1'")
    parser.add_argument("--p", type=int, default=5, help="Modulus p for Burau arithmetic")
    parser.add_argument("--max-length", type=int, required=True, help="Maximum Garside length of b to search")
    parser.add_argument(
        "--search-D",
        type=int,
        help="Projective support window used only for growing b itself; default is 4*max_length+1",
    )
    parser.add_argument("--bucket-size", type=int, default=4000, help="Reservoir size per score bucket")
    parser.add_argument("--num-buckets", type=int, default=32, help="Number of score buckets when --bucket-mode=score")
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
        default=512,
        help="How many candidate braids to score per commutator batch",
    )
    parser.add_argument(
        "--bucket-mode",
        choices=("exact", "score"),
        default="exact",
        help="Bucket either by exact commutator projlen or by normalized score buckets.",
    )
    parser.add_argument("--topk-save", type=int, default=50, help="How many best final candidates to save")
    parser.add_argument(
        "--save-kernel-hits",
        type=int,
        default=100,
        help="Maximum number of scalar-identity commutator hits to keep in the JSON output",
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

    search_D = args.search_D if args.search_D is not None else 4 * args.max_length + 1
    config = CommutatorSearchConfig(
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
        bucket_mode=args.bucket_mode,
        topk_save=args.topk_save,
        save_kernel_hits=args.save_kernel_hits,
        reference_braid_word=parse_signed_word(args.reference_braid_word),
        out_json=args.out_json,
    )

    set_seed(config.seed)
    search = ReservoirSearchCommutator(config)
    result = search.run()

    if config.out_json:
        out_path = Path(config.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
