#!/usr/bin/env python3
"""
Check whether [w, sigma_i] is in the kernel of the reduced Burau representation mod p.

This script accepts a signed Artin word in B_4, a generator index i in {1,2,3},
and a modulus p >= 2. It reports:

- the left Garside normal form of w,
- the left Garside normal form of [w, sigma_i] = w sigma_i w^{-1} sigma_i^{-1},
- whether Burau([w, sigma_i]) is exactly the identity mod p,
- whether Burau([w, sigma_i]) is at least a scalar multiple of the identity mod p.
"""

import argparse
import ast
import json
from typing import Dict, List, Optional, Sequence, Tuple

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from braid_data import GarsideFactor, _simple_braid_tables, burau_mod_p_polynomial_matrix


PolyEntry = Dict[int, int]
PolyMatrix = List[List[PolyEntry]]
Perm = Tuple[int, ...]


def parse_signed_word(spec: str) -> List[int]:
    text = spec.strip()
    if not text:
        raise ValueError("Word must not be empty")

    if "=" in text:
        _, rhs = text.split("=", 1)
        text = rhs.strip()

    if text.startswith("["):
        values = ast.literal_eval(text)
        if not isinstance(values, list):
            raise ValueError("List input must evaluate to a Python list")
        word = [int(x) for x in values]
    else:
        parts = text.replace(",", " ").split()
        word = [int(part) for part in parts]

    if not word:
        raise ValueError("Word must contain at least one generator")

    for g in word:
        if g == 0:
            raise ValueError("Generator index 0 is invalid")
        if abs(g) > 3:
            raise ValueError("This script currently expects B_4 generators in -3..-1 or 1..3")
    return word


def invert_signed_word(word: Sequence[int]) -> List[int]:
    return [-int(g) for g in reversed(word)]


class SignedWordLeftNormalForm:
    """Standalone copy of the normalizer logic used in reservoir_search_commutator.py."""

    def __init__(self, n: int = 4):
        if n != 4:
            raise ValueError("This normalizer currently expects B_4")
        self.tables = _simple_braid_tables(n)
        self.tau = self.tables["tau"]
        self.pair_table = self.tables["pair_table"]
        self.generator_to_perm = self.tables["generator_to_perm"]
        self.left_complements = self._build_left_complements()

    def _build_left_complements(self) -> Dict[Perm, Perm]:
        complements: Dict[Perm, Perm] = {}
        for perm in self.generator_to_perm.values():
            matches: List[Perm] = []
            for candidate in self.tables["simple_words"]:
                if candidate == self.tables["delta"]:
                    continue
                pair_d, pair_factors = self.pair_table[(candidate, perm)]
                if pair_d == 1 and tuple(pair_factors) == ():
                    matches.append(candidate)
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected a unique left complement for generator perm {perm}, got {matches}"
                )
            complements[perm] = matches[0]
        return complements

    def _append_simple(self, delta_power: int, factors: List[Perm], perm: Perm) -> Tuple[int, List[Perm]]:
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

    def normalize(self, word: Sequence[int]) -> Tuple[int, Tuple[Perm, ...]]:
        delta_power = 0
        factors: List[Perm] = []

        for g in word:
            perm = self.generator_to_perm[abs(int(g))]
            if g > 0:
                delta_power, factors = self._append_simple(delta_power, factors, perm)
            else:
                delta_power -= 1
                factors = [self.tau[p] for p in factors]
                delta_power, factors = self._append_simple(delta_power, factors, self.left_complements[perm])

        return delta_power, tuple(factors)


def scalar_identity_metadata(poly_mat: PolyMatrix) -> Optional[dict]:
    diagonal_entry: Optional[PolyEntry] = None
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

    ((degree, coeff),) = diagonal_entry.items()
    return {"scalar_degree": int(degree), "scalar_coeff_mod_p": int(coeff)}


def identity_metadata(poly_mat: PolyMatrix, p: int) -> bool:
    scalar = scalar_identity_metadata(poly_mat)
    if scalar is None:
        return False
    return scalar["scalar_degree"] == 0 and scalar["scalar_coeff_mod_p"] % p == 1


def first_non_scalar_witness(poly_mat: PolyMatrix) -> dict:
    for i in range(3):
        for j in range(3):
            entry = poly_mat[i][j]
            if i != j and entry:
                return {
                    "kind": "off_diagonal",
                    "row": i,
                    "col": j,
                    "terms": sorted((int(exp), int(coeff)) for exp, coeff in entry.items())[:12],
                }

    return {
        "kind": "diagonal_mismatch",
        "diag_entries": [
            sorted((int(exp), int(coeff)) for exp, coeff in poly_mat[i][i].items())[:12]
            for i in range(3)
        ],
    }


def factor_to_artin_word(perm: Perm) -> List[int]:
    return [int(idx) + 1 for idx in GarsideFactor(perm).artin_factors()]


def gnf_payload(delta_power: int, factors: Sequence[Perm]) -> dict:
    return {
        "delta_power": int(delta_power),
        "garside_length": int(len(factors)),
        "factor_perms": [list(perm) for perm in factors],
        "factor_artin_words": [factor_to_artin_word(perm) for perm in factors],
    }


def build_commutator(word: Sequence[int], generator: int) -> List[int]:
    return list(word) + [generator] + invert_signed_word(word) + [-generator]


def analyze(word: Sequence[int], generator: int, p: int) -> dict:
    normalizer = SignedWordLeftNormalForm(n=4)
    commutator = build_commutator(word, generator)

    w_delta, w_factors = normalizer.normalize(word)
    comm_delta, comm_factors = normalizer.normalize(commutator)
    burau = burau_mod_p_polynomial_matrix(commutator, p=p, n=4)

    scalar = scalar_identity_metadata(burau)
    result = {
        "input_word": [int(g) for g in word],
        "input_word_length": int(len(word)),
        "generator": int(generator),
        "prime": int(p),
        "commutator_word": commutator,
        "commutator_word_length": int(len(commutator)),
        "w_gnf": gnf_payload(w_delta, w_factors),
        "commutator_gnf": gnf_payload(comm_delta, comm_factors),
        "burau_mod_p_is_identity": identity_metadata(burau, p=p),
        "burau_mod_p_is_scalar": scalar is not None,
        "burau_mod_p_scalar": scalar,
    }
    if scalar is None:
        result["burau_mod_p_witness"] = first_non_scalar_witness(burau)
    return result


def print_gnf(label: str, payload: dict) -> None:
    print(f"{label}:")
    print(f"  delta_power: {payload['delta_power']}")
    print(f"  garside_length: {payload['garside_length']}")
    print("  factor_perms:")
    for perm in payload["factor_perms"]:
        print(f"    {perm}")
    print("  factor_artin_words:")
    for artin_word in payload["factor_artin_words"]:
        print(f"    {artin_word}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether [w, sigma_i] is in the kernel of the reduced Burau representation mod p, "
            "and print Garside data for w and the commutator."
        )
    )
    parser.add_argument(
        "--word",
        required=True,
        help="Signed Artin word in B_4. Accepts 'w = [-1,2,-3]', '[-1,2,-3]', or '-1,2,-3'.",
    )
    parser.add_argument("--generator", "-i", type=int, required=True, help="Generator index i in {1,2,3}.")
    parser.add_argument("--prime", "-p", type=int, required=True, help="Modulus p >= 2.")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    args = parser.parse_args()

    if args.generator < 1 or args.generator > 3:
        raise ValueError("--generator must lie in {1,2,3}")
    if args.prime < 2:
        raise ValueError("--prime must be at least 2")

    word = parse_signed_word(args.word)
    result = analyze(word, generator=args.generator, p=args.prime)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"input_word_length: {result['input_word_length']}")
    print(f"generator: sigma_{result['generator']}")
    print(f"prime: {result['prime']}")
    print(f"commutator_word_length: {result['commutator_word_length']}")
    print(f"burau_mod_p_is_identity: {result['burau_mod_p_is_identity']}")
    print(f"burau_mod_p_is_scalar: {result['burau_mod_p_is_scalar']}")
    if result["burau_mod_p_scalar"] is not None:
        scalar = result["burau_mod_p_scalar"]
        print(
            "burau_mod_p_scalar_data: "
            f"coeff={scalar['scalar_coeff_mod_p']} degree={scalar['scalar_degree']}"
        )
    else:
        print(f"burau_mod_p_witness: {result['burau_mod_p_witness']}")

    print_gnf("w_gnf", result["w_gnf"])
    print_gnf("commutator_gnf", result["commutator_gnf"])


if __name__ == "__main__":
    main()
