from itertools import permutations
import random
N = 4
perms = list(permutations(range(N)))


def _validate_perm(perm):
    """Return perm as a tuple and ensure it is a valid permutation of 0..n-1."""
    perm = tuple(perm)
    n = len(perm)
    if set(perm) != set(range(n)):
        raise ValueError(f"Not a valid permutation of 0..{n - 1}: {perm}")
    return perm

class GarsideFactor:
    def __init__(self, perm):
        self.perm = _validate_perm(perm)

    def left_descent(self):
        """
        Left descent set:
        {i | w^{-1}(i) > w^{-1}(i+1)}, with 0-based i in [0, n-2].
        """
        n = len(self.perm)
        inv = [0] * n
        for pos, value in enumerate(self.perm):
            inv[value] = pos
        return {i for i in range(n - 1) if inv[i] > inv[i + 1]}

    def right_descent(self):
        """
        Right descent set:
        {i | w(i) > w(i+1)}, with 0-based i in [0, n-2].
        """
        return {i for i in range(len(self.perm) - 1) if self.perm[i] > self.perm[i + 1]}

    def artin_factors(self):
        """
        Return a reduced Artin word for self.perm as adjacent transpositions.

        Output is a list of 0-based generator indices i, representing s_i that
        swaps positions i and i+1. Applying the list left-to-right to the
        identity permutation produces self.perm.
        """
        n = len(self.perm)

        # Convert perm to identity by right-multiplying adjacent transpositions.
        # Reverse that word to get a decomposition from identity to perm.
        working = list(self.perm)
        to_identity = []
        for target in range(n - 1, -1, -1):
            pos = working.index(target)
            while pos < target:
                working[pos], working[pos + 1] = working[pos + 1], working[pos]
                to_identity.append(pos)
                pos += 1

        return list(reversed(to_identity))


class GNF:
    """
    Garside normal form container:
    sigma = Delta^d * w1 * ... * w_ell

    Conditions enforced here:
    - d is an integer
    - all factors are permutations in S_n for the same n
    - ell >= 1
    - w1 != w0 (modeled as: first factor is not Delta permutation)
    - w_ell != e (last factor is not identity permutation)
    - R(w_k) superset L(w_{k+1}) for each adjacent pair
    """

    def __init__(self, d, factors):
        if not isinstance(d, int):
            raise TypeError("d must be an integer")
        self.d = d
        self.factors = [f if isinstance(f, GarsideFactor) else GarsideFactor(f) for f in factors]
        if not self.factors:
            raise ValueError("GNF requires at least one factor")

        self.n = len(self.factors[0].perm)
        for f in self.factors:
            if len(f.perm) != self.n:
                raise ValueError("All factors must lie in the same symmetric group S_n")

        self._validate_normal_form_conditions()

    @staticmethod
    def identity_perm(n):
        return tuple(range(n))

    @staticmethod
    def delta_perm(n):
        # Longest permutation in S_n: i -> n-1-i.
        return tuple(range(n - 1, -1, -1))

    def _validate_normal_form_conditions(self):
        first = self.factors[0]
        last = self.factors[-1]

        if first.perm == self.delta_perm(self.n):
            raise ValueError("Invalid GNF: w1 must not equal w0 (Delta)")
        if last.perm == self.identity_perm(self.n):
            raise ValueError("Invalid GNF: w_ell must not equal identity")

        for k in range(len(self.factors) - 1):
            left = self.factors[k]
            right = self.factors[k + 1]
            if not left.right_descent().issuperset(right.left_descent()):
                raise ValueError(
                    f"Invalid GNF at pair {k}, {k+1}: "
                    "R(w_k) must contain L(w_{k+1})"
                )

    @property
    def garside_length(self):
        return len(self.factors)

    def prefix(self, k):
        """
        Return the Garside prefix Delta^d * w1 * ... * w_k.
        k is 1-based and must satisfy 1 <= k <= ell.
        """
        ell = self.garside_length
        if k < 1 or k > ell:
            raise ValueError(f"k must be between 1 and {ell}")
        return GNF(self.d, self.factors[:k])

    def can_append_suffix(self, u):
        """
        True if u is a valid Garside suffix candidate for concatenation:
        R(w_ell) superset L(u).
        """
        u = u if isinstance(u, GarsideFactor) else GarsideFactor(u)
        if len(u.perm) != self.n:
            return False
        return self.factors[-1].right_descent().issuperset(u.left_descent())

    def append_suffix(self, u):
        """
        Return the concatenated normal form when R(w_ell) superset L(u).
        """
        u = u if isinstance(u, GarsideFactor) else GarsideFactor(u)
        if len(u.perm) != self.n:
            raise ValueError("Suffix must be in same S_n")
        if not self.can_append_suffix(u):
            raise ValueError("Not a Garside suffix: need R(w_ell) superset L(u)")
        return GNF(self.d, self.factors + [u])

    def __repr__(self):
        perms = [f.perm for f in self.factors]
        return f"GNF(d={self.d}, factors={perms})"


def _poly_const(c, p):
    c %= p
    if c == 0:
        return {}
    return {0: c}


def _poly_monomial(coeff, exp, p):
    coeff %= p
    if coeff == 0:
        return {}
    return {exp: coeff}


def _poly_add(a, b, p):
    out = dict(a)
    for exp, coeff in b.items():
        out[exp] = (out.get(exp, 0) + coeff) % p
        if out[exp] == 0:
            del out[exp]
    return out


def _poly_mul(a, b, p):
    if not a or not b:
        return {}
    out = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            e = ea + eb
            out[e] = (out.get(e, 0) + ca * cb) % p
            if out[e] == 0:
                del out[e]
    return out


def _poly_matrix_eye(m, p):
    eye = [[{} for _ in range(m)] for _ in range(m)]
    for i in range(m):
        eye[i][i] = _poly_const(1, p)
    return eye


def _poly_matrix_mul(a, b, p):
    rows = len(a)
    mid = len(a[0])
    cols = len(b[0])
    out = [[{} for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            acc = {}
            for k in range(mid):
                term = _poly_mul(a[i][k], b[k][j], p)
                if term:
                    acc = _poly_add(acc, term, p)
            out[i][j] = acc
    return out


def _burau_generator_matrix_poly(n, i, p, inverse=False):
    """
    Reduced Burau matrix for sigma_i (1-based i) as polynomial entries in v.
    If inverse=True, returns sigma_i^{-1}, introducing negative exponents.
    """
    if n < 2:
        raise ValueError("Need n >= 2")
    if i < 1 or i > n - 1:
        raise ValueError(f"Generator index i must be in 1..{n-1}")

    m = n - 1
    mat = _poly_matrix_eye(m, p)

    if not inverse:
        minus_v = _poly_monomial(-1, 1, p)
        minus_v2 = _poly_monomial(-1, 2, p)
    else:
        minus_v = _poly_monomial(-1, -1, p)
        minus_v2 = _poly_monomial(-1, -2, p)

    # Convert generator index to reduced Burau matrix coordinates.
    # sigma_1 uses rows/cols 0..1, sigma_{n-1} uses m-2..m-1,
    # interior sigma_i uses block on (i-2, i-1, i) in 0-based coordinates.
    if i == 1:
        mat[0][0] = minus_v2
        mat[0][1] = minus_v
        mat[1][0] = {}
        mat[1][1] = _poly_const(1, p)
    elif i == n - 1:
        mat[m - 2][m - 2] = _poly_const(1, p)
        mat[m - 2][m - 1] = {}
        mat[m - 1][m - 2] = minus_v
        mat[m - 1][m - 1] = minus_v2
    else:
        r0 = i - 2
        r1 = i - 1
        r2 = i
        mat[r0][r0] = _poly_const(1, p)
        mat[r0][r1] = {}
        mat[r0][r2] = {}
        mat[r1][r0] = minus_v
        mat[r1][r1] = minus_v2
        mat[r1][r2] = minus_v
        mat[r2][r0] = {}
        mat[r2][r1] = {}
        mat[r2][r2] = _poly_const(1, p)

    return mat


def burau_mod_p_polynomial_matrix(word, p, n=4):
    """
    Evaluate reduced Burau representation of a braid word modulo p.

    word: iterable of signed generator indices.
      k > 0 means sigma_k, k < 0 means sigma_|k|^{-1}.
    Returns an (n-1)x(n-1) matrix with polynomial dict entries:
      {exp: coeff_mod_p, ...}
    """
    if p <= 1:
        raise ValueError("p must be >= 2")
    if n < 2:
        raise ValueError("n must be >= 2")

    m = n - 1
    result = _poly_matrix_eye(m, p)
    for g in word:
        if g == 0:
            raise ValueError("Generator index 0 is invalid")
        i = abs(g)
        gen_mat = _burau_generator_matrix_poly(n, i, p, inverse=(g < 0))
        result = _poly_matrix_mul(result, gen_mat, p)
    return result


def burau_mod_p_tensor(word, p, D, n=4):
    """
    Convert Burau matrix to a degree-sliced tensor of shape D x 3 x 3 (for n=4).

    Exponent slice t stores coefficient of v^t modulo p.
    Raises ValueError if any term has exponent < 0 or exponent >= D.
    """
    if n != 4:
        raise ValueError("This tensor interface currently expects n=4 (3x3 matrices)")
    if D <= 0:
        raise ValueError("D must be positive")

    poly_mat = burau_mod_p_polynomial_matrix(word, p, n=n)
    tensor = [[[0 for _ in range(3)] for _ in range(3)] for _ in range(D)]

    for i in range(3):
        for j in range(3):
            for exp, coeff in poly_mat[i][j].items():
                if exp < 0 or exp >= D:
                    raise ValueError(
                        f"Tensor depth D={D} too small (or negative exponent present): "
                        f"entry ({i},{j}) has term v^{exp}"
                    )
                tensor[exp][i][j] = coeff % p

    return tensor


def gnf_to_braid_word(gnf):
    """
    Convert a GNF object to a signed Artin word (1-based generators).
    """
    if not isinstance(gnf, GNF):
        raise TypeError("gnf must be an instance of GNF")

    n = gnf.n
    delta = GarsideFactor(GNF.delta_perm(n)).artin_factors()
    word = []
    if gnf.d >= 0:
        for _ in range(gnf.d):
            word.extend([idx + 1 for idx in delta])
    else:
        inv_delta = [-(idx + 1) for idx in reversed(delta)]
        for _ in range(-gnf.d):
            word.extend(inv_delta)

    for factor in gnf.factors:
        word.extend([idx + 1 for idx in factor.artin_factors()])
    return word


def burau_mod_p_tensor_from_gnf(gnf, p, D):
    """
    Evaluate p-Burau tensor (D x 3 x 3) for a GNF object in B_4.
    """
    if not isinstance(gnf, GNF):
        raise TypeError("gnf must be an instance of GNF")
    if gnf.n != 4:
        raise ValueError("This tensor interface currently expects GNF in S_4")
    return burau_mod_p_tensor(gnf_to_braid_word(gnf), p, D, n=4)


class DataSetBuilder:
    """
    Build supervised data from random GNFs:
      input  = Burau tensor (D x 3 x 3),
      label1 = final Garside factor permutation,
      label2 = right descent set of that final factor.
    """

    def __init__(self, p, D, n=4, d_range=(0, 0), seed=None):
        if n != 4:
            raise ValueError("DataSetBuilder currently expects n=4 (3x3 Burau matrices)")
        if p <= 1:
            raise ValueError("p must be >= 2")
        if D <= 0:
            raise ValueError("D must be positive")
        d_min, d_max = d_range
        if d_min > d_max:
            raise ValueError("d_range must satisfy min <= max")
        if d_min < 0:
            raise ValueError("d_range min must be >= 0 for nonnegative tensor exponents")

        self.p = p
        self.D = D
        self.n = n
        self.d_range = d_range
        self.rng = random.Random(seed)
        self._all_perms = list(permutations(range(n)))

    def _random_int(self, low, high):
        return self.rng.randint(low, high)

    def _random_choice(self, seq):
        return seq[self.rng.randrange(len(seq))]

    def _valid_factor_candidates(self, required_left_subset=None, exclude_delta=False, exclude_identity=False):
        candidates = []
        delta = GNF.delta_perm(self.n)
        identity = GNF.identity_perm(self.n)
        required_left_subset = required_left_subset or set()

        for perm in self._all_perms:
            if exclude_delta and perm == delta:
                continue
            if exclude_identity and perm == identity:
                continue
            factor = GarsideFactor(perm)
            if factor.right_descent().issuperset(required_left_subset):
                candidates.append(factor)
        return candidates

    def random_gnf(self, L, max_attempts=200):
        """
        Sample a random valid GNF with Garside length L.
        """
        if L <= 0:
            raise ValueError("L must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

        for _ in range(max_attempts):
            d = self._random_int(self.d_range[0], self.d_range[1])

            # Build factors from right to left to enforce R(w_k) ⊇ L(w_{k+1}).
            factors = [None] * L
            last_candidates = self._valid_factor_candidates(exclude_identity=True)
            if not last_candidates:
                raise RuntimeError("No valid choices for final Garside factor")
            factors[-1] = self._random_choice(last_candidates)

            ok = True
            for k in range(L - 2, -1, -1):
                required = factors[k + 1].left_descent()
                candidates = self._valid_factor_candidates(
                    required_left_subset=required,
                    exclude_delta=(k == 0),
                    exclude_identity=False,
                )
                if not candidates:
                    ok = False
                    break
                factors[k] = self._random_choice(candidates)

            if ok:
                return GNF(d, factors)

        raise RuntimeError(f"Failed to sample valid GNF after {max_attempts} attempts")

    def sample(self, L, max_attempts=200):
        """
        Generate one training example.
        """
        gnf = self.random_gnf(L, max_attempts=max_attempts)
        tensor = burau_mod_p_tensor_from_gnf(gnf, p=self.p, D=self.D)
        final_factor = gnf.factors[-1]
        rdesc = sorted(final_factor.right_descent())

        return {
            "burau_tensor": tensor,
            "final_factor_perm": list(final_factor.perm),
            "final_factor_right_descent": rdesc,
            "gnf_d": gnf.d,
            "gnf_factors": [list(f.perm) for f in gnf.factors],
        }

    def build(self, num_samples, L, max_attempts=200):
        """
        Build a dataset list with num_samples iid random examples.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        return [self.sample(L, max_attempts=max_attempts) for _ in range(num_samples)]


if __name__ == "__main__":
    g = GarsideFactor((1,2,0,3))
    print(g.artin_factors())
    print(g.right_descent())
    print(g.left_descent())
