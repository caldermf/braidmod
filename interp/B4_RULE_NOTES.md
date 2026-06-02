# B4 Boundary Rule Notes

This note records the first algebraic rule-mining pass for the `B_4`
Burau/descent task. The goal was to look for a theorem-like analogue of the
`B_3` boundary-column rule.

## Clean Simple-Factor Rule

There is a clean rule at the level of a single final Garside factor.

Let `x` be a proper simple factor in `B_4`, and let `M = rho(x)` be the
reduced Burau matrix over `Z[v]` in our convention. For each degree `d`, write
`C_d(M)` for the bitmask of columns containing a nonzero entry in the
coefficient matrix `[v^d] M`. Let `D = maxdeg(M)`.

Exact enumeration of the 22 proper simples shows:

1. `C_D(M)` is always a subset of the right descent mask `R(x)`.
2. `C_D(M) = R(x)` for 18 of the 22 proper simples.
3. The ordered pair `(C_D(M), C_{D-1}(M))` determines `R(x)` for all 22 proper
   simples.

The same statement holds after reducing coefficients mod 2. This is the
closest `B_4` analogue of the `B_3` rule: the right descent of the final
simple factor is visible from a very small top-degree column-support
signature, but in `B_4` the top slice alone has four exceptional cases and
the second slice resolves them.

The four top-slice exceptions are:

| factor id | Artin word | right descent | top column mask |
|---:|---|---:|---:|
| `10` | `s1 s2 s3 s2` | `6` | `4` |
| `14` | `s1 s2 s1 s3` | `5` | `4` |
| `18` | `s2 s3 s2 s1` | `5` | `1` |
| `19` | `s1 s3 s2 s1` | `3` | `1` |

So the simple-factor rule is not "top column equals descent" but "top two
column masks determine descent."

## Full Product Over F2

For full length-25 normal forms, the final-factor signature is partially
obscured by multiplication by the preceding prefix. This is where the problem
becomes genuinely interesting.

On held-out length-25 data over `(Z/2)[v]`, explicit column-support rules are
not sufficient:

| Rule | Exact accuracy | Bit accuracy |
|---|---:|---:|
| trailing top column mask | `51.1%` | `80.9%` |
| best mined invariant: leading+trailing column-mask sequence, radius 2 | `63.6%` | `86.9%` |
| leading+trailing kernel/rowspace sequence, radius 2 | `63.5%` | `86.8%` |
| right-division frontier deltas for all generators | `60.3%` | `85.7%` |
| transformer, boundary-only radius 8 | `70.7%` | `89.2%` |
| transformer, full input | `71.4%` | `89.5%` |

The purity audit also rules out a clean radius-2 column-support theorem over
the full mod-2 matrix. For the best feature `both_col_r2`, only about `3.6%`
of examples lie on feature keys that are pure on the audit sample, and the
same-sample majority accuracy is only `64.7%`. There are high-count keys with
substantial mixtures of several descent masks.

## Interpretation

The evidence suggests the following picture.

The `B_3` result generalizes cleanly to the final simple factor, but not
directly to the observed full-product matrix over `(Z/2)[v]`. In `B_4`, the
final simple factor carries its right descent in a two-slice top-degree
column-support signature. The preceding Garside prefix acts on this signature
through the boundary coefficient matrices of its Burau image. Over `F_2`,
that action creates collisions and cancellations, so small hand-written
support rules become impure.

This matches the transformer evidence: the trained model reads a thicker
leading/trailing boundary band and outperforms the small finite invariants we
mined, but it does not solve the task perfectly. A plausible mechanistic
interpretation is that the model is learning a partial inverse to the prefix
boundary action, not merely reading the final factor signature directly.

## Next Mathematical Experiments

The next experiments should focus on reducing the aliasing introduced by
working over `F_2`:

1. Repeat the boundary-signature audit over `Z[v]`, keeping signs and integer
   coefficients in the top few slices.
2. Build a suffix-controlled dataset: fix the final simple factor, vary the
   preceding prefix, and measure exactly how prefix boundary maps transform
   the final two-slice signature.
3. Model the boundary-band update as a finite-state transducer on column
   subspaces/kernels. If this works, the transformer's circuit can be
   compared against a genuine recovered automaton.
4. Continue testing right-division frontier features, especially over `Z[v]`.
   Right descent is right divisibility, so `rho(beta) rho(s_i)^{-1}` remains
   the most algebraically natural quotient to inspect.

The punchline so far is therefore not "we found the full B4 theorem over
F2." The honest punchline is sharper: we found the exact final-simple-factor
rule, showed why the full-product F2 problem is harder, and identified the
frontier action that the transformer appears to be approximating.

## Integer Boundary Audit

We then repeated the boundary-rule audit over `Z[v]` instead of `(Z/2)[v]`,
using the stored factor sequences to replay the braid through the integer
Burau representation. The dense int64 implementation was checked against the
Python arbitrary-precision exact implementation on 16 length-25 examples, with
no mismatches. On the audited sample, coefficients remained small enough for
int64: the maximum absolute coefficient observed was `900,788`.

The result is much stronger than the mod-2 audit. Signs in the top boundary
band carry a large amount of the information that mod 2 destroys.

On `262,144` train examples and `65,536` held-out examples:

| Feature | Exact accuracy | Bit accuracy | Coverage |
|---|---:|---:|---:|
| trailing top column support over `Z[v]` | `60.5%` | `86.3%` | `100.0%` |
| best mod-2 mined feature: both boundary column masks, `r=2` | `63.6%` | `86.9%` | `99.9%` |
| signed/integer: both negative-column masks, `r=3` | `84.8%` | `94.5%` | `99.0%` |
| signed/integer: both positive-column masks, `r=3` | `84.2%` | `94.3%` | `98.9%` |
| signed/integer: trailing sign tokens, `r=2` | `84.1%` | `94.1%` | `98.7%` |

Increasing the radius beyond `3` or `4` did not produce a clean near-perfect
lookup. It mostly made keys sparser, reducing coverage or majority
generalization. Thus the best simple signed-boundary rule is not exact, but
it is far stronger than the mod-2 rule and stronger than the current
transformer trained only on mod-2 tokens.

This gives a sharper mathematical picture:

1. The final simple factor has an exact two-slice top-column signature.
2. In the full product, the prefix action obscures that signature.
3. Reducing mod 2 discards crucial sign information about this prefix action.
4. Over `Z[v]`, a radius-3 signed column-frontier recovers most of the
   descent information.

So the best current conjecture is that the natural B4 rule lives over the
signed boundary frontier, not over the support-only mod-2 frontier. The mod-2
transformer is trying to solve a lossy projection of this cleaner integer
frontier problem.
