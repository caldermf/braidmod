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

So the simple-factor rule is not "top column equals descent"; it is "top two
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

The current conclusion is not that we found the full `B_4` theorem over `F_2`.
It is sharper and more useful: we found the exact final-simple-factor rule,
showed why the full-product `F_2` problem is harder, and identified the
frontier action that the transformer appears to approximate.

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

## Integer Sign-Token Transformer

We then trained the same 3-layer CLS transformer on a signed integer
projection of the `Z[v]` Burau matrix. Each absolute-degree coefficient slice
is a `3 x 3` sign matrix with entries in `{0, -, +}`, encoded as a base-3
token. This is not the full integer coefficient data: it keeps exact support
and signs while discarding magnitudes. The vocabulary size is `3^9 = 19,683`.

The model was trained from the stored length-25 factor sequences, replaying
the integer Burau representation on GPU during training. The run used
`1,048,576` train examples per epoch, 16 epochs, and held-out validation/test
streams from the existing `16,777,216` example B4 corpus.

Held-out test performance:

| Model input | Exact accuracy | Bit accuracy | Micro-F1 |
|---|---:|---:|---:|
| `(Z/2)[v]` support tokens, 3-layer transformer | `71.4%` | `89.5%` | `89.8%` |
| `Z[v]` sign tokens, 3-layer transformer | `93.2%` | `97.7%` | `97.7%` |

This is a decisive improvement over both the mod-2 model and the best
hand-written signed-boundary lookup. The sign projection therefore captures a
large amount of the algebraic information lost mod 2.

The first-pass intervention result is not the same as the mod-2 boundary-band
story. On `8,192` held-out examples:

| Input intervention | Exact accuracy | Bit accuracy |
|---|---:|---:|
| Full sign-token matrix | `93.9%` | `98.0%` |
| Boundary only, radius `8` | `73.0%` | `90.3%` |
| Boundary only, radius `5` | `58.0%` | `83.6%` |
| Boundary only, radius `3` | `47.1%` | `77.1%` |
| Drop boundary, radius `8` | `19.9%` | `54.6%` |
| Drop boundary, radius `5` | `32.2%` | `62.9%` |

So the signed frontier is causally important, but a small boundary band is
not sufficient for this trained model. The model appears to use broader
signed information across the support, while still relying heavily on the
frontier: removing radius-8 boundary windows collapses the classifier.

Raw finite lookup baselines from the same sign-token representation:

| Lookup feature | Exact accuracy | Bit accuracy | Coverage |
|---|---:|---:|---:|
| leading token | `58.6%` | `86.2%` | `100.0%` |
| trailing token | `61.8%` | `86.8%` | `99.9%` |
| leading + trailing tokens | `63.1%` | `87.2%` | `99.8%` |
| trailing window, radius `2` | `82.6%` | `92.7%` | `95.8%` |
| both windows, radius `1` | `78.2%` | `91.4%` | `95.6%` |

The transformer is therefore not just implementing a tiny lookup over the
extremal sign slices. It beats the best simple lookup by about `10` exact
accuracy points.

Attention patterns still show boundary-oriented heads, but less cleanly than
the mod-2 model. The top CLS boundary heads are:

| Head | Leading mass | Trailing mass | Boundary mass | Support mass |
|---|---:|---:|---:|---:|
| `L1H4` | `0.003` | `0.230` | `0.233` | `0.932` |
| `L2H2` | `0.203` | `0.005` | `0.208` | `0.912` |
| `L0H5` | `0.001` | `0.171` | `0.172` | `0.991` |
| `L0H2` | `0.001` | `0.151` | `0.153` | `0.983` |

The conclusion is that the `Z[v]` sign-token model gives us a much better
predictive setting, but not yet a proof-style recovered algorithm. It strongly
supports the mathematical hypothesis that signs over `Z[v]` are the right next
object, and it gives a high-performing model worth interpreting. The next
target is a circuit-derived classifier or a hand-derived rule that closes the
gap between the `82-85%` signed-frontier lookup and the transformer's `93%`
exact accuracy.

## Z-Sign Deep Dive: Toward a Recovered Algorithm

We then ran a deeper interpretability pass on the trained `Z[v]` sign-token
transformer. The goal was to test the more specific hypothesis:

> The model is not simply reading a raw boundary lookup. It is using signed
> boundary information to infer a coarser algebraic state close to the final
> Garside factor or its right-divisibility frontier, and then reading descent
> from that state.

The full artifact is:

`interp/artifacts/b4_l25_zsign_deep_dive/results.json`

The run used `32,768` train examples for probes/lookups, `8,192` held-out
examples for evaluation, and `512` prefix-fixed counterfactual pairs.

### Circuit-Derived Classifiers

Linear probes from the model's late CLS stream recover most of the model's
decision. A direct three-bit descent probe almost matches the model:

| Representation | Probe exact | Probe bit | Agreement with model |
|---|---:|---:|---:|
| final hidden CLS | `93.2%` | `97.7%` | `97.3%` |
| layer-2 residual post CLS | `92.8%` | `97.5%` | `96.8%` |
| layer-1 residual post CLS | `91.4%` | `97.0%` | `95.0%` |
| layer-0 residual post CLS | `87.5%` | `95.7%` | `90.4%` |

More interestingly, a probe trained to decode the **final simple factor**,
followed by the exact simple-factor descent lookup, also gets close:

| Representation | Rule exact | Rule bit | Final-factor accuracy | Agreement with model |
|---|---:|---:|---:|---:|
| final hidden CLS | `90.9%` | `96.7%` | `75.2%` | `93.8%` |
| layer-2 residual post CLS | `90.8%` | `96.7%` | `76.4%` | `93.7%` |
| layer-1 residual post CLS | `89.2%` | `96.1%` | `75.5%` | `91.8%` |
| layer-0 residual post CLS | `84.7%` | `94.5%` | `72.4%` | `86.9%` |

This is not a perfect recovered algorithm, but it is a much stronger
mechanistic statement than "attention goes to the boundary." The late `CLS`
state contains a linearly decodable approximation to the final-factor
information, and composing that decoder with the exact algebraic lookup
recovers most of the transformer's behavior.

### Prefix-Fixed Counterfactuals

We constructed counterfactual pairs with the **same length-24 prefix** and a
different valid final simple factor. Thus the only algebraic change is the
final factor appended to the same prefix. On these pairs, the clean example is
classified at `93.2%` exact accuracy, while the corrupt example scored against
the clean label falls to `0.8%` exact / `38.9%` bit accuracy.

Patching input regions from clean into corrupt gives:

| Patched clean input region | Score recovery | Clean-label exact | Clean-label bit |
|---|---:|---:|---:|
| both boundary bands, radius `8` | `90.2%` | `82.6%` | `93.7%` |
| both boundary bands, radius `5` | `86.3%` | `75.6%` | `90.6%` |
| both boundary bands, radius `3` | `80.4%` | `70.1%` | `88.5%` |
| trailing band, radius `8` | `47.6%` | `37.1%` | `69.3%` |
| leading band, radius `8` | `27.5%` | `10.0%` | `49.4%` |
| interior except boundary bands, radius `8` | `7.0%` | `2.5%` | `41.7%` |

This is the strongest causal result so far. When the prefix is held fixed, the
signed boundary bands carry almost all of the final-factor change that the
model uses. The trailing side is more important than the leading side, but the
two-sided radius-8 band is the robust object.

Activation patching shows when this information has been collapsed into CLS:

| Activation patch | Score recovery | Clean-label exact |
|---|---:|---:|
| input embedding boundary bands, radius `8` | `90.2%` | `82.6%` |
| layer-0 residual post at CLS | `76.5%` | `64.6%` |
| layer-1 residual post at CLS | `96.4%` | `92.4%` |
| layer-2 residual post at CLS | `100.0%` | `93.2%` |

So the boundary information is read into CLS early, largely by the end of
layer 1, and the final block mostly sharpens an already available decision.

### Right-Quotient Frontier

Because descent is right divisibility, we also tested hand-written finite
lookups on the signed frontier of `rho(beta) rho(s_i)^{-1}` for each
generator `s_i`. This asks whether the quotient by a candidate generator
produces a recognizable signed boundary state.

The best quotient-frontier lookup is strong as a per-generator binary
classifier, but not yet a descent-set classifier:

| Quotient feature | Bit accuracy over `(beta, i)` | Set exact after recombining bits | Coverage |
|---|---:|---:|---:|
| trailing quotient window, radius `3` | `89.8%` | `73.9%` | `91.8%` |
| both quotient windows, radius `2` | `88.1%` | `67.6%` | `89.5%` |
| leading quotient window, radius `3` | `88.0%` | `64.2%` | `99.4%` |
| leading quotient token | `86.7%` | `60.3%` | `100.0%` |

This is mathematically useful but not yet the theorem. The right quotient
frontier is highly informative for each generator, but its bit errors compound
when recombined into full descent sets. It is probably part of the natural
rule, not the whole rule.

### Constrained Boundary-Only Model

Finally, we trained a deliberately smaller transformer that only ever sees the
radius-8 signed boundary bands. Everything outside the leading/trailing
support windows is zeroed before the model input. The model has 2 layers,
`d_model=96`, 4 heads per layer, and `2.1M` parameters. This is meant to test
whether the interpretable boundary mechanism is strong enough to learn as a
standalone model, not just as an intervention on the larger model.

Artifact:

`interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small/results.json`

Held-out test performance:

| Model | Visible input | Exact accuracy | Bit accuracy | Micro-F1 |
|---|---|---:|---:|---:|
| full Z-sign transformer | all `101` degree slices | `93.2%` | `97.7%` | `97.7%` |
| small boundary-only transformer | radius-8 signed frontier only | `90.6%` | `96.7%` | `96.7%` |

This is a strong sanity check on the causal patching result. The radius-8
signed frontier is not only sufficient under distribution shift for the
large model; it supports an independently trained small model with about
`91%` exact accuracy. The remaining `2-3%` gap is the same gap seen in the
circuit-derived final-factor classifier, suggesting that the larger model is
using a modest amount of extra information or a more efficient latent state
rather than a completely different algorithm.

## Hidden-Theorem Search Over `Z[v]`

We then pushed directly on the mathematical question: is there a clean
quotient/frontier theorem hiding behind the transformer's behavior?

The audit script is:

`interp/search_b4_hidden_theorem.py`

The main full artifact is:

`interp/artifacts/b4_l25_z_hidden_theorem_combo/results.json`

The run used `131,072` train examples and `32,768` held-out examples. It
tested:

1. signed leading/trailing boundary states;
2. generator quotient states from `rho(beta) rho(s_i)^{-1}`;
3. quotient-by-all-22-simple-factor signatures;
4. clipped coefficient-magnitude boundary states;
5. pair/triple combinations of the strongest states.

The first important result is negative but clarifying. The naive quotient
positivity rule does **not** work in reduced Burau. For every generator, the
crude tests "no negative Laurent terms" and "first exponent nonnegative" are
uninformative:

| Direct quotient predicate | Exact set accuracy | Bit accuracy |
|---|---:|---:|
| `rho(beta) rho(s_i)^-1` has no negative terms | `0.0%` | `50.0%` |
| first quotient exponent is nonnegative | `0.0%` | `50.0%` |

Likewise, quotienting by all 22 proper simple factors and recording which
quotients have zero negative support collapses to one feature key and gives no
descent information. Thus reduced Burau does not expose positive right
divisibility by a simple Laurent-polynomial positivity test.

The strongest finite hand states were informative but not theorem-like:

| Candidate state | Exact accuracy | Bit accuracy | Train pure mass | Coverage |
|---|---:|---:|---:|---:|
| signed boundary negative-column masks, radius `3` | `84.0%` | `94.1%` | `27.1%` | `98.4%` |
| quotient signed-column state, radius `1` | `84.0%` | `94.0%` | `30.5%` | `97.9%` |
| all-simple quotient width-delta state | `78.3%` | `92.5%` | `14.9%` | `100.0%` |
| best combined state: boundary neg-col `r=3` + simple quotient width-delta | `85.1%` | `94.2%` | `36.6%` | `97.1%` |

Clipped coefficient magnitudes did not close the gap; the best clipped
boundary feature in this search was well below the best signed-column state.

The high-count conflict tables matter. The best candidates are not only
failing because of sparse unseen keys. They have repeated feature keys with
multiple descent masks. So within this family of small signed frontier and
quotient states, there is no exact hidden theorem.

This updates the mathematical picture:

1. The exact final-simple top-two-column theorem remains real.
2. The full-product problem cannot be solved by naive right quotient
   polynomiality in reduced Burau.
3. Signed boundary and quotient-frontier states each recover about `84%`
   exact accuracy.
4. Combining them reaches only `85%`, far below the transformer's `93%`.
5. The model is probably using either a richer distributed state from the
   full sign matrix, a smoother statistical approximation to positive
   membership, or information not captured by these finite hand states.

### Current Summary

The story is now substantially clearer:

1. In `B_3`, descent is exactly a visible extremal-column rule.
2. In `B_4`, the exact final-simple-factor rule still exists, but the prefix
   transports and obscures that signature.
3. Over `(Z/2)[v]`, too much sign information is destroyed.
4. Over `Z[v]` sign tokens, a transformer reaches `93%` exact accuracy.
5. In prefix-fixed counterfactuals, radius-8 signed boundary bands recover
   `90%` of the model's clean-corrupt score.
6. By layer 1, that boundary information is mostly collapsed into CLS.
7. A final-CLS probe, composed with the exact final-factor descent lookup,
   recovers `90.9%` exact accuracy and agrees with the model on `93.8%` of
   examples.
8. A small transformer trained only on the radius-8 signed frontier reaches
   `90.6%` exact accuracy.
9. A direct hidden-theorem search over quotient/frontier states did not find
   an exact finite rule; the best combined hand state reached `85.1%` exact,
   with genuine high-count collisions.

This is close to the result we want, but the remaining gap is important. The
best next target is to identify the missing `2-3%` between
the final-factor-derived circuit classifier and the model itself, or to find a
more natural latent state than the literal final factor. The evidence points
to a transported signed boundary state: something coarser than the full final
factor but more stable than a small raw lookup table.
