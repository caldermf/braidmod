# Learning Braid Group Representations: the `B_4` Case

This report summarizes the `B_4` phase of the project. It is written for a
reader who has not followed the experiment log.

The high-level question is:

> Can a small transformer learn algebraic information about a braid from its
> Burau representation matrix, and can we reverse engineer what it learned?

For `B_3`, the answer was clean: the right descent set of the final Garside
factor is visible from a single extremal coefficient slice of the Burau
matrix, and the transformer learned to read that slice. The `B_4` case is
substantially harder and more interesting. The final result is not yet a
complete theorem, but we now have a coherent mechanistic story:

> Over `Z[v]`, the signed leading/trailing boundary bands of the Burau matrix
> carry most of the information needed to recover the final descent set. The
> transformer learns to read this signed frontier, collapse it into the CLS
> stream by roughly layer 1, and use it to approximate the final simple factor
> or an equivalent descent-relevant state.

## Task

We work in the braid group `B_4`. Each example is a positive braid in Garside
normal form of length `25`:

```text
beta = x_1 x_2 ... x_25
```

where each `x_j` is a proper simple factor. The prediction target is the right
descent set of the final factor `x_25`:

```text
R(x_25) subset {s_1, s_2, s_3}.
```

Equivalently, the label is a three-bit vector:

```text
[s_1 in R(x_25), s_2 in R(x_25), s_3 in R(x_25)].
```

The model is not given the factor sequence. It only sees the reduced Burau
matrix of the full braid.

## Dataset

We generated `16,777,216` valid random positive `B_4` Garside normal forms of
length `25`. The generator samples from right to left so that adjacent Garside
factors satisfy the normal-form condition:

```text
R(x_j) superset L(x_{j+1}).
```

Each example stores:

- the factor IDs, for audit and analysis;
- the final factor ID;
- the three-bit descent label;
- the Burau matrix over `(Z/2)[v]`, packed into absolute degree slices.

For length `25`, the absolute-degree depth is `101`, and each coefficient
slice is a `3 x 3` matrix.

## First Representation: `(Z/2)[v]`

The first `B_4` model used the Burau matrix reduced mod 2. Each degree slice
is a `3 x 3` binary matrix encoded as a token in `0..511`.

The model was a 3-layer absolute-degree transformer:

| Hyperparameter | Value |
|---|---:|
| Layers | `3` |
| Heads per layer | `6` |
| `d_model` | `192` |
| Readout | CLS token |
| Parameters | about `1.45M` |

Held-out performance:

| Input | Exact descent-set accuracy | Bit accuracy | Micro-F1 |
|---|---:|---:|---:|
| `(Z/2)[v]` support tokens | `71.4%` | `89.5%` | `89.8%` |

This was nontrivial, but far from saturated.

## Mod-2 Mechanistic Result

The mod-2 transformer learned a boundary-band computation. Let `r` denote a
radius around the minimum and maximum occupied Burau degrees. Keeping only
the two boundary windows and zeroing the rest gave:

| Input variant | Exact accuracy | Bit accuracy |
|---|---:|---:|
| Full matrix | `71.5%` | `89.4%` |
| Boundary only, `r=8` | `70.7%` | `89.2%` |
| Boundary only, `r=5` | `68.4%` | `88.4%` |
| Boundary only, `r=3` | `64.5%` | `87.0%` |
| Boundary only, `r=0` | `52.1%` | `82.1%` |
| Drop boundary, `r=8` | `19.0%` | `49.3%` |

Matched boundary patching strengthened this into a causal claim. On pairs with
the same support interval but different labels, transplanting clean boundary
bands into corrupt examples recovered most of the clean decision:

| Patch into corrupt input | Score recovery | Clean-label exact | Bit accuracy |
|---|---:|---:|---:|
| Both boundary bands, `r=8` | `82.5%` | `54.5%` | `79.0%` |
| Both boundary bands, `r=5` | `80.4%` | `50.0%` | `77.0%` |
| Leading band only, `r=8` | `54.8%` | `30.1%` | `65.2%` |
| Trailing band only, `r=8` | `23.5%` | `17.6%` | `51.4%` |
| Interior excluding boundary bands, `r=8` | `17.8%` | `14.5%` | `46.1%` |

Attention analysis also found boundary-specialized heads. In one seed,
`L2H0` was a sharp leading-frontier pointer, with mean CLS attention `0.9027`
to the leading edge. A seed-7 replicate learned the same role with a different
head index (`L2H5`). The role was stable; the head number was not.

The mod-2 conclusion was:

> The model uses a learned frontier circuit, but mod 2 is too lossy to reveal
> a clean algebraic rule.

## Algebraic Rule Mining

We then asked what rule might be hiding underneath the neural computation.

At the level of a single proper simple factor `x` in `B_4`, there is a clean
rule. Let `M = rho(x)` over `Z[v]`. For each degree `d`, let `C_d(M)` be the
bitmask of columns with nonzero entries in `[v^d] M`, and let `D = maxdeg(M)`.
Exact enumeration of the 22 proper simple factors shows:

1. `C_D(M)` is always a subset of `R(x)`.
2. `C_D(M) = R(x)` for 18 of the 22 proper simples.
3. The pair `(C_D(M), C_{D-1}(M))` determines `R(x)` for all 22 proper
   simples.

So the `B_3` rule generalizes at the final-factor level: the final simple
factor has a tiny top-degree signature. The problem is that the observed input
is the Burau matrix of the full product:

```text
rho(x_1 ... x_24) rho(x_25).
```

The prefix transports and obscures the final-factor signature. Over
`(Z/2)[v]`, cancellations create collisions.

We tested simple finite lookup rules over mod-2 boundary features:

| Rule | Exact accuracy | Bit accuracy |
|---|---:|---:|
| trailing top column mask | `51.1%` | `80.9%` |
| best mined mod-2 boundary feature, radius `2` | `63.6%` | `86.9%` |
| right-division frontier deltas | `60.3%` | `85.7%` |
| mod-2 transformer | `71.4%` | `89.5%` |

The hand rules were informative but not close to exact.

## Moving To `Z[v]`

The natural next step was to stop throwing away signs. We replayed the stored
factor sequences through the integer Burau representation over `Z[v]`.

An integer boundary audit showed that signs matter:

| Feature | Exact accuracy | Bit accuracy | Coverage |
|---|---:|---:|---:|
| trailing top column support over `Z[v]` | `60.5%` | `86.3%` | `100.0%` |
| best mod-2 boundary feature | `63.6%` | `86.9%` | `99.9%` |
| signed negative-column masks, `r=3` | `84.8%` | `94.5%` | `99.0%` |
| signed positive-column masks, `r=3` | `84.2%` | `94.3%` | `98.9%` |
| trailing sign tokens, `r=2` | `84.1%` | `94.1%` | `98.7%` |

This changed the project direction. The useful information was not just
support; it was signed boundary structure.

## Z-Sign Tokens

We trained a second transformer on a compressed signed version of the integer
Burau matrix. Each coefficient slice is a `3 x 3` sign matrix with entries:

```text
0 = zero, 1 = negative, 2 = positive.
```

The slice is encoded as a base-3 token, so the vocabulary size is:

```text
3^9 = 19,683.
```

This representation keeps support and sign, but discards coefficient
magnitudes.

The model uses the same 3-layer `d_model=192` CLS architecture as before. It
computes sign tokens on GPU from the stored factor IDs during training.

Held-out performance:

| Input | Exact descent-set accuracy | Bit accuracy | Micro-F1 |
|---|---:|---:|---:|
| `(Z/2)[v]` support tokens | `71.4%` | `89.5%` | `89.8%` |
| `Z[v]` sign tokens | `93.2%` | `97.7%` | `97.7%` |

This was the main predictive breakthrough. The sign-token representation made
the task much cleaner.

## First-Pass Z-Sign Interventions

On `8,192` held-out examples, the trained Z-sign transformer behaved as
follows under input interventions:

| Input intervention | Exact accuracy | Bit accuracy |
|---|---:|---:|
| Full sign-token matrix | `93.9%` | `98.0%` |
| Boundary only, radius `8` | `73.0%` | `90.3%` |
| Boundary only, radius `5` | `58.0%` | `83.6%` |
| Boundary only, radius `3` | `47.1%` | `77.1%` |
| Drop boundary, radius `8` | `19.9%` | `54.6%` |
| Drop boundary, radius `5` | `32.2%` | `62.9%` |

This told us two things:

1. The signed frontier is causally important.
2. The full model also uses information outside the tiny boundary windows
   when evaluated under this direct zeroing intervention.

Simple lookup baselines remained weaker than the model:

| Lookup feature | Exact accuracy | Bit accuracy | Coverage |
|---|---:|---:|---:|
| leading token | `58.6%` | `86.2%` | `100.0%` |
| trailing token | `61.8%` | `86.8%` | `99.9%` |
| leading + trailing tokens | `63.1%` | `87.2%` | `99.8%` |
| trailing window, radius `2` | `82.6%` | `92.7%` | `95.8%` |
| both windows, radius `1` | `78.2%` | `91.4%` | `95.6%` |

So the transformer is doing more than a tiny extremal-token lookup.

## Deep Mechanistic Experiments

We then ran a deeper pass designed to test whether the model had learned a
recoverable algebraic computation.

The deep-dive run used:

- `32,768` train examples for probes and lookup tables;
- `8,192` held-out examples for evaluation;
- `512` prefix-fixed counterfactual pairs.

Artifact:

```text
interp/artifacts/b4_l25_zsign_deep_dive/results.json
```

### Circuit-Derived Classifiers

First, we asked whether internal activations linearly expose the model's
semantic state.

A direct three-bit descent probe from the late CLS stream nearly matches the
model:

| Representation | Probe exact | Probe bit | Agreement with model |
|---|---:|---:|---:|
| final hidden CLS | `93.2%` | `97.7%` | `97.3%` |
| layer-2 residual post CLS | `92.8%` | `97.5%` | `96.8%` |
| layer-1 residual post CLS | `91.4%` | `97.0%` | `95.0%` |
| layer-0 residual post CLS | `87.5%` | `95.7%` | `90.4%` |

This shows that the late CLS representation makes the descent decision almost
linearly available.

More interestingly, we trained a probe to decode the final simple factor, then
composed that probe with the exact simple-factor descent lookup:

| Representation | Rule exact | Rule bit | Final-factor accuracy | Agreement with model |
|---|---:|---:|---:|---:|
| final hidden CLS | `90.9%` | `96.7%` | `75.2%` | `93.8%` |
| layer-2 residual post CLS | `90.8%` | `96.7%` | `76.4%` | `93.7%` |
| layer-1 residual post CLS | `89.2%` | `96.1%` | `75.5%` | `91.8%` |
| layer-0 residual post CLS | `84.7%` | `94.5%` | `72.4%` | `86.9%` |

This is our closest current version of a recovered algorithm:

```text
late CLS -> approximate final simple factor -> exact descent lookup.
```

It is not perfect, but it recovers most of the model's behavior and is much
more interpretable than the raw neural classifier.

### Prefix-Fixed Counterfactuals

We constructed counterfactual pairs:

```text
clean   = prefix * final_factor_a
corrupt = prefix * final_factor_b
```

The prefix is identical. The final factor is changed to another valid simple
factor with a different descent set.

This is a stronger test than matching examples by support interval. It asks:
when the algebraic change is only the final factor, where does the model see
that change in the Burau matrix?

On these pairs:

| Run | Clean-label exact | Clean-label bit |
|---|---:|---:|
| clean example | `93.2%` | `97.6%` |
| corrupt example scored as clean | `0.8%` | `38.9%` |

Patching clean input regions into the corrupt example:

| Patched clean input region | Score recovery | Clean-label exact | Clean-label bit |
|---|---:|---:|---:|
| both boundary bands, radius `8` | `90.2%` | `82.6%` | `93.7%` |
| both boundary bands, radius `5` | `86.3%` | `75.6%` | `90.6%` |
| both boundary bands, radius `3` | `80.4%` | `70.1%` | `88.5%` |
| trailing band, radius `8` | `47.6%` | `37.1%` | `69.3%` |
| leading band, radius `8` | `27.5%` | `10.0%` | `49.4%` |
| interior except boundary bands, radius `8` | `7.0%` | `2.5%` | `41.7%` |

This is the strongest causal result in the `B_4` work:

> Holding the prefix fixed, the signed radius-8 boundary bands carry about
> 90% of the model's clean-corrupt score difference.

The boundary is not only a correlational feature; it causally transmits the
final-factor change used by the model.

Activation patching shows where the information becomes explicit:

| Activation patch | Score recovery | Clean-label exact |
|---|---:|---:|
| input embedding boundary bands, radius `8` | `90.2%` | `82.6%` |
| layer-0 residual post at CLS | `76.5%` | `64.6%` |
| layer-1 residual post at CLS | `96.4%` | `92.4%` |
| layer-2 residual post at CLS | `100.0%` | `93.2%` |

By the end of layer 1, the boundary-derived information is mostly collapsed
into CLS. The final block largely sharpens the already available decision.

### Right-Quotient Frontier

Descent is right divisibility. Algebraically, `s_i` is in the right descent
set when right-division by `s_i` succeeds. We therefore tested finite lookup
features on:

```text
rho(beta) rho(s_i)^(-1)
```

for each `i = 1, 2, 3`.

The quotient frontier is informative:

| Quotient feature | Bit accuracy over `(beta, i)` | Set exact after recombining bits | Coverage |
|---|---:|---:|---:|
| trailing quotient window, radius `3` | `89.8%` | `73.9%` | `91.8%` |
| both quotient windows, radius `2` | `88.1%` | `67.6%` | `89.5%` |
| leading quotient window, radius `3` | `88.0%` | `64.2%` | `99.4%` |
| leading quotient token | `86.7%` | `60.3%` | `100.0%` |

This is mathematically meaningful but not yet the full theorem. The quotient
frontier is good at the per-generator bit task, but its bit errors compound
when recombined into full descent sets.

## Constrained Boundary-Only Model

Finally, we trained a deliberately smaller transformer that only ever sees the
radius-8 signed boundary bands. Everything outside the leading/trailing
support windows is zeroed before the model input.

| Hyperparameter | Value |
|---|---:|
| Layers | `2` |
| Heads per layer | `4` |
| `d_model` | `96` |
| Parameters | `2.1M` |
| Visible input | radius-8 signed boundary bands only |

Held-out performance:

| Model | Visible input | Exact accuracy | Bit accuracy | Micro-F1 |
|---|---|---:|---:|---:|
| full Z-sign transformer | all `101` degree slices | `93.2%` | `97.7%` | `97.7%` |
| small boundary-only transformer | radius-8 signed frontier only | `90.6%` | `96.7%` | `96.7%` |

This result shows that the signed frontier is not only sufficient under a
patching intervention on the larger model. It is rich enough to support an
independently trained model with about `91%` exact accuracy.

Artifact:

```text
interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small/results.json
```

## Boundary-Only Model Interpretability

We then ran the same deep-dive harness on the small boundary-only model. The
analysis uses the same input transform the model was trained on: all tokens
outside the radius-8 signed leading/trailing frontier are zeroed before the
model sees the example.

Artifact:

```text
interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small_deep_dive/results.json
```

On the `8,192` example analysis split, the boundary-only model gets:

| Model | Exact accuracy | Bit accuracy | Micro-F1 |
|---|---:|---:|---:|
| boundary-only transformer, analysis split | `91.1%` | `96.9%` | `96.9%` |

The late CLS stream again exposes the task almost linearly:

| Representation | Probe exact | Probe bit | Agreement with model |
|---|---:|---:|---:|
| final hidden CLS | `90.1%` | `96.6%` | `96.0%` |
| layer-1 residual post CLS | `89.8%` | `96.4%` | `95.5%` |
| layer-0 residual post CLS | `68.4%` | `88.8%` | `68.5%` |

The causal circuit is also easier to read than in the full model:

| Intervention | Exact accuracy after ablation | Exact drop |
|---|---:|---:|
| full boundary-only model | `91.1%` | `0.0%` |
| zero layer-1 attention | `48.4%` | `42.7%` |
| zero layer-0 MLP | `48.8%` | `42.3%` |
| zero layer-0 attention | `75.7%` | `15.5%` |
| zero layer-1 MLP | `81.2%` | `10.0%` |
| zero `L1H2` at CLS | `80.6%` | `10.6%` |
| zero `L1H1` at CLS | `81.3%` | `9.9%` |

The strongest CLS boundary heads are all in layer 1:

| Head | Leading mass | Trailing mass | Boundary mass | Support mass |
|---|---:|---:|---:|---:|
| `L1H1` | `0.001` | `0.385` | `0.385` | `0.957` |
| `L1H2` | `0.004` | `0.356` | `0.359` | `0.903` |
| `L1H3` | `0.246` | `0.015` | `0.260` | `0.854` |
| `L1H0` | `0.002` | `0.248` | `0.249` | `0.889` |

Two heads, `L1H1` and `L1H2`, specialize strongly to the trailing boundary
and are also the two largest single-head ablation effects. `L1H3` is the
leading-boundary counterpart. The emerging circuit is:

```text
layer-0 attention/MLP builds local signed-frontier features
    -> layer-1 attention reads frontier features into CLS
    -> layer-1 MLP sharpens the descent decision
```

Prefix-fixed counterfactual patching is especially clean in this model because
the only visible input is the frontier:

| Patched clean input region | Score recovery | Clean-label exact | Clean-label bit |
|---|---:|---:|---:|
| both boundary bands, radius `8` | `99.9%` | `89.5%` | `96.4%` |
| both boundary bands, radius `5` | `97.6%` | `88.3%` | `95.8%` |
| both boundary bands, radius `3` | `93.2%` | `84.2%` | `94.4%` |
| trailing band, radius `8` | `53.2%` | `44.3%` | `73.2%` |
| leading band, radius `8` | `45.2%` | `19.7%` | `61.0%` |
| interior except boundary bands, radius `8` | `0.3%` | `1.6%` | `39.4%` |

The small model is therefore the clearest current circuit target: it is strong,
causally boundary-only, and has a visible layer-1 frontier readout.

We also trained a seed-7 boundary-only replicate with the same architecture.
It reached `90.36%` exact accuracy, `96.66%` bit accuracy, and `96.61%`
micro-F1 on the held-out test set. On the deep-dive split, its late-`CLS`
direct descent probe reached `88.68%` exact accuracy, and radius-8 boundary
patching recovered `99.7%` of the clean-corrupt score difference. The
component-level dependencies shifted somewhat: this seed leaned more heavily
on the layer-0 MLP and layer-1 attention, with the largest single-head effect
at `L1H0`. The robustness lesson is that the invariant is the signed-frontier
computation and late-`CLS` readout, not a particular numbered head.

## Latent-State Search

We also searched for a better circuit-derived latent state than the literal
final factor. For each final simple factor, we computed algebraic labels such
as its right descent mask, left/right descent pair, top column mask, top-two
column masks `(C_D, C_{D-1})`, and final factor ID. We trained linear probes
for each latent target and mapped the predicted latent class to a descent set.

For the full Z-sign model, the best non-tautological latent probes from final
CLS were:

| Latent target | Rule exact | Rule bit | Latent accuracy | Agreement with model |
|---|---:|---:|---:|---:|
| left/right descent masks | `91.2%` | `96.9%` | `74.7%` | `94.1%` |
| top-two column masks | `91.1%` | `96.8%` | `76.5%` | `94.0%` |
| final factor ID | `90.9%` | `96.7%` | `75.2%` | `93.8%` |

For the boundary-only model:

| Latent target | Rule exact | Rule bit | Latent accuracy | Agreement with model |
|---|---:|---:|---:|---:|
| left/right descent masks | `86.8%` | `95.3%` | `66.4%` | `90.1%` |
| top-two column masks | `86.7%` | `95.2%` | `68.2%` | `90.4%` |
| final factor ID | `86.5%` | `95.2%` | `68.0%` | `89.5%` |

The latent search did not find a dramatically better state than final factor
ID. But it did find a useful refinement: the final simple's top-two column
signature and the left/right descent pair are slightly more natural than the
literal 22-way factor ID. This fits the algebraic picture: the model is closer
to recovering a transported descent-relevant signature than to memorizing
factor identities.

## Sparse Autoencoder Pass

We then trained TopK sparse autoencoders on the boundary-only Z-sign model's
internal activations. The point was not just to see whether an SAE can
reconstruct a residual stream. The real question was whether a small family of
sparse features carries the descent computation causally and robustly.

Artifacts:

```text
interp/artifacts/b4_l25_zsign_boundary_r8_sae_final/results.json
interp/artifacts/b4_l25_zsign_boundary_r8_sae_final/SUMMARY.md
interp/B4_SAE_FINAL.md
```

The final analysis used the seed-42 and seed-7 boundary-only models, `8,192`
held-out examples, `32,768` train examples for feature-only classifiers, and
`512` prefix-fixed counterfactual pairs.

The main result is:

> A few dozen late-CLS SAE features recover almost all of the boundary-only
> model's descent-set computation. They beat count-matched random active
> features, recur across seeds, and are causally fed by layer-1 CLS attention
> paths.

Sparse-feature classifiers:

| Seed | Site | Feature set | k | Exact | Bit | Agreement with transformer | Random exact |
|---|---|---|---:|---:|---:|---:|---:|
| seed 7 | `final_hidden_cls` | binary | `29` | `89.4%` | `96.3%` | `96.0%` | `49.6%` |
| seed 7 | `final_hidden_cls` | descent | `16` | `89.2%` | `96.3%` | `95.7%` | `36.1%` |
| seed 42 | `final_hidden_cls` | descent | `17` | `89.0%` | `96.1%` | `94.4%` | `36.0%` |
| seed 42 | `final_hidden_cls` | binary | `27` | `88.9%` | `96.1%` | `94.4%` | `46.9%` |
| seed 7 | `l1_resid_post_cls` | binary | `32` | `88.0%` | `95.7%` | `92.9%` | `52.8%` |
| seed 42 | `l1_resid_post_cls` | binary | `31` | `87.0%` | `95.4%` | `91.4%` | `52.6%` |

The teacher models are about `90.6-90.8%` exact on the same evaluation slice.
So selected sparse features recover nearly all of the model; random active
features do not.

The same feature families recur across seeds. Matching selected seed-42
features to seed-7 features by activation correlation gives mean best-match
correlations of `0.649` at `final_hidden_cls` and `0.613` at
`l1_resid_post_cls`, with most best matches also selected by the seed-7
feature-label analysis.

Path patching gives the circuit-level connection. On prefix-fixed pairs,
patching all layer-1 attention-head outputs at `CLS` recovers the selected
late SAE features and much of the clean decision:

| Seed | Target site | Selected features | Feature recovery | Logit recovery |
|---|---|---:|---:|---:|
| seed 42 | `l1_resid_post_cls` | `31` | `81.1%` | `76.2%` |
| seed 42 | `final_hidden_cls` | `27` | `76.6%` | `76.2%` |
| seed 7 | `final_hidden_cls` | `29` | `69.0%` | `64.6%` |
| seed 7 | `l1_resid_post_cls` | `32` | `65.1%` | `64.6%` |

The SAE result therefore sharpens the mechanistic story:

```text
signed boundary tokens
    -> layer-1 CLS attention paths
    -> sparse late-CLS descent features
    -> output logits
```

This is not a single-feature theorem. Individual features are meaningful but
small. The causal unit is a sparse distributed feature family.

## Hidden-Theorem Search

We then explicitly searched for a cleaner algebraic rule over `Z[v]`.

Artifact:

```text
interp/artifacts/b4_l25_z_hidden_theorem_combo/results.json
```

This audit used `131,072` train examples and `32,768` held-out examples. It
tested signed boundary states, generator quotient states
`rho(beta) rho(s_i)^{-1}`, quotient signatures for all 22 proper simple
factors, clipped coefficient-magnitude states, and combinations of the best
features.

The naive right-divisibility theorem does not work in reduced Burau:

| Direct quotient predicate | Exact set accuracy | Bit accuracy |
|---|---:|---:|
| no negative Laurent terms in `rho(beta) rho(s_i)^{-1}` | `0.0%` | `50.0%` |
| first quotient exponent nonnegative | `0.0%` | `50.0%` |

The best finite states were useful but not close to exact:

| Candidate state | Exact accuracy | Bit accuracy |
|---|---:|---:|
| signed boundary negative-column masks, radius `3` | `84.0%` | `94.1%` |
| generator quotient signed-column state, radius `1` | `84.0%` | `94.0%` |
| all-simple quotient width-delta state | `78.3%` | `92.5%` |
| best combined state | `85.1%` | `94.2%` |

The conflict tables show repeated feature keys with multiple descent masks, not
just sparse unseen-key failures. Within this family of small signed frontier or
quotient states, there is no exact rule.

## Interpretation

The `B_4` story now has a clear shape.

In `B_3`, the exact rule is visible from one extremal matrix unit. In `B_4`,
the final simple factor also has a small top-degree signature, but the prefix
transports and obscures it. Mod 2 destroys too much sign information, leaving
only a weak frontier signal. Over `Z[v]`, even after discarding magnitudes and
keeping only signs, the signal becomes strong.

The transformer appears to learn the following kind of computation:

```text
signed Burau matrix
    -> read signed leading/trailing boundary frontier
    -> collapse boundary-derived state into CLS
    -> infer a descent-relevant latent state close to the final simple factor
    -> output the three descent bits
```

Evidence for each step:

| Claim | Evidence |
|---|---|
| Signs matter | Z-sign model jumps from `71.4%` to `93.2%` exact accuracy |
| Boundary matters | dropping radius-8 boundary windows collapses accuracy |
| Boundary is causal | prefix-fixed boundary patching recovers `90.2%` of clean-corrupt score |
| Boundary can stand alone | small boundary-only model gets `90.6%` exact |
| The model forms a late semantic state | final CLS descent probe gets `93.2%` exact |
| That state is close to final-factor information | final-factor probe plus exact lookup gets `90.9%` exact |
| The small circuit is readable | boundary-only layer-1 heads read trailing/leading frontier into CLS |
| The best nontrivial latent state is algebraic | top-two final-simple column masks slightly beat final-factor ID |
| The obvious quotient theorem fails | direct quotient polynomiality gives `50%` bit accuracy |
| Small hand states are not enough | best combined quotient/frontier lookup gets `85.1%` exact |

The current result is not a complete theorem. The remaining gap is the
`2-3%` between the boundary-only/final-factor-derived explanations and the
full Z-sign model. That gap may be extra non-boundary information, coefficient
magnitude information lost in sign tokens, or a more natural latent state than
the literal final simple factor.

The result is mechanistically strong:

> We trained a transformer to predict a braid-theoretic descent set from a
> Burau matrix, found that the successful representation lives over signed
> `Z[v]` boundary data rather than mod-2 support, causally localized the
> final-factor signal to the signed frontier, and recovered most of the
> model's computation as a final-factor-style algebraic classifier.

The compact summary is:

> `B_4` descent is not directly visible from a single Burau coefficient, but
> it becomes highly learnable from the signed boundary frontier over `Z[v]`.
> A small boundary-only transformer reaches `90.6%` exact accuracy, and its
> main circuit consists of layer-1 attention heads reading signed frontier
> features into CLS. The late CLS state is well explained by algebraic final
> simple-factor signatures, especially the top-two column signature that
> exactly determines descent for isolated simple factors.

## Artifacts

Public-facing summaries:

- Project overview: `interp/README.md`
- Tracked metrics: `interp/RESULTS.md`
- Mechanism diagram: `interp/figures/b4_signed_frontier.svg`

Main datasets and models:

- B4 dataset:
  `interp/data/generated/b4_l25_p2_n16777216`
- Mod-2 transformer:
  `interp/artifacts/b4_l25_p2_xfmr3_abs/results.json`
- Z-sign transformer:
  `interp/artifacts/b4_l25_zsign_xfmr3_abs/results.json`
- Boundary-only Z-sign transformer:
  `interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small/results.json`

Mechanistic experiments:

- Mod-2 first-pass interp:
  `interp/artifacts/b4_l25_p2_firstpass_interp/results.json`
- Mod-2 matched boundary patching:
  `interp/artifacts/b4_l25_p2_matched_boundary_patching/results.json`
- Mod-2 boundary-head circuit:
  `interp/artifacts/b4_l25_p2_boundary_head_circuit/results.json`
- Integer boundary audit:
  `interp/artifacts/b4_l25_p2_integer_boundary/results.json`
- Z-sign first-pass interp:
  `interp/artifacts/b4_l25_zsign_firstpass_interp/results.json`
- Z-sign deep dive:
  `interp/artifacts/b4_l25_zsign_deep_dive/results.json`
- Boundary-only Z-sign deep dive:
  `interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small_deep_dive/results.json`
- Boundary-only Z-sign seed-7 robustness:
  `interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small_seed7/results.json`
- Boundary-only Z-sign seed-7 deep dive:
  `interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small_seed7_deep_dive/results.json`
- Boundary-only Z-sign SAE final analysis:
  `interp/artifacts/b4_l25_zsign_boundary_r8_sae_final/results.json`
- Hidden-theorem search:
  `interp/artifacts/b4_l25_z_hidden_theorem_combo/results.json`

Primary scripts:

- `interp/generate_b4_dataset.py`
- `interp/train_b4_transformer.py`
- `interp/train_b4_z_sign_transformer.py`
- `interp/run_b4_firstpass_interp.py`
- `interp/run_b4_z_sign_firstpass_interp.py`
- `interp/run_b4_z_sign_deep_dive.py`
- `interp/run_b4_sae_experiments.py`
- `interp/stress_b4_sae_controls.py`
- `interp/analyze_b4_sae_final.py`
- `interp/audit_b4_integer_boundary.py`
- `interp/search_b4_hidden_theorem.py`

## Next Steps

The most promising follow-up is now narrower: explain the remaining gap
between the clean boundary-only circuit and the full Z-sign model.

1. Interpret the boundary-only model's layer-0 MLP features. This is now the
   largest dependency in the small model besides layer-1 attention.
2. Turn the top-two-column latent probe into a circuit-derived classifier:
   decode the final-simple top signature, then apply the exact column rule.
3. Move beyond small finite lookups: test a learned but interpretable
   automaton/state-space model over the signed matrix frontier.
4. Add coefficient magnitudes in a model-facing way, not just clipped lookup
   features, and see whether the model's extra `2-3%` comes from magnitude
   cues.

The ideal end state would be a theorem-like statement: a finite signed
boundary state, computable from the `Z[v]` Burau matrix, determines or nearly
determines the final right descent set for these length-25 `B_4` normal forms.
