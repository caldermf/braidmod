# B4 SAE Final Analysis

This is the final sparse-autoencoder pass on the `B_4` Z-sign boundary-only
transformer. The purpose was to stress-test whether the SAE story is a real
mechanistic result or just a reconstruction trick.

The short answer:

> The result is real, but it is a sparse distributed code rather than a single
> magic feature. A few dozen late-CLS SAE features recover most of the
> boundary-only model's descent-set computation, beat random feature controls
> by a wide margin, recur across independently trained seeds, and are fed
> primarily by layer-1 CLS attention paths.

## Setup

The model is the boundary-only Z-sign transformer:

```text
interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small/best_model.pt
```

with a seed-7 replicate:

```text
interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small_seed7/best_model.pt
```

Both models see only radius-8 signed boundary windows from the integer Burau
matrix over `Z[v]`, encoded as sign tokens. The target is the right descent
set of the final Garside factor in `B_4`.

The SAE analysis uses:

- `8,192` held-out evaluation examples;
- `32,768` training examples for sparse-feature classifiers;
- `512` prefix-fixed counterfactual pairs;
- two independently trained model seeds;
- SAEs trained at `final_hidden_cls`, `l1_resid_post_cls`, and
  `l1_attn_out_cls`.

Main artifact:

```text
interp/artifacts/b4_l25_zsign_boundary_r8_sae_final/results.json
interp/artifacts/b4_l25_zsign_boundary_r8_sae_final/SUMMARY.md
```

Runner:

```text
interp/analyze_b4_sae_final.py
interp/jobs/analyze_b4_sae_final.sh
```

## Stress-Test Result

The earlier SAE stress controls showed that full SAE reconstruction is not
the interesting part. Full-feature patching is expected to work if the SAE
faithfully reconstructs the activation.

The nontrivial result is that small semantically selected feature sets work
far better than random active feature sets.

For seed 7 at `final_hidden_cls`, patching 16 descent-labeled SAE features on
prefix-fixed clean/corrupt pairs recovered:

| Feature set | Score recovery | Clean-label exact |
|---|---:|---:|
| selected descent features | `0.753` | `0.637` |
| random 16 active features | `0.032` | `0.024` |

For seed 42 at `final_hidden_cls`, 16 descent-labeled features recovered:

| Feature set | Score recovery | Clean-label exact |
|---|---:|---:|
| selected descent features | `0.556` | `0.424` |
| random 16 active features | `0.022` | `0.020` |

Permuting the clean rows before all-feature patching collapsed recovery to
about `0.18-0.22`, so the patch is not just injecting a generic high-quality
SAE reconstruction. The features have to come from the right example.

## Sparse Feature Classifiers

We trained tiny linear classifiers on only the selected SAE feature
activations. This asks whether the selected sparse features contain enough
information to reproduce the model's descent prediction without access to the
full residual stream.

Best results:

| Seed | Site | Feature set | k | Exact | Bit | Agreement with transformer | Random exact |
|---|---|---|---:|---:|---:|---:|---:|
| seed 7 | `final_hidden_cls` | binary | `29` | `0.894` | `0.963` | `0.960` | `0.496` |
| seed 7 | `final_hidden_cls` | descent | `16` | `0.892` | `0.963` | `0.957` | `0.361` |
| seed 42 | `final_hidden_cls` | descent | `17` | `0.890` | `0.961` | `0.944` | `0.360` |
| seed 42 | `final_hidden_cls` | binary | `27` | `0.889` | `0.961` | `0.944` | `0.469` |
| seed 7 | `l1_resid_post_cls` | binary | `32` | `0.880` | `0.957` | `0.929` | `0.528` |
| seed 42 | `l1_resid_post_cls` | binary | `31` | `0.870` | `0.954` | `0.914` | `0.526` |

The teacher models are around `0.906-0.908` exact on this eval slice, so the
selected `final_hidden_cls` SAE features recover nearly all of the model's
performance. Count-matched random active features do not.

This is the cleanest SAE result:

> The model's late descent computation is almost linearly readable from a few
> dozen sparse features, and those features are not interchangeable with random
> active SAE features.

## Cross-Seed Recurrence

We matched features across the seed-42 and seed-7 models by activation
correlation on the same held-out examples. This is a stronger test than
matching feature indices, since the two models do not share a hidden-space
basis.

Average best-match absolute correlations:

| Site | Matched seed-42 features | Mean best corr | Best match also selected in seed 7 |
|---|---:|---:|---:|
| `final_hidden_cls` | `48` | `0.649` | `37` |
| `l1_resid_post_cls` | `48` | `0.613` | `37` |
| `l1_attn_out_cls` | `53` | `0.552` | `36` |

Many of the strongest feature matches have correlations above `0.8`. For
example, seed-42 final-hidden feature `1283` matches seed-7 feature `675`
with correlation `0.916`, and both are selected descent/binary features.

The important point is not one-to-one identity. The feature basis is not
canonical. The important point is recurrence of the same sparse concepts:
late features that fire on the same algebraic examples, carry the same
descent-relevant information, and appear in both independently trained runs.

## Path Patching

We then asked where the selected late SAE features come from. On prefix-fixed
clean/corrupt pairs, we patched clean attention-head outputs at the `CLS`
position into corrupt runs and measured recovery of the selected SAE feature
activations.

The strongest intervention is patching all layer-1 attention heads at `CLS`.

| Seed | Target site | Selected features | Feature recovery | Logit recovery | Clean-label exact |
|---|---|---:|---:|---:|---:|
| seed 42 | `l1_resid_post_cls` | `31` | `0.811` | `0.762` | `0.719` |
| seed 42 | `final_hidden_cls` | `27` | `0.766` | `0.762` | `0.719` |
| seed 7 | `final_hidden_cls` | `29` | `0.690` | `0.646` | `0.506` |
| seed 7 | `l1_resid_post_cls` | `32` | `0.651` | `0.646` | `0.506` |

Single heads matter too, especially `L1H1` in both seeds, but no single head
explains the whole code. The pattern is distributed:

```text
signed boundary tokens
    -> layer-1 CLS attention paths
    -> late CLS sparse descent features
    -> output logits
```

This is exactly the kind of mechanistic structure we hoped to see: not merely
good probes, but a causal path from upstream attention into the sparse
features that drive the decision.

## Feature Atlas

Individual features are meaningful but not individually sufficient. Some fire
for `s_i` being present, others fire for `s_i` being absent, and several write
mixed logit directions. For instance, final-hidden SAE features often have
top-activation precision near `0.0` or `1.0` for a descent bit against a base
rate around `0.5`.

Examples:

| Seed | Site | Feature | Top label behavior | Individual patch |
|---|---|---:|---|---:|
| seed 42 | `final_hidden_cls` | `1188` | top examples have `descent_s1 = 0` | `0.053` |
| seed 42 | `final_hidden_cls` | `545` | top examples have `descent_s1 = 1` | `0.021` |
| seed 7 | `final_hidden_cls` | `1103` | top examples have `descent_s1 = 1` | `0.063` |
| seed 7 | `final_hidden_cls` | `1036` | top examples have `descent_s3 = 0` | `0.128` |

The feature-level picture is therefore not "one feature equals one answer."
It is a sparse evidence code: many crisp features, each carrying part of the
descent decision, with the output head combining them.

## Interpretation

The B4 SAE pass upgrades the project in an important way.

Before this pass, the story was:

> The boundary-only transformer uses signed frontier information, and its late
> CLS state linearly probes for descent/final-factor information.

After this pass, the story is sharper:

> The late CLS state decomposes into a small sparse feature family. A few dozen
> selected SAE features almost recover the model's descent prediction, beat
> random active features by a large margin, recur across seeds, and receive
> causal input from layer-1 CLS attention.

This is a solid mechanistic-interpretability result. It is not yet a theorem
about braid groups, but it is a convincing recovered-algorithm result for the
trained model: the model builds a sparse descent-relevant code from signed
Burau boundary data.

## Caveats

The honest caveats are:

- The code is distributed. We did not find a single feature or one-line rule
  that exactly determines the answer.
- The strongest selected-feature classifiers reach about `0.89` exact, just
  below the teacher's `0.91` exact on the same slice.
- Feature bases are not canonical, so cross-seed matching is concept-level
  recurrence, not literal feature-index recurrence.
- The result is for the boundary-only Z-sign model. The full Z-sign model has
  a little extra performance and may use additional non-boundary information.

These caveats are fine. They make the claim precise rather than weaker.

## Punchline

The punchline for the repo is:

> We trained a transformer to infer `B_4` descent sets from Burau matrices,
> found that signed `Z[v]` boundary data is the right representation, and then
> used SAEs to recover the model's internal descent code. A few dozen sparse
> late-CLS features recover almost all of the boundary-only model, survive
> random and permutation controls, recur across seeds, and are causally fed by
> layer-1 attention heads reading the boundary frontier.
