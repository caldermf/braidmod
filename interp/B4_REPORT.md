# Learning Braid Group Representations: the `B_4` Case

This note summarizes the first mechanistic interpretability pass on the
`B_4` transformer. The task is: given only the reduced Burau matrix of a
positive length-25 `B_4` braid over `(Z/2)[v]`, predict the three-bit right
descent set of the final Garside factor, i.e. membership of `s_1`, `s_2`,
and `s_3`.

The headline result is that the `B_4` model does not reduce to the `B_3`
single-boundary-token rule. Instead, it learns a boundary-band algorithm:
almost all of its performance is preserved when the input is restricted to
moderately thick windows around the minimum and maximum occupied Burau
degrees, and performance collapses when those windows are removed. The
decision is then consolidated into the final CLS residual stream by a small
set of boundary-specialized attention heads.

## Dataset and Model

The `B_4` corpus contains `16,777,216` randomly sampled valid positive
Garside normal forms of length `25`. Each example stores the reduced Burau
matrix over `(Z/2)[v]` as `101` absolute-degree coefficient-slice tokens.
Each token is a `3 x 3` binary matrix encoded as an integer in `0..511`.

The model is a 3-layer transformer with `d_model=192`, 6 heads per layer,
and a CLS readout. It has about `1.45M` parameters.

Held-out test performance for the main seed-42 model:

| Metric | Value |
|---|---:|
| Exact descent-set accuracy | `71.37%` |
| Bit accuracy | `89.45%` |
| Micro-F1 | `89.82%` |

This is intentionally a harder regime than `B_3`: the model is strong but not
saturated, so the circuit has visible failure modes and distributed structure.

## Boundary-Band Sufficiency

A direct intervention on the input shows that the model mostly uses a thick
frontier around the support of the Burau polynomial matrix. Let `r` denote a
window radius around both the minimum and maximum occupied degrees. Keeping
only those windows and zeroing the rest of the matrix gives:

| Input variant | Exact accuracy | Bit accuracy | Mean logit score |
|---|---:|---:|---:|
| Full matrix | `71.46%` | `89.41%` | `4.259` |
| Boundary only, `r=8` | `70.70%` | `89.17%` | `4.053` |
| Boundary only, `r=5` | `68.38%` | `88.41%` | `3.873` |
| Boundary only, `r=3` | `64.48%` | `87.00%` | `3.672` |
| Boundary only, `r=0` | `52.06%` | `82.08%` | `2.970` |
| Drop boundary, `r=8` | `18.96%` | `49.32%` | `-0.003` |

So the model is not merely reading the two extremal coefficient slices. It
needs a band of nearby coefficient slices. But that band is nearly sufficient:
with only `34` of the `101` degree tokens retained, the model recovers almost
all of its full-input accuracy.

This is the clean conceptual contrast with `B_3`. In `B_3`, a single extremal
matrix unit determines the answer exactly. In `B_4`, the learned algorithm
appears to read a local frontier of the Burau matrix at both ends of the
degree support.

## Matched Patching

To make the causal test sharper, we formed matched pairs with the same
minimum and maximum occupied degrees but different descent masks. We then ran
the corrupt example while patching input windows from the clean example and
scored logits against the clean label.

On `512` matched pairs:

| Patch into corrupt input | Normalized score recovery | Exact clean-label accuracy | Bit accuracy |
|---|---:|---:|---:|
| Corrupt baseline | `0.000` | `6.84%` | `44.40%` |
| Both boundary bands, `r=8` | `0.825` | `54.49%` | `79.04%` |
| Both boundary bands, `r=5` | `0.804` | `50.00%` | `76.95%` |
| Both boundary bands, `r=3` | `0.531` | `31.25%` | `63.35%` |
| Leading band only, `r=8` | `0.548` | `30.08%` | `65.23%` |
| Trailing band only, `r=8` | `0.235` | `17.58%` | `51.37%` |
| Interior excluding boundary bands, `r=8` | `0.178` | `14.45%` | `46.09%` |

The `r=8` boundary bands carry most of the clean decision. The complement
with those bands removed carries little. This rules out the weaker
interpretation that the boundary-only result is just an out-of-distribution
artifact: on paired examples with the same support interval, transplanting
the boundary bands moves the model strongly toward the clean answer.

Activation patching localizes where this information becomes explicit:

| Patched activation | Normalized score recovery |
|---|---:|
| `blocks.2.hook_resid_mid` at CLS | `1.000` |
| `blocks.2.hook_resid_post` at CLS | `1.000` |
| `hook_final_hidden` at CLS | `1.000` |
| `blocks.1.hook_resid_mid`, CLS plus boundary `r=3` | `0.775` |
| `blocks.2.hook_mlp_out` at CLS | `0.752` |

By the final block, the relevant boundary-band information has been collapsed
into the CLS stream.

## Boundary-Specialized Heads

Attention patterns reveal a striking division of labor. In the final layer,
two CLS heads specialize to opposite ends of the Burau support:

| Head | Mean CLS attention to leading edge | Mean CLS attention to trailing edge | Mean attention to support | Entropy |
|---|---:|---:|---:|---:|
| `L2H0` | `0.9027` | `0.0002` | `0.9848` | `0.255` |
| `L2H1` | `0.0004` | `0.5756` | `0.7444` | `1.028` |
| `L1H2` | `0.3710` | `0.0010` | `0.9768` | `1.451` |
| `L1H5` | `0.0042` | `0.3045` | `0.9592` | `1.784` |

The strongest head, `L2H0`, is almost a pointer to the leading frontier: on
typical examples it puts nearly all CLS attention on the first occupied
degree and its immediate neighbors.

Targeted ablations show that these heads are causally involved but not the
whole story:

| Intervention | Exact accuracy | Bit accuracy | Mean logit score |
|---|---:|---:|---:|
| Full model | `71.46%` | `89.41%` | `4.259` |
| Zero all layer-0 attention | `46.39%` | `75.88%` | `2.114` |
| Zero all layer-2 attention | `62.39%` | `85.41%` | `2.730` |
| Zero layer-2 MLP | `65.09%` | `87.09%` | `2.282` |
| Zero `L2H0` at CLS | `67.98%` | `87.64%` | `3.094` |
| Zero `L2H1` at CLS | `70.21%` | `89.04%` | `4.072` |

The final-layer leading-edge head is the largest single head effect, while
layer-0 attention and the final MLP are broader shared dependencies. The
circuit is therefore not a one-head trick: early attention builds useful
frontier features, final attention reads them into CLS, and the final MLP
substantially sharpens the readout.

Head-path patching agrees with this. On matched pairs, patching only `L2H0`
at the CLS destination recovers about one third of the clean-corrupt score:

| Patched path | Normalized score recovery |
|---|---:|
| `L2H0` projected head output at CLS | `0.354` |
| `L2H0` `z` vector at CLS | `0.354` |
| `L2H0` clean CLS pattern plus clean boundary values, `r=8` | `0.354` |
| `L2H0` clean boundary values only, `r=8` | `0.330` |
| `L2H1` projected head output at CLS | `0.095` |

This isolates a real path from boundary values through `L2H0` into CLS, but
also shows that the full boundary-band computation is distributed across more
than one component.

## Semantic Decoding

Linear probes on activations show that the late CLS stream linearly represents
the task-level descent information:

| Representation | Probe exact-set accuracy | Probe bit accuracy | Descent-mask accuracy | Final-factor accuracy |
|---|---:|---:|---:|---:|
| `L2` residual post at CLS | `69.73%` | `88.94%` | `69.98%` | `58.58%` |
| Final hidden CLS | `69.54%` | `88.85%` | `69.87%` | `58.46%` |
| `L1` residual post at CLS | `65.11%` | `87.12%` | `67.80%` | `56.49%` |
| `L0` residual post at CLS | `60.79%` | `85.21%` | `63.43%` | `50.48%` |

The probe is close to the model's own exact-set accuracy, meaning the late CLS
state makes the descent-set computation mostly linearly available. The final
factor itself is only partially decoded, which is also informative: the model
is not simply reconstructing the 22-way Garside factor and applying a lookup.
It seems to represent a coarser algebraic summary sufficient for descent
membership.

Raw feature lookup baselines are much weaker:

| Raw feature lookup | Exact accuracy | Bit accuracy |
|---|---:|---:|
| Leading boundary token | `57.81%` | `85.10%` |
| Leading + trailing boundary tokens | `57.73%` | `84.62%` |
| Leading window `r=1` lookup | `53.72%` | `81.85%` |
| Both boundary windows `r=0` with degrees | `49.00%` | `75.94%` |

This matters because it separates the `B_4` result from the `B_3` result. The
model is not just exploiting a tiny exact lookup rule visible from one
extremal coefficient. It is computing a learned boundary-band function.

## Robustness

We trained one full replicate with the same architecture and seed `7`. It
reaches essentially the same held-out performance:

| Model | Exact descent-set accuracy | Bit accuracy | Micro-F1 |
|---|---:|---:|---:|
| Seed 42 | `71.37%` | `89.45%` | `89.82%` |
| Seed 7 | `71.15%` | `89.40%` | `89.79%` |

The boundary-band result replicates almost exactly:

| Input variant | Seed 42 exact | Seed 7 exact |
|---|---:|---:|
| Full matrix | `71.46%` | `71.56%` |
| Boundary only, `r=8` | `70.70%` | `70.92%` |
| Boundary only, `r=5` | `68.38%` | `68.58%` |
| Boundary only, `r=0` | `52.06%` | `52.78%` |
| Drop boundary, `r=8` | `18.96%` | `19.40%` |

Matched patching also replicates:

| Patch into corrupt input | Seed 42 recovery | Seed 7 recovery |
|---|---:|---:|
| Both boundary bands, `r=8` | `0.825` | `0.845` |
| Both boundary bands, `r=5` | `0.804` | `0.814` |
| Leading band only, `r=8` | `0.548` | `0.565` |
| Trailing band only, `r=8` | `0.235` | `0.218` |
| Interior excluding boundary bands, `r=8` | `0.161` | `0.148` |

The exact head identities are not stable, which is the right lesson. The
mechanistic role is stable. In seed 42, the sharp final leading-edge pointer
is `L2H0`; in seed 7, it is `L2H5`.

| Model | Leading-edge final head | Mean CLS attention to leading edge | Matched path recovery |
|---|---:|---:|---:|
| Seed 42 | `L2H0` | `0.9027` | `0.354` |
| Seed 7 | `L2H5` | `0.917` | `0.344` |

Thus the robust claim is not that a particular numbered head matters. The
robust claim is that training produces a final-layer head that points sharply
to the leading Burau frontier, routes boundary-band values into CLS, and
accounts for about one third of the clean-corrupt score gap by itself. The
full model remains a distributed frontier circuit, with early attention and
the final MLP carrying additional shared computation.

## Interpretation

The current `B_4` picture is:

1. The descent set is largely determined, for the trained model, by two
   moderate-width boundary bands of the Burau matrix.
2. The model's decision collapses into the final CLS residual stream.
3. Attention heads specialize to the leading and trailing frontiers of the
   Burau support. Across seeds, a final-layer head consistently emerges as a
   sharp leading-edge pointer, though its numerical head index changes.
4. The final MLP and early attention layers are important shared computation,
   so this is a distributed frontier circuit rather than a single-head
   mechanism.
5. Late CLS activations linearly expose the descent mask nearly as well as the
   model itself, while the full final factor is only partially represented.

This is a more interesting result than the `B_3` case in exactly the way we
wanted. `B_3` gave a clean calibration story: the exact algebraic rule was
simple, and the transformer learned it. `B_4` gives a richer frontier story:
the model appears to discover that the relevant algebraic signal lives near
the leading and trailing degree boundaries, then uses specialized attention
heads to move that information into CLS.

## Remaining Work

The strongest next steps are:

1. Try to turn the boundary-band observation into a mathematical statement:
   characterize which coefficient slices near the extremal degrees determine
   the right descent set for length-25 sampled normal forms.
2. Train a stronger `B_4` model or longer run. If accuracy improves, test
   whether the same boundary circuit sharpens or whether new non-boundary
   mechanisms appear.

## Artifacts

- Model: `interp/artifacts/b4_l25_p2_xfmr3_abs/results.json`
- First-pass probes and ablations:
  `interp/artifacts/b4_l25_p2_firstpass_interp/results.json`
- Matched boundary patching:
  `interp/artifacts/b4_l25_p2_matched_boundary_patching/results.json`
- Targeted boundary-head circuit:
  `interp/artifacts/b4_l25_p2_boundary_head_circuit/results.json`
- Seed-7 robustness model:
  `interp/artifacts/b4_l25_p2_xfmr3_abs_seed7/results.json`
- Seed-7 first-pass probes and ablations:
  `interp/artifacts/b4_l25_p2_firstpass_interp_seed7/results.json`
- Seed-7 matched boundary patching:
  `interp/artifacts/b4_l25_p2_matched_boundary_patching_seed7/results.json`
- Seed-7 targeted boundary-head circuit:
  `interp/artifacts/b4_l25_p2_boundary_head_circuit_seed7/results.json`
