# Learning Braid Group Representations: the `B_3` Case

This report summarizes the first complete case study for the braid
representation project. The task is deliberately narrow: given only the
reduced Burau matrix of a positive `B_3` braid over `(Z/2)[v]`, predict whether
the final Garside factor has descent set `{s_1}` or `{s_2}`.

The main result is that this is an algorithm-recovery example, not just a
high-accuracy classifier. In `B_3`, the descent label is exactly recoverable
from a boundary coefficient of the Burau matrix. A small transformer learns an
internal representation very close to that algebraic object: a linear probe
decodes the extremal matrix unit from the model's late `CLS` residual stream,
and the exact unit-column rule applied to that decoded unit gives `99.99%`
label accuracy.

## Dataset and Exact Rule

We generated the full length-25 `B_3` positive Garside-normal-form corpus:
`67,108,864` braids, each represented by its reduced Burau matrix over
`(Z/2)[v]`. The matrix is stored as degree-indexed `2 x 2` coefficient slices.

For every length-25 example, the minimum occupied degree and maximum occupied
degree each contain a single nonzero matrix entry, i.e. a matrix unit. The
descent label is determined exactly by either extremal slice:

| Boundary slice | Exact rule for label `{s_2}` | Full length-25 accuracy |
|---|---:|---:|
| minimum degree | unique nonzero entry is in column `0` | `100.00%` |
| maximum degree | unique nonzero entry is in column `1` | `100.00%` |

This was also audited exhaustively for lengths `1` through `19`, and on a
`1,048,576` example sample at length `20`, again with `100.00%` accuracy.

The reason is simple. The leading and trailing Burau coefficients of each
proper simple factor are matrix units. In `B_3` normal form, the allowed factor
transitions make the product of extremal matrix units nonzero, so the column of
the extremal unit at the end of the product is inherited from the final factor.
That column is exactly the descent label.

## Models

The main model is a 2-layer, 4-head transformer over absolute degree tokens.
It reaches `99.81%` test accuracy on held-out length-25 examples. A one-hidden
layer MLP reaches `99.95%`, which is a useful reminder that this task is not
architecturally hard once the boundary rule is known. We still use the
transformer as the main interpretability target because its attention structure
gives a circuit we can inspect.

| Model | Test examples | Test accuracy |
|---|---:|---:|
| 2-layer transformer, `d_model=128` | `671,088` | `99.81%` |
| 1-hidden-layer MLP, width `128` | `671,088` | `99.95%` |
| smaller seed-7 transformer, `d_model=96` | `262,144` | `98.51%` |

The smaller seed-7 transformer is a robustness check. With less capacity and a
different seed, the same kind of late `CLS` representation appears.

## Mechanistic Result

A direct semantic probe on the main transformer's late `CLS` residual stream
shows that the model represents the mathematical object, not only the
binary label:

| Representation | Label / column accuracy | Four-way unit-token accuracy |
|---|---:|---:|
| initial CLS embedding | `50.34%` | `25.18%` |
| layer-0 head 2 output at CLS | `78.32%` | `60.53%` |
| layer-1 head 0 value stream at CLS | `90.61%` | `88.60%` |
| layer-1 head 1 output at CLS | `94.63%` | `91.83%` |
| layer-1 residual stream at CLS | `99.98%` | `99.95%` |

The stronger test is to compose the decoded representation with the algebraic
rule. We trained a ridge probe to decode the four possible extremal matrix
units from `blocks.1.hook_resid_post[:, 0]`, then applied the exact
unit-column rule by hand.

| Classifier | Accuracy vs true labels | Agreement with model predictions |
|---|---:|---:|
| full transformer on same eval set | `99.790%` | `100.000%` |
| true leading boundary rule | `100.000%` | `99.790%` |
| true trailing boundary rule | `100.000%` | `99.790%` |
| decoded leading unit, then rule | `99.989%` | `99.791%` |
| decoded trailing unit, then rule | `99.989%` | `99.791%` |

This is the central mechanistic result. A simple linear decoder recovers the
extremal matrix unit from the model, and the hand-written algebraic rule
applied to that decoded object almost exactly recovers the correct labels. The
decoded rule is slightly more accurate than the transformer head itself on this
evaluation set, which is consistent with the representation being cleaner than
the final readout.

## Circuit Evidence

Matched-support interventions make the causal story sharper. On 256
opposite-label matched pairs with the same support interval, swapping only the
two boundary tokens into the corrupt input recovers `100.00%` clean-label
accuracy. Swapping only the leading boundary gives `63.87%`; swapping only the
trailing boundary gives `70.70%`.

Activation patching localizes the decision to the late CLS stream:

| Patched activation on matched pairs | Normalized recovery |
|---|---:|
| layer-0 residual, both boundary tokens | `0.745` |
| layer-0 residual, CLS plus both boundaries | `0.891` |
| layer-1 attention output at CLS | `0.974` |
| layer-1 MLP output at CLS | `0.547` |
| layer-1 residual stream at CLS | `1.000` |

Head-level results point to a compact attention circuit rather than an MLP
memorization story:

| Intervention on `131,072` test examples | Accuracy | Mean logit score |
|---|---:|---:|
| full transformer | `99.790%` | `11.472` |
| zero layer-1 attention | `83.145%` | `5.462` |
| zero layer-0 attention | `88.476%` | `8.570` |
| zero layer-0 MLP | `99.839%` | `11.455` |
| zero layer-1 MLP | `99.872%` | `8.795` |
| keep layer-1 heads `{0,1,2}`, zero both MLPs | `99.779%` | `8.839` |
| keep layer-0 head `2` and layer-1 heads `{0,1,2}`, zero both MLPs | `96.608%` | `8.822` |
| keep layer-0 head `2` and layer-1 heads `{0,1,2}` | `95.693%` | `10.777` |
| drop layer-1 head `0` | `93.279%` | `9.713` |
| drop layer-1 heads `{0,1}` | `86.192%` | `6.927` |

The exact head identities should not be overclaimed, but the broad mechanism
is stable: attention reads boundary-derived information into `CLS`, while the
MLPs mostly polish or rescale the readout. Layer-1 attention is the decisive
readout path.

## Robustness

We trained one constrained replicate: a smaller 2-layer transformer with
`d_model=96`, 4 heads, and seed `7`. Its best checkpoint reached `98.55%`
validation accuracy and `98.51%` held-out test accuracy. The same late-CLS
semantic structure appears:

| Seed-7 smaller model representation | Label / column accuracy | Four-way unit-token accuracy |
|---|---:|---:|
| initial CLS embedding | `50.34%` | `25.18%` |
| layer-0 head 2 output at CLS | `77.42%` | `63.17%` |
| layer-1 head 0 output at CLS | `93.46%` | `81.65%` |
| layer-1 residual stream at CLS | `99.80%` | `99.38%` |

This is not a full seed sweep, but it gives a useful guardrail against the
concern that the main result depends on a single accidental representation.

## Interpretation

The `B_3` case is now a compact end-to-end example of mechanistic
interpretability on a nontrivial algebraic task:

1. The mathematical algorithm is known and exact: read the extremal Burau
   matrix unit and apply a column rule.
2. A transformer trained only on matrix tokens learns to solve the task.
3. The model's late CLS stream linearly represents the extremal matrix unit.
4. Composing that decoded representation with the exact algebraic rule
   reconstructs the task algorithm with `99.99%` accuracy.
5. Causal patching and ablations identify attention, especially late CLS
   attention, as the core readout mechanism.

The caveat is part of the value of the example: `B_3` over `(Z/2)[v]` is a
controlled setting. Once the boundary-unit theorem is visible, the task is
simple. That makes it a useful calibration case. The value of the `B_3` result
is that it gives a complete template: find the exact algebra, train the model,
decode the algebraic object inside the network, and test the causal path.

## Artifacts

- Project overview: `interp/README.md`
- Tracked metrics: `interp/RESULTS.md`
- Mechanism diagram: `interp/figures/b3_algorithm_recovery.svg`
- Exact boundary rule: `interp/artifacts/b3_l25_p2_boundary_rule/results.json`
- Length audit: `interp/artifacts/b3_boundary_rule_lengths/results.json`
- Main transformer: `interp/artifacts/b3_l25_p2_xfmr2_abs/results.json`
- Circuit-derived classifier: `interp/artifacts/b3_l25_p2_circuit_classifier/results.json`
- Circuit sufficiency table: `interp/artifacts/b3_l25_p2_circuit_sufficiency/results.json`
- Matched-support patching: `interp/artifacts/b3_l25_p2_matched_boundary_patching/results.json`
- Semantic probes: `interp/artifacts/b3_l25_p2_semantic_probes/results.json`
- Seed-7 robustness replicate: `interp/artifacts/b3_l25_p2_xfmr2_abs_seed7_small/`
