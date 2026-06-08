# Results Summary

This file is the tracked metric snapshot for the interpretability project. The
large JSON artifacts live under `interp/artifacts/`, which is intentionally
gitignored.

## Public Claim

The repo demonstrates exact algorithm recovery in `B_3` and a strong
signed-frontier sparse-code mechanism in `B_4`.

```text
B3: exact theorem -> trained transformer -> decoded internal algorithm.
B4: no exact small theorem found -> signed frontier is causally central and
    supports a readable high-performing boundary-only transformer.
B4 SAE: selected sparse late-CLS features recover nearly all of the
    boundary-only model and survive controls.
```

## B3: Exact Algorithm Recovery

Task: predict whether the final `B_3` Garside factor has right descent
`{s_1}` or `{s_2}` from the reduced Burau matrix over `(Z/2)[v]`.

Dataset:

| Quantity | Value |
|---|---:|
| Garside length | `25` |
| Full corpus size | `67,108,864` |
| Test examples for main models | `671,088` |

Exact algebraic rule:

| Rule | Accuracy |
|---|---:|
| leading extremal matrix-unit column | `100.00%` |
| trailing extremal matrix-unit column | `100.00%` |

Models:

| Model | Test accuracy |
|---|---:|
| 2-layer transformer, `d_model=128` | `99.81%` |
| 1-hidden-layer MLP, width `128` | `99.95%` |
| smaller seed-7 transformer, `d_model=96` | `98.51%` |

Circuit-derived classifier:

| Classifier | Accuracy vs true labels | Agreement with model |
|---|---:|---:|
| full transformer on same eval set | `99.790%` | `100.000%` |
| true leading boundary rule | `100.000%` | `99.790%` |
| true trailing boundary rule | `100.000%` | `99.790%` |
| decoded leading unit, then exact rule | `99.989%` | `99.791%` |
| decoded trailing unit, then exact rule | `99.989%` | `99.791%` |

Key causal result:

| Intervention | Result |
|---|---:|
| matched-pair input swap of both boundary tokens | `100.00%` clean-label accuracy |
| layer-1 residual stream at CLS patch | `1.000` normalized recovery |
| layer-1 attention output at CLS patch | `0.974` normalized recovery |

Interpretation:

> The model's late CLS residual stream linearly represents the extremal matrix
> unit, and the hand-derived unit-column rule applied to that decoded object
> recovers the algorithm.

## B4: From Mod 2 To Signed Z[v]

Task: predict the three-bit right descent set of the final `B_4` Garside factor
from the full braid's reduced Burau matrix.

Dataset:

| Quantity | Value |
|---|---:|
| Garside length | `25` |
| Random corpus size | `16,777,216` |
| Absolute degree depth | `101` |
| Held-out test examples for main models | `167,772` |

Main model comparison:

| Model input | Exact set accuracy | Bit accuracy | Micro-F1 |
|---|---:|---:|---:|
| `(Z/2)[v]` support tokens | `71.37%` | `89.45%` | `89.82%` |
| `Z[v]` sign tokens | `93.25%` | `97.69%` | `97.67%` |
| `Z[v]` sign-token boundary-only radius `8` | `90.59%` | `96.74%` | `96.70%` |

This is the main representation result: signs over `Z[v]` are essential.

## B4: Full Z-Sign Mechanistic Evidence

Deep-dive artifact:

```text
interp/artifacts/b4_l25_zsign_deep_dive/results.json
```

Analysis split:

| Metric | Value |
|---|---:|
| exact set accuracy | `93.59%` |
| bit accuracy | `97.81%` |
| micro-F1 | `97.78%` |

Late CLS probes:

| Probe from final hidden CLS | Exact | Bit | Agreement with model |
|---|---:|---:|---:|
| direct descent-bit probe | `93.19%` | `97.68%` | `97.27%` |
| final-factor-style latent then rule | `90.92%` | `96.73%` | `93.84%` |
| left/right descent latent then rule | `91.20%` | `96.85%` | `94.13%` |
| top-two-column latent then rule | `91.09%` | `96.79%` | `93.98%` |

Prefix-fixed counterfactual patching:

| Patched clean input region | Score recovery | Clean-label exact | Clean-label bit |
|---|---:|---:|---:|
| both boundary bands, radius `8` | `90.2%` | `82.6%` | `93.7%` |
| both boundary bands, radius `5` | `86.3%` | `75.6%` | `90.6%` |
| both boundary bands, radius `3` | `80.4%` | `70.1%` | `88.5%` |
| interior excluding boundary bands, radius `8` | `7.0%` | `2.5%` | `41.7%` |

Top full-model ablations on the analysis split:

| Intervention | Exact drop |
|---|---:|
| zero layer-0 attention | `51.3%` |
| zero layer-0 MLP | `20.9%` |
| zero layer-0 head 2 at CLS | `5.2%` |
| zero layer-2 MLP | `3.8%` |
| zero layer-1 attention | `3.5%` |

Interpretation:

> The full Z-sign model uses broad signed information, but prefix-fixed
> counterfactuals show that the signed boundary bands causally carry most of
> the final-factor signal.

## B4: Boundary-Only Circuit

Boundary-only model artifact:

```text
interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small/results.json
```

Boundary-only deep-dive artifact:

```text
interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small_deep_dive/results.json
```

Boundary-only seed-7 robustness artifacts:

```text
interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small_seed7/results.json
interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small_seed7_deep_dive/results.json
```

Architecture:

| Hyperparameter | Value |
|---|---:|
| layers | `2` |
| heads per layer | `4` |
| `d_model` | `96` |
| visible input | radius-8 signed boundary bands only |

Analysis split:

| Metric | Value |
|---|---:|
| exact set accuracy | `91.14%` |
| bit accuracy | `96.94%` |
| micro-F1 | `96.89%` |

Late CLS probes:

| Probe from final hidden CLS | Exact | Bit | Agreement with model |
|---|---:|---:|---:|
| direct descent-bit probe | `90.11%` | `96.55%` | `96.00%` |
| right-descent latent then rule | `88.92%` | `95.96%` | `93.38%` |
| left/right descent latent then rule | `86.79%` | `95.26%` | `90.15%` |
| top-two-column latent then rule | `86.72%` | `95.24%` | `90.39%` |

Top ablations:

| Intervention | Exact accuracy after ablation | Exact drop |
|---|---:|---:|
| full boundary-only model | `91.14%` | `0.0%` |
| zero layer-1 attention | `48.41%` | `42.72%` |
| zero layer-0 MLP | `48.79%` | `42.35%` |
| zero layer-0 attention | `75.66%` | `15.48%` |
| zero layer-1 MLP | `81.15%` | `9.99%` |
| zero `L1H2` at CLS | `80.57%` | `10.57%` |
| zero `L1H1` at CLS | `81.26%` | `9.88%` |

Prefix-fixed patching:

| Patched clean input region | Score recovery | Clean-label exact | Clean-label bit |
|---|---:|---:|---:|
| both boundary bands, radius `8` | `99.9%` | `89.5%` | `96.4%` |
| both boundary bands, radius `5` | `97.6%` | `88.3%` | `95.8%` |
| both boundary bands, radius `3` | `93.2%` | `84.2%` | `94.4%` |
| interior excluding boundary bands, radius `8` | `0.3%` | `1.6%` | `39.4%` |

Interpretation:

> The boundary-only model is the clearest `B_4` circuit target: layer-1
> attention heads read signed frontier information into `CLS`, and the late `CLS`
> stream exposes the descent decision almost linearly.

Robustness replicate:

| Model | Exact | Bit | Micro-F1 |
|---|---:|---:|---:|
| boundary-only seed 42, held-out test | `90.59%` | `96.74%` | `96.70%` |
| boundary-only seed 7, held-out test | `90.36%` | `96.66%` | `96.61%` |
| boundary-only seed 7, deep-dive split | `89.89%` | `96.51%` | `96.46%` |

Seed-7 late CLS probes:

| Probe from final hidden CLS | Exact | Bit | Agreement with model |
|---|---:|---:|---:|
| direct descent-bit probe | `88.68%` | `96.13%` | `95.61%` |
| right-descent latent then rule | `87.56%` | `95.54%` | `93.18%` |

Seed-7 prefix-fixed patching:

| Patched clean input region | Score recovery | Clean-label exact | Clean-label bit |
|---|---:|---:|---:|
| both boundary bands, radius `8` | `99.7%` | `91.6%` | `97.1%` |
| both boundary bands, radius `5` | `96.3%` | `89.1%` | `96.3%` |
| both boundary bands, radius `3` | `90.4%` | `83.6%` | `94.0%` |
| interior excluding boundary bands, radius `8` | `0.1%` | `2.0%` | `39.3%` |

Seed-7 top ablations:

| Intervention | Exact accuracy after ablation | Exact drop |
|---|---:|---:|
| full seed-7 boundary-only model | `89.89%` | `0.0%` |
| zero layer-0 MLP | `56.18%` | `33.72%` |
| zero layer-1 attention | `64.50%` | `25.39%` |
| zero layer-0 attention | `64.55%` | `25.34%` |
| zero layer-1 MLP | `78.85%` | `11.05%` |
| zero `L1H0` at CLS | `81.51%` | `8.39%` |

This replicate supports the main claim while also showing that component
identities are not the invariant. The stable object is the boundary-only
frontier computation and its late-CLS readout, not a specific numbered head.

## B4: Sparse Autoencoder Feature Code

Final SAE artifact:

```text
interp/artifacts/b4_l25_zsign_boundary_r8_sae_final/results.json
interp/artifacts/b4_l25_zsign_boundary_r8_sae_final/SUMMARY.md
```

Final SAE report:

```text
interp/B4_SAE_FINAL.md
```

The SAE pass asks whether the boundary-only model's late `CLS` descent
computation decomposes into sparse, causal features. It uses the seed-42 and
seed-7 boundary-only models, `8,192` held-out examples, `32,768` feature-probe
train examples, and `512` prefix-fixed counterfactual pairs.

Selected sparse-feature classifiers:

| Seed | Site | Feature set | k | Exact | Bit | Agreement with transformer | Random exact |
|---|---|---|---:|---:|---:|---:|---:|
| seed 7 | `final_hidden_cls` | binary | `29` | `89.4%` | `96.3%` | `96.0%` | `49.6%` |
| seed 7 | `final_hidden_cls` | descent | `16` | `89.2%` | `96.3%` | `95.7%` | `36.1%` |
| seed 42 | `final_hidden_cls` | descent | `17` | `89.0%` | `96.1%` | `94.4%` | `36.0%` |
| seed 42 | `final_hidden_cls` | binary | `27` | `88.9%` | `96.1%` | `94.4%` | `46.9%` |
| seed 7 | `l1_resid_post_cls` | binary | `32` | `88.0%` | `95.7%` | `92.9%` | `52.8%` |
| seed 42 | `l1_resid_post_cls` | binary | `31` | `87.0%` | `95.4%` | `91.4%` | `52.6%` |

Cross-seed recurrence:

| Site | Matched seed-42 features | Mean best corr | Best match also selected in seed 7 |
|---|---:|---:|---:|
| `final_hidden_cls` | `48` | `0.649` | `37` |
| `l1_resid_post_cls` | `48` | `0.613` | `37` |
| `l1_attn_out_cls` | `53` | `0.552` | `36` |

Path patching into selected sparse features:

| Seed | Target site | Selected features | Layer-1 CLS-head feature recovery | Logit recovery |
|---|---|---:|---:|---:|
| seed 42 | `l1_resid_post_cls` | `31` | `81.1%` | `76.2%` |
| seed 42 | `final_hidden_cls` | `27` | `76.6%` | `76.2%` |
| seed 7 | `final_hidden_cls` | `29` | `69.0%` | `64.6%` |
| seed 7 | `l1_resid_post_cls` | `32` | `65.1%` | `64.6%` |

Interpretation:

> The boundary-only model's late `CLS` state contains a sparse distributed
> descent code. A few dozen selected SAE features recover nearly all of the
> model, random active features do not, the same feature families recur across
> seeds, and layer-1 `CLS` attention causally feeds the code.

## B4: Hidden-Theorem Search

Artifact:

```text
interp/artifacts/b4_l25_z_hidden_theorem_combo/results.json
```

The search used `131,072` train examples and `32,768` held-out examples. It
tested signed boundary states, generator quotient states, all-simple quotient
states, clipped coefficient states, and pair/triple combinations.

Naive quotient positivity fails in reduced Burau:

| Direct quotient predicate | Exact set accuracy | Bit accuracy |
|---|---:|---:|
| no negative Laurent terms in `rho(beta) rho(s_i)^-1` | `0.0%` | `50.0%` |
| first quotient exponent nonnegative | `0.0%` | `50.0%` |

Best hand states:

| Candidate state | Exact | Bit |
|---|---:|---:|
| signed boundary negative-column masks, radius `3` | `84.0%` | `94.1%` |
| generator quotient signed-column state, radius `1` | `84.0%` | `94.0%` |
| all-simple quotient width-delta state | `78.3%` | `92.5%` |
| best combined state | `85.1%` | `94.2%` |

Interpretation:

> The obvious finite signed-frontier theorem was not found. The best hand
> states have genuine high-count collisions, so the transformer's `93%` solution
> likely uses a richer distributed state than these small signatures.

## Final Framing

The repo should make this distinction clear:

- `B_3`: exact algorithm recovered inside a transformer.
- `B_4`: signed-frontier mechanism found, causally localized, and bounded by a
  serious negative theorem search.
- `B_4` SAE: the boundary-only model's late state decomposes into a sparse
  distributed descent code that is causally testable and seed-robust.

That combination is the result.
