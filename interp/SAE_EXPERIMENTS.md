# B4 SAE Experiments

This is the sparse-autoencoder pass on the `B_4` signed-frontier transformer.
The goal is to move from "the boundary circuit is localized" to "the boundary
circuit decomposes into sparse algebraic features."

## Experiment

The runner is:

```text
interp/run_b4_sae_experiments.py
```

The Slurm wrapper is:

```text
interp/jobs/run_b4_sae_suite.sh
```

For each activation site, the script:

1. trains a TopK SAE;
2. checks reconstruction quality and whether SAE reconstruction preserves model
   behavior;
3. labels sparse features against algebraic variables such as descent bits,
   final factor, final-simple top signatures, and signed boundary masks;
4. ablates and keeps labeled feature sets through the SAE reconstruction path;
5. runs prefix-fixed clean-to-corrupt feature patching.

## Submitted Jobs

Smoke job:

```text
13902561
```

Full seed-42 boundary-only model:

```text
13902564
```

Full seed-7 boundary-only model:

```text
13902565
```

The two full jobs depend on the smoke job with `afterok:13902561`.

Those first full jobs completed successfully but only ran `l1_resid_post_cls`,
because Slurm split the comma-separated `SITES` value in `--export`.  The runner
now accepts plus-separated site lists, and the corrected expanded jobs are:

```text
13904530  seed 42, six-site run
13904531  seed 7, six-site run
```

## Artifacts

Smoke:

```text
interp/artifacts/b4_l25_zsign_boundary_r8_sae_smoke/results.json
```

Seed 42:

```text
interp/artifacts/b4_l25_zsign_boundary_r8_sae_seed42/results.json
interp/artifacts/b4_l25_zsign_boundary_r8_sae_seed42_v2/results.json
```

Seed 7:

```text
interp/artifacts/b4_l25_zsign_boundary_r8_sae_seed7/results.json
interp/artifacts/b4_l25_zsign_boundary_r8_sae_seed7_v2/results.json
```

Each output directory also stores one SAE checkpoint per activation site.

Final post-training analysis:

```text
interp/artifacts/b4_l25_zsign_boundary_r8_sae_final/results.json
interp/artifacts/b4_l25_zsign_boundary_r8_sae_final/SUMMARY.md
interp/B4_SAE_FINAL.md
```

Runner:

```text
interp/analyze_b4_sae_final.py
interp/jobs/analyze_b4_sae_final.sh
```

## Sites

The full jobs train SAEs on:

```text
l1_resid_post_cls
final_hidden_cls
l1_attn_out_cls
l0_mlp_out_cls
l0_mlp_out_leading
l0_mlp_out_trailing
```

These are the main `CLS` sites where the boundary-only model's descent
information is available or causally important.

## What Success Looks Like

Minimum useful result:

- SAE reconstruction preserves most model accuracy.
- Several high-activation sparse features have clear algebraic labels.
- Ablating those labeled features moves the relevant descent logits.

Strong result:

- A small set of labeled SAE features explains a large fraction of the
  prefix-fixed clean-corrupt score difference.
- The same feature families appear in both seed 42 and seed 7, even if feature
  indices differ.

Best result:

- The boundary-only model decomposes into a sparse signed-frontier evidence
  circuit: layer-0/attention features build local signed-boundary evidence,
  late `CLS` features represent generator-specific descent evidence, and
  feature patching causally swaps the model's predicted descent set.

## Final Analysis

The final SAE pass uses the trained seed-42 and seed-7 boundary-only Z-sign
models. It does not train new SAEs. It evaluates the existing SAEs with:

- `8,192` held-out examples;
- `32,768` train examples for sparse-feature classifiers;
- `512` prefix-fixed counterfactual pairs;
- random active-feature classifier controls;
- clean-row permutation controls from `stress_controls.json`;
- cross-seed activation-correlation matching;
- head-to-feature path patching.

Main result:

> The SAE story is not just reconstruction. A few dozen selected late-CLS
> features almost recover the boundary-only model, beat count-matched random
> active features by a large margin, recur across seeds, and are causally fed
> by layer-1 CLS attention paths.

Sparse-feature classifier results:

| Seed | Site | Feature set | k | Exact | Bit | Agreement with transformer | Random exact |
|---|---|---|---:|---:|---:|---:|---:|
| seed 7 | `final_hidden_cls` | binary | `29` | `0.894` | `0.963` | `0.960` | `0.496` |
| seed 7 | `final_hidden_cls` | descent | `16` | `0.892` | `0.963` | `0.957` | `0.361` |
| seed 42 | `final_hidden_cls` | descent | `17` | `0.890` | `0.961` | `0.944` | `0.360` |
| seed 42 | `final_hidden_cls` | binary | `27` | `0.889` | `0.961` | `0.944` | `0.469` |
| seed 7 | `l1_resid_post_cls` | binary | `32` | `0.880` | `0.957` | `0.929` | `0.528` |
| seed 42 | `l1_resid_post_cls` | binary | `31` | `0.870` | `0.954` | `0.914` | `0.526` |

Cross-seed recurrence:

| Site | Matched seed-42 features | Mean best corr | Best match also selected in seed 7 |
|---|---:|---:|---:|
| `final_hidden_cls` | `48` | `0.649` | `37` |
| `l1_resid_post_cls` | `48` | `0.613` | `37` |
| `l1_attn_out_cls` | `53` | `0.552` | `36` |

Path patching:

| Seed | Target site | Selected features | Layer-1 CLS-head feature recovery | Logit recovery |
|---|---|---:|---:|---:|
| seed 42 | `l1_resid_post_cls` | `31` | `0.811` | `0.762` |
| seed 42 | `final_hidden_cls` | `27` | `0.766` | `0.762` |
| seed 7 | `final_hidden_cls` | `29` | `0.690` | `0.646` |
| seed 7 | `l1_resid_post_cls` | `32` | `0.651` | `0.646` |

The honest caveat is that this is a distributed sparse code, not a single
feature theorem. Individual features are crisp but small; feature families are
the causal unit.
