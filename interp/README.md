# Learning Braid Group Representations

This directory contains a mechanistic-interpretability project on braid group
representations. The task is deliberately algebraic: given only the reduced
Burau matrix of a positive braid, predict descent information about the final
Garside factor.

The project has two linked case studies:

```text
B3: exact recovered algorithm.
B4: signed-frontier circuit plus sparse SAE feature code.
```

The goal is not just high accuracy. The goal is to understand what the model
learned, recover the relevant algebraic objects inside the network, and test
the proposed mechanism causally.

## Headline

In `B_3`, the story is clean. The final descent label is exactly determined
by an extremal matrix unit in the Burau matrix over `(Z/2)[v]`. A small
transformer learns to represent that matrix unit in its late `CLS` stream. A
linear decoder recovers the unit, and applying the hand-derived column rule
to the decoded unit gives `99.99%` label accuracy.

In `B_4`, the corresponding one-slice rule no longer survives the prefix
action. But the same frontier idea remains highly informative once signs over
`Z[v]` are retained. A Z-sign transformer reaches `93.2%` exact descent-set
accuracy. A smaller boundary-only transformer, seeing only radius-8 signed
leading/trailing frontier bands, reaches `90.6%` exact accuracy and gives the
cleanest circuit target.

Sparse autoencoders then sharpen the `B_4` story. A few dozen late-`CLS` SAE
features recover almost all of the boundary-only model's descent computation,
beat count-matched random active features, recur across independent seeds, and
are causally fed by layer-1 `CLS` attention paths.

The compact punchline is:

> `B_3` gives an exact algorithm-recovery example. `B_4` shows the harder
> generalization: signed Burau boundary data is read into `CLS`, organized
> into a sparse descent-relevant code, and used to predict the final descent
> set.

## Read This First

1. [RESULTS.md](RESULTS.md) is the metric snapshot.
2. [B3_REPORT.md](B3_REPORT.md) explains the exact recovered algorithm.
3. [B4_REPORT.md](B4_REPORT.md) explains the signed-frontier mechanism.
4. [B4_SAE_FINAL.md](B4_SAE_FINAL.md) explains the sparse-feature result.
5. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) maps claims to scripts, jobs, and
   JSON artifacts.

For the algebraic false starts and negative theorem search, read
[B4_RULE_NOTES.md](B4_RULE_NOTES.md). For the local hook and patching harness,
read [INTERP_HARNESS.md](INTERP_HARNESS.md).

## Figures

The tracked figures are intentionally lightweight and repo-native:

- [figures/project_story.svg](figures/project_story.svg): the whole project in
  one diagram.
- [figures/b3_vs_b4_comparison.svg](figures/b3_vs_b4_comparison.svg): the
  contrast between exact `B_3` recovery and the harder `B_4` sparse-code case.
- [figures/b3_algorithm_recovery.svg](figures/b3_algorithm_recovery.svg):
  exact `B_3` unit-column recovery.
- [figures/b4_signed_frontier.svg](figures/b4_signed_frontier.svg): signed
  `B_4` frontier mechanism.
- [figures/b4_hidden_theorem_gap.svg](figures/b4_hidden_theorem_gap.svg): the
  gap between hand-written finite rules and the learned Z-sign models.
- [figures/b4_sae_circuit.svg](figures/b4_sae_circuit.svg): SAE sparse-code
  circuit.
- [figures/b4_sae_controls.svg](figures/b4_sae_controls.svg): selected SAE
  features versus controls.
- [figures/b4_sae_feature_atlas.svg](figures/b4_sae_feature_atlas.svg): how
  positive, negative, and mixed SAE evidence features combine.
- [figures/reproducibility_map.svg](figures/reproducibility_map.svg): how data,
  jobs, artifacts, and public reports fit together.

## Main Results

### B3: exact algorithm recovery

Task: predict whether the final `B_3` factor has right descent `{s_1}` or
`{s_2}` from the Burau matrix over `(Z/2)[v]`.

| Result | Number |
|---|---:|
| full length-25 corpus | `67,108,864` examples |
| exact extremal-unit rule | `100.00%` accuracy |
| main transformer | `99.81%` test accuracy |
| decoded unit plus exact rule | `99.99%` accuracy |
| matched boundary-token patching | `100.00%` clean-label accuracy |

Interpretation:

```text
Burau boundary matrix unit
    -> late CLS linearly represents the unit
    -> hand-derived column rule
    -> descent label
```

This is the calibration case: the theorem is exact, the model learns the
object used by the theorem, and causal interventions locate the computation in
late `CLS` attention.

### B4: signed frontier

Task: predict the three-bit right descent set of the final `B_4` Garside
factor from the full braid's reduced Burau matrix.

| Input representation | Exact | Bit | Micro-F1 |
|---|---:|---:|---:|
| `(Z/2)[v]` support tokens | `71.4%` | `89.5%` | `89.8%` |
| `Z[v]` sign tokens | `93.2%` | `97.7%` | `97.7%` |
| radius-8 boundary-only Z-sign tokens | `90.6%` | `96.7%` | `96.7%` |

The sign-token result is the key representation shift. Mod-2 support is too
lossy. Signed boundary data over `Z[v]` retains the useful final-factor signal.

Mechanistic evidence:

| Claim | Evidence |
|---|---|
| signs matter | exact accuracy jumps from `71.4%` to `93.2%` |
| boundary matters | dropping radius-8 boundary windows collapses accuracy |
| boundary is causal | prefix-fixed boundary patching recovers `90.2%` score |
| boundary can stand alone | boundary-only model gets `90.6%` exact |
| late `CLS` is semantic | direct descent probe gets about `93.2%` exact |
| exact small rule not found | best finite hand state gets about `85.1%` exact |

### B4: SAE sparse code

The final SAE pass asks whether the boundary-only model's late `CLS` state
decomposes into sparse, causal features.

| Seed | Site | Feature set | k | Exact | Bit | Agreement with transformer | Random exact |
|---|---|---|---:|---:|---:|---:|---:|
| seed 7 | `final_hidden_cls` | binary | `29` | `89.4%` | `96.3%` | `96.0%` | `49.6%` |
| seed 7 | `final_hidden_cls` | descent | `16` | `89.2%` | `96.3%` | `95.7%` | `36.1%` |
| seed 42 | `final_hidden_cls` | descent | `17` | `89.0%` | `96.1%` | `94.4%` | `36.0%` |
| seed 42 | `final_hidden_cls` | binary | `27` | `88.9%` | `96.1%` | `94.4%` | `46.9%` |

The teacher models are about `90.6-90.8%` exact on the same slice. So the
selected sparse features recover nearly all of the model; random active
features do not.

Path patching shows where these features come from:

```text
signed boundary tokens
    -> layer-1 CLS attention paths
    -> sparse late-CLS descent features
    -> output logits
```

Patching all layer-1 `CLS` attention-head outputs recovers up to `81.1%` of
the selected SAE feature activations and `76.2%` of the clean logit score.

## What Is Solved And What Is Not

Solved:

- `B_3` has an exact theorem and an exact recovered model algorithm.
- The repo has a complete `B_3` chain: theorem, model, linear decoder, causal
  patching, and robustness replicate.

Strongly explained:

- `B_4` over signed `Z[v]` is much more learnable than mod-2 support.
- The signed leading/trailing frontier is causally central.
- The boundary-only model has a readable sparse late-`CLS` descent code.
- SAEs recover that code across seeds with random and permutation controls.

Not solved:

- No exact `B_4` theorem is currently known.
- The `B_4` code is distributed, not a single-feature rule.
- The full Z-sign model has a small performance edge over the boundary-only
  model, so it may use some non-boundary information too.

## Reproducibility

Large generated files are intentionally ignored by git:

```text
interp/data/generated/
interp/artifacts/
interp/slurm_logs/
```

The reports copy out the key metrics, and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) gives the script/job/artifact map.

Basic local checks:

```bash
python -m py_compile interp/analyze_b4_sae_final.py \
  interp/run_b4_sae_experiments.py \
  interp/stress_b4_sae_controls.py

bash -n interp/jobs/analyze_b4_sae_final.sh \
  interp/jobs/run_b4_sae_suite.sh \
  interp/jobs/stress_b4_sae_controls.sh
```

Representative Slurm entry points:

```bash
sbatch interp/jobs/train_b3_transformer_l25_p2.sh
sbatch interp/jobs/run_b3_circuit_classifier.sh
sbatch interp/jobs/train_b4_z_sign_transformer.sh
sbatch interp/jobs/train_b4_z_sign_boundary_small.sh
sbatch interp/jobs/run_b4_z_sign_deep_dive.sh
sbatch interp/jobs/search_b4_hidden_theorem.sh
sbatch interp/jobs/run_b4_sae_suite.sh
sbatch interp/jobs/stress_b4_sae_controls.sh
sbatch interp/jobs/analyze_b4_sae_final.sh
```
