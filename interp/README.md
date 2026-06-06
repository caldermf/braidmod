# Learning Braid Group Representations

This directory contains the mechanistic-interpretability project built inside
`braidmod`. The basic task is to predict descent information about the final
Garside factor of a positive braid using only the braid's reduced Burau matrix.

The project has two parts: a solved calibration case in `B_3`, and a richer
signed-frontier case in `B_4`.

## Headline

In `B_3`, the project is an exact algorithm-recovery case study. The descent
label is determined by an extremal matrix unit in the Burau matrix, and a
trained transformer linearly represents that matrix unit in its late `CLS`
stream. Decoding the unit and applying the hand-derived column rule gives
`99.99%` label accuracy.

In `B_4`, the one-slice theorem no longer survives the prefix action, but the
same frontier principle remains useful. Over signed `Z[v]` Burau matrices, a
small transformer reaches `93.2%` exact descent-set accuracy. A constrained
boundary-only transformer, seeing only radius-8 signed leading/trailing
frontier bands, reaches `90.6%` exact accuracy and has a readable layer-1
attention readout circuit.

The point is not that `B_4` is solved. The result is more specific:

> We recover an exact algebraic algorithm inside a transformer in `B_3`, then
> show that the analogous `B_4` computation becomes a signed-frontier circuit:
> highly learnable, causally localized, and partially explained, but not
> reducible to the obvious finite hand rule.

## Main Documents

- `B3_REPORT.md`: final-facing writeup for the exact `B_3` recovery.
- `B4_REPORT.md`: final-facing writeup for the `B_4` signed-frontier case.
- `RESULTS.md`: compact tracked summary of headline metrics.
- `BOUNDARY_RULE.md`: proof sketch and detailed notes for the exact `B_3`
  boundary rule.
- `B4_RULE_NOTES.md`: exploratory algebra notes and the hidden-theorem search.
- `INTERP_HARNESS.md`: map of the local TransformerLens-style hook/patch tools.
- `FINAL_CHECKLIST.md`: final claim stack, completed polish, and remaining
  optional work.
- `PLAN.md`: original project plan.

## Mechanism Diagrams

- `figures/b3_algorithm_recovery.svg`
- `figures/b4_signed_frontier.svg`

These are lightweight tracked diagrams for readers who want the core mechanism
before reading the experiment reports.

## What Is Solved

`B_3` is the solved calibration case.

- Full length-25 corpus audited: `67,108,864` examples.
- Boundary theorem: extremal matrix-unit column determines descent with
  `100%` accuracy.
- Transformer test accuracy: `99.81%`.
- Circuit-derived classifier: decode late-`CLS` matrix unit, apply exact column
  rule, get `99.99%` label accuracy.
- Matched boundary patching: swapping both boundary tokens recovers `100%`
  clean-label accuracy on matched pairs.

This is the result to lead with: the mathematical rule is exact, and the model
internally represents the object used by that rule.

## What Is Partially Explained

`B_4` is the richer case.

- Mod-2 support-token transformer: `71.4%` exact accuracy.
- Signed `Z[v]` sign-token transformer: `93.2%` exact accuracy.
- Boundary-only sign-token transformer: `90.6%` exact accuracy.
- Prefix-fixed boundary patching in the full model: radius-8 signed boundary
  bands recover `90.2%` of the clean-corrupt score difference.
- Late-`CLS` direct descent probe: `93.2%` exact accuracy.
- Final-factor/top-signature probes recover most, but not all, of the model's
  behavior.

The clearest current `B_4` mechanism is:

```text
signed Burau matrix
  -> signed leading/trailing frontier bands
  -> layer-1 attention reads frontier information into CLS
  -> late CLS represents a descent-relevant algebraic state
  -> three descent bits
```

## What Is Not Solved

The exact `B_4` theorem is not currently known. A targeted hidden-theorem
search over signed boundary states, generator quotients, all-simple quotient
signatures, clipped coefficient states, and their combinations did not find an
exact finite rule.

Best hand states in that search:

- signed boundary negative-column masks, radius `3`: `84.0%` exact;
- generator quotient signed-column state, radius `1`: `84.0%` exact;
- best combined state: `85.1%` exact.

The conflict tables show repeated feature keys with multiple descent masks.
This is evidence that the transformer's `93%` solution uses a richer
state than these small hand-designed signatures.

## Reproducibility Notes

Large datasets and model artifacts are intentionally ignored by git:

- `interp/data/generated/`
- `interp/artifacts/`
- `interp/slurm_logs/`

The tracked reports copy the important metrics out of those ignored artifacts.
The scripts are designed for the Roberts cluster and use the PyTorch module in
the Slurm wrappers.

Basic syntax checks:

```bash
python -m py_compile interp/search_b4_hidden_theorem.py \
  interp/run_b4_z_sign_deep_dive.py \
  interp/train_b4_z_sign_transformer.py

bash -n interp/jobs/search_b4_hidden_theorem.sh \
  interp/jobs/run_b4_z_sign_deep_dive.sh \
  interp/jobs/train_b4_z_sign_boundary_small.sh
```

Representative Slurm entry points:

```bash
sbatch interp/jobs/train_b3_transformer_l25_p2.sh
sbatch interp/jobs/run_b3_circuit_classifier.sh
sbatch interp/jobs/train_b4_z_sign_transformer.sh
sbatch interp/jobs/train_b4_z_sign_boundary_small.sh
sbatch interp/jobs/run_b4_z_sign_deep_dive.sh
sbatch interp/jobs/search_b4_hidden_theorem.sh
```

For fast smoke testing, override example counts and artifact names through
environment variables, as the job scripts do internally:

```bash
sbatch --export=ALL,ARTIFACT_NAME=b4_hidden_smoke,TRAIN_EXAMPLES=4096,EVAL_EXAMPLES=1024 \
  interp/jobs/search_b4_hidden_theorem.sh
```

## Reading Order

1. Read `RESULTS.md`.
2. Read `B3_REPORT.md` for the exact recovered algorithm.
3. Read `B4_REPORT.md` for the signed-frontier generalization.
4. Read `B4_RULE_NOTES.md` only if you want the algebraic false starts and
   negative theorem search.
