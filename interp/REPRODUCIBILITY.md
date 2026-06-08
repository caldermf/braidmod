# Reproducibility And Claim Audit

This page maps the public claims in the `interp/` project to the scripts,
Slurm entry points, and JSON artifacts that support them.

Large generated files are intentionally not tracked by git:

```text
interp/data/generated/
interp/artifacts/
interp/slurm_logs/
```

The tracked Markdown reports copy the important metrics out of those ignored
artifacts. The scripts and job wrappers are tracked.

The public workflow figure is:

```text
interp/figures/reproducibility_map.svg
```

## Environment

The project was run on the Roberts cluster. The Slurm wrappers load the local
PyTorch module:

```bash
module load PyTorch/2.7.1-foss-2024a-CUDA-12.8.0
```

Several scripts also work in the `burau_gpu` conda environment:

```bash
conda activate burau_gpu
```

The Slurm wrappers are the preferred way to rerun expensive jobs. Do not run
training or large inference on the login node.

## Fast Checks

These checks do not regenerate the datasets or models. They verify that the
tracked Python and Slurm entry points parse.

```bash
python -m py_compile \
  interp/generate_b3_dataset.py \
  interp/train_b3_transformer.py \
  interp/run_b3_circuit_classifier.py \
  interp/generate_b4_dataset.py \
  interp/train_b4_z_sign_transformer.py \
  interp/run_b4_z_sign_deep_dive.py \
  interp/search_b4_hidden_theorem.py \
  interp/run_b4_sae_experiments.py \
  interp/stress_b4_sae_controls.py \
  interp/analyze_b4_sae_final.py
```

```bash
bash -n \
  interp/jobs/generate_b3_l25_p2_full.sh \
  interp/jobs/train_b3_transformer_l25_p2.sh \
  interp/jobs/run_b3_circuit_classifier.sh \
  interp/jobs/generate_b4_l25_p2_n16777216.sh \
  interp/jobs/train_b4_z_sign_transformer.sh \
  interp/jobs/train_b4_z_sign_boundary_small.sh \
  interp/jobs/run_b4_z_sign_deep_dive.sh \
  interp/jobs/search_b4_hidden_theorem.sh \
  interp/jobs/run_b4_sae_suite.sh \
  interp/jobs/stress_b4_sae_controls.sh \
  interp/jobs/analyze_b4_sae_final.sh
```

## Claim Audit

| Claim | Main report | Script or job | Artifact |
|---|---|---|---|
| `B_3` length-25 boundary rule is exact | `B3_REPORT.md` | `interp/analyze_b3_boundary_rule.py` | `interp/artifacts/b3_l25_p2_boundary_rule/results.json` |
| `B_3` boundary rule holds across shorter lengths | `B3_REPORT.md` | `interp/audit_b3_boundary_rule_lengths.py` | `interp/artifacts/b3_boundary_rule_lengths/results.json` |
| `B_3` transformer reaches `99.81%` | `RESULTS.md` | `interp/jobs/train_b3_transformer_l25_p2.sh` | `interp/artifacts/b3_l25_p2_xfmr2_abs/results.json` |
| decoded `B_3` unit plus exact rule reaches `99.99%` | `B3_REPORT.md` | `interp/jobs/run_b3_circuit_classifier.sh` | `interp/artifacts/b3_l25_p2_circuit_classifier/results.json` |
| `B_3` boundary patching recovers clean labels | `B3_REPORT.md` | `interp/jobs/run_b3_matched_boundary_patching.sh` | `interp/artifacts/b3_l25_p2_matched_boundary_patching/results.json` |
| mod-2 `B_4` model reaches `71.4%` | `B4_REPORT.md` | `interp/jobs/train_b4_transformer_l25_p2.sh` | `interp/artifacts/b4_l25_p2_xfmr3_abs/results.json` |
| Z-sign `B_4` model reaches `93.2%` | `B4_REPORT.md` | `interp/jobs/train_b4_z_sign_transformer.sh` | `interp/artifacts/b4_l25_zsign_xfmr3_abs/results.json` |
| signed boundary bands are causally central | `B4_REPORT.md` | `interp/jobs/run_b4_z_sign_deep_dive.sh` | `interp/artifacts/b4_l25_zsign_deep_dive/results.json` |
| boundary-only Z-sign model reaches `90.6%` | `B4_REPORT.md` | `interp/jobs/train_b4_z_sign_boundary_small.sh` | `interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small/results.json` |
| boundary-only seed-7 replicate is robust | `B4_REPORT.md` | `interp/jobs/train_b4_z_sign_boundary_small.sh` | `interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small_seed7/results.json` |
| small finite `B_4` theorem search does not find an exact rule | `B4_RULE_NOTES.md` | `interp/jobs/search_b4_hidden_theorem.sh` | `interp/artifacts/b4_l25_z_hidden_theorem_combo/results.json` |
| SAEs reconstruct and label late boundary-model features | `B4_SAE_FINAL.md` | `interp/jobs/run_b4_sae_suite.sh` | `interp/artifacts/b4_l25_zsign_boundary_r8_sae_seed42_v2/results.json` |
| selected SAE features beat random and permutation controls | `B4_SAE_FINAL.md` | `interp/jobs/stress_b4_sae_controls.sh` | `interp/artifacts/b4_l25_zsign_boundary_r8_sae_seed42_v2/stress_controls.json` |
| selected SAE features recover nearly all of the boundary model | `B4_SAE_FINAL.md` | `interp/jobs/analyze_b4_sae_final.sh` | `interp/artifacts/b4_l25_zsign_boundary_r8_sae_final/results.json` |
| layer-1 `CLS` attention feeds the selected SAE features | `B4_SAE_FINAL.md` | `interp/jobs/analyze_b4_sae_final.sh` | `interp/artifacts/b4_l25_zsign_boundary_r8_sae_final/results.json` |

## Reproduction Order

The full end-to-end rerun is expensive. This is the logical order.

### B3

Generate and audit the dataset:

```bash
sbatch interp/jobs/generate_b3_l25_p2_full.sh
sbatch interp/jobs/audit_b3_l25_p2_full.sh
sbatch interp/jobs/audit_b3_boundary_rule_lengths.sh
```

Train and interpret the transformer:

```bash
sbatch interp/jobs/train_b3_transformer_l25_p2.sh
sbatch interp/jobs/run_b3_boundary_counterfactuals.sh
sbatch interp/jobs/run_b3_matched_boundary_patching.sh
sbatch interp/jobs/run_b3_semantic_probes.sh
sbatch interp/jobs/run_b3_circuit_classifier.sh
sbatch interp/jobs/run_b3_circuit_sufficiency.sh
sbatch interp/jobs/run_b3_head_circuit.sh
```

Optional baseline:

```bash
sbatch interp/jobs/train_b3_mlp_l25_p2.sh
```

### B4 Dataset And Models

Generate and audit the length-25 `B_4` dataset:

```bash
sbatch interp/jobs/generate_b4_l25_p2_n16777216.sh
sbatch interp/jobs/audit_b4_l25_p2_n16777216.sh
```

Train the mod-2 and Z-sign models:

```bash
sbatch interp/jobs/train_b4_transformer_l25_p2.sh
sbatch interp/jobs/train_b4_z_sign_transformer.sh
sbatch interp/jobs/train_b4_z_sign_boundary_small.sh
```

Run the main mechanistic analyses:

```bash
sbatch interp/jobs/run_b4_firstpass_interp.sh
sbatch interp/jobs/run_b4_matched_boundary_patching.sh
sbatch interp/jobs/run_b4_z_sign_firstpass_interp.sh
sbatch interp/jobs/run_b4_z_sign_deep_dive.sh
```

The boundary-only deep dives use the same
`interp/jobs/run_b4_z_sign_deep_dive.sh` wrapper with checkpoint and output
overrides for the boundary-only model.

### B4 Algebraic Rule Search

Run the integer-boundary and hidden-theorem audits:

```bash
sbatch interp/jobs/audit_b4_integer_boundary.sh
sbatch interp/jobs/search_b4_hidden_theorem.sh
```

### B4 SAE Analysis

Train the SAEs:

```bash
sbatch interp/jobs/run_b4_sae_suite.sh
```

Run skeptical controls:

```bash
sbatch interp/jobs/stress_b4_sae_controls.sh
```

Run the final post-training analysis:

```bash
sbatch interp/jobs/analyze_b4_sae_final.sh
```

## Smoke Runs

Most Slurm wrappers support environment-variable overrides. A typical smoke
run reduces example counts and writes to a separate output directory:

```bash
sbatch --partition=gpu_devel --time=00:30:00 \
  --export=ALL,OUT_DIR=$PWD/interp/artifacts/b4_sae_final_smoke,EVAL_EXAMPLES=512,TRAIN_EXAMPLES=1024,PREFIX_PAIRS=64,RANDOM_CLASSIFIER_TRIALS=2,CLASSIFIER_STEPS=50,MAX_ATLAS_FEATURES=8 \
  interp/jobs/analyze_b4_sae_final.sh
```

For rule-search smoke tests:

```bash
sbatch --export=ALL,ARTIFACT_NAME=b4_hidden_smoke,TRAIN_EXAMPLES=4096,EVAL_EXAMPLES=1024 \
  interp/jobs/search_b4_hidden_theorem.sh
```

## Expected Final Artifacts

The public docs expect these paths to exist after the full project run:

```text
interp/artifacts/b3_l25_p2_boundary_rule/results.json
interp/artifacts/b3_l25_p2_xfmr2_abs/results.json
interp/artifacts/b3_l25_p2_circuit_classifier/results.json
interp/artifacts/b4_l25_p2_xfmr3_abs/results.json
interp/artifacts/b4_l25_zsign_xfmr3_abs/results.json
interp/artifacts/b4_l25_zsign_deep_dive/results.json
interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small/results.json
interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small_seed7/results.json
interp/artifacts/b4_l25_z_hidden_theorem_combo/results.json
interp/artifacts/b4_l25_zsign_boundary_r8_sae_final/results.json
```

## Notes For Reviewers

The most important thing to check is not a single metric. It is the chain of
evidence:

```text
mathematical task
    -> trained model
    -> internal representation
    -> causal intervention
    -> control comparison
    -> robustness replicate
```

For `B_3`, the chain ends in an exact theorem and a decoded internal algorithm.
For `B_4`, it ends in a signed-frontier mechanism and a sparse distributed
feature code, with an explicit negative result for small hand-written finite
rules.
