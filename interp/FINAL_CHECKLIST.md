# Final Project Checklist

This is the repo-facing checklist for the `interp/` project. It records what
has been wrapped into the public research narrative and what remains optional.

## Done

- Wrote a public project entrypoint: `interp/README.md`.
- Added a tracked metric snapshot: `interp/RESULTS.md`.
- Added a reproducibility and claim-audit page: `interp/REPRODUCIBILITY.md`.
- Added a root README pointer so new readers find the `interp/` work.
- Polished the `B_3` report around the exact algorithm-recovery result.
- Polished the `B_4` report around the signed-frontier mechanism.
- Added a public SAE report: `interp/B4_SAE_FINAL.md`.
- Added mechanism diagrams:
  - `interp/figures/project_story.svg`
  - `interp/figures/b3_algorithm_recovery.svg`
  - `interp/figures/b4_signed_frontier.svg`
  - `interp/figures/b4_sae_circuit.svg`
  - `interp/figures/b4_sae_controls.svg`
- Added B4 signed-frontier deep-dive tooling:
  - `interp/run_b4_z_sign_deep_dive.py`
  - `interp/jobs/run_b4_z_sign_deep_dive.sh`
- Added B4 hidden-theorem search tooling:
  - `interp/search_b4_hidden_theorem.py`
  - `interp/jobs/search_b4_hidden_theorem.sh`
- Added B4 boundary-only training entrypoint:
  - `interp/jobs/train_b4_z_sign_boundary_small.sh`
- Ran syntax checks for the new Python and Slurm files.
- B4 boundary-only robustness replicate:
  - Training Slurm job: `13896534`
  - Dependent deep-dive Slurm job: `13897122`
  - Output directory:
    `interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small_seed7`
  - Deep-dive output:
    `interp/artifacts/b4_l25_zsign_boundary_r8_xfmr2_small_seed7_deep_dive/results.json`
  - Held-out test result: `90.36%` exact, `96.66%` bit, `96.61%` micro-F1.
  - Deep-dive result: late-`CLS` descent probe gets `88.68%` exact, and
    radius-8 boundary patching recovers `99.7%` of the clean-corrupt score.
- Added B4 SAE tooling:
  - `interp/run_b4_sae_experiments.py`
  - `interp/stress_b4_sae_controls.py`
  - `interp/analyze_b4_sae_final.py`
  - `interp/jobs/run_b4_sae_suite.sh`
  - `interp/jobs/stress_b4_sae_controls.sh`
  - `interp/jobs/analyze_b4_sae_final.sh`
- Ran final SAE analysis:
  - Final Slurm job: `13920979`
  - Output directory:
    `interp/artifacts/b4_l25_zsign_boundary_r8_sae_final`
  - Best selected-feature classifier: `89.4%` exact, `96.3%` bit,
    `96.0%` agreement with the transformer.
  - Cross-seed best-match correlations: `0.649` at `final_hidden_cls`,
    `0.613` at `l1_resid_post_cls`.
  - Layer-1 `CLS` attention patching recovers up to `81.1%` of selected SAE
    feature activations and `76.2%` logit recovery.

## Optional Polish

- Add a `requirements` or module note if moving off the Roberts cluster.
- Add one raster overview figure for talks or README previews.
- Package a tiny artifact-free smoke dataset if this needs to run outside the
  cluster.

## Claim Stack

Lead with this order:

1. `B_3` exact theorem: extremal Burau matrix unit determines descent.
2. `B_3` interp result: late CLS linearly represents that unit; decoded unit
   plus exact rule gives `99.99%` accuracy.
3. `B_4` representation result: signed `Z[v]` tokens raise exact accuracy from
   `71.4%` to `93.2%`.
4. `B_4` causal result: prefix-fixed boundary patching recovers `90.2%` of the
   clean-corrupt score difference.
5. `B_4` clean circuit target: a boundary-only transformer reaches `90.6%`
   exact accuracy and uses layer-1 attention heads to read signed frontier
   information into `CLS`.
6. `B_4` SAE result: a few dozen late-`CLS` sparse features recover almost
   all of the boundary-only model, beat random active-feature controls, recur
   across seeds, and are causally fed by layer-1 `CLS` attention.
7. Honest boundary: the hidden-theorem search did not find an exact small
   finite rule for `B_4`; the current result is a strong signed-frontier
   mechanism, not a solved theorem.
