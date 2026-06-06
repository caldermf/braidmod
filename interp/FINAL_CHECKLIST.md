# Final Project Checklist

This is the repo-facing checklist for the `interp/` project. It records what
has been wrapped into the public research narrative and what remains optional.

## Done

- Wrote a public project entrypoint: `interp/README.md`.
- Added a tracked metric snapshot: `interp/RESULTS.md`.
- Added a root README pointer so new readers find the `interp/` work.
- Polished the `B_3` report around the exact algorithm-recovery result.
- Polished the `B_4` report around the signed-frontier mechanism.
- Added mechanism diagrams:
  - `interp/figures/b3_algorithm_recovery.svg`
  - `interp/figures/b4_signed_frontier.svg`
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

## Optional Polish

- Add one raster overview figure for talks or README previews.
- Add a short abstract paragraph at the top of `interp/README.md` if this will
  be shared as a standalone project.
- Add a `requirements` or module note if moving off the Roberts cluster.
- Run one more small seed only if the current seed-7 replicate is weak.

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
6. Honest boundary: the hidden-theorem search did not find an exact small
   finite rule for `B_4`; the current result is a strong signed-frontier
   mechanism, not a solved theorem.
