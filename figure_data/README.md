# Figure data

This directory contains the small tracked JSON inputs used to reproduce the
public confusion figures.

- `confusion_suite_tuned/`
  Saved GWY-kernel-plus-random prefix trajectories for the tuned transformer.
- `search/kernel_hits_len60.json`
  Saved kernel-hit search output used for the kernel-vs-random overlays.

The curated PNGs in `figures/` were rendered from these inputs together with
the public transformer checkpoint in `checkpoints/best_transformer/`.
