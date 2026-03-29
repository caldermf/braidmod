# Figure guide

This directory contains the curated public plots for `braidmod`.

## Training figures

- [mlp_training_curves.png](mlp_training_curves.png)
  Baseline MLP training and validation curves.
- [transformer_training_curves.png](transformer_training_curves.png)
  Best public transformer training and validation curves.
- [mlp_vs_transformer_validation.png](mlp_vs_transformer_validation.png)
  Direct validation comparison between the original MLP and the best transformer.

## Averaged kernel-vs-random overlays

- [kernel_avg_first5_vs_random_avg7.png](kernel_avg_first5_vs_random_avg7.png)
  Light smoothing. Preserves more local movement in the kernel trajectories.
- [kernel_avg_first5_vs_random_avg10.png](kernel_avg_first5_vs_random_avg10.png)
  Balanced summary figure with moderate smoothing.
- [kernel_avg_first5_vs_random_avg15.png](kernel_avg_first5_vs_random_avg15.png)
  Best single presentation figure for the public story.
- [kernel_avg_first5_vs_random_avg20.png](kernel_avg_first5_vs_random_avg20.png)
  Heavy smoothing. Shows that the separation survives substantial averaging.

Each of these averages the first five saved kernel-hit trajectories from the
tracked search output against the average of five saved random braids.

## Individual kernel-hit overlays

- [kernel_hits_vs_random_avg10.png](kernel_hits_vs_random_avg10.png)
  Individual kernel-hit trajectories against random controls with moderate smoothing.
- [kernel_hits_vs_random_avg15.png](kernel_hits_vs_random_avg15.png)
  Cleaner individual overlay for public presentation.
- [kernel_hits_vs_random_avg20.png](kernel_hits_vs_random_avg20.png)
  Strongest smoothing for the individual-trajectory view.

These plots show that the separation is not created by averaging alone.

## Geordie plots

- [geordie_vs_random_cumulative_xent.png](geordie_vs_random_cumulative_xent.png)
  Cumulative target cross-entropy comparison for the saved Geordie kernel word.
- [geordie_target_cross_entropy_avg5.png](geordie_target_cross_entropy_avg5.png)
  Smoothed target cross-entropy along Geordie prefixes.
- [geordie_entropy_confusion.png](geordie_entropy_confusion.png)
  Entropy-based confusion view for the same example.

## Provenance

The figure inputs used to reproduce the public confusion plots live in:

- [`../figure_data/confusion_suite_tuned`](../figure_data/confusion_suite_tuned)
- [`../figure_data/search/kernel_hits_len60.json`](../figure_data/search/kernel_hits_len60.json)

The public model checkpoint used for those renders is:

- [`../checkpoints/best_transformer/best_model.pt`](../checkpoints/best_transformer/best_model.pt)
