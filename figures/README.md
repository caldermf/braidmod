# Figure guide

This directory contains the curated public plots for `braidmod`.

## Training figures

- [mlp_and_transformer_training_curves.png](mlp_and_transformer_training_curves.png)
  The main front-page figure: MLP and transformer training/validation curves side by side.
- [mlp_training_curves.png](mlp_training_curves.png)
  Baseline MLP training and validation curves.
- [transformer_training_curves.png](transformer_training_curves.png)
  Best public transformer training and validation curves.
- [mlp_vs_transformer_validation.png](mlp_vs_transformer_validation.png)
  Direct validation comparison between the original MLP and the best transformer.

## Known-kernel overlays

- [known_kernel_elements_vs_random_target_cross_entropy_running_average_15_steps.png](known_kernel_elements_vs_random_target_cross_entropy_running_average_15_steps.png)
  Mean target cross-entropy for the first five known kernel elements against the mean of five random braids, using a 15-step running average.
- [individual_known_kernel_elements_vs_random_target_cross_entropy_running_average_15_steps.png](individual_known_kernel_elements_vs_random_target_cross_entropy_running_average_15_steps.png)
  Individual known-kernel trajectories against five random controls, again using a 15-step running average.

These are the clearest public plots showing that the confusion signal is not an artifact of one example.

## GWY kernel element plots

- [gwy_kernel_element_vs_random_cumulative_target_cross_entropy.png](gwy_kernel_element_vs_random_cumulative_target_cross_entropy.png)
  Cumulative target cross-entropy for the GWY kernel element against the mean of five random braids.
- [gwy_kernel_element_vs_random_target_cross_entropy_running_average_5_steps.png](gwy_kernel_element_vs_random_target_cross_entropy_running_average_5_steps.png)
  Target cross-entropy for the GWY kernel element with a 5-step running average.
- [gwy_kernel_element_vs_random_entropy_confusion.png](gwy_kernel_element_vs_random_entropy_confusion.png)
  Entropy-based confusion for the GWY kernel element against the random mean.

## Provenance

The figure inputs used to reproduce the public confusion plots live in:

- [`../figure_data/confusion_suite_tuned`](../figure_data/confusion_suite_tuned)
- [`../figure_data/search/kernel_hits_len60.json`](../figure_data/search/kernel_hits_len60.json)

The public model checkpoint used for those renders is:

- [`../checkpoints/best_transformer/best_model.pt`](../checkpoints/best_transformer/best_model.pt)
