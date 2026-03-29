# Model Confusion as a Statistical Kernel Signal

## Problem

We work in `B_4` with the reduced Burau representation mod `p`. For a braid in
left Garside normal form

`Delta^d w_1 ... w_\ell`,

the supervised task is to predict the final Garside factor `w_\ell` from the
Burau matrix alone.

That target is deliberately **not** kernel membership. The goal is to learn a
structural algebraic feature of ordinary braids and then ask what happens when
a prefix stops looking ordinary to the model.

## Core idea

For each prefix, the model produces logits over possible final Garside factors.
From those logits we extract model-confusion signals such as:

- entropy of the factor distribution
- cross-entropy against the actual final factor
- smoothed versions of those signals along prefix trajectories

The working heuristic is simple: if a prefix comes from a kernel element, it
should often look abnormal to a model trained only on standard Garside data.

## Why this is useful

This avoids direct kernel supervision. The workflow is:

1. sample ordinary random Garside data
2. train on a mathematically meaningful prediction target
3. reuse model confusion as a probe for unusual prefixes

That keeps the learning problem cleaner while still producing a practical
signal for search.

## Public models

### Original MLP baseline

The MLP flattens the Burau tensor and applies discrete embeddings followed by a
structured feedforward stack.

- validation factor accuracy: `0.7266`
- curves: `figures/mlp_training_curves.png`

Even this baseline already produces usable confusion curves on saved kernel
examples.

### Best transformer

The best public model is a hierarchical transformer:

- local attention inside each `3 x 3` degree slice
- global attention across degrees
- direct training on final-factor prediction
- model selection by validation loss

- validation loss: `0.2178`
- validation factor accuracy: `0.9388`
- curves: `figures/transformer_training_curves.png`
- comparison: `figures/mlp_vs_transformer_validation.png`

## Prefix-level behavior

The clearest public figures are the kernel-vs-random overlays based on target
cross-entropy.

- `figures/known_kernel_elements_vs_random_target_cross_entropy_running_average_15_steps.png`
- `figures/individual_known_kernel_elements_vs_random_target_cross_entropy_running_average_15_steps.png`

The first averages five saved kernel-hit trajectories and compares them to the
mean of five random braids under a 15-step running average. The second shows
the individual known-kernel trajectories against the same random controls.

For the single-element case study we keep:

- `figures/gwy_kernel_element_vs_random_cumulative_target_cross_entropy.png`
- `figures/gwy_kernel_element_vs_random_target_cross_entropy_running_average_5_steps.png`
- `figures/gwy_kernel_element_vs_random_entropy_confusion.png`

These summarize the GWY kernel element against random controls and show that
the target-cross-entropy signal remains the clearest public proxy.

## Search

`reservoir_search_braidmod.py` turns these signals into a search heuristic. It
can score frontier expansions with:

- projective length alone
- model target cross-entropy alone
- combined frontier scores that blend projlen and model confusion

That is the main point of the repo: not just that the model fits the factor
prediction task, but that its confusion profile becomes useful for locating
promising regions of braid space.

## Reproducibility

Tracked public inputs for the confusion figures live in:

- `figure_data/confusion_suite_tuned/`
- `figure_data/search/kernel_hits_len60.json`

Tracked public model artifacts live in:

- `checkpoints/original_mlp/`
- `checkpoints/best_transformer/`

Large training corpora are intentionally not checked into GitHub. Regenerate
them under `data/generated/` using `generate_dataset.py` or `jobs/build_reference_dataset.sh`.

Historical ablations, older wrappers, and side experiments were moved to
`prototypes/` to keep the top-level repo focused on the public narrative.
