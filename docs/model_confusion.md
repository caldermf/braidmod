# Model Confusion as a Statistical Kernel Signal

## Setup

We work in `B_4` with the reduced Burau representation mod `p`. For a braid in
left Garside normal form

`Delta^d w_1 ... w_\ell`,

we train a model to predict the final Garside factor `w_\ell` from the Burau
matrix alone.

The training target is deliberately not kernel membership. The point is to
learn a structural algebraic feature that ordinary braids exhibit and then ask
when a prefix looks atypical to that model.

## The central idea

Given a prefix, the model produces logits over the possible final Garside
factors. We then study quantities such as:

- entropy of the factor distribution
- cross-entropy against the actual final factor
- smoothed versions of those signals along prefixes

We refer to these as model-confusion signals. The working heuristic is that
kernel-like prefixes should systematically induce abnormal uncertainty or
abnormal target surprise.

## Why this is interesting

This approach avoids supervised kernel labels. Instead, it uses:

1. ordinary random Garside data
2. a prediction task with algebraic meaning
3. model confusion as a downstream probe

That makes the search procedure conceptually cleaner: the model is not trained
to say "kernel", but its failure mode still becomes informative.

## Models

### Original MLP baseline

The MLP treats the Burau tensor as a flattened structured input with discrete
embeddings. It works surprisingly well as a first-pass structural predictor and
already yields useful confusion signals.

Public baseline summary:

- validation factor accuracy: `0.7266`
- curves: `figures/mlp_training_curves.png`

### Best transformer

The final public transformer is a hierarchical encoder:

- local encoder across each `3 x 3` degree slice
- global encoder across degrees
- trained directly on final-factor prediction
- checkpoint selection by validation loss

Public best-transformer summary:

- validation loss: `0.2178`
- validation factor accuracy: `0.9388`
- curves: `figures/transformer_training_curves.png`
- comparison: `figures/mlp_vs_transformer_validation.png`

## Prefix-level behavior

The most useful public figures are the kernel-vs-random overlays.

Two especially clean summary plots are:

- `figures/kernel_avg_first5_vs_random_avg10.png`
- `figures/kernel_avg_first5_vs_random_avg15.png`

These average the first five saved kernel-hit trajectories and compare them to
the average of five saved random braids. The transformer's target
cross-entropy remains markedly elevated on the kernel side across long prefix
intervals.

We also keep the full multi-curve overlay:

- `figures/kernel_hits_vs_random_avg10.png`

and the suite-level Geordie/random summaries:

- `figures/geordie_vs_random_cumulative_xent.png`
- `figures/geordie_entropy_confusion.png`

## Search

The search code in `reservoir_search_braidmod.py` does not require direct
kernel supervision either. It can score frontier expansions using:

- Burau projective length alone
- model target cross-entropy alone
- frontier-based combinations of projlen and model confusion

This is the part that turns the learned signal into a practical mathematical
tool rather than a classification demo.

## Reproducibility

Tracked inputs for the public confusion figures live in:

- `figure_data/confusion_suite_tuned/`
- `figure_data/search/kernel_hits_len60.json`

Tracked model artifacts live in:

- `checkpoints/original_mlp/`
- `checkpoints/best_transformer/`

Curated cluster wrappers live in:

- `jobs/`

Older ablations, historical wrappers, and side experiments have been moved to
`prototypes/`.
