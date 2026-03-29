# Burau Model Confusion in `B_4`

This repository studies a simple idea for probing the kernel of the reduced
Burau representation mod `p`:

1. Train a model to predict an algebraic property of a braid from its Burau
   matrix, rather than training directly on kernel membership.
2. Measure the model's uncertainty or target surprise along prefixes.
3. Use that "model confusion" signal as a statistical guide for search.

In this project, the supervised task is:

- input: a projectively normalized Burau tensor in `B_4`
- label: the final Garside factor of the braid's left normal form

The baseline is an MLP. The stronger model is a hierarchical transformer that
encodes local structure inside each `3 x 3` degree slice and then aggregates
across degrees.

## Why this repo exists

The reduced Burau representation is a classical object, but the kernel problem
is difficult. Instead of asking a model to predict "kernel or not" directly, we
ask it to infer structure that honest braids should exhibit. The working
hypothesis is:

- kernel-like prefixes should look atypical to a model trained on ordinary
  Garside data
- that atypicality should appear as elevated target cross-entropy or broader
  uncertainty
- a search procedure can exploit that signal without ever training on a kernel
  label

The result is a practical workflow that statistically separates saved kernel
examples from random braids and plugs directly into reservoir search.

## What to look at first

- `docs/model_confusion.md`: the public-facing writeup
- `figures/`: curated training and confusion plots
- `checkpoints/`: public model artifacts and logs
- `generate_dataset.py`: reproducible dataset generation CLI
- `train_garside_mlp.py`: unified trainer for both the MLP and transformer
- `reservoir_search_braidmod.py`: search with projlen and model-confusion scores

## Public repo layout

- `braid_data.py`
  Garside normal form utilities, Burau evaluation, and the core dataset builder.
- `generate_dataset.py`
  Public CLI for generating Burau/Garside training data.
- `train_garside_mlp.py`
  Trainer for the original MLP and the final transformer.
- `garside_models.py`, `garside_transformer.py`
  Model definitions and checkpoint-aware construction.
- `predict_garside_mlp.py`
  Inference CLI for saved checkpoints.
- `reservoir_search_braidmod.py`
  Reservoir search with projlen, target cross-entropy, and frontier-style
  combined scoring.
- `plot_prefix_confusion.py`, `track_confusion_prefix.py`,
  `render_kernel_random_xent_overlay.py`,
  `render_average_kernel_random_xent_overlay.py`
  Prefix-confusion analysis and figure generation.
- `jobs/`
  Clean Slurm entrypoints for the public workflow.
- `checkpoints/`
  Public baseline and best-transformer artifacts.
- `figure_data/`
  Small tracked JSON inputs used to reproduce the public confusion figures.
- `figures/`
  Curated public plots.
- `prototypes/`
  Archived experiments, cluster wrappers, notes, and prototype-only scripts.

## Quick start

Create an environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-ml.txt
```

Generate a dataset:

```bash
.venv/bin/python generate_dataset.py \
  --output-path data/burau_gnf_L30to60_p5_D140_N200000_uniform_corrected.json \
  --num-samples 200000 \
  --length-min 30 \
  --length-max 60 \
  --p 5 \
  --D 140
```

Train the original MLP baseline:

```bash
.venv/bin/python train_garside_mlp.py \
  --data-path data/burau_gnf_L30to60_p5_D140_N200000_uniform_corrected.json \
  --p 5 \
  --task multitask \
  --batch-size 512 \
  --epochs 40 \
  --embed-dim 32 \
  --hidden-dim 1024 \
  --blocks 3 \
  --dropout 0.1 \
  --aux-weight 0.2 \
  --out-dir artifacts/public_original_mlp
```

Train the best transformer:

```bash
.venv/bin/python train_garside_mlp.py \
  --data-path data/burau_gnf_L30to60_p5_D140_N200000_uniform_corrected.json \
  --p 5 \
  --model-type transformer \
  --task final_factor \
  --batch-size 256 \
  --epochs 30 \
  --d-model 256 \
  --ffn-mult 4 \
  --num-local-blocks 2 \
  --num-local-heads 4 \
  --num-global-blocks 6 \
  --num-global-heads 8 \
  --dropout 0.05 \
  --label-smoothing 0.03 \
  --selection-objective loss \
  --out-dir artifacts/public_best_transformer
```

Run model-confusion search:

```bash
.venv/bin/python reservoir_search_braidmod.py \
  --p 5 \
  --max-length 60 \
  --bucket-size 100000 \
  --use-best 300000 \
  --bootstrap-length 5 \
  --num-buckets 100 \
  --score-type frontier_target_xent \
  --checkpoint checkpoints/best_transformer/best_model.pt \
  --device cuda \
  --out-json artifacts/public_frontier_search.json
```

Render the public averaged kernel-vs-random confusion curves:

```bash
.venv/bin/python render_average_kernel_random_xent_overlay.py \
  --search-json figure_data/search/kernel_hits_len60.json \
  --checkpoint checkpoints/best_transformer/best_model.pt \
  --suite-dir figure_data/confusion_suite_tuned \
  --out-png figures/generated/kernel_avg_first5_vs_random_avg_target_xent_avg15.png \
  --device cuda \
  --mode avg5 \
  --window 15 \
  --max-length 60 \
  --num-kernels 5
```

## Public artifacts

### Baseline MLP

- curves: `checkpoints/original_mlp/training_curves.png`
- log: `checkpoints/original_mlp/train.log`
- headline validation accuracy: `0.7266`

The full baseline checkpoint is intentionally not tracked because the raw
PyTorch file exceeds a normal GitHub-friendly size. The exact training job and
log are included, so the baseline remains reproducible.

### Best transformer

- checkpoint: `checkpoints/best_transformer/best_model.pt`
- curves: `checkpoints/best_transformer/training_curves.png`
- comparison plot: `checkpoints/best_transformer/mlp_vs_transformer_validation.png`
- headline validation loss: `0.2178`
- headline validation factor accuracy: `0.9388`

### Public figure set

- `figures/kernel_avg_first5_vs_random_avg10.png`
- `figures/kernel_avg_first5_vs_random_avg15.png`
- `figures/kernel_hits_vs_random_avg10.png`
- `figures/geordie_vs_random_cumulative_xent.png`
- `figures/geordie_entropy_confusion.png`

## Cluster use

If you are running on Yale Bouchet, use the curated scripts in `jobs/`.
The older experiment-specific wrappers are archived under `prototypes/slurm/`.

## Prototypes

Everything that did not belong in the public narrative was moved under
`prototypes/`: abandoned ablations, one-off scripts, internal notes, cluster
wrappers, and other research backlog. The point is to keep the top-level repo
focused on the main mathematical story rather than the full notebook of
everything tried along the way.
