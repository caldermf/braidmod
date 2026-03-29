# braidmod

`braidmod` studies a simple strategy for probing the kernel of the reduced
Burau representation in `B_4` mod `p`:

1. Train on an algebraic prediction problem that does **not** use kernel labels.
2. Evaluate the model along prefixes of candidate braids.
3. Use model confusion as a statistical signal for search.

The supervised task is:

- input: a projectively normalized Burau tensor
- label: the final Garside factor of the braid's left normal form

The baseline model is an MLP. The strongest public model is a hierarchical
transformer that encodes local structure inside each `3 x 3` degree slice and
then aggregates across degrees.

![MLP vs transformer validation curves](figures/mlp_vs_transformer_validation.png)
![Average kernel-vs-random confusion overlay](figures/kernel_avg_first5_vs_random_avg15.png)

## Why this is interesting

The kernel problem is hard. Instead of training a classifier to say "kernel" or
"not kernel", this repo trains models to infer honest algebraic structure from
Burau matrices. The working hypothesis is:

- ordinary braids should look structurally predictable
- kernel-like prefixes should look atypical
- that atypicality should show up as target surprise or uncertainty
- a search procedure can exploit that signal without direct kernel supervision

This turns model failure into a mathematical tool.

## Headline results

- original MLP validation factor accuracy: `0.7266`
- best transformer validation loss: `0.2178`
- best transformer validation factor accuracy: `0.9388`
- saved kernel prefixes remain statistically separated from random braids by
  smoothed target cross-entropy

The most presentation-ready figures are:

- [figures/kernel_avg_first5_vs_random_avg15.png](figures/kernel_avg_first5_vs_random_avg15.png)
- [figures/geordie_vs_random_cumulative_xent.png](figures/geordie_vs_random_cumulative_xent.png)
- [figures/mlp_vs_transformer_validation.png](figures/mlp_vs_transformer_validation.png)

## Model comparison

| Model | Input handling | Core trunk | Training target | Public metric |
| --- | --- | --- | --- | --- |
| MLP baseline | Embed each mod-`p` coefficient, then flatten the full `D x 3 x 3` tensor | Residual feedforward blocks | final factor + auxiliary descent class | `0.7266` val factor accuracy |
| Best transformer | Embed coefficients as tokens, summarize each degree slice, then attend across degrees | hierarchical local/global self-attention | final factor only | `0.2178` val loss, `0.9388` val factor accuracy |

The transformer wins for a structural reason, not just a scaling reason. The
Burau tensor already has a natural hierarchy:

- inside one degree there is a small `3 x 3` matrix with local row/column structure
- across degrees there is a polynomial support pattern with variable occupied width

The MLP sees that tensor only after flattening. The transformer keeps both
levels of structure explicit.

## Shared input representation

Both public models consume the same projectively normalized Burau tensor:

- shape: `D x 3 x 3`
- entries: integers mod `p`
- extra scalar feature: `burau_min_degree`
- target class: one of the `24` permutations in `S_4`, representing the final
  Garside factor

The dataset zero-pads the tensor out to the fixed training depth `D`. For the
transformer, the code infers a valid-degree mask from the last occupied degree
so that internal zero slices remain legal while trailing padding is ignored.

## MLP baseline

The original MLP is intentionally simple.

1. Embed each coefficient by value, degree, row, and column.
2. Flatten the full tensor into one long vector.
3. Project once into a hidden state.
4. Add a projected `burau_min_degree` feature.
5. Pass the result through residual MLP blocks.
6. Predict the final factor, with an optional auxiliary head for descent type.

This baseline already works surprisingly well. It shows that the Burau matrix
really does carry usable information about the final Garside factor, and it was
enough to produce the first useful model-confusion plots. But it has a real
limitation: once the tensor is flattened, the network has to rediscover both
matrix-local and degree-local structure from absolute positions alone.

## Hierarchical transformer

The public transformer keeps the architecture close to the algebraic object.

### Stage 1: tokenization inside each degree

For each occupied degree slice, the model creates `9` tokens, one for each
entry of the `3 x 3` matrix. Every token is the sum of:

- a value embedding for the mod-`p` coefficient
- a row embedding
- a column embedding
- a local degree embedding

So the model never sees a coefficient in isolation; each token already knows
which matrix entry and which polynomial degree it came from.

### Stage 2: local encoder over one `3 x 3` slice

Each degree slice receives a learned local CLS token and then passes through:

- `2` local transformer blocks
- `4` attention heads per block
- feedforward width multiplier `4`

The local CLS output is the summary vector for that degree. This is the key
compression step: instead of flattening all `9D` entries directly into one huge
vector, the model first converts each degree into one structured summary.

### Stage 3: global encoder across degrees

The degree summaries are then fed to a second transformer with:

- a learned global CLS token
- a separate global degree embedding
- `6` global transformer blocks
- `8` attention heads per block

The projected `burau_min_degree` scalar is added as a bias to the global CLS
token. That lets the model compare the coarse location of the polynomial support
with the content of the degree summaries.

### Stage 4: final prediction head

The global CLS representation is layer-normalized and sent to a `24`-way linear
classifier for the final Garside factor. The code still supports the older
auxiliary descent head, but the best public model does **not** use it; the best
run is factor-only.

### Why this architecture

This design is deliberately modest:

- no giant token sequence over all coefficients
- no decoder
- no positional gimmicks beyond row, column, and degree embeddings
- no task-specific hand engineering beyond the hierarchy already present in the
  Burau tensor

What it does add is the right inductive bias. The model can learn relations
within a single polynomial degree before it reasons across degrees, which is
much closer to how the data is actually organized than the flat MLP.

## Why the transformer outperforms the MLP

- The MLP must memorize interactions after flattening; the transformer models
  them directly with attention.
- The local encoder shares one computation pattern across every degree slice,
  which is a much stronger bias than treating every flattened coordinate as
  unrelated.
- The global encoder can attend across occupied degrees while ignoring padded
  tail degrees through the inferred mask.
- The best transformer trains on the exact target we care about, without the
  auxiliary descent head that made the older setup more diffuse.

The result is not just prettier training curves. The better factor predictor
still leaves a useful confusion signal on kernel examples, especially in the
smoothed target cross-entropy plots.

## Figure gallery

The full figure inventory is documented in [figures/README.md](figures/README.md).
The core public story uses four groups of plots.

### 1. Training curves

![Transformer training curves](figures/transformer_training_curves.png)
![MLP vs transformer validation comparison](figures/mlp_vs_transformer_validation.png)

These show the basic modeling result: the transformer learns the factor
prediction task much more cleanly than the original MLP.

### 2. Averaged kernel-vs-random overlays

![Average kernel-vs-random overlay, avg7](figures/kernel_avg_first5_vs_random_avg7.png)
![Average kernel-vs-random overlay, avg15](figures/kernel_avg_first5_vs_random_avg15.png)
![Average kernel-vs-random overlay, avg20](figures/kernel_avg_first5_vs_random_avg20.png)

These average the first five saved kernel-hit trajectories and compare them to
five random braids. `avg7` keeps more local wiggle, `avg15` is the cleanest
presentation figure, and `avg20` shows that the separation survives even under
heavy smoothing.

### 3. Individual kernel-hit trajectories

![Kernel-hit vs random overlay, avg15](figures/kernel_hits_vs_random_avg15.png)

The averaged view is not hiding one anomalous example. The individual
kernel-hit curves also sit above the random controls for long prefix intervals.

### 4. Geordie case study

![Geordie cumulative target cross-entropy](figures/geordie_vs_random_cumulative_xent.png)
![Geordie smoothed target cross-entropy](figures/geordie_target_cross_entropy_avg5.png)
![Geordie entropy confusion](figures/geordie_entropy_confusion.png)

These plots focus on the saved Geordie kernel word. The target cross-entropy
signal is the clearest separation, while entropy is a weaker but still visible
secondary view.

## Start here

- `docs/model_confusion.md`
  Public writeup of the core idea and the main plots.
- `checkpoints/`
  Public model artifacts, logs, and training curves.
- `figures/`
  Curated plots for the public story.
- `figures/README.md`
  Short guide to what each public figure is showing.
- `jobs/`
  Clean cluster entrypoints for dataset generation, training, figure rendering,
  and search.

## Repository tour

- `braid_data.py`
  Garside normal form utilities, Burau evaluation, and dataset construction.
- `generate_dataset.py`
  CLI for generating Burau/Garside training data.
- `train_garside_mlp.py`
  Unified trainer for the original MLP and the final transformer.
- `garside_models.py`, `garside_transformer.py`
  Public model definitions and checkpoint-aware construction.
- `predict_garside_mlp.py`
  Inference CLI for saved checkpoints.
- `reservoir_search_braidmod.py`
  Search that combines projective length and model-confusion scores.
- `plot_prefix_confusion.py`, `track_confusion_prefix.py`,
  `render_kernel_random_xent_overlay.py`,
  `render_average_kernel_random_xent_overlay.py`
  Prefix-confusion analysis and figure generation.
- `figure_data/`
  Tracked JSON inputs used to reproduce the public confusion figures.
- `prototypes/`
  Archived experiments, historical cluster wrappers, and research backlog.

## Quick start

Create an environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-ml.txt
```

Generate the reference dataset locally:

```bash
.venv/bin/python generate_dataset.py \
  --output-path data/generated/burau_gnf_L30to60_p5_D140_N200000_uniform_corrected.json \
  --num-samples 200000 \
  --length-min 30 \
  --length-max 60 \
  --p 5 \
  --D 140
```

Train the original MLP baseline:

```bash
.venv/bin/python train_garside_mlp.py \
  --data-path data/generated/burau_gnf_L30to60_p5_D140_N200000_uniform_corrected.json \
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
  --data-path data/generated/burau_gnf_L30to60_p5_D140_N200000_uniform_corrected.json \
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

The raw MLP checkpoint is intentionally omitted from GitHub. The public repo
keeps the log, the curves, and the exact training recipe needed to regenerate
it.

### Best transformer

- checkpoint: `checkpoints/best_transformer/best_model.pt`
- curves: `checkpoints/best_transformer/training_curves.png`
- comparison plot: `checkpoints/best_transformer/mlp_vs_transformer_validation.png`
- headline validation loss: `0.2178`
- headline validation factor accuracy: `0.9388`

### Public figure set

- `figures/mlp_training_curves.png`
- `figures/transformer_training_curves.png`
- `figures/mlp_vs_transformer_validation.png`
- `figures/kernel_avg_first5_vs_random_avg7.png`
- `figures/kernel_avg_first5_vs_random_avg10.png`
- `figures/kernel_avg_first5_vs_random_avg15.png`
- `figures/kernel_avg_first5_vs_random_avg20.png`
- `figures/kernel_hits_vs_random_avg10.png`
- `figures/kernel_hits_vs_random_avg15.png`
- `figures/kernel_hits_vs_random_avg20.png`
- `figures/geordie_vs_random_cumulative_xent.png`
- `figures/geordie_target_cross_entropy_avg5.png`
- `figures/geordie_entropy_confusion.png`

## Data policy

The repository tracks small figure inputs and the public transformer checkpoint,
but not the large generated training corpora. Put regenerated datasets under
`data/generated/`. See `data/README.md` for the default paths and regeneration
commands.

## Cluster use

If you are running on Yale Bouchet, use the curated scripts in `jobs/`. Older
experiment-specific wrappers are archived under `prototypes/slurm/`.

## License

This repository is released under the MIT License. See `LICENSE`.
