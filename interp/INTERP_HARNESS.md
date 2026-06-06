# Transformer Interp Harness

This is the working map of the local TransformerLens-style machinery for the
`B_3`, `p=2`, length-25 transformer.

## Core Wrapper

`interp/b3_interp.py` provides a manual forward pass with named hook points:

```text
hook_token_embed
hook_embed
hook_resid_embed
blocks.{L}.hook_resid_pre
blocks.{L}.hook_attn_norm
blocks.{L}.hook_q
blocks.{L}.hook_k
blocks.{L}.hook_v
blocks.{L}.hook_pattern
blocks.{L}.hook_z
blocks.{L}.hook_attn_head_out
blocks.{L}.hook_attn_out
blocks.{L}.hook_resid_mid
blocks.{L}.hook_mlp_norm
blocks.{L}.hook_mlp_out
blocks.{L}.hook_resid_post
hook_final_hidden
hook_logits
```

The important addition beyond the first-pass harness is
`hook_attn_head_out`, which stores each attention head's projected residual
stream contribution separately, with shape `[batch, head, token, d_model]`.
Summing this over heads and adding the output bias reconstructs the normal
attention output. A synthetic check gives a cached-vs-direct max logit
difference of `1.8e-7`, so the no-hook path is faithful.

## Patch/Ablation Tools

The harness currently supports:

- token-position activation patching from clean to corrupt examples;
- multi-position activation patching, e.g. leading plus trailing boundary;
- single-head and head-set patching at selected destination positions;
- single-head and head-set zero ablation at selected destination positions;
- boundary support utilities: first support degree, last support degree,
  boundary windows, boundary-only counterfactuals.

This is intentionally small and repo-native, not a dependency on
TransformerLens internals. The model architecture is simple enough that a
manual forward pass is clearer and easier to audit.

## Current Experiment Scripts

`interp/run_b3_interp_experiments.py`

First-pass feature lookups, boundary masking, attention summaries, and residual
token patching.

`interp/run_b3_matched_boundary_patching.py`

Matched-support clean/corrupt pairs. This is the sharpest causal test for the
boundary rule because clean and corrupt examples share the same support
interval but have opposite labels.

`interp/run_b3_head_circuit.py`

Head-level circuit analysis:

- CLS attention mass to leading/trailing boundary positions;
- per-head zero ablation;
- matched clean-to-corrupt per-head patching;
- head-subset patching and ablation.

The Slurm entry point is `interp/jobs/run_b3_head_circuit.sh`, with a 5-minute
`scavenge_gpu` time limit.

## Current Circuit Candidate

The exact algebraic rule is still the strongest mathematical fact:

```text
leading boundary column 0 -> {s_2}
leading boundary column 1 -> {s_1}
trailing boundary column 1 -> {s_2}
trailing boundary column 0 -> {s_1}
```

The model-level evidence now points to a simple aggregation circuit rather than
an opaque global polynomial computation:

- matched input swaps: replacing both boundary tokens recovers `100%` clean
  behavior;
- layer-0 residual patching at both boundary tokens gives `0.745` normalized
  recovery;
- layer-1 CLS residual patching gives `1.000` recovery;
- layer-1 CLS attention has head specialization:
  - head 0 puts about `0.458` attention on the trailing boundary;
  - head 1 puts about `0.490` attention on the leading boundary;
- ablating layer-1 head 0 at CLS costs about `1.73` logit-score points and
  `9.5%` accuracy on the sampled test batch;
- ablating layer-1 head 1 at CLS costs about `3.51` logit-score points and
  `8.4%` accuracy.

Head-set and path patching refine this picture:

- patching all layer-1 attention head outputs at CLS gives `0.974` normalized
  recovery;
- patching layer-1 heads `{0, 1}` at CLS gives `0.790` recovery;
- patching layer-1 head 0 at CLS gives `0.582` recovery;
- path patching shows layer-1 head 0's effect comes from the trailing boundary
  value vector:
  - `v_trailing_value_source`: `0.452` recovery;
  - `pattern_cls_row_plus_v_trailing_value_source`: `0.520`;
  - `z_cls_dest` / projected head output at CLS: `0.582`;
- layer-0 head 2 carries leading-boundary information:
  - `v_leading_value_source`: `0.331` recovery;
  - full layer-0 head-2 projected output: `0.360`;
  - CLS-only layer-0 head-2 projected output: `0.115`.

The layer-0 head-2 destination sweep shows that this leading-boundary signal is
not localized to the boundary tokens themselves. The largest single destination
is CLS (`0.115` recovery), while the full-head output recovers `0.360`.
The remaining effect is spread over many small destination contributions. This
is a useful warning: the layer-0 part of the circuit is not a single clean
copy-to-CLS operation.

Candidate-subcircuit ablations on `131,072` held-out examples give the current
sufficiency picture:

- full model: `99.79%` accuracy;
- zero layer-1 attention: `83.15%`;
- zero layer-0 attention: `88.48%`;
- drop layer-1 head 0: `93.28%`;
- drop layer-1 heads `{0, 1}`: `86.19%`;
- keep only layer-0 head 2 plus layer-1 heads `{0, 1, 2}`:
  `95.69%`;
- keep only layer-0 head 2 plus layer-1 heads `{0, 1}`:
  `91.73%`;
- keep only layer-1 heads `{0, 1}` and no layer-0 attention:
  `85.20%`.

This supports the causal story but also shows the circuit is distributed: the
small candidate head set is highly informative but not fully sufficient for the
trained model's near-perfect behavior.

So the current circuit hypothesis is:

1. Layer 0 head 2 reads the leading boundary coefficient and writes a distributed
   leading-boundary feature into the residual stream.
2. Layer 1 head 0 reads the trailing boundary coefficient directly into CLS.
3. Layer 1 attention as a whole aggregates the boundary evidence at CLS.
4. The final MLP/head converts this boundary evidence into the descent logit.

This is a concrete circuit candidate, but it still needs two checks before it is
publication-grade: a corrected probe/decoder showing exactly which matrix-unit
column is represented in the relevant head outputs, and a circuit-derived
hand-coded classifier or model-edit that preserves the transformer's behavior.
The probe script is `interp/run_b3_semantic_probes.py`. The first probe run
sampled from too few shards and was discarded; the corrected all-shard ridge
probe job `13546782` completed successfully.

Corrected semantic probe highlights:

- `embed_cls` is at chance: `50.34%` label/column accuracy and `25.18%`
  four-way boundary-unit accuracy;
- `l0_resid_post_cls` linearly decodes label/column at `98.13%` and four-way
  leading/trailing unit token at `96.23%`;
- `l0h2_headout_cls` decodes label/column at `78.32%`, rows at `73.00%`, and
  four-way unit tokens at `60.53%`;
- `l0h2_headout_support_mean` decodes label/column at `77.28%` and four-way
  unit tokens at `64.39%`;
- `l1h0_z_cls` decodes label/column at `90.61%` and four-way unit tokens at
  `88.60%`;
- `l1h1_headout_cls` decodes label/column at `94.63%` and four-way unit tokens
  at `91.83%`;
- `l1_resid_post_cls` decodes four-way unit tokens at `99.95%`.

This supports the current mechanism: the model does not merely carry a scalar
label direction; by layer 1, the `CLS` stream linearly exposes nearly the full
extremal matrix-unit identity.
