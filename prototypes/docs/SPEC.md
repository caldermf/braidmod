# Transformer Design Spec for Polynomial-Matrix -> Final Garside Word

## Purpose

Design and implement a **hierarchical transformer encoder** for inputs that are polynomial matrices over \(\mathbf F_p\), represented as a tensor of shape:

```python
(B, L, M, M)
```

where:
- `B` = batch size
- `L` = number of degree slices kept for the polynomial
- `M` = matrix dimension (often `3`, but the code should be written so that `M` is configurable)
- entries are integers in `{0, ..., p-1}` and should be treated as **discrete tokens**, not continuous scalars

The model output is the **final Garside word**. You can assume the dataset/output-side machinery already exists; focus on implementing the transformer architecture cleanly and correctly.

---

## High-level design

The model should be **hierarchical**, with two levels of reasoning:

1. **Local encoder over each degree slice**
   - Each degree slice is an `M x M` coefficient matrix.
   - The model should read the entries within a single matrix and summarize them into one vector.
   - The **same local encoder weights must be shared across all degrees**.

2. **Global encoder over the polynomial sequence**
   - After summarizing each degree slice into one vector, run a transformer over the sequence of degree summaries.
   - This stage captures interactions across degrees.

This is **not** a causal language model. Use a **bidirectional transformer encoder** at both local and global stages.

---

## Core architectural choices

### 1. Input tokenization and embeddings

The raw input should stay simple. The dataloader should just return integer tensors of shape `(B, L, M, M)` plus labels/masks as needed.

The **model** should construct embeddings internally.

For each coefficient entry `x[b, k, i, j]`, create an embedding by summing:

- **value embedding**: embedding lookup for the coefficient value in `{0, ..., p-1}`
- **row embedding**: embedding for row index `i`
- **column embedding**: embedding for column index `j`
- **local degree embedding**: embedding for the degree index `k`

So the first embedding block should conceptually implement:

```python
h[b, k, i, j] = E_val[x[b, k, i, j]] + E_row[i] + E_col[j] + E_deg_local[k]
```

Do **not** use a CNN here. The matrix entries are not image patches; the row/column positions have algebraic meaning.

### 2. Matrix position encoding

Use **row embeddings + column embeddings**, not just a flattened slot id from `0` to `M*M-1`.

This bakes in the fact that the local object is a matrix, not merely a short sequence.

### 3. Shared local encoder across degree

The local encoder should be identical for every degree slice. In other words, the model should learn one reusable way of reading an `M x M` coefficient matrix.

Degree-specific behavior should come from the **degree embedding**, not from using different local weights per degree.

### 4. CLS-based summarization

Within each degree slice, prepend a learned **local CLS token** before applying the local transformer.

After the local transformer finishes, use the output corresponding to the local CLS token as the summary vector for that degree.

Then prepend a learned **global CLS token** before the global transformer. The output corresponding to the global CLS token is the representation used by the final prediction head.

---

## Recommended baseline hyperparameters

Use these as the initial implementation defaults:

- `d_model = 256`
- `ffn_mult = 4`
- **local transformer**:
  - `num_local_blocks = 2`
  - `num_local_heads = 4`
- **global transformer**:
  - `num_global_blocks = 6`
  - `num_global_heads = 8`
- dropout: `0.05` or `0.1`
- pre-norm transformer blocks
- GELU or SwiGLU feedforward

These values should be configurable via a clean config object or dataclass.

---

## Tensor flow and shapes

Assume the input tensor is:

```python
x: (B, L, M, M)   # dtype torch.long
```

### Stage A: embedding the raw polynomial matrix

1. Value embedding:

```python
val = val_emb(x)   # (B, L, M, M, d_model)
```

2. Add row/column/local-degree embeddings by broadcasting.

Result:

```python
h: (B, L, M, M, d_model)
```

3. Flatten the matrix positions inside each degree slice:

```python
h = h.view(B, L, M*M, d_model)
```

Now each degree slice has `M*M` local tokens.

---

### Stage B: local transformer over one degree slice

1. Prepend a learned local CLS token to each degree slice:

```python
h_local: (B, L, 1 + M*M, d_model)
```

2. Reshape to batch the degree slices together, so the same local encoder is shared:

```python
h_local = h_local.view(B * L, 1 + M*M, d_model)
```

3. Apply the local transformer encoder (shared weights):

```python
h_local_out: (B * L, 1 + M*M, d_model)
```

4. Extract the local CLS token:

```python
deg_tokens = h_local_out[:, 0, :]   # (B * L, d_model)
deg_tokens = deg_tokens.view(B, L, d_model)
```

This yields one summary vector per degree.

---

### Stage C: global transformer over degree sequence

1. Add a global degree embedding (separate from the local degree embedding if desired).

2. Prepend a learned global CLS token:

```python
g: (B, L + 1, d_model)
```

3. Apply the global transformer encoder:

```python
g_out: (B, L + 1, d_model)
```

4. Extract the global CLS token:

```python
poly_repr = g_out[:, 0, :]   # (B, d_model)
```

5. Feed `poly_repr` into the prediction head for final Garside word output.

---

## Detailed implementation requirements

### Module breakdown

Please structure the code into clearly separated modules.

Suggested components:

- `PolynomialMatrixEmbedder`
  - takes `(B, L, M, M)` integer tensor
  - returns `(B, L, M*M, d_model)` local token embeddings
  - owns `val_emb`, `row_emb`, `col_emb`, `deg_emb_local`

- `TransformerBlock`
  - standard pre-norm transformer encoder block
  - multi-head self-attention + feedforward + residuals
  - usable for both local and global stages

- `LocalMatrixEncoder`
  - prepends local CLS
  - reshapes `(B, L, M*M, d)` to `(B*L, 1+M*M, d)`
  - applies `num_local_blocks` shared transformer blocks
  - returns `(B, L, d)`

- `GlobalPolynomialEncoder`
  - adds global degree embeddings
  - prepends global CLS
  - applies `num_global_blocks`
  - returns `(B, d)`

- `PolynomialMatrixTransformer`
  - top-level model that combines embedder + local encoder + global encoder + head

### Attention mask support

Support variable-length polynomial sequences via a degree mask.

If some examples are padded to a common `L_max`, the model should accept a mask like:

```python
degree_mask: (B, L)
```

where `True` / `1` means valid degree and `False` / `0` means padding.

Requirements:
- local stage: padding only matters if padded degrees are present; it is acceptable to zero out or mask out padded degree summaries before the global stage
- global stage: attention must ignore padded degrees
- the global CLS token should remain active

Please implement masking cleanly rather than assuming every sample has the same true polynomial length.

### Data types

- input coefficients should be `torch.long`
- embeddings are learned
- coefficients should **not** be converted to floats before embedding lookup

### Efficiency

Implementation should be straightforward and readable, but not sloppy.

Recommended pattern:
- embed once
- use broadcasting for row/column/degree embeddings
- batch local slices together via shape `(B*L, local_seq_len, d_model)`
- avoid Python loops over degree where possible

---

## Design principles to preserve

These are important. Do not drift away from them.

### Preserve the hierarchy

The point of the architecture is that there are **two natural scales**:
- local interactions within one coefficient matrix
- global interactions across degrees

Do not collapse this into a single flat transformer over all `(degree, row, col)` tokens unless explicitly asked later.

### Preserve weight sharing in the local stage

The same local encoder should read every degree slice. This is deliberate.

Do not create separate attention weights for degree `k=0`, `k=1`, etc.

### Keep positional structure meaningful

Use:
- row embedding
- column embedding
- degree embedding

Do not use a CNN or image-style locality assumptions.

### Treat coefficients as discrete tokens

The coefficients are elements of `F_p` represented as integers. Use an embedding table.

Do not treat them as ordinary scalar features unless explicitly experimenting with an ablation.

---

## Recommended prediction head

You can assume the output-side logic for the final Garside word already exists.

Still, the model should expose a clean final representation:

```python
poly_repr: (B, d_model)
```

and then hand this to a configurable prediction head.

If useful, structure the code so the head can be swapped between:
- classification-style output
- sequence decoder head
- custom existing Garside word head

But the trunk architecture above is the priority.

---

## Ablations worth keeping easy

Please make it easy to test the following with minimal code changes:

1. **No local transformer**
   - flatten each `M x M` degree slice and map directly with an MLP

2. **Mean pooling instead of local CLS**
   - after local blocks, average over the non-CLS tokens

3. **Flat slot embeddings instead of row+column embeddings**
   - useful as an ablation, but not the default

4. **Smaller / larger widths**
   - e.g. `d_model = 192`, `256`, `384`

Please keep the implementation modular enough that these are easy to try.

---

## Suggested coding style

- Use PyTorch
- Use a config dataclass for model hyperparameters
- Write clean docstrings with shapes in the comments
- Keep tensor reshaping explicit and readable
- Avoid magical shape assumptions hidden deep in the code
- Add a small smoke test / example forward pass in the file or test suite

---

## Concrete target behavior

The finished model should:

1. accept an integer tensor of shape `(B, L, M, M)`
2. embed coefficients + row/column/degree information inside the model
3. summarize each degree slice with a shared local transformer
4. summarize the polynomial with a global transformer
5. expose the final representation for predicting the final Garside word

---

## One-line summary

Implement a **hierarchical bidirectional transformer** for polynomial matrices over `F_p`, with:
- discrete coefficient embeddings
- row/column/local-degree embeddings
- shared local transformer blocks over each `M x M` degree slice
- CLS-based per-degree summaries
- a global transformer over the degree sequence
- a final head for the Garside word output

