# B_3 Boundary-Slice Rule

This note records the exact boundary-slice criterion found in the `B_3`,
`p=2` corpus. It is a supporting research note for the public `B_3` report.

## Empirical Statement

For every generated `B_3` positive Garside-normal-form braid of length `25`,
the normalized reduced Burau matrix over `(Z/2)[v, v^{-1}]` has:

- a unique nonzero `2 x 2` coefficient entry at the minimum occupied degree;
- a unique nonzero `2 x 2` coefficient entry at the maximum occupied degree.

The descent label is determined exactly by either boundary slice.

Minimum-degree rule:

```text
label = {s_2} iff the unique nonzero leading coefficient lies in column 0.
label = {s_1} iff it lies in column 1.
```

Maximum-degree rule:

```text
label = {s_2} iff the unique nonzero trailing coefficient lies in column 1.
label = {s_1} iff it lies in column 0.
```

The full-corpus audit over `67,108,864` length-25 examples verifies both rules
with accuracy `1.0`.

The same rule has also been checked across shorter lengths: lengths `1` through
`19` were exhaustive, and length `20` was checked on a `1,048,576` example
sample. In every checked case, the leading and trailing boundary slices were
unit tokens and both rules had accuracy `1.0`.

## Why This Is Plausible

The four proper simple factors in `B_3` have leading and trailing Burau
coefficient matrices that are matrix units. With factor ids used by the dataset:

```text
0: s_2       leading E00, trailing E11, label {s_2}
1: s_1       leading E11, trailing E00, label {s_1}
2: s_1 s_2   leading E10, trailing E01, label {s_2}
3: s_2 s_1   leading E01, trailing E10, label {s_1}
```

Thus the leading matrix unit column encodes the factor's right descent:

```text
leading column 0 -> {s_2}
leading column 1 -> {s_1}
```

The trailing matrix unit encodes the same information with the opposite column
convention:

```text
trailing column 1 -> {s_2}
trailing column 0 -> {s_1}
```

For a product in `B_3` left normal form, the allowed transitions appear to keep
the extreme-degree matrix-unit products nonzero. The extreme coefficient of the
product is then the product of extreme coefficient matrix units, so its output
column is inherited from the final factor. This explains why the boundary slice
can recover the final descent.

## Proof Sketch

Write the leading coefficient matrix of a simple factor `f` as a matrix unit
`L(f) = E_{r(f), c(f)}`. The minimum-degree coefficient of a product
`f_1 ... f_L` is the product

```text
L(f_1) L(f_2) ... L(f_L),
```

provided this matrix-unit product is nonzero. Matrix units multiply by

```text
E_{a,b} E_{c,d} = E_{a,d} iff b = c,
E_{a,b} E_{c,d} = 0 otherwise.
```

For the four factor ids, the leading units are:

```text
0: s_2       E00, c=0
1: s_1       E11, c=1
2: s_1 s_2   E10, c=0
3: s_2 s_1   E01, c=1
```

The normal-form transition table is:

```text
0 -> {0, 3}
1 -> {1, 2}
2 -> {0, 3}
3 -> {1, 2}
```

For every allowed transition `f -> g`, the leading column of `f` equals the
leading row of `g`. Therefore the leading matrix-unit product is always
nonzero, and equals `E_{r(f_1), c(f_L)}`. Thus the column of the leading
coefficient of the whole Burau matrix is exactly the leading column of the
final factor.

The label is also exactly this column:

```text
final leading column 0 -> {s_2}
final leading column 1 -> {s_1}
```

The trailing-coefficient proof is the same. The trailing units are:

```text
0: s_2       E11, c=1
1: s_1       E00, c=0
2: s_1 s_2   E01, c=1
3: s_2 s_1   E10, c=0
```

Again, every allowed transition has trailing column of `f` equal to trailing row
of `g`, so the maximum-degree coefficient of the whole product is a nonzero
matrix unit whose column is the trailing column of the final factor. For the
trailing unit, the label convention is:

```text
final trailing column 1 -> {s_2}
final trailing column 0 -> {s_1}
```

There is no cancellation issue at the extreme degree: the minimum-degree term
comes only from multiplying minimum-degree terms, and the maximum-degree term
comes only from multiplying maximum-degree terms. Once those matrix-unit
products are nonzero, they are the unique extremal coefficients.

## Current Model Evidence

The first-pass transformer interp run found:

- keeping only both boundary tokens gives about `99.99%` accuracy;
- keeping only the leading boundary token gives about `63%` accuracy for the
  transformer, even though a discrete lookup using the leading token is exact;
- keeping only the trailing boundary token gives about `97%` accuracy for the
  transformer;
- zeroing both boundary tokens drives transformer accuracy close to chance;
- activation patching localizes much of the final decision to layer-1 CLS.

The off-manifold boundary-token flip experiment is not a clean causal proof:
flipping boundary token columns alone does not reliably flip either trained
model. That suggests the trained models are sensitive to manifold-consistent
boundary evidence, not arbitrary one-token edits.

The matched-support intervention gives a cleaner causal test. We paired
opposite-label examples with the same support interval (`first=13`, `last=63`)
and used the corrupt example as the base input. Replacing only the leading
boundary token recovered `63.87%` clean-label accuracy; replacing only the
trailing boundary token recovered `70.70%`; replacing both boundary tokens
recovered `100%` accuracy and a clean-side logit score.

For the transformer, activation patching on the same matched pairs shows the
main path:

- layer-0 `hook_resid_post`, both boundary tokens patched together:
  `0.745` normalized recovery;
- layer-0 `hook_resid_post`, CLS plus both boundary tokens:
  `0.891` normalized recovery;
- layer-1 `hook_resid_post`, CLS:
  `1.000` normalized recovery;
- layer-1 `hook_attn_out`, CLS:
  `0.974` normalized recovery;
- layer-1 `hook_mlp_out`, CLS:
  `0.547` normalized recovery.

This is consistent with a simple two-stage circuit: boundary-position features
are represented in token residual streams after layer 0, then layer 1 aggregates
that information into the `CLS` stream for the binary decision.

Head-level path patching gives a sharper candidate circuit:

- patching all layer-1 attention head outputs at CLS gives `0.974` normalized
  recovery on matched clean/corrupt pairs;
- patching layer-1 heads `{0, 1}` at CLS gives `0.790` recovery;
- patching layer-1 head 0 at CLS gives `0.582` recovery;
- layer-1 head 0's recovery is mostly the trailing boundary value path:
  `v_trailing_value_source` gives `0.452`, and patching the CLS pattern row
  plus that trailing value gives `0.520`;
- layer-0 head 2 carries leading-boundary information: patching its leading
  source value gives `0.331`, and patching its full projected output gives
  `0.360`.

The mechanistic hypothesis is therefore more specific than "the model uses a
boundary token": layer 0 head 2 broadcasts leading-boundary information, layer
1 head 0 reads trailing-boundary information into `CLS`, and the layer-1 `CLS`
stream combines these signals before the final classifier.

Semantic probes give independent evidence that the circuit is carrying the
mathematical object, not only an opaque class bit. On corrected all-shard train
and test samples, ridge probes from the constant initial CLS embedding are at
chance (`50.34%` label accuracy, `25.18%` four-way unit-token accuracy). In
contrast:

- `l0h2_headout_cls` decodes the label/boundary-column direction at `78.32%`
  and the four-way boundary unit token at `60.53%`;
- `l1h0_z_cls` decodes the label/boundary-column direction at `90.61%` and the
  four-way boundary unit token at `88.60%`;
- `l1h1_headout_cls` decodes the label/boundary-column direction at `94.63%`
  and the four-way boundary unit token at `91.83%`;
- `l1_resid_post_cls` decodes the four-way boundary unit token at `99.95%`.

Thus the late `CLS` representation contains an almost perfect linear encoding of
the extremal matrix-unit identity, which is stronger than merely encoding the
binary descent label.

## Next Checks

1. Determine whether the transformer actually computes the leading/trailing
   matrix-unit column, or whether it uses a correlated absolute-degree shortcut.
2. Train probes or build explicit decoders for matrix-unit column/row inside
   the relevant head outputs.
3. Compare the transformer circuit against the stronger one-hidden-layer MLP,
   which reaches higher held-out accuracy on the same task.
4. Train a deliberately smaller or more constrained transformer that must use
   the boundary rule more transparently.
