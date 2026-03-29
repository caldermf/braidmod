# Original MLP baseline

This directory contains the public baseline training artifacts:

- `train.log`
- `training_curves.png`

Headline validation factor accuracy: `0.7266`.

The raw PyTorch checkpoint is not tracked here because the file is too large for
the intended public GitHub layout. Reproduce it with `jobs/train_original_mlp.sh`
after generating the reference dataset under `data/generated/`.
