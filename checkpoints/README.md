# Public checkpoints

This directory holds the model artifacts that belong in the public story.

## `original_mlp/`

This is the baseline MLP training run used for comparison in the writeup.

- tracked: `train.log`, `training_curves.png`
- headline validation factor accuracy: `0.7266`

The full raw MLP checkpoint is not checked in because the PyTorch file is too
large for a normal GitHub-friendly repository. The exact training recipe is
preserved in `jobs/train_original_mlp.sh`, and the baseline remains fully
reproducible from the tracked dataset and log.

## `best_transformer/`

This is the final public transformer.

- tracked checkpoint: `best_model.pt`
- tracked log: `train.log`
- tracked curves: `training_curves.png`
- comparison figure: `mlp_vs_transformer_validation.png`

Headline metrics:

- validation loss: `0.2178`
- validation factor accuracy: `0.9388`
