# Best transformer

This directory contains the final public transformer artifacts:

- `best_model.pt`
- `train.log`
- `training_curves.png`
- `mlp_vs_transformer_validation.png`

Headline metrics:

- validation loss: `0.2178`
- validation factor accuracy: `0.9388`

The training corpus used for this run is intentionally not tracked in GitHub.
Regenerate it locally under `data/generated/` with `jobs/build_reference_dataset.sh`.
