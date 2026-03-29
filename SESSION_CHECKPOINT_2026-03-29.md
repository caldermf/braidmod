# Session Checkpoint: 2026-03-29

This file is a handoff for the next developer working in `braidmod`.

## Current Goal

We reverted to the original transformer idea and tuned it for the best possible validation loss while keeping the model simple enough to present cleanly on GitHub.

The main conclusion so far is:

- Keep the hierarchical transformer.
- Drop the auxiliary right-descent head for the best final-factor loss.
- Do not use the true Garside length input.
- Do not use EMA for checkpoint selection or curve presentation.
- Select checkpoints by `val_loss`, not by `val_factor_acc`.

## Best Results So Far

### Original Transformer Baseline

Baseline log:

- `artifacts/garside_transformer_L30to60_p5_D140_N200000_e20/train.log`

Best early baseline point we were comparing against:

- epoch 6: `val_loss=0.2575`, `val_factor_acc=0.9181`

### Tuned Factor-Only Pilot (Completed)

Completed tuned run:

- `artifacts/garside_transformer_tuned_factoronly_noema_pilot_L30to60_p5_D140_N200000_e10/train.log`

Best completed pilot metrics:

- epoch 9: `val_loss=0.2357`, `val_factor_acc=0.9278`
- epoch 10: `val_loss=0.2358`, `val_factor_acc=0.9280`
- logged best: `best_val_loss=0.2357`

Best pilot checkpoints:

- `artifacts/garside_transformer_tuned_factoronly_noema_pilot_L30to60_p5_D140_N200000_e10/best_model.pt`
- `artifacts/garside_transformer_tuned_factoronly_noema_pilot_L30to60_p5_D140_N200000_e10/best_loss_model.pt`

### Tuned Factor-Only Base Run (Still Running)

Active Slurm job:

- job id: `6772086`

Live log:

- `artifacts/garside_transformer_tuned_factoronly_base_noema_L30to60_p5_D140_N200000_e30/train.log`

Latest checked metrics from the running job:

- epoch 10: `val_loss=0.2300`, `val_factor_acc=0.9307`
- epoch 11: `val_loss=0.2300`, `val_factor_acc=0.9333`

Important note:

- `best_model.pt` in this directory is mutable because the job is still running.
- I froze a snapshot at the time of handoff:
  - `artifacts/garside_transformer_tuned_factoronly_base_noema_L30to60_p5_D140_N200000_e30/best_model_epoch11_snapshot.pt`

## Winning Recipe

The best-performing simple recipe so far is:

- `--model-type transformer`
- `--task final_factor`
- `--d-model 256`
- `--ffn-mult 4`
- `--num-local-blocks 2`
- `--num-local-heads 4`
- `--num-global-blocks 6`
- `--num-global-heads 8`
- `--dropout 0.05`
- `--weight-decay 5e-3`
- `--label-smoothing 0.03`
- `--selection-objective loss`
- no EMA
- no Garside length
- keep `min_degree`

## What Was Tried And What Happened

### 1. Hierarchical Transformer Implementation

Files added / updated:

- `garside_transformer.py`
- `garside_models.py`
- `train_garside_mlp.py`
- `predict_garside_mlp.py`
- `train_transformer_L30to60_p5_D140_N200000_e20.sh`

This was a strong improvement over the MLP and became the base architecture.

### 2. Prefix-Confusion Evaluation Tooling

Files added / updated:

- `rescore_saved_prefix_suite.py`
- `run_transformer_prefix_gpu_eval.sh`
- `plot_prefix_confusion.py`
- `track_confusion_prefix.py`
- `render_kernel_random_xent_overlay.py`
- `reservoir_search_braidmod.py`

This produced the Geordie/random and kernel-hit overlay plots.

### 3. Garside-Length Conditioning Ablation

Files added / updated for the ablation:

- `train_transformer_with_length_L30to60_p5_D140_N200000_e20.sh`
- `run_transformer_with_length_prefix_gpu_eval.sh`
- `render_length54_confusion_suite_transformer_with_length.sh`

Conclusion:

- It did not help the target-cross-entropy proxy.
- It modestly improved raw entropy/confusion separation in some comparisons.
- We explicitly decided to abandon this direction for the main model.

### 4. Tuning Sweep

I tested several ideas:

- multitask + tuned regularization
- deeper transformer
- factor-only objective
- EMA

Conclusions:

- EMA made early validation curves misleading and was not worth keeping.
- The deeper model underperformed the base model at matched epochs and was canceled.
- The best simple gain came from switching to `task=final_factor` and tuning regularization / checkpoint selection.

## Code Changes Relevant To The Tuning/Handoff

### Trainer Improvements

Main file:

- `train_garside_mlp.py`

Key additions:

- `label_smoothing`
- `ema_decay`
- `selection_objective`
- separate optimization loss vs logged loss so train/val loss remain comparable
- saving `best_model.pt`, `best_loss_model.pt`, `best_metric_model.pt`
- writing `history.json`

Practical takeaway:

- For presentation and model selection, treat `val_loss` as the main quantity.

### Plotting Improvements

Files:

- `plot_training_curves.py`
- `plot_training_log_comparison.py`

These now produce decent-looking static plots for GitHub / README use.

## Important Artifacts

### Completed tuned pilot

- checkpoint:
  - `artifacts/garside_transformer_tuned_factoronly_noema_pilot_L30to60_p5_D140_N200000_e10/best_model.pt`
- curves:
  - `artifacts/garside_transformer_tuned_factoronly_noema_pilot_L30to60_p5_D140_N200000_e10/training_curves.png`
- old vs tuned:
  - `artifacts/garside_transformer_tuned_factoronly_noema_pilot_L30to60_p5_D140_N200000_e10/old_vs_tuned_comparison.png`

### Running stronger base run

- mutable best:
  - `artifacts/garside_transformer_tuned_factoronly_base_noema_L30to60_p5_D140_N200000_e30/best_model.pt`
- frozen snapshot:
  - `artifacts/garside_transformer_tuned_factoronly_base_noema_L30to60_p5_D140_N200000_e30/best_model_epoch11_snapshot.pt`
- current curves:
  - `artifacts/garside_transformer_tuned_factoronly_base_noema_L30to60_p5_D140_N200000_e30/training_curves_current.png`
- current old vs tuned:
  - `artifacts/garside_transformer_tuned_factoronly_base_noema_L30to60_p5_D140_N200000_e30/old_vs_tuned_comparison_current.png`

## Current Slurm State

Still running:

- `6772086` for `train_transformer_tuned_factoronly_base_L30to60_p5_D140_N200000_e30.sh`

Canceled because they lost:

- `6772087` deeper factor-only run
- earlier multitask tuned runs
- length-conditioned run was completed only as an ablation

## Recommended Next Steps

1. Monitor job `6772086` until it stops improving meaningfully.
2. Freeze the final best checkpoint with another immutable snapshot name once the job finishes.
3. Re-render the final polished training curves from the completed `e30` run.
4. Re-run the prefix confusion suite with the tuned factor-only checkpoint.
5. Compare old transformer vs tuned factor-only on:
   - Geordie vs random prefix x-ent
   - kernel-hit overlays
   - search behavior if needed
6. Update `README.md` once the final tuned run is chosen.

## Useful Commands For The Next Developer

Monitor the running job:

```bash
squeue -j 6772086
tail -f artifacts/garside_transformer_tuned_factoronly_base_noema_L30to60_p5_D140_N200000_e30/train.log
```

Render comparison plots after the run finishes:

```bash
/home/com36/.conda/envs/burau_gpu/bin/python plot_training_curves.py \
  --log artifacts/garside_transformer_tuned_factoronly_base_noema_L30to60_p5_D140_N200000_e30/train.log \
  --out artifacts/garside_transformer_tuned_factoronly_base_noema_L30to60_p5_D140_N200000_e30/training_curves_final.png \
  --title "Tuned Burau Transformer Curves (factor-only base, final)"

/home/com36/.conda/envs/burau_gpu/bin/python plot_training_log_comparison.py \
  --log old=artifacts/garside_transformer_L30to60_p5_D140_N200000_e20/train.log \
  --log tuned=artifacts/garside_transformer_tuned_factoronly_base_noema_L30to60_p5_D140_N200000_e30/train.log \
  --out artifacts/garside_transformer_tuned_factoronly_base_noema_L30to60_p5_D140_N200000_e30/old_vs_tuned_comparison_final.png \
  --title "Original Transformer vs Tuned Factor-Only Transformer"
```

Rescore the saved prefix suite with a tuned checkpoint:

```bash
CHECKPOINT_PATH=/nfs/roberts/project/pi_com36/com36/burau-experiments/braidmod/artifacts/garside_transformer_tuned_factoronly_base_noema_L30to60_p5_D140_N200000_e30/best_model.pt \
OUT_DIR=/nfs/roberts/project/pi_com36/com36/burau-experiments/braidmod/artifacts/length54_confusion_suite_transformer_tuned_factoronly \
./run_transformer_prefix_gpu_eval.sh
```

## Worktree Note

The git root is one level above `braidmod` and contains unrelated untracked work in `curvemod/`.

When committing, only stage the relevant `braidmod` files and this checkpoint note. Do not sweep in unrelated `curvemod` files.
