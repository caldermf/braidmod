# braidmod

Small research repo for generating Burau-mod-p braid data in `B_4` and training an MLP to predict the final Garside factor or its right descent set.

The code now uses the reduced Burau normalization
`sigma_1 -> [[-v^2, -v], [0, 1]]`,
`sigma_{n-1} -> [[1, 0], [-v, -v^2]]`,
and
`sigma_i -> [[1, 0, 0], [-v, -v^2, -v], [0, 0, 1]]`
for `1 < i < n - 1`.
Older datasets and artifacts generated before this correction are not compatible with the current code unless they are regenerated.

## What's here

- `braid_data.py`: Garside-factor utilities, GNF validation, Burau polynomial/tensor evaluation, and random dataset generation.
- `train_garside_mlp.py`: training entry point for `final_factor`, `right_descent`, or `multitask`.
- `predict_garside_mlp.py`: inference CLI for trained checkpoints.
- `plot_training_curves.py`: plot loss and task metric from training logs.
- `data/`: local JSON datasets.

## Environment

Install the ML dependencies into a local virtualenv:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-ml.txt
```

## Dataset format

Training expects a JSON list of records with:

- `burau_tensor`: integer tensor shaped `[D, 3, 3]`
- `final_factor_perm`: permutation of `0..3`
- `final_factor_right_descent`: subset of `{0, 1, 2}`

`DataSetBuilder` in `braid_data.py` can generate compatible samples.

## Train

Example training run on the checked-in `p=5`, `D=96` dataset:

```bash
.venv/bin/python train_garside_mlp.py \
  --data-path data/burau_gnf_L20_p5_D96_N20000.json \
  --p 5 \
  --task multitask \
  --out-dir artifacts
```

Tasks:

- `final_factor`: predict the final Garside factor permutation
- `right_descent`: predict the right descent set only
- `multitask`: predict the final factor with an auxiliary right-descent loss

Checkpoints include `p` and `D`; inference inputs must match both.

## Predict

Run inference on a record from a dataset:

```bash
.venv/bin/python predict_garside_mlp.py \
  --checkpoint artifacts/best_model.pt \
  --dataset-path data/burau_gnf_L20_p5_D96_N20000.json \
  --index 0
```

Or pass a JSON file containing either `[D, 3, 3]` directly or `{"burau_tensor": ...}`.

## Plot logs

The trainer prints lines like:

```text
epoch=001 train_loss=... train_factor_acc=... val_loss=... val_factor_acc=... lr=...
```

Plot them with:

```bash
.venv/bin/python plot_training_curves.py --log train.log --out training_curves.png
```

The plotter supports both older `*_acc` logs and newer task-specific metrics such as `factor_acc` and `desc_exact`.

## Repo conventions

- Treat `artifacts/`, `tmp_artifacts/`, and `tmp_artifacts_rd/` as generated outputs.
- The large `N100000` dataset is intentionally ignored.
- This directory is a prototype workspace, not a packaged library.

## Yale Bouchet / Slurm instructions for Codex

This repo often runs on the Yale YCRC Bouchet cluster. Follow these rules exactly.

### Current Bouchet partitions

Public partitions:

- `day`
- `devel`
- `week`
- `gpu`
- `gpu_h200`
- `gpu_devel`
- `bigmem`
- `mpi`
- `scavenge`
- `scavenge_gpu`

Priority partitions:

- `priority`
- `priority_gpu`
- `priority_mpi`

Private partition currently listed on Bouchet docs:

- `pi_co54`

### Core policy

1. Always pass `--partition` explicitly.
   Never rely on defaults.
   On Yale, omitted partitions can land you somewhere unintended.
   For this repo, the defaults are:
   CPU jobs: `--partition=scavenge`
   GPU jobs: `--partition=scavenge_gpu`

2. Prefer scavenge partitions almost always.
   Default to `scavenge` or `scavenge_gpu`.
   Do not use `day`, `week`, `gpu`, `gpu_devel`, `bigmem`, `mpi`, `priority*`, or `pi_*` unless the human explicitly asks.

3. GPU jobs must request GPUs explicitly.
   Always pass `--gpus=...` for GPU work.
   Default GPU request is usually:
   `--gpus=1`
   A generic GPU request is fine.

4. Do not explicitly request H200 unless the human explicitly tells you to.
   Do not use:
   `--gpus=h200:...`
   `--partition=gpu_h200`
   `--constraint=h200`
   A generic request like `--gpus=1` on `scavenge_gpu` is fine, even if Slurm happens to place the job on an H200.
   The goal is to never explicitly ask for H200s unless instructed.

5. Prefer short walltimes.
   Calder prefers requesting about 1.5x the estimated runtime, not giant padded requests.
   Example: if something should take about 25-30 minutes, ask for about 40-45 minutes, not 24 hours.
   Do not request long times "just in case" unless explicitly told.

6. Use scavenge with good hygiene.
   `scavenge` and `scavenge_gpu` are preemptable.
   If the code can restart safely, prefer:
   `--requeue`
   `--signal=B:USR1@120`
   Save checkpoints regularly.
   Write outputs incrementally and flush logs.

7. Do not use `--mem-per-gpu`.
   Use `--mem=...` or `--mem-per-cpu=...` instead.

8. Use a real Python executable, not assumptions.
   Some jobs need an explicit Python path.
   If the repo or human provides a specific path like `PYTHON_PATH="/home/com36/.conda/envs/burau_gpu/bin/python"`, then use that exact path.
   Do not replace a known working explicit Python path with bare `python` unless the human says to.
   If no explicit path is provided, then load the expected environment and resolve Python deliberately.

9. Do not do heavy setup on the login node.
   If work is nontrivial, get an allocation with `salloc` or submit with `sbatch`.

10. Check submitted jobs proactively.
    After `sbatch`, capture the job ID.
    Check queue state with:
    `squeue --me`
    and/or `squeue -j JOBID`
    Inspect logs early after launch:
    `tail -n 50 slurm_logs/<file>`
    or `tail -f slurm_logs/<file>`
    Confirm the job actually started correctly and is producing expected output.
    If the job is pending, read the reason field from `squeue`.
    If the job is running, make sure logs are advancing and the process is behaving sensibly.

11. For Yale monitoring/debugging, prefer the usual cluster tools.
    `squeue --me` to see current jobs
    `jobstats JOBID` to inspect CPU, memory, and GPU utilization
    `sacct -j JOBID --duplicates` if requeues or preemptions matter
    `htnm` before longer jobs to see time until next maintenance

12. Avoid idle GPU allocations.
    Do not leave jobs sitting on GPUs while doing nothing.
    Start the actual compute quickly.
    If a job is meant to use GPUs, logs and utilization should reflect that.

### Default command patterns

#### CPU interactive

```bash
salloc \
  --partition=scavenge \
  --time=00:45:00 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=16G
```
