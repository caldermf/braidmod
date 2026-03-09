# Bouchet SLURM Reference

This reference captures the Bouchet and repo-specific SLURM guidance from [README.md](../../../README.md).

## Partitions

Public partitions listed in the repo:

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

Private partition currently listed in the repo:

- `pi_co54`

## Core policy

1. Always pass `--partition` explicitly.
   Use `scavenge` for CPU jobs and `scavenge_gpu` for GPU jobs unless the human explicitly requests something else.

2. Prefer scavenge partitions almost always.
   Do not choose `day`, `week`, `gpu`, `gpu_devel`, `bigmem`, `mpi`, `priority*`, or `pi_*` unless the human explicitly asks.

3. GPU jobs must request GPUs explicitly.
   Default to `--gpus=1` when no more specific request is provided.

4. Do not explicitly request H200s unless the human explicitly tells you to.
   Avoid `--gpus=h200:...`, `--partition=gpu_h200`, and `--constraint=h200` by default.

5. Prefer short walltimes.
   Use roughly 1.5x the expected runtime, not large padded requests.

6. Treat `scavenge` and `scavenge_gpu` as preemptable.
   Prefer `--requeue` and `--signal=B:USR1@120` when the job can resume safely.
   Encourage checkpointing and incremental writes.

7. Do not use `--mem-per-gpu`.
   Use `--mem=...` or `--mem-per-cpu=...`.

8. Use a real Python executable, not assumptions.
   If the repo or human gives an explicit Python path, keep using it.

9. Do not do heavy setup on the login node.
   For nontrivial work, obtain an allocation with `salloc` or submit with `sbatch`.

10. Check submitted jobs proactively.
    Capture the job ID, inspect `squeue`, and watch logs early.

11. Preferred monitoring/debugging commands:
    `squeue --me`
    `jobstats JOBID`
    `sacct -j JOBID --duplicates`
    `htnm`

12. Avoid idle GPU allocations.
    Start real compute quickly and confirm logs or utilization reflect GPU work.

## Default command patterns

CPU interactive pattern from the repo:

```bash
salloc \
  --partition=scavenge \
  --time=00:45:00 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=16G
```

GPU interactive pattern inferred from the repo policy:

```bash
salloc \
  --partition=scavenge_gpu \
  --time=00:45:00 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=16G \
  --gpus=1
```

Use the inferred GPU pattern only when the human requests GPU work and has not provided a different partition or GPU count.
