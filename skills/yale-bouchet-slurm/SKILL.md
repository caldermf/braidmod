---
name: yale-bouchet-slurm
description: Prepare, review, or debug SLURM commands and job scripts for the Yale YCRC Bouchet cluster, especially when Codex is asked to request allocations, submit jobs, choose partitions, monitor runs, or avoid login-node misuse. Use when work mentions Bouchet, YCRC, SLURM, sbatch, salloc, srun, partitions, scavenge, scavenge_gpu, gpu_devel, or cluster job hygiene for this repo.
---

# Yale Bouchet SLURM

Use this skill for Bouchet-specific job planning and command generation.

Read [references/bouchet-slurm.md](references/bouchet-slurm.md) before proposing or editing SLURM commands. It contains the repo's cluster policy and default choices.

## Follow these defaults

- Always pass `--partition` explicitly.
- Default CPU work to `--partition=scavenge`.
- Default GPU work to `--partition=scavenge_gpu --gpus=1`.
- Do not request H200-specific resources unless the human explicitly asks.
- Prefer short walltimes, roughly 1.5x the estimated runtime.
- Do not use `--mem-per-gpu`; use `--mem` or `--mem-per-cpu`.
- Do not replace an explicitly provided Python path with bare `python`.

## Choose the execution mode

- Use `salloc` for nontrivial interactive work.
- Use `sbatch` for longer or restartable runs.
- Avoid heavy setup on the login node.
- If the user explicitly asks for a specific partition such as `gpu_devel`, honor that request.

## Preemptable partition hygiene

- Treat `scavenge` and `scavenge_gpu` as preemptable.
- Prefer `--requeue` and `--signal=B:USR1@120` when the workload can resume safely.
- Encourage checkpointing, incremental outputs, and log flushing for longer jobs.

## After submission

- Capture the job ID after `sbatch`.
- Check state with `squeue --me` or `squeue -j JOBID`.
- Inspect logs early with `tail -n 50` or `tail -f`.
- Use `jobstats JOBID` for utilization, `sacct -j JOBID --duplicates` for requeues, and `htnm` before longer jobs.

## Response style

- Give the exact command or job script the human can run.
- Keep Bouchet-specific reasoning explicit when rejecting an unsafe partition, long walltime, login-node setup, or H200-specific request.
- If you infer a value from repo policy rather than an explicit user request, say so briefly.
