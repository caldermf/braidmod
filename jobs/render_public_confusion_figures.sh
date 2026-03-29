#!/usr/bin/env bash
# Compatibility wrapper for the old public confusion-figure entrypoint.
# The curated public figure set now lives behind render_public_story_figures.sh.

#SBATCH --job-name=braidmod-public-figures
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec bash "$REPO_ROOT/jobs/render_public_story_figures.sh"
