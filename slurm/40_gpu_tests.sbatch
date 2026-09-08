#!/usr/bin/env bash
#SBATCH --job-name=cfi-gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
source "$(dirname "$0")/common.sh"
run_uv pytest -m gpu tests/gpu
