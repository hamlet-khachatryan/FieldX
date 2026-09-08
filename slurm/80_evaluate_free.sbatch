#!/usr/bin/env bash
#SBATCH --job-name=fieldx-free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
# SLURM stages this script outside the repository, so "$0" must never be used
# to locate project files. Everything resolves from SLURM_SUBMIT_DIR.
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/slurm/common.sh"
fieldx_load_cuda
# MANUAL, ONE-SHOT. Never submitted automatically by scripts/submit.sh.
# Requires MODEL_LOCK.json and consumes the held-out free set exactly once.
run_cfi device-report
run_cfi evaluate-free
