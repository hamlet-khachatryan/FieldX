#!/usr/bin/env bash
#SBATCH --job-name=cfi-free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
source "$(dirname "$0")/common.sh"
run_cfi device-report
run_cfi evaluate-free
