#!/usr/bin/env bash
#SBATCH --job-name=cfi-finalfit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
source "$(dirname "$0")/common.sh"
run_cfi device-report
run_cfi derivative-check
run_cfi fit-map
