#!/usr/bin/env bash
#SBATCH --job-name=cfi-info
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
source "$(dirname "$0")/common.sh"
run_cfi device-report
run_cfi information-spectrum
