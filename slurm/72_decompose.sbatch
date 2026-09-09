#!/usr/bin/env bash
#SBATCH --job-name=fieldx-decompose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
# SLURM stages this script outside the repository, so "$0" must never be used
# to locate project files. Everything resolves from SLURM_SUBMIT_DIR.
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/slurm/common.sh"

# Read-only analysis: Gemmi densities, sparse linear algebra and FFTs. No accelerator is
# required, so this stage does not compete for GPU allocation and does not load CUDA.
# It is a leaf: the model lock does not depend on it, so a failure here cannot block the
# freeze. With decomposition.enabled=false the command returns immediately.
run_cfi decompose
