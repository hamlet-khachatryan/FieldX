#!/usr/bin/env bash
#SBATCH --job-name=fieldx-freeze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
# SLURM stages this script outside the repository, so "$0" must never be used
# to locate project files. Everything resolves from SLURM_SUBMIT_DIR.
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/slurm/common.sh"

# Hashes the configuration, prepared split, starting density, work scaling, fitted
# latent field and information spectrum into MODEL_LOCK.json. This is the last stage
# the automated pipeline runs; the free set is evaluated manually afterwards.
run_cfi freeze-model
