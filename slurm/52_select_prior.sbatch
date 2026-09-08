#!/usr/bin/env bash
#SBATCH --job-name=fieldx-select
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

: "${CFI_PRIOR_MANIFEST:?Set CFI_PRIOR_MANIFEST}"
: "${CFI_SELECTED_CONFIG:?Set CFI_SELECTED_CONFIG to the output path for the selected config}"
fieldx_require_config

# Candidates are ranked on the tune subset only. The free set is not read here.
run_uv fieldrefine select-prior "$CFI_CONFIG" "$CFI_PRIOR_MANIFEST" "$CFI_SELECTED_CONFIG"
