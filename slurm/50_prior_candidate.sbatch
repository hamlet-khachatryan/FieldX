#!/usr/bin/env bash
#SBATCH --job-name=fieldx-prior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -euo pipefail
# SLURM stages this script outside the repository, so "$0" must never be used
# to locate project files. Everything resolves from SLURM_SUBMIT_DIR.
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/slurm/common.sh"
fieldx_load_cuda

: "${CFI_PRIOR_MANIFEST:?Set CFI_PRIOR_MANIFEST to the prior-candidate manifest}"
: "${SLURM_ARRAY_TASK_ID:?This job must run as a SLURM array}"

# Each array task fits exactly one prior candidate on the train subset. Candidates are
# independent, so they parallelise across the array with no shared state.
CFI_CONFIG="$(run_uv python - "$CFI_PRIOR_MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json
import sys

manifest = json.loads(open(sys.argv[1]).read())
index = int(sys.argv[2])
if not 0 <= index < len(manifest):
    raise SystemExit(f"array index {index} out of range for {len(manifest)} candidates")
print(manifest[index]["config"])
PY
)"
export CFI_CONFIG
echo "candidate $SLURM_ARRAY_TASK_ID: $CFI_CONFIG"
run_cfi fit-map
