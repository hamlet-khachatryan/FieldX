#!/usr/bin/env bash
#SBATCH --job-name=cfi-prior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -euo pipefail
: "${CFI_PRIOR_MANIFEST:?Set CFI_PRIOR_MANIFEST}"
: "${SLURM_ARRAY_TASK_ID:?This job must run as a SLURM array}"
: "${CFI_UV_ENV:?Set CFI_UV_ENV}"
PROJECT_ROOT="${CFI_PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export UV_PROJECT_ENVIRONMENT="$CFI_UV_ENV"
CFI_CONFIG="$(uv run --project "$PROJECT_ROOT" --frozen --no-sync python - "$CFI_PRIOR_MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1]))
print(manifest[int(sys.argv[2])]["config"])
PY
)"
export CFI_CONFIG CFI_PROJECT_ROOT
source "$(dirname "$0")/common.sh"
run_cfi fit-map
