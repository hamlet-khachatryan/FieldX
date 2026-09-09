#!/usr/bin/env bash
#
# Submit the FieldX v3 pipeline for one dataset.
#
#     scripts/submit.sh CONFIG.yaml [PRIOR_GRID.yaml]
#
# The dependency chain ends at the model freeze. The free-set evaluation is never
# submitted automatically; the exact command is printed at the end instead.
#
# Site options without editing job files:
#     export CFI_SBATCH_ARGS="--account=MYACCOUNT --partition=MYGPU"
#     export CFI_ARRAY_LIMIT=4

set -euo pipefail

usage() {
  echo "Usage: scripts/submit.sh CONFIG.yaml [PRIOR_GRID.yaml]" >&2
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../slurm/common.sh
source "$PROJECT_ROOT/slurm/common.sh"

CONFIG="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
[[ -f "$CONFIG" ]] || { echo "Configuration not found: $1" >&2; exit 2; }

# The prior grid is dataset-specific and mandatory. There is deliberately no fallback:
# silently reusing another dataset's grid would silently change the experiment.
if [[ $# -eq 2 ]]; then
  GRID="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
else
  GRID="$(dirname "$CONFIG")/prior_grid.yaml"
fi
if [[ ! -f "$GRID" ]]; then
  echo "Prior grid not found: $GRID" >&2
  echo "Every dataset needs its own prior grid. Create one, or regenerate it with:" >&2
  echo "  uv run fieldrefine init-pdb <PDBID> --force" >&2
  exit 2
fi

command -v sbatch >/dev/null 2>&1 || { echo "sbatch not found; run this on a SLURM submission host." >&2; exit 2; }

# Lightweight work only on the submission host: validate, then expand the prior grid.
run_uv fieldrefine config-check "$CONFIG" >/dev/null
RUN_ROOT="$(run_uv python - "$CONFIG" <<'PY'
import sys

from crystal_field.config import load_config

print(load_config(sys.argv[1]).run.output_dir)
PY
)"
LOGS="$RUN_ROOT/logs"
mkdir -p "$LOGS"

CAND_DIR="$RUN_ROOT/prior_configs"
run_uv fieldrefine expand-priors "$CONFIG" "$GRID" "$CAND_DIR" >/dev/null
MANIFEST="$CAND_DIR/manifest.json"
N="$(run_uv python - "$MANIFEST" <<'PY'
import json
import sys

print(len(json.loads(open(sys.argv[1]).read())))
PY
)"
(( N > 0 )) || { echo "No prior candidates in $GRID" >&2; exit 2; }
SELECTED="$RUN_ROOT/selected.yaml"

read -r -a EXTRA_SBATCH <<< "${CFI_SBATCH_ARGS:-}"
BASE_ENV="ALL,CFI_CONFIG=$CONFIG,CFI_PROJECT_ROOT=$PROJECT_ROOT"
ARRAY_ENV="ALL,CFI_PROJECT_ROOT=$PROJECT_ROOT,CFI_PRIOR_MANIFEST=$MANIFEST"
SELECT_ENV="$BASE_ENV,CFI_PRIOR_MANIFEST=$MANIFEST,CFI_SELECTED_CONFIG=$SELECTED"
FINAL_ENV="ALL,CFI_CONFIG=$SELECTED,CFI_PROJECT_ROOT=$PROJECT_ROOT"

# sbatch is invoked from PROJECT_ROOT so that SLURM_SUBMIT_DIR is the repository, which
# is how every job locates slurm/common.sh.
cd "$PROJECT_ROOT"
submit() {
  local job="$1"; shift
  sbatch "${EXTRA_SBATCH[@]}" --parsable \
    --output="$LOGS/slurm-%x-%j.out" --error="$LOGS/slurm-%x-%j.err" \
    "$@" "$PROJECT_ROOT/slurm/$job"
}

j_inspect=$(submit  00_inspect.sbatch            --export="$BASE_ENV")
j_prepare=$(submit  10_prepare.sbatch            --export="$BASE_ENV" --dependency=afterok:"$j_inspect")
j_rho0=$(submit     20_make_rho0.sbatch          --export="$BASE_ENV" --dependency=afterok:"$j_prepare")
j_scaletr=$(submit  25_scaling_train.sbatch      --export="$BASE_ENV" --dependency=afterok:"$j_rho0")
j_numerics=$(submit 30_numerics.sbatch           --export="$BASE_ENV" --dependency=afterok:"$j_scaletr")
j_gputest=$(submit  40_gpu_tests.sbatch          --export="$BASE_ENV" --dependency=afterok:"$j_numerics")
j_atomic=$(submit   45_atomic_benchmark.sbatch   --export="$BASE_ENV" --dependency=afterok:"$j_gputest")

LIMIT="${CFI_ARRAY_LIMIT:-4}"
j_priors=$(sbatch "${EXTRA_SBATCH[@]}" --parsable \
  --output="$LOGS/slurm-%x-%A_%a.out" --error="$LOGS/slurm-%x-%A_%a.err" \
  --export="$ARRAY_ENV" --array="0-$((N-1))%$LIMIT" --dependency=afterok:"$j_atomic" \
  "$PROJECT_ROOT/slurm/50_prior_candidate.sbatch")

# afterok on the array job id waits for every task in the array.
j_select=$(submit   52_select_prior.sbatch       --export="$SELECT_ENV"  --dependency=afterok:"$j_priors")
j_scalewk=$(submit  55_scaling_work.sbatch       --export="$FINAL_ENV"   --dependency=afterok:"$j_select")
j_final=$(submit    60_final_fit.sbatch          --export="$FINAL_ENV"   --dependency=afterok:"$j_scalewk")
j_info=$(submit     70_information.sbatch        --export="$FINAL_ENV"   --dependency=afterok:"$j_final")
# Read-only analysis, parallel to the information spectrum. A leaf: the freeze does not
# depend on it, so a decomposition failure cannot block the model lock.
j_decompose=$(submit 72_decompose.sbatch         --export="$FINAL_ENV"   --dependency=afterok:"$j_final")
# The freeze records the fit and the information spectrum, so it waits for both.
j_freeze=$(submit   75_freeze_model.sbatch       --export="$FINAL_ENV"   --dependency=afterok:"$j_final":"$j_info")

cat <<SUMMARY
Submitted FieldX v3 pipeline for $CONFIG
  prior grid        $GRID ($N candidates, array limit $LIMIT)
  run root          $RUN_ROOT
  logs              $LOGS

  00 inspect              $j_inspect
  10 prepare              $j_prepare
  20 rho0 + solvent       $j_rho0
  25 scaling (train)      $j_scaletr
  30 numerics gates       $j_numerics
  40 GPU tests            $j_gputest
  45 atomic benchmark     $j_atomic
  50 prior array          $j_priors
  52 select prior         $j_select
  55 scaling (work)       $j_scalewk
  60 final fit            $j_final
  70 information          $j_info
  72 decomposition        $j_decompose
  75 freeze model         $j_freeze

Monitor with:
  squeue -u "\$USER"
  sacct -u "\$USER" --starttime today

The free set is untouched until you run the one-shot evaluation BY HAND, after
$j_freeze has written MODEL_LOCK.json:

  cd $PROJECT_ROOT
  sbatch ${CFI_SBATCH_ARGS:-} --export=$FINAL_ENV $PROJECT_ROOT/slurm/80_evaluate_free.sbatch
SUMMARY
