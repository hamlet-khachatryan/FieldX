#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?Usage: scripts/submit.sh CONFIG.yaml [PRIOR_GRID.yaml]}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$(realpath "$CONFIG")"
GRID="${2:-$(dirname "$CONFIG")/prior_grid.yaml}"
[[ -f "$GRID" ]] || GRID="$ROOT/configs/6o2h/prior_grid.yaml"
: "${CFI_UV_ENV:?Run scripts/bootstrap_cluster.sh and source environment/cluster.env}"
export UV_PROJECT_ENVIRONMENT="$CFI_UV_ENV"

readarray -t CFG_META < <(PYTHONPATH="$ROOT/src" uv run --project "$ROOT" --frozen --no-sync python - "$CONFIG" <<'PY'
import sys
from pathlib import Path
from crystal_field.config import load_config
cfg = load_config(Path(sys.argv[1]))
print(cfg.run.output_dir.resolve())
print(cfg.run.data_dir.resolve())
print(cfg.input.model.resolve())
PY
)
RUN_ROOT="${CFG_META[0]}"
mkdir -p "$RUN_ROOT"
CAND_DIR="$RUN_ROOT/prior_configs"
uv run --project "$ROOT" --frozen --no-sync cfi expand-priors "$CONFIG" "$GRID" "$CAND_DIR" >/dev/null
MANIFEST="$CAND_DIR/manifest.json"
N="$(uv run --project "$ROOT" --frozen --no-sync python - "$MANIFEST" <<'PY'
import json, sys
print(len(json.loads(open(sys.argv[1]).read())))
PY
)"
(( N > 0 )) || { echo "No prior candidates" >&2; exit 2; }
SELECTED="$RUN_ROOT/selected.yaml"
read -r -a EXTRA_SBATCH <<< "${CFI_SBATCH_ARGS:-}"
EXPORT_BASE="ALL,CFI_CONFIG=$CONFIG,CFI_PROJECT_ROOT=$ROOT,CFI_UV_ENV=$CFI_UV_ENV"

submit() {
  sbatch "${EXTRA_SBATCH[@]}" --parsable "$@"
}

j0=$(submit --export="$EXPORT_BASE" "$ROOT/slurm/00_inspect.sbatch")
j1=$(submit --dependency=afterok:"$j0" --export="$EXPORT_BASE" "$ROOT/slurm/10_prepare.sbatch")
j2=$(submit --dependency=afterok:"$j1" --export="$EXPORT_BASE" "$ROOT/slurm/20_make_rho0.sbatch")
j25=$(submit --dependency=afterok:"$j2" --export="$EXPORT_BASE" "$ROOT/slurm/25_scaling_train.sbatch")
j3=$(submit --dependency=afterok:"$j25" --export="$EXPORT_BASE" "$ROOT/slurm/30_numerics.sbatch")
j4=$(submit --dependency=afterok:"$j3" --export="$EXPORT_BASE" "$ROOT/slurm/40_gpu_tests.sbatch")
j45=$(submit --dependency=afterok:"$j4" --export="$EXPORT_BASE" "$ROOT/slurm/45_atomic_benchmark.sbatch")
LIMIT="${CFI_ARRAY_LIMIT:-8}"
EXPORT_ARRAY="ALL,CFI_PRIOR_MANIFEST=$MANIFEST,CFI_PROJECT_ROOT=$ROOT,CFI_UV_ENV=$CFI_UV_ENV"
ja=$(submit --dependency=afterok:"$j45" --array="0-$((N-1))%$LIMIT" --export="$EXPORT_ARRAY" "$ROOT/slurm/50_prior_candidate.sbatch")

SELECT_SCRIPT="$RUN_ROOT/select_prior_${ja}.sh"
cat > "$SELECT_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export UV_PROJECT_ENVIRONMENT="$CFI_UV_ENV"
uv run --project "$ROOT" --frozen --no-sync cfi select-prior "$CONFIG" "$MANIFEST" "$SELECTED"
EOF
chmod +x "$SELECT_SCRIPT"
js=$(submit --job-name=fieldrefine-select --cpus-per-task=1 --mem=2G --time=00:10:00 --dependency=afterok:"$ja" --output="$RUN_ROOT/slurm-%x-%j.out" "$SELECT_SCRIPT")

EXPORT_SELECTED="ALL,CFI_CONFIG=$SELECTED,CFI_PROJECT_ROOT=$ROOT,CFI_UV_ENV=$CFI_UV_ENV"
j55=$(submit --dependency=afterok:"$js" --export="$EXPORT_SELECTED" "$ROOT/slurm/55_scaling_work.sbatch")
jf=$(submit --dependency=afterok:"$j55" --export="$EXPORT_SELECTED" "$ROOT/slurm/60_final_fit.sbatch")
ji=$(submit --dependency=afterok:"$jf" --export="$EXPORT_SELECTED" "$ROOT/slurm/70_information.sbatch")

FREEZE_SCRIPT="$RUN_ROOT/freeze_${jf}.sh"
cat > "$FREEZE_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export UV_PROJECT_ENVIRONMENT="$CFI_UV_ENV"
uv run --project "$ROOT" --frozen --no-sync cfi freeze-model "$SELECTED"
EOF
chmod +x "$FREEZE_SCRIPT"
jlock=$(submit --job-name=fieldrefine-freeze --cpus-per-task=1 --mem=2G --time=00:10:00 --dependency=afterok:"$jf" --output="$RUN_ROOT/slurm-%x-%j.out" "$FREEZE_SCRIPT")

echo "Submitted: inspect=$j0 prepare=$j1 rho0=$j2 scale-train=$j25 numerics=$j3 gpu-tests=$j4 atomic-benchmark=$j45 priors=$ja select=$js scale-work=$j55 final=$jf info=$ji freeze=$jlock"
echo "After model lock, run one-shot free evaluation with:"
echo "  sbatch ${CFI_SBATCH_ARGS:-} --dependency=afterok:$jlock --export=$EXPORT_SELECTED $ROOT/slurm/80_evaluate_free.sbatch"
