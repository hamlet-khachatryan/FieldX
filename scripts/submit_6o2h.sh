#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-configs/6o2h/default.yaml}"
GRID="${2:-configs/6o2h/prior_grid.yaml}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="$(realpath "$BASE")"
GRID="$(realpath "$GRID")"
: "${CFI_UV_ENV:?Set CFI_UV_ENV; source environment/cluster.env after bootstrap}"
export UV_PROJECT_ENVIRONMENT="$CFI_UV_ENV"

CAND_DIR="$ROOT/run/6O2H/prior_configs"
uv run --project "$ROOT" --frozen --no-sync cfi expand-priors "$BASE" "$GRID" "$CAND_DIR" >/dev/null
MANIFEST="$CAND_DIR/manifest.json"
N="$(uv run --project "$ROOT" --frozen --no-sync python - "$MANIFEST" <<'PY'
import json
import sys
print(len(json.load(open(sys.argv[1]))))
PY
)"
[[ "$N" -gt 0 ]] || { echo "No prior candidates" >&2; exit 2; }
SELECTED="$ROOT/run/6O2H/selected.yaml"

read -r -a EXTRA_SBATCH <<< "${CFI_SBATCH_ARGS:-}"
EXPORT_BASE="ALL,CFI_CONFIG=$BASE,CFI_PROJECT_ROOT=$ROOT,CFI_UV_ENV=$CFI_UV_ENV"

j0=$(sbatch "${EXTRA_SBATCH[@]}" --parsable --export="$EXPORT_BASE" "$ROOT/slurm/00_inspect.sbatch")
j1=$(sbatch "${EXTRA_SBATCH[@]}" --parsable --dependency=afterok:"$j0" --export="$EXPORT_BASE" "$ROOT/slurm/10_prepare.sbatch")
j2=$(sbatch "${EXTRA_SBATCH[@]}" --parsable --dependency=afterok:"$j1" --export="$EXPORT_BASE" "$ROOT/slurm/20_make_rho0.sbatch")
j25=$(sbatch "${EXTRA_SBATCH[@]}" --parsable --dependency=afterok:"$j2" --export="$EXPORT_BASE" "$ROOT/slurm/25_scaling_train.sbatch")
j3=$(sbatch "${EXTRA_SBATCH[@]}" --parsable --dependency=afterok:"$j25" --export="$EXPORT_BASE" "$ROOT/slurm/30_numerics.sbatch")
j4=$(sbatch "${EXTRA_SBATCH[@]}" --parsable --dependency=afterok:"$j3" --export="$EXPORT_BASE" "$ROOT/slurm/40_gpu_tests.sbatch")
j45=$(sbatch "${EXTRA_SBATCH[@]}" --parsable --dependency=afterok:"$j4" --export="$EXPORT_BASE" "$ROOT/slurm/45_atomic_benchmark.sbatch")

LIMIT="${CFI_ARRAY_LIMIT:-8}"
EXPORT_ARRAY="ALL,CFI_PRIOR_MANIFEST=$MANIFEST,CFI_PROJECT_ROOT=$ROOT,CFI_UV_ENV=$CFI_UV_ENV"
ja=$(sbatch "${EXTRA_SBATCH[@]}" --parsable --dependency=afterok:"$j45" --array="0-$((N-1))%$LIMIT" --export="$EXPORT_ARRAY" "$ROOT/slurm/50_prior_candidate.sbatch")

SELECT_SCRIPT="$ROOT/run/6O2H/select_prior_${ja}.sh"
mkdir -p "$(dirname "$SELECT_SCRIPT")"
cat > "$SELECT_SCRIPT" <<EOF2
#!/usr/bin/env bash
set -euo pipefail
export UV_PROJECT_ENVIRONMENT="$CFI_UV_ENV"
uv run --project "$ROOT" --frozen --no-sync cfi select-prior "$BASE" "$MANIFEST" "$SELECTED"
EOF2
chmod +x "$SELECT_SCRIPT"
js=$(sbatch "${EXTRA_SBATCH[@]}" --parsable --job-name=cfi-select --cpus-per-task=1 --mem=2G --time=00:10:00 --dependency=afterok:"$ja" --output=slurm-%x-%j.out "$SELECT_SCRIPT")

EXPORT_SELECTED="ALL,CFI_CONFIG=$SELECTED,CFI_PROJECT_ROOT=$ROOT,CFI_UV_ENV=$CFI_UV_ENV"
j55=$(sbatch "${EXTRA_SBATCH[@]}" --parsable --dependency=afterok:"$js" --export="$EXPORT_SELECTED" "$ROOT/slurm/55_scaling_work.sbatch")
jf=$(sbatch "${EXTRA_SBATCH[@]}" --parsable --dependency=afterok:"$j55" --export="$EXPORT_SELECTED" "$ROOT/slurm/60_final_fit.sbatch")
ji=$(sbatch "${EXTRA_SBATCH[@]}" --parsable --dependency=afterok:"$jf" --export="$EXPORT_SELECTED" "$ROOT/slurm/70_information.sbatch")

FREEZE_SCRIPT="$ROOT/run/6O2H/freeze_${jf}.sh"
cat > "$FREEZE_SCRIPT" <<EOF2
#!/usr/bin/env bash
set -euo pipefail
export UV_PROJECT_ENVIRONMENT="$CFI_UV_ENV"
uv run --project "$ROOT" --frozen --no-sync cfi freeze-model "$SELECTED"
EOF2
chmod +x "$FREEZE_SCRIPT"
jlock=$(sbatch "${EXTRA_SBATCH[@]}" --parsable --job-name=cfi-freeze --cpus-per-task=1 --mem=2G --time=00:10:00 --dependency=afterok:"$jf" --output=slurm-%x-%j.out "$FREEZE_SCRIPT")

echo "Submitted: inspect=$j0 prepare=$j1 rho0=$j2 scale-train=$j25 numerics=$j3 gpu-tests=$j4 atomic-benchmark=$j45 priors=$ja select=$js scale-work=$j55 final=$jf info=$ji freeze=$jlock"
echo "Free evaluation is intentionally manual. After the model is frozen:"
echo "  sbatch ${CFI_SBATCH_ARGS:-} --dependency=afterok:$jlock --export=$EXPORT_SELECTED $ROOT/slurm/80_evaluate_free.sbatch"
