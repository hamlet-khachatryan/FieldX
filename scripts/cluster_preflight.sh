#!/usr/bin/env bash
#
# Login-node preflight. Checks that the repository environment can drive the pipeline.
# It needs no GPU and no CUDA: on a login node JAX correctly reports a CPU device, and
# that is not a failure. GPU and CUDA validation belong to slurm/30_numerics.sbatch and
# slurm/40_gpu_tests.sbatch.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../slurm/common.sh
source "$PROJECT_ROOT/slurm/common.sh"

fail=0
ok()   { printf '  ok    %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fail=1; }
note() { printf '  note  %s\n' "$*"; }

echo "FieldX preflight"
echo "  host          $(hostname)"
echo "  project root  $PROJECT_ROOT"
echo "  uv            $UV"
echo "  environment   $UV_PROJECT_ENVIRONMENT"
echo

echo "Environment"
[[ -d "$PROJECT_ROOT/.venv" ]] && ok ".venv exists" || bad ".venv is missing; run: uv sync --locked --extra cpu --group dev"
"$UV" --version >/dev/null 2>&1 && ok "uv runs ($("$UV" --version))" || bad "uv is not executable"
run_uv python -c 'import sys; print(sys.version.split()[0])' >/dev/null 2>&1 \
  && ok "python runs ($(run_uv python -c 'import sys; print(sys.version.split()[0])'))" \
  || bad "python does not run in the repository environment"
[[ -n "${UV_CACHE_DIR:-}" ]] && note "uv cache: $UV_CACHE_DIR" || note "uv cache: default (set FIELDX_UV_CACHE to keep it off a home quota)"
echo

echo "Scientific stack"
for module in gemmi reciprocalspaceship optax jax numpy pydantic; do
  if version="$(run_uv python -c "import $module; print($module.__version__)" 2>/dev/null)"; then
    ok "$module $version"
  else
    bad "$module does not import"
  fi
done
echo

echo "Accelerator"
if devices="$(run_uv python -c 'import jax; print(jax.default_backend(), jax.devices())' 2>/dev/null)"; then
  note "JAX devices = $devices"
  note "A CPU-only device on a login node is expected and is not an error."
else
  bad "JAX does not initialise"
fi
echo

echo "Test and control plane"
run_uv pytest --version >/dev/null 2>&1 && ok "pytest available ($(run_uv pytest --version 2>&1 | head -1))" || bad "pytest is not installed"
run_uv ruff --version >/dev/null 2>&1 && ok "ruff available ($(run_uv ruff --version))" || note "ruff is not installed (dev group)"
run_uv fieldrefine --help >/dev/null 2>&1 && ok "fieldrefine CLI runs" || bad "fieldrefine CLI does not run"
if command -v sbatch >/dev/null 2>&1; then
  ok "sbatch found ($(sbatch --version 2>&1 | head -1))"
else
  bad "sbatch not found; production jobs cannot be submitted from this host"
fi
if [[ -f "$PROJECT_ROOT/slurm/dls/cuda.sh" ]]; then
  ok "CUDA init script present (slurm/dls/cuda.sh, sourced inside GPU jobs only)"
else
  bad "slurm/dls/cuda.sh is missing"
fi
echo

if (( fail )); then
  echo "Preflight FAILED."
  exit 1
fi
echo "Preflight passed. Submit with: scripts/submit.sh configs/<dataset>/default.yaml"
