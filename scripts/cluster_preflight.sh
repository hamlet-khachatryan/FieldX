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
echo "  workspace     ${FIELDX_WORKSPACE:-<unset; paths set individually>}"
echo

echo "Environment"
[[ -d "$PROJECT_ROOT/.venv" ]] && ok ".venv exists" || bad ".venv is missing; run: uv sync --locked --extra cpu --group dev"
"$UV" --version >/dev/null 2>&1 && ok "uv runs ($("$UV" --version))" || bad "uv is not executable"
run_uv python -c 'import sys; print(sys.version.split()[0])' >/dev/null 2>&1 \
  && ok "python runs ($(run_uv python -c 'import sys; print(sys.version.split()[0])'))" \
  || bad "python does not run in the repository environment"
echo

# The CUDA wheels total several GB. A cache on a small home quota fails partway through
# extraction with "Disk quota exceeded", after the download has already been paid for.
# On a laptop $HOME is the right place, so this only fails on a submission host.
echo "Storage"
if command -v sbatch >/dev/null 2>&1; then on_cluster=1; else on_cluster=0; fi
check_location() {
  local label="$1" path="$2" fix="$3"
  if [[ -z "$path" ]]; then
    note "$label: unknown"
  elif [[ -n "${HOME:-}" && ("$path" == "$HOME" || "$path" == "$HOME"/*) ]]; then
    if (( on_cluster )); then
      bad "$label is under \$HOME ($path); multi-GB accelerator wheels will exhaust a home quota"
      [[ -z "$fix" ]] || printf '        fix: export %s=/path/on/shared/filesystem\n' "$fix"
    else
      note "$label: $path (fine off-cluster)"
    fi
  else
    ok "$label: $path"
  fi
}
check_location "uv cache"      "$("$UV" cache dir 2>/dev/null || true)"  UV_CACHE_DIR
check_location "uv python dir" "$("$UV" python dir 2>/dev/null || true)" UV_PYTHON_INSTALL_DIR
check_location "project root"  "$PROJECT_ROOT"                          ""
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
# The accelerator extra has to match the GPU, not the site's default toolkit. Getting
# this wrong stays invisible until the first GPU job, where it surfaces as "ptxas too old".
plugin="$(run_uv python -c 'import importlib.util
for major in (13, 12):
    if importlib.util.find_spec(f"jax_cuda{major}_plugin"):
        print(major); break' 2>/dev/null || true)"
if [[ -z "$plugin" ]]; then
  note "no JAX CUDA plugin installed (cpu extra) -- correct for a login node or CI"
else
  ok "JAX CUDA plugin: cuda$plugin"
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  if [[ -z "$cc" ]]; then
    note "no GPU visible here, so the cuda$plugin choice cannot be checked from this host"
    note "verify on a GPU node: nvidia-smi --query-gpu=name,compute_cap --format=csv"
  elif [[ "${cc/./}" =~ ^[0-9]+$ ]] && (( ${cc/./} < 75 && plugin >= 13 )); then
    bad "GPU compute capability $cc is not supported by CUDA $plugin (CUDA 13 dropped Volta)"
    printf '        fix: rm -rf .venv && uv sync --locked --extra cuda12 --group dev\n'
  else
    ok "compute capability $cc is supported by cuda$plugin"
  fi
fi
echo

# Honour the same override that fieldx_load_cuda uses, so a non-DLS site pointing
# FIELDX_CUDA_INIT at its own script is not reported as broken.
cuda_init="${FIELDX_CUDA_INIT:-$PROJECT_ROOT/slurm/dls/cuda.sh}"
if [[ -f "$cuda_init" ]]; then
  ok "CUDA init script: $cuda_init (sourced inside GPU jobs only)"
else
  bad "CUDA init script not found: $cuda_init"
  printf '        GPU jobs would run without CUDA loaded and fall back to a CPU backend.\n'
  printf '        fix: restore the file from the repository (git status; git pull), or\n'
  printf '             export FIELDX_CUDA_INIT=/path/to/your/site/cuda-init.sh\n'
fi
echo

if (( fail )); then
  echo "Preflight FAILED."
  exit 1
fi
echo "Preflight passed. Submit with: scripts/submit.sh configs/<dataset>/default.yaml"
