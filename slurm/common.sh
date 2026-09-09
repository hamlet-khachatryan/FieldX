#!/usr/bin/env bash
# Shared runtime for FieldX SLURM jobs and for login-node helper scripts.
#
# PATH RULE (mandatory): sbatch may stage the submitted script under a spool
# directory such as /var/spool/slurm/job12345/slurm_script, so "$0" does NOT point
# into the repository. Every SLURM job must therefore start with
#
#     cd "$SLURM_SUBMIT_DIR"
#     source "$SLURM_SUBMIT_DIR/slurm/common.sh"
#
# and never with source "$(dirname "$0")/common.sh".
#
# No site environment variable and no generated environment file name the Python
# environment: it is the repository-local .venv, resolved deterministically below.

set -euo pipefail

_FIELDX_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_fieldx_is_root() { [[ -f "$1/pyproject.toml" && -f "$1/slurm/common.sh" ]]; }

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && _fieldx_is_root "$SLURM_SUBMIT_DIR"; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
elif [[ -n "${CFI_PROJECT_ROOT:-}" ]] && _fieldx_is_root "$CFI_PROJECT_ROOT"; then
  PROJECT_ROOT="$CFI_PROJECT_ROOT"
else
  PROJECT_ROOT="$(cd "$_FIELDX_COMMON_DIR/.." && pwd)"
fi

if ! _fieldx_is_root "$PROJECT_ROOT"; then
  echo "FieldX: '$PROJECT_ROOT' is not a FieldX repository (no pyproject.toml + slurm/common.sh)." >&2
  echo "Submit from the repository root so SLURM_SUBMIT_DIR points at it." >&2
  exit 2
fi
export PROJECT_ROOT CFI_PROJECT_ROOT="$PROJECT_ROOT"

# The repository environment. FIELDX_UV names an explicit uv executable when uv is not
# on PATH and is not installed inside .venv.
UV="${FIELDX_UV:-}"
if [[ -z "$UV" && -x "$PROJECT_ROOT/.venv/bin/uv" ]]; then
  UV="$PROJECT_ROOT/.venv/bin/uv"
elif [[ -z "$UV" ]]; then
  UV="$(command -v uv || true)"
fi
if [[ -z "$UV" || ! -x "$UV" ]]; then
  echo "FieldX: no usable uv executable. Install uv, or set FIELDX_UV to an absolute path." >&2
  exit 2
fi
export UV
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PROJECT_ROOT/.venv}"
# Keep uv's caches off a small home quota. UV_CACHE_DIR and UV_PYTHON_INSTALL_DIR are
# uv's own variables and so also apply to bare `uv sync` on a login node; FIELDX_UV_CACHE
# is an alias for jobs that only export the FieldX variables. The CUDA wheels are several
# GB, so a home-quota cache fails partway through extraction.
[[ -z "${FIELDX_UV_CACHE:-}" ]] || export UV_CACHE_DIR="$FIELDX_UV_CACHE"
[[ -z "${FIELDX_UV_PYTHON_DIR:-}" ]] || export UV_PYTHON_INSTALL_DIR="$FIELDX_UV_PYTHON_DIR"

# One workspace path fills in every location that is not already set. The aliases above
# run first so an explicit FIELDX_UV_CACHE still wins over the workspace default. Jobs
# inherit this because submit.sh exports the whole environment (--export=ALL,...).
if [[ -n "${FIELDX_WORKSPACE:-}" && -f "$PROJECT_ROOT/scripts/workspace-env.sh" ]]; then
  FIELDX_WORKSPACE_QUIET=1
  # shellcheck source=../scripts/workspace-env.sh
  source "$PROJECT_ROOT/scripts/workspace-env.sh"
  unset FIELDX_WORKSPACE_QUIET
fi

export PYTHONUNBUFFERED=1
# Preallocation is OFF by default. Reserving most of the device up front leaves CUDA no
# room outside the arena to load compiled modules, which fails as
#
#   RESOURCE_EXHAUSTED: Failed to load in-memory CUBIN (compiled for a different GPU?):
#   CUDA_ERROR_OUT_OF_MEMORY [executable_name='jit_copy']
#
# even when the problem itself needs a fraction of a GiB. FieldX grids are modest and
# `fieldrefine estimate-memory` sizes them in advance, so on-demand allocation is the
# safer default. Turn it back on with a fraction that leaves headroom (0.7-0.8) if a
# large information solve fragments memory.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${FIELDX_JAX_CACHE:-$PROJECT_ROOT/.jax-cache}}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# Every GPU job loads CUDA itself; an interactive shell's module environment is never
# inherited by a batch job. Login-node control commands never call this.
fieldx_load_cuda() {
  local init="${FIELDX_CUDA_INIT:-$PROJECT_ROOT/slurm/dls/cuda.sh}"
  if [[ -f "$init" ]]; then
    # shellcheck disable=SC1090
    source "$init"
  else
    echo "FieldX: no CUDA init script at $init; assuming CUDA is already on PATH." >&2
  fi
}

fieldx_require_config() {
  : "${CFI_CONFIG:?Set CFI_CONFIG to an absolute configuration path}"
  [[ -f "$CFI_CONFIG" ]] || { echo "FieldX: CFI_CONFIG does not exist: $CFI_CONFIG" >&2; exit 2; }
}

if [[ -n "${CFI_APPTAINER_IMAGE:-}" ]]; then
  # Optional. The normal DLS workflow is `module load cuda` + the repository .venv.
  _fieldx_binds="$PROJECT_ROOT:$PROJECT_ROOT"
  [[ -z "${CFI_CONFIG:-}" ]] || _fieldx_binds="$_fieldx_binds,$(dirname "$CFI_CONFIG"):$(dirname "$CFI_CONFIG")"
  [[ -z "${CFI_APPTAINER_BINDS:-}" ]] || _fieldx_binds="$_fieldx_binds,$CFI_APPTAINER_BINDS"
  _fieldx_apptainer=(apptainer exec --nv --bind "$_fieldx_binds" --pwd "$PROJECT_ROOT" "$CFI_APPTAINER_IMAGE")
  run_cfi() { fieldx_require_config; "${_fieldx_apptainer[@]}" fieldrefine "$@" "$CFI_CONFIG"; }
  run_uv() { "${_fieldx_apptainer[@]}" "$@"; }
else
  run_cfi() { fieldx_require_config; "$UV" run --project "$PROJECT_ROOT" --frozen --no-sync fieldrefine "$@" "$CFI_CONFIG"; }
  run_uv() { "$UV" run --project "$PROJECT_ROOT" --frozen --no-sync "$@"; }
fi
