#!/usr/bin/env bash
set -euo pipefail

: "${CFI_CONFIG:?Set CFI_CONFIG to an absolute YAML configuration path}"
: "${CFI_UV_ENV:?Set CFI_UV_ENV to the uv project environment path}"
PROJECT_ROOT="${CFI_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export UV_PROJECT_ENVIRONMENT="$CFI_UV_ENV"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PROJECT_ROOT/run/xla_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

if [[ -n "${CFI_APPTAINER_IMAGE:-}" ]]; then
  BIND_SPEC="$PROJECT_ROOT:$PROJECT_ROOT,$(dirname "$CFI_CONFIG"):$(dirname "$CFI_CONFIG")"
  [[ -z "${CFI_APPTAINER_BINDS:-}" ]] || BIND_SPEC="$BIND_SPEC,$CFI_APPTAINER_BINDS"
  CFI_CLI=(apptainer exec --nv --bind "$BIND_SPEC" --pwd "$PROJECT_ROOT" "$CFI_APPTAINER_IMAGE" cfi)
  run_cfi() { "${CFI_CLI[@]}" "$@" "$CFI_CONFIG"; }
  run_uv() { apptainer exec --nv --bind "$BIND_SPEC" --pwd "$PROJECT_ROOT" "$CFI_APPTAINER_IMAGE" "$@"; }
else
  run_cfi() { uv run --project "$PROJECT_ROOT" --frozen --no-sync cfi "$@" "$CFI_CONFIG"; }
  run_uv() { uv run --project "$PROJECT_ROOT" --frozen --no-sync "$@"; }
fi
