#!/usr/bin/env bash
#
# One path in, every FieldX location out.
#
#   source scripts/workspace-env.sh /path/to/your/workspace
#   FIELDX_WORKSPACE=/path/to/your/workspace source scripts/workspace-env.sh
#
# Source this BEFORE the first `uv sync`. UV_CACHE_DIR and UV_PYTHON_INSTALL_DIR are uv's
# own variables and must be set before uv downloads anything: the CUDA extras pull several
# GB of NVIDIA wheels, and a cache left on a small home quota dies mid-extraction with
# "Disk quota exceeded" after the download has already been paid for. slurm/common.sh
# sources this file too, so exporting FIELDX_WORKSPACE once covers submitted jobs as well.
#
# Anything you set yourself wins: this only fills in what you have not already chosen.

if [[ -n "${1:-}" ]]; then
  FIELDX_WORKSPACE="$1"
fi

if [[ -n "${FIELDX_WORKSPACE:-}" ]]; then
  mkdir -p "$FIELDX_WORKSPACE" 2>/dev/null || true
  if [[ ! -d "$FIELDX_WORKSPACE" ]]; then
    echo "FieldX: FIELDX_WORKSPACE is not a usable directory: $FIELDX_WORKSPACE" >&2
    return 1 2>/dev/null || exit 1
  fi
  # Absolute, so a later `cd` cannot silently change where data and runs land.
  FIELDX_WORKSPACE="$(cd "$FIELDX_WORKSPACE" && pwd)"
  export FIELDX_WORKSPACE

  export UV_CACHE_DIR="${UV_CACHE_DIR:-$FIELDX_WORKSPACE/uv-cache}"
  export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$FIELDX_WORKSPACE/uv-python}"
  export FIELDX_DATA_ROOT="${FIELDX_DATA_ROOT:-$FIELDX_WORKSPACE/data}"
  export FIELDX_RUNS_ROOT="${FIELDX_RUNS_ROOT:-$FIELDX_WORKSPACE/runs}"
  export FIELDX_JAX_CACHE="${FIELDX_JAX_CACHE:-$FIELDX_WORKSPACE/jax-cache}"

  mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$FIELDX_DATA_ROOT" \
           "$FIELDX_RUNS_ROOT" "$FIELDX_JAX_CACHE" 2>/dev/null || true

  if [[ -z "${FIELDX_WORKSPACE_QUIET:-}" ]]; then
    printf 'FieldX workspace %s\n' "$FIELDX_WORKSPACE" >&2
    printf '  uv cache      %s\n' "$UV_CACHE_DIR" >&2
    printf '  uv python     %s\n' "$UV_PYTHON_INSTALL_DIR" >&2
    printf '  data          %s\n' "$FIELDX_DATA_ROOT" >&2
    printf '  runs          %s\n' "$FIELDX_RUNS_ROOT" >&2
    printf '  jax cache     %s\n' "$FIELDX_JAX_CACHE" >&2
  fi
fi
