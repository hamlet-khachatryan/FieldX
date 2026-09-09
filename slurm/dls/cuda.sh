#!/usr/bin/env bash
# Diamond Light Source GPU runtime initialisation.
#
# The only site-specific file in the repository. The generic scientific code contains
# no DLS paths; point FIELDX_CUDA_INIT at your own equivalent on another cluster.
#
# Verified on a DLS GPU node: `module load cuda` provides /dls_sw/apps/cuda/13.3.1 and
# CUPTI at extras/CUPTI/lib64/libcupti.so, which JAX 0.10+ with the CUDA 13 plugin needs.
#
# The pip CUDA wheels ship their own ptxas under
# .venv/lib/python3.*/site-packages/nvidia/cu13/bin/ptxas, and XLA prefers it. When that
# copy predates the GPU's compute capability the compile fails with
#
#   UNIMPLEMENTED: .../nvidia/cu13/bin/ptxas ptxas too old. Falling back to the driver
#
# so this script points XLA at the site toolkit, which is the one matched to the driver.

if command -v module >/dev/null 2>&1; then
  module load cuda || echo "FieldX: 'module load cuda' failed; continuing with the ambient environment." >&2
elif [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
  module load cuda || echo "FieldX: 'module load cuda' failed; continuing with the ambient environment." >&2
else
  echo "FieldX: no module system found; assuming CUDA is already on PATH." >&2
fi

# Locate the toolkit root the module just provided. Different sites export different
# variables, so fall back to the location of nvcc/ptxas on PATH.
_fieldx_cuda_root="${FIELDX_XLA_CUDA_DIR:-${CUDA_HOME:-${CUDA_ROOT:-${CUDA_PATH:-${CUDA_DIR:-}}}}}"
if [[ -z "$_fieldx_cuda_root" ]]; then
  for _tool in nvcc ptxas; do
    _path="$(command -v "$_tool" 2>/dev/null || true)"
    if [[ -n "$_path" ]]; then
      _fieldx_cuda_root="$(cd "$(dirname "$_path")/.." && pwd)"
      break
    fi
  done
fi

if [[ -n "$_fieldx_cuda_root" && -x "$_fieldx_cuda_root/bin/ptxas" ]]; then
  export CUDA_HOME="$_fieldx_cuda_root"
  export CUDA_DIR="$_fieldx_cuda_root"          # XLA also consults CUDA_DIR
  export PATH="$_fieldx_cuda_root/bin:$PATH"
  # XLA searches <dir>/bin/ptxas. Without this it uses the wheel's bundled copy.
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=$_fieldx_cuda_root ${XLA_FLAGS:-}"
  echo "FieldX: XLA CUDA toolkit = $_fieldx_cuda_root ($("$_fieldx_cuda_root/bin/ptxas" --version 2>/dev/null | tail -1))" >&2
else
  echo "FieldX: no CUDA toolkit with bin/ptxas found; XLA will use the ptxas bundled in the wheels." >&2
  echo "        If compilation fails with 'ptxas too old', set FIELDX_XLA_CUDA_DIR to a toolkit root." >&2
fi

# JAX resolves libcupti through LD_LIBRARY_PATH; some module files export CUDA_HOME only.
for _dir in "${CUDA_HOME:-}/extras/CUPTI/lib64" "${CUDA_HOME:-}/lib64"; do
  [[ -d "$_dir" ]] && export LD_LIBRARY_PATH="$_dir:${LD_LIBRARY_PATH:-}"
done
unset _fieldx_cuda_root _tool _path _dir

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
