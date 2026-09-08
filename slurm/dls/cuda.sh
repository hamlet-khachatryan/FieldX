#!/usr/bin/env bash
# Diamond Light Source GPU runtime initialisation.
#
# The only site-specific file in the repository. The generic scientific code contains
# no DLS paths; point FIELDX_CUDA_INIT at your own equivalent on another cluster.
#
# Verified on a DLS GPU node: `module load cuda` provides /dls_sw/apps/cuda/13.3.1 and
# CUPTI at extras/CUPTI/lib64/libcupti.so, which JAX 0.10+ with the CUDA 13 plugin needs.

if command -v module >/dev/null 2>&1; then
  module load cuda || echo "FieldX: 'module load cuda' failed; continuing with the ambient environment." >&2
elif [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
  module load cuda || echo "FieldX: 'module load cuda' failed; continuing with the ambient environment." >&2
else
  echo "FieldX: no module system found; assuming CUDA is already on PATH." >&2
fi

# JAX resolves libcupti through LD_LIBRARY_PATH; some module files export CUDA_HOME only.
if [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME/extras/CUPTI/lib64" ]]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"
fi

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
