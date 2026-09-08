#!/usr/bin/env bash
set -euo pipefail
: "${CFI_UV_ENV:?Set CFI_UV_ENV}"
ROOT="${CFI_PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export UV_PROJECT_ENVIRONMENT="$CFI_UV_ENV"
echo "host: $(hostname)"
echo "uv: $(uv --version)"
command -v sbatch >/dev/null && sbatch --version || true
command -v nvidia-smi >/dev/null && nvidia-smi -L || true
uv run --project "$ROOT" --frozen --no-sync python - <<'PY'
import json
import gemmi
import jax
import optax
import reciprocalspaceship
print(json.dumps({
    "jax": jax.__version__,
    "backend": jax.default_backend(),
    "devices": [str(d) for d in jax.devices()],
    "gemmi": gemmi.__version__,
    "reciprocalspaceship": reciprocalspaceship.__version__,
    "optax": optax.__version__,
}, indent=2))
PY
