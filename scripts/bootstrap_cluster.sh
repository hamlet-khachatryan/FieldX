#!/usr/bin/env bash
set -euo pipefail

ACCEL="${1:-cuda13}"
ENV_DIR="${2:-${SCRATCH:-$HOME}/cfi-uv}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UV_VERSION="${CFI_UV_VERSION:-0.10.0}"

case "$ACCEL" in
  cpu|cuda12|cuda13) ;;
  *) echo "Usage: $0 [cpu|cuda12|cuda13] [ENV_DIR]" >&2; exit 2 ;;
esac

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user "uv==$UV_VERSION"
  export PATH="$HOME/.local/bin:$PATH"
fi

export UV_PROJECT_ENVIRONMENT="$ENV_DIR"
if [[ ! -f "$ROOT/uv.lock" ]]; then
  uv lock --project "$ROOT" --python 3.11
fi
uv sync --project "$ROOT" --locked --extra "$ACCEL" --group dev --no-editable --python 3.11
uv run --project "$ROOT" --frozen --no-sync python - <<'PY'
import json
import platform

import gemmi
import jax
import numpy
import optax
import reciprocalspaceship

print(json.dumps({
    "python": platform.python_version(),
    "gemmi": gemmi.__version__,
    "reciprocalspaceship": reciprocalspaceship.__version__,
    "numpy": numpy.__version__,
    "jax": jax.__version__,
    "optax": optax.__version__,
    "backend": jax.default_backend(),
    "devices": [str(device) for device in jax.devices()],
}, indent=2))
PY

mkdir -p "$ROOT/environment"
cat > "$ROOT/environment/cluster.env" <<EOF2
export CFI_UV_ENV="$ENV_DIR"
export CFI_ACCEL="$ACCEL"
export CFI_PROJECT_ROOT="$ROOT"
EOF2

echo "Environment ready. Run: source $ROOT/environment/cluster.env"
