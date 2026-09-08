#!/usr/bin/env bash
#
# Archive the repository for transfer to a machine without network access to GitHub.
# Experimental data, run products and local environments are excluded.

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-fieldx-transfer.tar.gz}"
cd "$(dirname "$PROJECT_ROOT")"
tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='.jax-cache' \
  --exclude='.pytest_cache' \
  --exclude='.ruff_cache' \
  --exclude='.idea' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.egg-info' \
  --exclude='runs' \
  --exclude='run' \
  --exclude='data' \
  --exclude='paper/*.aux' \
  --exclude='paper/*.bbl' \
  --exclude='paper/*.blg' \
  --exclude='paper/*.fdb_latexmk' \
  --exclude='paper/*.fls' \
  --exclude='paper/*.log' \
  --exclude='paper/*.out' \
  --exclude='paper/*.pdf' \
  -czf "$OUT" "$(basename "$PROJECT_ROOT")"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$OUT" > "$OUT.sha256"
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "$OUT" > "$OUT.sha256"
fi
printf '%s\n' "$OUT"
