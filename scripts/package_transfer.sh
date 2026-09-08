#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-crystal-field-inference-transfer.tar.gz}"
cd "$(dirname "$ROOT")"
tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='.pytest_cache' \
  --exclude='.ruff_cache' \
  --exclude='__pycache__' \
  --exclude='run' \
  --exclude='data' \
  --exclude='environment/cluster.env' \
  --exclude='environment/locked-from-cluster.txt' \
  --exclude='*.pyc' \
  --exclude='paper/*.aux' \
  --exclude='paper/*.bbl' \
  --exclude='paper/*.blg' \
  --exclude='paper/*.fdb_latexmk' \
  --exclude='paper/*.fls' \
  --exclude='paper/*.log' \
  --exclude='paper/*.out' \
  --exclude='paper/rendered' \
  --exclude='paper/contact_sheet.png' \
  -czf "$OUT" "$(basename "$ROOT")"
sha256sum "$OUT" > "$OUT.sha256"
printf '%s\n' "$OUT"
