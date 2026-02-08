#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d ".venv" ]]; then
  echo "ERROR: .venv not found. Run: bash scripts/setup_venv.sh"
  exit 2
fi

source .venv/bin/activate

UVICORN_ARGS=(src.server:app --host 127.0.0.1 --port 8000)
if [[ "${RELOAD:-0}" == "1" ]]; then
  UVICORN_ARGS+=(--reload)
fi

uvicorn "${UVICORN_ARGS[@]}"
