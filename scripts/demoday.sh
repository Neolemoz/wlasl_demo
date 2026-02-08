#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d ".venv" ]]; then
  echo "ERROR: .venv not found. Run: bash scripts/setup_venv.sh"
  exit 2
fi

API_URL="http://127.0.0.1:8000/health"
WEB_URL="http://127.0.0.1:3000"
API_STARTED=0
WEB_STARTED=0
API_PID=""
WEB_PID=""

api_is_up() {
  curl -s --max-time 1 "$API_URL" | rg -q '"ok"\s*:\s*true'
}

web_is_up() {
  curl -s --max-time 1 -I "$WEB_URL" >/dev/null 2>&1 || curl -s --max-time 1 "$WEB_URL" >/dev/null 2>&1
}

cleanup() {
  if [[ $API_STARTED -eq 1 && -n "$API_PID" ]]; then
    kill "$API_PID" >/dev/null 2>&1 || true
  fi
  if [[ $WEB_STARTED -eq 1 && -n "$WEB_PID" ]]; then
    kill "$WEB_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup INT TERM

if ! api_is_up; then
  ./scripts/run_server.sh &
  API_PID=$!
  API_STARTED=1
fi

if ! web_is_up; then
  NPM_CONFIG_AUDIT=false NPM_CONFIG_FUND=false ./scripts/run_web.sh &
  WEB_PID=$!
  WEB_STARTED=1
fi

API_TIMEOUT=30
WEB_TIMEOUT=30
API_WAITED=0
WEB_WAITED=0

until api_is_up; do
  sleep 1
  API_WAITED=$((API_WAITED + 1))
  if [[ $API_WAITED -ge $API_TIMEOUT ]]; then
    echo "ERROR: API did not become ready within ${API_TIMEOUT}s."
    exit 1
  fi
done

until web_is_up; do
  sleep 1
  WEB_WAITED=$((WEB_WAITED + 1))
  if [[ $WEB_WAITED -ge $WEB_TIMEOUT ]]; then
    echo "ERROR: Web UI did not become ready within ${WEB_TIMEOUT}s."
    exit 1
  fi
done

echo "READY: http://127.0.0.1:3000"
echo "Press Ctrl+C to stop."

wait
