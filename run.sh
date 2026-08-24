#!/usr/bin/env bash
# Start the local development server.
#
#   ./run.sh              # http://0.0.0.0:8000 — reachable from the local network
#   DECIPHER_PORT=9000 ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  echo "No .venv found. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

HOST="${DECIPHER_HOST:-0.0.0.0}"
PORT="${DECIPHER_PORT:-8000}"

echo "Serving on http://${HOST}:${PORT}  (docs at /docs)"
exec .venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
