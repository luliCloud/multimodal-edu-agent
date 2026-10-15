#!/usr/bin/env bash
# Start the EduTok Studio webapp: FastAPI (port 8000) + Next.js (port 3000).
#
#   ./webapp/start.sh            # dev mode with hot reload
#   ./webapp/start.sh --prod     # production build of the frontend
#
# Pipeline settings pass straight through, e.g.:
#   PLANNER_BACKEND=qwen RENDER_BACKEND=wan ./webapp/start.sh
# Ports can be changed with API_PORT / WEB_PORT. Ctrl+C stops both servers.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND="$ROOT/webapp/frontend"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-3000}"
MODE="dev"
[[ "${1:-}" == "--prod" ]] && MODE="prod"

cd "$ROOT"

# ---- Python environment -------------------------------------------------------
if [[ ! -d .venv ]]; then
  echo "› Creating .venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if ! python -c "import fastapi, multipart, backend" >/dev/null 2>&1; then
  echo "› Installing Python dependencies"
  pip install -q -e ".[dev]" -r webapp/backend/requirements.txt
fi

# ---- Node dependencies --------------------------------------------------------
command -v npm >/dev/null || { echo "✗ npm not found; install Node.js 20+"; exit 1; }
if [[ ! -d "$FRONTEND/node_modules" ]]; then
  echo "› Installing frontend dependencies"
  (cd "$FRONTEND" && npm install --no-fund --no-audit)
fi

# ---- Start both servers -------------------------------------------------------
pids=()
# Kill a process and all its descendants (npx/uvicorn --reload spawn children).
kill_tree() {
  local child
  for child in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$child"; done
  kill "$1" 2>/dev/null || true
}
cleanup() {
  trap - EXIT INT TERM
  echo
  echo "› Stopping servers"
  for pid in "${pids[@]:-}"; do
    [[ -n "$pid" ]] && kill_tree "$pid"
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "› API  → http://127.0.0.1:$API_PORT  (docs at /docs)"
if [[ "$MODE" == "dev" ]]; then
  uvicorn webapp.backend.app.main:app --reload --reload-dir backend --reload-dir webapp/backend \
    --port "$API_PORT" &
else
  uvicorn webapp.backend.app.main:app --port "$API_PORT" &
fi
pids+=($!)

export BACKEND_URL="http://127.0.0.1:$API_PORT"
echo "› UI   → http://localhost:$WEB_PORT"
if [[ "$MODE" == "dev" ]]; then
  (cd "$FRONTEND" && exec npx next dev -p "$WEB_PORT") &
else
  (cd "$FRONTEND" && npx next build && exec npx next start -p "$WEB_PORT") &
fi
pids+=($!)

# Exit (and stop the other server) as soon as either one dies.
# A polling loop rather than `wait -n`, which macOS's bash 3.2 lacks.
while kill -0 "${pids[0]}" 2>/dev/null && kill -0 "${pids[1]}" 2>/dev/null; do
  sleep 1
done
