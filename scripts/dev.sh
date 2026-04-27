#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8989}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-8999}"

cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/uvicorn" ]]; then
  echo "Missing .venv. Run setup first:" >&2
  echo "  uv venv .venv --python 3.10 && source .venv/bin/activate && uv sync --active --dev --extra sam2" >&2
  exit 1
fi

if [[ ! -d "frontend/node_modules" ]]; then
  echo "Missing frontend/node_modules. Run setup first:" >&2
  echo "  cd frontend && pnpm install" >&2
  exit 1
fi

pid_in_this_repo() {
  local pid="$1"
  local cwd
  cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
  [[ "$cwd" == "$ROOT_DIR" || "$cwd" == "$ROOT_DIR/frontend" ]]
}

wait_for_port() {
  local port="$1"
  local i
  for i in {1..30}; do
    if ! lsof -i ":${port}" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.1
  done
  return 1
}

ensure_port_available() {
  local label="$1"
  local port="$2"
  local pids pid

  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi

  mapfile -t pids < <(lsof -ti ":${port}" -sTCP:LISTEN -n -P 2>/dev/null || true)
  if [[ "${#pids[@]}" -eq 0 ]]; then
    return 0
  fi

  for pid in "${pids[@]}"; do
    if ! pid_in_this_repo "$pid"; then
      echo "Port ${port} is already in use by a process outside this repo. Stop it or set ${label}_PORT." >&2
      lsof -i ":${port}" -sTCP:LISTEN -n -P >&2 || true
      exit 1
    fi
  done

  echo "Stopping existing ${label,,} process on port ${port}: ${pids[*]}"
  for pid in "${pids[@]}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done

  if ! wait_for_port "$port"; then
    echo "Port ${port} did not become available after stopping existing ${label,,} process." >&2
    exit 1
  fi
}

ensure_port_available "BACKEND" "$BACKEND_PORT"
ensure_port_available "FRONTEND" "$FRONTEND_PORT"

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT INT TERM

echo "Backend:  http://${BACKEND_HOST}:${BACKEND_PORT}"
echo "Frontend: http://${FRONTEND_HOST}:${FRONTEND_PORT}"

".venv/bin/uvicorn" backend.app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload &
pids+=("$!")

(
  cd frontend
  pnpm exec vite --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
) &
pids+=("$!")

wait -n "${pids[@]}"
exit_code=$?
exit "$exit_code"
