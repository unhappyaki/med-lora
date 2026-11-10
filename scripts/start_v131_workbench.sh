#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/medical-v5
WORKBENCH_ROOT="$ROOT/inference/v131/workbench"
PYTHON=/usr/local/miniconda3/envs/py312/bin/python
PORT="${PORT:-9876}"
PID_FILE="$WORKBENCH_ROOT/server.pid"
TOKEN_FILE="$WORKBENCH_ROOT/access.token"
LOG_FILE="$WORKBENCH_ROOT/server.log"

mkdir -p "$WORKBENCH_ROOT"
chmod 700 "$WORKBENCH_ROOT"

if [[ ! -s "$TOKEN_FILE" ]]; then
  umask 077
  "$PYTHON" - <<'PY' > "$TOKEN_FILE"
import secrets
print(secrets.token_urlsafe(24))
PY
fi
chmod 600 "$TOKEN_FILE"

if [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "already running pid=$(cat "$PID_FILE") port=$PORT"
  exit 0
fi

if "$PYTHON" - "$PORT" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(0.5)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
then
  echo "port $PORT is already in use" >&2
  exit 1
fi

export MODEL_DIR="${MODEL_DIR:-$ROOT/inference/v131/merged-model}"
export ADAPTER_DIR="$ROOT/inference/v131/adapter"
export SCHEMA_PATH="$ROOT/schemas/section_extraction_schema_v1.json"
export SERVED_MODEL_NAME=medical-section-v131
export WORKBENCH_TOKEN="$(cat "$TOKEN_FILE")"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:9877/v1}"
export VLLM_MODEL_NAME=medical-section-v131

nohup "$PYTHON" -m uvicorn v131_inference_workbench:app \
  --app-dir "$ROOT/scripts" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers 1 \
  --timeout-keep-alive 30 \
  --no-access-log \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "started pid=$(cat "$PID_FILE") port=$PORT"
echo "health=http://127.0.0.1:$PORT/health"
echo "log=$LOG_FILE"
