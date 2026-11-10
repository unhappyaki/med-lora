#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/medical-v5
VLLM_ROOT="$ROOT/inference/v131/vllm"
WORKBENCH_ROOT="$ROOT/inference/v131/workbench"
MODEL_DIR="$ROOT/inference/v131/merged-model"
PYTHON=/usr/local/miniconda3/envs/py312/bin/python
PORT="${VLLM_PORT:-9877}"
PID_FILE="$VLLM_ROOT/server.pid"
LOG_FILE="$VLLM_ROOT/server.log"
TOKEN_FILE="$WORKBENCH_ROOT/access.token"

mkdir -p "$VLLM_ROOT" "$WORKBENCH_ROOT"
chmod 700 "$VLLM_ROOT" "$WORKBENCH_ROOT"

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

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="/usr/local/cuda-13.2/bin:/usr/local/miniconda3/envs/py312/bin:$PATH"

nohup "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name medical-section-v131 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --api-key "$(cat "$TOKEN_FILE")" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.55 \
  --trust-remote-code \
  --no-enable-log-requests \
  --disable-log-stats \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "started pid=$(cat "$PID_FILE") port=$PORT"
echo "health=http://127.0.0.1:$PORT/health"
echo "log=$LOG_FILE"
