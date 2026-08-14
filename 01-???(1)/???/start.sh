#!/usr/bin/env sh
set -eu
SERVICE_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SERVICE_DIR/.." && pwd)
RUNTIME="$ROOT/runtime"
mkdir -p "$RUNTIME/logs/retrieval-service" "$RUNTIME/cache/python" "$RUNTIME/cache/pytest" "$RUNTIME/history" "$RUNTIME/pids"
"$ROOT/scripts/housekeep-runtime.sh"
export PYTHONPYCACHEPREFIX="$RUNTIME/cache/python"
export CHAT_API_TRACE_PATH="$RUNTIME/history/chat-api.trace.jsonl"
: "${RAGV6_API_HOST:=127.0.0.1}"
: "${RAGV6_API_PORT:=8011}"
cd "$SERVICE_DIR"
python -m uvicorn api_server:app --host "$RAGV6_API_HOST" --port "$RAGV6_API_PORT" --workers 1
