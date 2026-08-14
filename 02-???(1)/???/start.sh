#!/usr/bin/env sh
set -eu
SERVICE_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SERVICE_DIR/.." && pwd)
mkdir -p "$ROOT/runtime/logs/web-client" "$ROOT/runtime/cache/python" "$ROOT/runtime/cache/pytest" "$ROOT/runtime/history" "$ROOT/runtime/pids"
"$ROOT/scripts/housekeep-runtime.sh"
cd "$SERVICE_DIR"
node server.js
