#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate


export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-9000}
export DATA_DIR=${DATA_DIR:-data}

if [[ -n "${BOT_JSON:-}" ]]; then
  echo "[import] Importing JSON(s): $BOT_JSON"
  python import_from_file.py "$BOT_JSON" | tee /tmp/labchat_import.log
  # Extract slugs summary line
  if grep -q '^IMPORT_DONE:' /tmp/labchat_import.log; then
    SLUGS=$(grep '^IMPORT_DONE:' /tmp/labchat_import.log | tail -n1 | sed 's/IMPORT_DONE://')
    IFS=',' read -r -a arr <<< "$SLUGS"
    echo "[import] Ready links:"
    for s in "${arr[@]}"; do
      echo "  -> http://${HOST}:${PORT}/b/${s}"
    done
  else
    echo "[import] Import script did not complete; check output above."
  fi
fi

echo "[serve] Starting on http://${HOST}:${PORT}"
uvicorn app:app --host "$HOST" --port "$PORT"

