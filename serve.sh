#!/usr/bin/env bash
# Launch the Piper Novel Studio web UI (LAN-accessible).
# Usage: ./serve.sh           -> http://<lan-ip>:8000
#        ./serve.sh --reload  -> auto-reload on file changes

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "Piper Novel Studio"
echo "  local : http://127.0.0.1:8000"
[ -n "$LAN_IP" ] && echo "  lan   : http://$LAN_IP:8000"
echo "  login : 137 / 137  (change from the UI menu)"
echo

exec .venv/bin/python -m uvicorn webapp.server:app \
  --host 0.0.0.0 --port 8000 "$@"
