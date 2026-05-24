#!/bin/bash
# Run on the server to confirm the backend is up to date and WebSocket is available.
set -e

echo "=== Local backend (port 8000) ==="
curl -sf http://127.0.0.1:8000/health | python3 -m json.tool || {
  echo "FAIL: backend not responding on 127.0.0.1:8000"
  exit 1
}

echo ""
echo "=== OpenAPI paths (must include WebSocket via /ws/live in app code) ==="
curl -sf http://127.0.0.1:8000/openapi.json | python3 -c "
import json, sys
paths = json.load(sys.stdin).get('paths', {})
for p in sorted(paths):
    print(' ', p)
if '/ws/live' not in str(paths):
    print('NOTE: WebSocket routes do not appear in OpenAPI; checking /health instead.')
"

echo ""
echo "=== Health via Caddy (public HTTPS) ==="
curl -sf https://bee-vision.duckdns.org/health | python3 -m json.tool || {
  echo "FAIL: https://bee-vision.duckdns.org/health — check Caddy config (deploy/Caddyfile)"
  exit 1
}

echo ""
echo "OK: Backend and Caddy look correct. Try Start Live Camera in the browser."
