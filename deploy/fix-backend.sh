#!/bin/bash
# Run on the server to install WebSocket deps and restart the correct backend.
set -e

cd ~/bee-project
git pull

source ~/bee_env/bin/activate
pip install -q 'uvicorn[standard]' websockets

echo "=== main.py has WebSocket route? ==="
grep -n "add_api_websocket_route\|ws/live" backend/main.py

echo "=== Stopping old backend processes on port 8000 ==="
sudo systemctl stop bee-backend 2>/dev/null || true
sleep 1
if command -v fuser >/dev/null  then
  sudo fuser -k 8000/tcp 2>/dev/null || true
fi
sleep 1

echo "=== Installing systemd unit (adjust User/ paths if needed) ==="
sudo cp deploy/bee-backend.service /etc/systemd/system/bee-backend.service
sudo systemctl daemon-reload
sudo systemctl enable bee-backend
sudo systemctl start bee-backend
sleep 4

echo "=== Health check ==="
curl -s http://127.0.0.1:8000/health | python3 -m json.tool

echo ""
echo "If websocket_ready is false or websocket_paths is empty, check:"
echo "  sudo journalctl -u bee-backend -n 40 --no-pager"
