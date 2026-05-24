# Deploying with Caddy (HTTPS + live camera WebSocket)

The live camera uses `wss://bee-vision.duckdns.org/ws/live`. Caddy must route API and WebSocket traffic to the **backend (port 8000)**, not only to Vite (port 5173).

## One-time setup on the server

```bash
cd ~/bee-project
git pull

# Install the Caddy config
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy

# Ensure services are running
sudo systemctl restart bee-backend
sudo systemctl restart bee-frontend
sudo systemctl status bee-backend bee-frontend caddy
```

## Verify

```bash
# Backend should respond
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/get-result

# Through Caddy (from the server)
curl -s -o /dev/null -w "%{http_code}\n" https://bee-vision.duckdns.org/get-result
```

In the browser, open DevTools → Network → WS and confirm `wss://bee-vision.duckdns.org/ws/live` connects with status **101**.

## If WebSocket still fails

1. Confirm `bee-backend` is active: `sudo systemctl status bee-backend`
2. Check Caddy logs: `sudo journalctl -u caddy -n 50 --no-pager`
3. Ensure `/etc/caddy/Caddyfile` includes the `@backend_paths` block (not only `reverse_proxy :5173`)
