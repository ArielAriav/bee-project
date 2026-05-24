/**
 * API base URL for HTTP requests and WebSocket connections.
 * On HTTPS (e.g. bee-vision.duckdns.org), uses same origin so Caddy/Vite can proxy
 * to the backend without mixed-content blocking.
 */
export function getApiBase() {
  if (import.meta.env.VITE_API_BASE) {
    return import.meta.env.VITE_API_BASE.replace(/\/$/, '');
  }
  if (typeof window !== 'undefined' && window.location.protocol === 'https:') {
    return window.location.origin;
  }
  return 'http://178.63.89.118:8000';
}

export function getWsBase() {
  const api = getApiBase();
  if (api.startsWith('https://')) {
    return api.replace(/^https:\/\//, 'wss://');
  }
  if (api.startsWith('http://')) {
    return api.replace(/^http:\/\//, 'ws://');
  }
  return api;
}
