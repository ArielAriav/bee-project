import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const backendProxy = {
  '/ws': {
    target: 'http://127.0.0.1:8000',
    ws: true,
    changeOrigin: true,
  },
  '/upload-video': { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/get-result': { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/video-feed': { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/stop-session': { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/health': { target: 'http://127.0.0.1:8000', changeOrigin: true },
};

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    allowedHosts: ['bee-vision.duckdns.org'],
    proxy: backendProxy,
  },
  preview: {
    host: true,
    allowedHosts: ['bee-vision.duckdns.org'],
    proxy: backendProxy,
  },
});
