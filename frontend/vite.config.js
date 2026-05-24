import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: ['bee-vision.duckdns.org'],
    proxy: {
      '/ws': {
        target: 'http://127.0.0.1:8000',
        ws: true,
        changeOrigin: true,
      },
      '/upload-video': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/get-result': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/video-feed': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/stop-session': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
});
