import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    server: {
        host: '0.0.0.0', // Expose to network for phone connection
        port: 3000,
        // Allow all hosts including ngrok
        allowedHosts: [
            '.ngrok-free.dev',
            '.ngrok.io',
            '.ngrok.app',
            'localhost',
            '.localhost'
        ],
        proxy: {
            // Proxy API requests and WebSocket to backend
            '/api': {
                target: 'http://backend:8000',
                changeOrigin: true,
                secure: false,
                ws: true, // Enable WebSocket proxying
                timeout: 60000, // Increase timeout for large requests
                proxyTimeout: 60000,
                configure: (proxy, options) => {
                    proxy.on('error', (err, req, res) => {
                        console.error('Proxy ERROR:', err.message, 'for', req.method, req.url);
                    });
                    proxy.on('proxyReq', (proxyReq, req, res) => {
                        console.log('Proxying:', req.method, req.url, '→ http://backend:8000' + req.url);
                    });
                    proxy.on('proxyRes', (proxyRes, req, res) => {
                        if (req.url.includes('/frame')) {
                            console.log('Proxy response for /frame:', proxyRes.statusCode);
                        }
                    });
                    proxy.on('upgrade', (req, socket, head) => {
                        console.log('WebSocket UPGRADE request:', req.url);
                    });
                    proxy.on('proxyReqWs', (proxyReq, req, socket, options, head) => {
                        console.log('WebSocket PROXY request:', req.url);
                    });
                }
            }
        }
    }
})
