import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  
  return {
    plugins: [react()],
    base: env.VITE_BASE_PATH || '/sri-lanka-employment-predictor-UI/',
    server: {
      host: '0.0.0.0',  // Allow external access
      port: 3000,
      proxy: {
        '/api': {
          target: env.VITE_API_URL || 'https://ffb6a6813594.ngrok-free.app',
          changeOrigin: true,
        }
      }
    }
  }
})
