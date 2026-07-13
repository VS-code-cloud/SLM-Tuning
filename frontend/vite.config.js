import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Plain Vite + React. VITE_API_URL (optional) sets the default server URL; the
// user can also paste the ngrok URL into the UI at runtime.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
})
