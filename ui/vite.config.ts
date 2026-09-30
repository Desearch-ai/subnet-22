import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

const DEV_PORT = 5173

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { tsconfigPaths: true },
  server: { port: DEV_PORT, strictPort: true },
  preview: { port: DEV_PORT, strictPort: true },
  test: { include: ['src/**/*.test.ts'] },
})
