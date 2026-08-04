/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  build: {
    target: 'es2022',
    // three/R3F land in Phase 5 behind React.lazy, which produces its own chunk.
    // Declaring a manual chunk before anything imports it only ships dead bytes.
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
});
