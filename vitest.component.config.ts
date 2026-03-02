import { defineConfig } from 'vitest/config';
import vue from '@vitejs/plugin-vue';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
  plugins: [
    vue(),
    wasm(),
    topLevelAwait()
  ],
  test: {
    environment: 'jsdom',
    include: ['src/**/*.component.test.ts'],
    globals: true,
    setupFiles: ['./src/test/setup-vitest.ts'],
  },
});
