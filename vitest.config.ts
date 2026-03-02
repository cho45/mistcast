import { defineConfig } from 'vitest/config';
import vue from '@vitejs/plugin-vue';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
  plugins: [
    wasm(),
    topLevelAwait()
  ],
  test: {
    exclude: ['refs/**', 'node_modules/**', 'dist/**'],
    projects: [
      // Node.js 環境の既存テスト
      {
        test: {
          environment: 'node',
          include: ['tests/**/*.test.ts', 'src/**/*.test.ts'],
          exclude: ['src/**/*.component.test.ts'],
        },
      },
      // ブラウザ環境のコンポーネントテスト
      {
        plugins: [vue()],
        test: {
          environment: 'jsdom',
          include: ['src/**/*.component.test.ts'],
          globals: true,
          setupFiles: ['./src/test/setup-vitest.ts'],
        },
      },
    ],
  },
});
