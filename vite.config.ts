import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import wasm from 'vite-plugin-wasm'
import topLevelAwait from 'vite-plugin-top-level-await'
import { spawn } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import fs from 'node:fs'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Rust ファイルの変更時に wasm を dev ビルドするプラグイン
function rustWatchPlugin() {
  return {
    name: 'rust-watch',
    configureServer(server: any) {
      const dspDir = path.resolve(__dirname, 'dsp')
      let isBuilding = false

      const buildWasm = () => {
        if (isBuilding) {
          console.log('⏳ wasm build already in progress, skipping...')
          return
        }
        isBuilding = true
        console.log('\n🦀 Rust files changed, building wasm (dev mode)...')
        const wasmPack = path.join(process.env.HOME || '', '.cargo', 'bin', 'wasm-pack')

        const args = ['build', '--dev', '--target', 'web', '--out-dir', '../pkg-simd']
        const rustflags = [process.env.RUSTFLAGS, '-C', 'target-feature=+simd128'].filter(Boolean).join(' ')
        const child = spawn(wasmPack, args, {
          cwd: dspDir,
          stdio: 'inherit',
          env: {
            ...process.env,
            RUSTFLAGS: rustflags,
            PATH: `${path.join(process.env.HOME || '', '.cargo', 'bin')}:${process.env.PATH}`
          }
        })

        child.on('close', (code) => {
          isBuilding = false
          if (code === 0) {
            console.log('✅ wasm build complete')
            // ブラウザにリロードを促す
            server.ws.send({ type: 'full-reload' })
          } else {
            console.log('❌ wasm build failed')
          }
        })
      }

      // .rs ファイルを監視（ディレクトリ単位）
      server.watcher.add(dspDir)
      server.watcher.on('change', (filePath: string) => {
        if (filePath.endsWith('.rs')) {
          buildWasm()
        }
      })

      // 初期ビルド（pkg-simd がなければ）
      if (!fs.existsSync(path.resolve(__dirname, 'pkg-simd', 'dsp_bg.wasm'))) {
        buildWasm()
      }
    }
  }
}

// https://vite.dev/config/
export default defineConfig({
  base: "./",
  plugins: [
    vue(),
    wasm(),
    topLevelAwait(),
    rustWatchPlugin(),
  ],
  server: {
    allowedHosts: ['.trycloudflare.com', '.stfuawsc.com'],
  },
})
