import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const DSP_ROOT = path.join(ROOT, "dsp");

const run = (cmd, args, extraEnv = {}) => {
    console.log(`[build-wasm] running: ${cmd} ${args.join(" ")}`);
    const result = spawnSync(cmd, args, {
        cwd: DSP_ROOT,
        stdio: "inherit",
        env: {
            ...process.env,
            ...extraEnv,
        },
    });
    if (typeof result.status === "number" && result.status !== 0) {
        process.exit(result.status);
    }
    if (result.error) {
        throw result.error;
    }
};

const buildWasm = () => {
    // Build for web (compatible with vitest + vite-plugin-wasm)
    // WASM_SIMD=0 で明示的に無効化しない限り SIMD を有効にしてビルドする。
    const enableSimd = process.env.WASM_SIMD !== "0";
    const simdFlag = "-C target-feature=+simd128";
    const mergedRustFlags = enableSimd
        ? [process.env.RUSTFLAGS, simdFlag].filter(Boolean).join(" ")
        : process.env.RUSTFLAGS;
    const extraEnv = mergedRustFlags ? { RUSTFLAGS: mergedRustFlags } : {};
    console.log(`[build-wasm] wasm simd: ${enableSimd ? "enabled" : "disabled"}`);
    run("wasm-pack", ["build", "--target", "web", "--out-dir", "../pkg"], extraEnv);
};

buildWasm();
