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
    // refs/radio と同様に、既存ツールチェインのみで完結する no-install を既定にする。
    const wasmMode = process.env.WASM_MODE ?? "no-install";
    const noOptRaw = String(process.env.WASM_NO_OPT ?? "1").toLowerCase();
    const noOpt = noOptRaw === "1" || noOptRaw === "true" || noOptRaw === "on";
    // WASM_SIMD=0 で明示的に無効化しない限り SIMD を有効にしてビルドする。
    const enableSimd = process.env.WASM_SIMD !== "0";
    const simdFlag = "-C target-feature=+simd128";
    const mergedRustFlags = enableSimd
        ? [process.env.RUSTFLAGS, simdFlag].filter(Boolean).join(" ")
        : process.env.RUSTFLAGS;
    const extraEnv = mergedRustFlags ? { RUSTFLAGS: mergedRustFlags } : {};
    const args = ["build", "--target", "web", "--out-dir", "../pkg", "--mode", wasmMode];
    if (noOpt) args.push("--no-opt");
    console.log(
        `[build-wasm] wasm simd: ${enableSimd ? "enabled" : "disabled"} mode=${wasmMode} no_opt=${noOpt}`
    );
    run("wasm-pack", args, extraEnv);
};

buildWasm();
