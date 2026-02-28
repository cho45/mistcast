import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const DSP_ROOT = path.join(ROOT, "dsp");
const flavor = String(process.env.WASM_FLAVOR ?? "both").toLowerCase();

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

const pickFlavors = () => {
    if (flavor === "base") return ["base"];
    if (flavor === "simd") return ["simd"];
    if (flavor === "both") return ["base", "simd"];
    throw new Error(`invalid WASM_FLAVOR=${flavor} (expected: base|simd|both)`);
};

const buildWasm = () => {
    // Build for web (compatible with vitest + vite-plugin-wasm)
    // refs/radio と同様に、既存ツールチェインのみで完結する no-install を既定にする。
    const wasmMode = process.env.WASM_MODE ?? "no-install";
    const noOptRaw = String(process.env.WASM_NO_OPT ?? "1").toLowerCase();
    const noOpt = noOptRaw === "1" || noOptRaw === "true" || noOptRaw === "on";
    console.log(`[build-wasm] flavor=${flavor} mode=${wasmMode} no_opt=${noOpt}`);

    for (const f of pickFlavors()) {
        const args = [
            "build",
            "--target",
            "web",
            "--out-dir",
            f === "base" ? "../pkg" : "../pkg-simd",
            "--mode",
            wasmMode,
        ];
        if (noOpt) args.push("--no-opt");
        console.log(`[build-wasm] building ${f}`);
        if (f === "simd") {
            const rustflags = [process.env.RUSTFLAGS, "-C target-feature=+simd128"]
                .filter(Boolean)
                .join(" ");
            run("wasm-pack", args, { RUSTFLAGS: rustflags });
        } else {
            run("wasm-pack", args);
        }
    }
};

buildWasm();
