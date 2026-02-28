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
    run("wasm-pack", ["build", "--target", "web", "--out-dir", "../pkg"]);
};

buildWasm();
