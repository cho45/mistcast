import { performance } from "node:perf_hooks";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import initBase, {
  WasmDsssEncoder as WasmDsssEncoderBase,
  WasmDsssDecoder as WasmDsssDecoderBase,
  WasmMaryEncoder as WasmMaryEncoderBase,
  WasmMaryDecoder as WasmMaryDecoderBase,
} from "../pkg/dsp.js";
import initSimd, {
  WasmDsssEncoder as WasmDsssEncoderSimd,
  WasmDsssDecoder as WasmDsssDecoderSimd,
  WasmMaryEncoder as WasmMaryEncoderSimd,
  WasmMaryDecoder as WasmMaryDecoderSimd,
} from "../pkg-simd/dsp.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const BENCH_ITERS = Math.max(1, Number.parseInt(process.env.BENCH_ITERS ?? "30", 10));
const BENCH_WARMUP_MIN_PAIRS = Math.max(
  1,
  Number.parseInt(process.env.BENCH_WARMUP_MIN_PAIRS ?? "4", 10),
);
const BENCH_WARMUP_MAX_PAIRS = Math.max(
  BENCH_WARMUP_MIN_PAIRS,
  Number.parseInt(process.env.BENCH_WARMUP_MAX_PAIRS ?? "30", 10),
);
const BENCH_STABLE_WINDOW = Math.max(
  3,
  Number.parseInt(process.env.BENCH_STABLE_WINDOW ?? "5", 10),
);
const BENCH_STABLE_CV = Math.max(
  0.0001,
  Number.parseFloat(process.env.BENCH_STABLE_CV ?? "0.03"),
);
const FRAMES = Math.max(1, Number.parseInt(process.env.BENCH_FRAMES ?? "12", 10));

const BINDINGS = [
  {
    flavor: "base",
    wasmPath: path.join(ROOT, "pkg", "dsp_bg.wasm"),
    init: initBase,
    WasmDsssEncoder: WasmDsssEncoderBase,
    WasmDsssDecoder: WasmDsssDecoderBase,
    WasmMaryEncoder: WasmMaryEncoderBase,
    WasmMaryDecoder: WasmMaryDecoderBase,
  },
  {
    flavor: "simd",
    wasmPath: path.join(ROOT, "pkg-simd", "dsp_bg.wasm"),
    init: initSimd,
    WasmDsssEncoder: WasmDsssEncoderSimd,
    WasmDsssDecoder: WasmDsssDecoderSimd,
    WasmMaryEncoder: WasmMaryEncoderSimd,
    WasmMaryDecoder: WasmMaryDecoderSimd,
  },
];

const percentile = (xs, p) => {
  if (xs.length === 0) return NaN;
  const sorted = [...xs].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p));
  return sorted[idx];
};

const median = (xs) => percentile(xs, 0.5);
const mean = (xs) => xs.reduce((acc, x) => acc + x, 0) / xs.length;
const stdev = (xs) => {
  if (xs.length <= 1) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((acc, x) => acc + (x - m) ** 2, 0) / xs.length);
};
const cv = (xs) => {
  const m = mean(xs);
  if (m === 0) return Number.POSITIVE_INFINITY;
  return stdev(xs) / m;
};

async function initBindings() {
  for (const binding of BINDINGS) {
    const wasmBuffer = fs.readFileSync(binding.wasmPath);
    await binding.init({ module_or_path: wasmBuffer });
  }
}

function buildDsssSignal(binding, sampleRate) {
  const encoder = new binding.WasmDsssEncoder(sampleRate, 1);
  encoder.set_data(new TextEncoder().encode("SIMD benchmark payload for DSSS."));
  let allSamples = new Float32Array(0);
  for (let i = 0; i < FRAMES; i++) {
    const frame = encoder.pull_frame();
    if (!frame) break;
    const next = new Float32Array(allSamples.length + frame.length + 4800);
    next.set(allSamples);
    next.set(frame, allSamples.length);
    allSamples = next;
  }
  return allSamples;
}

function buildMarySignal(binding, sampleRate) {
  const encoder = new binding.WasmMaryEncoder(sampleRate, 3);
  encoder.set_data(new TextEncoder().encode("SIMD benchmark payload for Mary."));
  let allSamples = new Float32Array(0);
  for (let i = 0; i < FRAMES; i++) {
    const frame = encoder.pull_frame();
    if (!frame) break;
    const next = new Float32Array(allSamples.length + frame.length + 4800);
    next.set(allSamples);
    next.set(frame, allSamples.length);
    allSamples = next;
  }
  return allSamples;
}

function benchOne(flavorBinding, signal) {
  if (typeof global.gc === "function") {
    global.gc();
  }
  const start = performance.now();
  flavorBinding.decoder.reset();
  flavorBinding.decoder.process_samples(signal);
  return performance.now() - start;
}

function runAlternatingPair(bindingsByFlavor, signal, pairIndex) {
  const order = pairIndex % 2 === 0 ? ["base", "simd"] : ["simd", "base"];
  const result = {};
  for (const flavor of order) {
    result[flavor] = benchOne(bindingsByFlavor[flavor], signal);
  }
  return result;
}

function warmupUntilStable(bindingsByFlavor, signal) {
  const warm = { base: [], simd: [] };
  let pairs = 0;
  for (; pairs < BENCH_WARMUP_MAX_PAIRS; pairs++) {
    const r = runAlternatingPair(bindingsByFlavor, signal, pairs);
    warm.base.push(r.base);
    warm.simd.push(r.simd);

    if (pairs + 1 < BENCH_WARMUP_MIN_PAIRS) continue;
    if (warm.base.length < BENCH_STABLE_WINDOW || warm.simd.length < BENCH_STABLE_WINDOW) continue;

    const b = warm.base.slice(-BENCH_STABLE_WINDOW);
    const s = warm.simd.slice(-BENCH_STABLE_WINDOW);
    if (cv(b) <= BENCH_STABLE_CV && cv(s) <= BENCH_STABLE_CV) {
      pairs += 1;
      break;
    }
  }

  return {
    warmup_pairs: pairs,
    warmup_base_cv: Number(cv(warm.base.slice(-BENCH_STABLE_WINDOW)).toFixed(4)),
    warmup_simd_cv: Number(cv(warm.simd.slice(-BENCH_STABLE_WINDOW)).toFixed(4)),
  };
}

async function main() {
  await initBindings();
  if (typeof global.gc !== "function") {
    console.warn(
      "[bench] global.gc が無効です。より安定した計測には `node --expose-gc` で実行してください。",
    );
  }

  const sampleRate = 48_000;
  const signalBase = BINDINGS[0];
  const dsssSignal = buildDsssSignal(signalBase, sampleRate);
  const marySignal = buildMarySignal(signalBase, sampleRate);

  const scenarios = [
    { id: "dsss_decode", signal: dsssSignal, kind: "dsss" },
    { id: "mary_decode", signal: marySignal, kind: "mary" },
  ];

  const report = {
    config: {
      BENCH_ITERS,
      BENCH_WARMUP_MIN_PAIRS,
      BENCH_WARMUP_MAX_PAIRS,
      BENCH_STABLE_WINDOW,
      BENCH_STABLE_CV,
      FRAMES,
      sampleRate,
    },
    scenarios: {},
  };

  const bindingsByFlavor = {
    base: BINDINGS.find((b) => b.flavor === "base"),
    simd: BINDINGS.find((b) => b.flavor === "simd"),
  };

  for (const scenario of scenarios) {
    for (const binding of Object.values(bindingsByFlavor)) {
      binding.decoder =
        scenario.kind === "dsss"
          ? new binding.WasmDsssDecoder(sampleRate, 1)
          : new binding.WasmMaryDecoder(sampleRate, 3);
    }

    const warmupMeta = warmupUntilStable(bindingsByFlavor, scenario.signal);

    const timesByFlavor = { base: [], simd: [] };
    for (let i = 0; i < BENCH_ITERS; i++) {
      const r = runAlternatingPair(bindingsByFlavor, scenario.signal, i);
      timesByFlavor.base.push(r.base);
      timesByFlavor.simd.push(r.simd);
    }

    report.scenarios[scenario.id] = {
      warmup: warmupMeta,
      base: {
        median_ms: Number(median(timesByFlavor.base).toFixed(3)),
        p95_ms: Number(percentile(timesByFlavor.base, 0.95).toFixed(3)),
        min_ms: Number(Math.min(...timesByFlavor.base).toFixed(3)),
        max_ms: Number(Math.max(...timesByFlavor.base).toFixed(3)),
        cv: Number(cv(timesByFlavor.base).toFixed(4)),
      },
      simd: {
        median_ms: Number(median(timesByFlavor.simd).toFixed(3)),
        p95_ms: Number(percentile(timesByFlavor.simd, 0.95).toFixed(3)),
        min_ms: Number(Math.min(...timesByFlavor.simd).toFixed(3)),
        max_ms: Number(Math.max(...timesByFlavor.simd).toFixed(3)),
        cv: Number(cv(timesByFlavor.simd).toFixed(4)),
      },
    };
  }

  for (const [scenario, byFlavor] of Object.entries(report.scenarios)) {
    const base = byFlavor.base.median_ms;
    const simd = byFlavor.simd.median_ms;
    const speedup = base / simd;
    console.log(`\n[${scenario}]`);
    console.log(
      `  warmup pairs: ${byFlavor.warmup.warmup_pairs} (base_cv=${byFlavor.warmup.warmup_base_cv}, simd_cv=${byFlavor.warmup.warmup_simd_cv})`,
    );
    console.log(`  base median: ${base.toFixed(3)} ms`);
    console.log(`  simd median: ${simd.toFixed(3)} ms`);
    console.log(`  speedup: ${speedup.toFixed(3)}x`);
  }

  console.log("\n--- JSON ---");
  console.log(JSON.stringify(report, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
