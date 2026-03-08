#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/dsp/eval/profiles/wasm-node}"
SKIP_BUILD="${SKIP_BUILD:-0}"
BENCH_ITERS="${BENCH_ITERS:-30}"
BENCH_WARMUP_MIN_PAIRS="${BENCH_WARMUP_MIN_PAIRS:-5}"
BENCH_WARMUP_MAX_PAIRS="${BENCH_WARMUP_MAX_PAIRS:-20}"
BENCH_STABLE_WINDOW="${BENCH_STABLE_WINDOW:-5}"
BENCH_STABLE_CV="${BENCH_STABLE_CV:-0.03}"
BENCH_FRAMES="${BENCH_FRAMES:-12}"

mkdir -p "$OUT_DIR"

cd "$ROOT"
if [ "$SKIP_BUILD" != "1" ]; then
  echo "[profile-wasm-node] building wasm (base + simd)"
  npm run build:wasm >/dev/null
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CPU_PROFILE_NAME="bench-simd-${TIMESTAMP}.cpuprofile"

echo "[profile-wasm-node] writing profile to $OUT_DIR/$CPU_PROFILE_NAME"
BENCH_ITERS="$BENCH_ITERS" \
BENCH_WARMUP_MIN_PAIRS="$BENCH_WARMUP_MIN_PAIRS" \
BENCH_WARMUP_MAX_PAIRS="$BENCH_WARMUP_MAX_PAIRS" \
BENCH_STABLE_WINDOW="$BENCH_STABLE_WINDOW" \
BENCH_STABLE_CV="$BENCH_STABLE_CV" \
BENCH_FRAMES="$BENCH_FRAMES" \
node \
  --expose-gc \
  --cpu-prof \
  --cpu-prof-dir="$OUT_DIR" \
  --cpu-prof-name="$CPU_PROFILE_NAME" \
  scripts/bench-simd.js

echo "[profile-wasm-node] done: $OUT_DIR/$CPU_PROFILE_NAME"
