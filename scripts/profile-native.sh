#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DSP_DIR="$ROOT/dsp"
BIN="$DSP_DIR/target/release/dsss_e2e_eval"
OUT_DIR="${OUT_DIR:-$ROOT/dsp/eval/profiles/native}"
PROFILE_TOOL="${PROFILE_TOOL:-auto}" # auto|samply|perf|none
TOTAL_SIM_SEC="${TOTAL_SIM_SEC:-20}"
SAMPLY_SAVE_ONLY="${SAMPLY_SAVE_ONLY:-1}"

mkdir -p "$OUT_DIR"

echo "[profile-native] building dsss_e2e_eval (release)"
cargo build --release --bin dsss_e2e_eval --manifest-path "$DSP_DIR/Cargo.toml" >/dev/null

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

ARGS=(
  --phy dsss
  --mode point
  --total-sim-sec "$TOTAL_SIM_SEC"
  --payload-bytes 64
  --chunk-samples 2048
  --output csv
)

if [ "$#" -gt 0 ]; then
  ARGS+=("$@")
fi

pick_tool() {
  if [ "$PROFILE_TOOL" != "auto" ]; then
    echo "$PROFILE_TOOL"
    return
  fi
  if command -v samply >/dev/null 2>&1; then
    echo "samply"
    return
  fi
  if command -v perf >/dev/null 2>&1; then
    echo "perf"
    return
  fi
  echo "none"
}

TOOL="$(pick_tool)"
echo "[profile-native] tool=$TOOL"
echo "[profile-native] command: $BIN ${ARGS[*]}"

case "$TOOL" in
  samply)
    OUT_FILE="$OUT_DIR/dsss_e2e_${TIMESTAMP}.json"
    SAMPLY_ARGS=(-o "$OUT_FILE")
    if [ "$SAMPLY_SAVE_ONLY" = "1" ]; then
      SAMPLY_ARGS+=(-s)
    fi
    samply record "${SAMPLY_ARGS[@]}" -- "$BIN" "${ARGS[@]}"
    echo "[profile-native] wrote: $OUT_FILE"
    ;;
  perf)
    OUT_FILE="$OUT_DIR/dsss_e2e_${TIMESTAMP}.perf.data"
    PERF_FREQ="${PERF_FREQ:-999}"
    perf record -F "$PERF_FREQ" -g -o "$OUT_FILE" -- "$BIN" "${ARGS[@]}"
    echo "[profile-native] wrote: $OUT_FILE"
    echo "[profile-native] inspect with: perf report -i $OUT_FILE"
    ;;
  none)
    "$BIN" "${ARGS[@]}"
    echo "[profile-native] profiler not found (ran without profiler)"
    ;;
  *)
    echo "[profile-native] unknown PROFILE_TOOL=$TOOL" >&2
    exit 2
    ;;
esac
