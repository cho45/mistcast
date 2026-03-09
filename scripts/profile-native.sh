#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DSP_DIR="$ROOT/dsp"
BUILD_PROFILE="${BUILD_PROFILE:-profiling}"
BIN="$DSP_DIR/target/$BUILD_PROFILE/dsss_e2e_eval"
OUT_DIR="${OUT_DIR:-$ROOT/dsp/eval/profiles/native}"
PROFILE_TOOL="${PROFILE_TOOL:-auto}" # auto|samply|perf|none
PROFILE_PHY="${PROFILE_PHY:-mary}" # mary|dsss
TOTAL_SIM_SEC="${TOTAL_SIM_SEC:-60}"
MIN_SAMPLE_COUNT="${MIN_SAMPLE_COUNT:-1500}"
SAMPLY_SAVE_ONLY="${SAMPLY_SAVE_ONLY:-1}"
ALLOC_PROFILE="${ALLOC_PROFILE:-1}"

mkdir -p "$OUT_DIR"

echo "[profile-native] building dsss_e2e_eval (profile=$BUILD_PROFILE)"
cargo build --profile "$BUILD_PROFILE" --bin dsss_e2e_eval --manifest-path "$DSP_DIR/Cargo.toml" >/dev/null

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

ARGS=(
  --phy "$PROFILE_PHY"
  --mode point
  --total-sim-sec "$TOTAL_SIM_SEC"
  --payload-bytes 64
  --chunk-samples 2048
  --output csv
)

if [ "$#" -gt 0 ]; then
  ARGS+=("$@")
fi

if [ "$ALLOC_PROFILE" = "1" ]; then
  ARGS+=(--alloc-profile)
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
    OUT_FILE="$OUT_DIR/${PROFILE_PHY}_e2e_${TIMESTAMP}.json"
    SAMPLY_ARGS=(-o "$OUT_FILE")
    if [ "$SAMPLY_SAVE_ONLY" = "1" ]; then
      SAMPLY_ARGS+=(-s)
    fi
    samply record "${SAMPLY_ARGS[@]}" -- "$BIN" "${ARGS[@]}"
    SAMPLE_COUNT="$(jq '.threads[0].samples.stack | map(select(. != null and . >= 0)) | length' "$OUT_FILE")"
    if [ "${SAMPLE_COUNT:-0}" -lt "$MIN_SAMPLE_COUNT" ]; then
      echo "[profile-native] sample count too small: ${SAMPLE_COUNT} (< ${MIN_SAMPLE_COUNT})" >&2
      echo "[profile-native] hint: increase TOTAL_SIM_SEC (current=${TOTAL_SIM_SEC})" >&2
      rm -f "$OUT_FILE"
      exit 1
    fi
    if command -v python3 >/dev/null 2>&1; then
      python3 "$ROOT/scripts/profile-native-summary.py" "$OUT_FILE" || true
    fi
    echo "[profile-native] wrote: $OUT_FILE"
    ;;
  perf)
    OUT_FILE="$OUT_DIR/${PROFILE_PHY}_e2e_${TIMESTAMP}.perf.data"
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
