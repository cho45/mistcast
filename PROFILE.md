# Profiling 手順

このプロジェクトは、目的に応じて CPU プロファイルと alloc プロファイルを分けて取得する。

## 前提

- ルートディレクトリ: `mistcast/`
- Node (`npm` / `node`) と Rust (`cargo`) が使えること
- `dsp/Cargo.toml` の `profile.profiling` を使うこと（`debug=2`, `strip=none`）

## 1. Native CPU プロファイル（samply / perf）

### 実行（推奨）

```bash
npm run profile:native:dsss
```

または:

```bash
make profile-native-dsss
```

### 動作

- `cargo build --profile profiling --bin dsss_e2e_eval`
- 既定 PHY は `mary`（`PROFILE_PHY` で変更可）
- 既定 `TOTAL_SIM_SEC=60`
- profiler は自動選択（`samply` → `perf` → なし）
- 既定で `--alloc-profile` を有効化（`tx/channel/rx` の phase 別集計）
- 出力先: `dsp/eval/profiles/native/`

### よく使うオプション

```bash
PROFILE_TOOL=samply npm run profile:native:dsss
PROFILE_TOOL=perf npm run profile:native:dsss
PROFILE_PHY=dsss npm run profile:native:dsss
TOTAL_SIM_SEC=30 npm run profile:native:dsss
MIN_SAMPLE_COUNT=1000 npm run profile:native:dsss
ALLOC_PROFILE=0 npm run profile:native:dsss
BUILD_PROFILE=release npm run profile:native:dsss
```

### 取得後の見方

`samply`:

```bash
samply load dsp/eval/profiles/native/<file>.json
```

`perf`:

```bash
perf report -i dsp/eval/profiles/native/<file>.perf.data
perf report -i dsp/eval/profiles/native/<file>.perf.data --stdio | head -n 200
```

## 2. Native alloc プロファイル（dhat-rs）

alloc の発生源特定は `dhat-rs` を使う。これは外部計装に近い運用で、本番ライブラリに恒久的な計装を入れずに callsite 単位で確認できる。

### 実行

```bash
cd dsp
cargo run --profile profiling --features alloc-prof-dhat --bin dsss_e2e_eval -- \
  --phy mary --mode point --total-sim-sec 20 --payload-bytes 64 --chunk-samples 2048 --seed 42 --output csv
```

`dsss` を測る場合:

```bash
cargo run --profile profiling --features alloc-prof-dhat --bin dsss_e2e_eval -- \
  --phy dsss --mode point --total-sim-sec 20 --payload-bytes 64 --chunk-samples 2048 --seed 43 --output csv
```

### 出力

- 実行終了時に `dsp/dhat-heap.json` が生成される
- 連続実行時は上書きされるため、即座に退避すること

```bash
mv dsp/dhat-heap.json dsp/eval/profiles/native/dhat-<phy>-<timestamp>.json
```

### 集計例（関数別の累積 alloc bytes）

```bash
cd dsp
jq -r '
  . as $r
  | ( [ .pps[].tb ] | add ) as $total
  | [ .pps[]
      | . as $p
      | (([$p.fs[] | $r.ftbl[.] | select(test("dsp::|dsss_e2e_eval::"))][0]) // $r.ftbl[$p.fs[0]]) as $raw
      | ($raw | sub("^0x[0-9a-f]+: "; "") | split(" (")[0]) as $func
      | {func:$func, tb:$p.tb, tbk:$p.tbk}
    ]
  | group_by(.func)
  | map({func:.[0].func, tb:(map(.tb)|add), tbk:(map(.tbk)|add), pct: ((map(.tb)|add) * 100.0 / $total)})
  | sort_by(-.tb)
  | .[:15]
  | .[]
  | "\(.pct|tostring)% tb=\(.tb) tbk=\(.tbk) \(.func)"
' dhat-heap.json
```

## 3. WASM(Node) CPU プロファイル

### 実行（推奨）

```bash
npm run profile:wasm:node
```

または:

```bash
make profile-wasm-node
```

### 動作

- 既定で `npm run build:wasm` 後にベンチを実行
- `node --cpu-prof --expose-gc` で CPU profile を保存
- 出力先: `dsp/eval/profiles/wasm-node/*.cpuprofile`

### オプション

```bash
SKIP_BUILD=1 npm run profile:wasm:node
BENCH_ITERS=50 BENCH_WARMUP_MIN_PAIRS=8 BENCH_WARMUP_MAX_PAIRS=30 npm run profile:wasm:node
```

## 4. FEC List Viterbi マイクロベンチ

`List Viterbi` の `list_size (K)` を増やしたときのスケーリングを確認する。
ベンチ実装は `dsp/benches/native_dsp.rs` の
`fec/decode_soft_list_into/packet_llr348/k={1,2,4,8,16,32}`。

### 実行

```bash
cargo bench --manifest-path dsp/Cargo.toml --bench native_dsp -- 'fec/decode_soft_list_into'
```

### 特定 K のみ実行

`criterion` のフィルタは部分一致のため、`k=1` は `k=16` も拾う。
正確に 1 本だけ回す場合はアンカー付き正規表現を使う。

```bash
cargo bench --manifest-path dsp/Cargo.toml --bench native_dsp -- '^fec/decode_soft_list_into/packet_llr348/k=1$'
```

### 見るべき値

- `time`（1 codeword あたりの処理時間）
- `thrpt`（bit/s ではなく、ベンチ設定上の elements/s）
- K 増加時の増え方（線形に近いか、急激に悪化するか）

## 5. 比較時のポイント

- 同じ入力条件（seed / total_sim_sec / payload-bytes / chunk-samples）で比較する
- CPU は Self % / Total % を見る
- alloc は `tb`（累積 bytes）と `tbk`（累積 blocks）を関数別に比較する
- 差分 5% 未満はノイズの可能性があるため、反復して再測定する

## 6. 注意点

- `alloc-prof-dhat` 有効時は `dhat` が global allocator を使用する
- そのため `--alloc-profile` は同時利用しても有効な内訳にならない（dhat 計測時は付けない）
- wasm 比較前に `build:wasm` を実行し `pkg` / `pkg-simd` を更新する
