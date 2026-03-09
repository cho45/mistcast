# Profiling 手順

このプロジェクトのプロファイル取得は、まず native で支配率を見て、その後 wasm(Node) で確認する。

## 前提

- ルートディレクトリ: `mistcast/`
- Node が使えること（`npm` / `node`）
- Rust ツールチェインが使えること（`cargo`）

## 1. Native (Rust) の e2e プロファイル

### 実行（推奨）

```bash
npm run profile:native:dsss
```

または:

```bash
make profile-native-dsss
```

### 動作

- 既定で `dsp/target/profiling/dsss_e2e_eval` をビルドして実行
- 既定 PHY は `mary`（`PROFILE_PHY` で変更可）
- 既定 `TOTAL_SIM_SEC=60` で、1回計測でもサンプル数を確保しやすくしている
- profiler は自動選択（`samply` → `perf` → なし）
- 出力先: `dsp/eval/profiles/native/`

### オプション

- profiler 指定:

```bash
PROFILE_TOOL=samply npm run profile:native:dsss
PROFILE_TOOL=perf npm run profile:native:dsss
PROFILE_TOOL=none npm run profile:native:dsss
```

- PHY 指定:

```bash
PROFILE_PHY=mary npm run profile:native:dsss
PROFILE_PHY=dsss npm run profile:native:dsss
```

- 計測時間変更:

```bash
TOTAL_SIM_SEC=30 npm run profile:native:dsss
```

- サンプル数下限（既定 1500）:

```bash
MIN_SAMPLE_COUNT=1000 npm run profile:native:dsss
```

- ビルドプロファイル指定（必要時）:

```bash
BUILD_PROFILE=release npm run profile:native:dsss
```

## 2. WASM (Node) の e2e プロファイル

### 実行（推奨）

```bash
npm run profile:wasm:node
```

または:

```bash
make profile-wasm-node
```

### 動作

- 既定で `npm run build:wasm` 実行後にベンチを回す
- `node --cpu-prof --expose-gc` で CPU profile を保存
- 出力先: `dsp/eval/profiles/wasm-node/*.cpuprofile`

### オプション

- wasm 再ビルドを省略:

```bash
SKIP_BUILD=1 npm run profile:wasm:node
```

- ベンチ条件調整:

```bash
BENCH_ITERS=50 BENCH_WARMUP_MIN_PAIRS=8 BENCH_WARMUP_MAX_PAIRS=30 npm run profile:wasm:node
```

## 3. まず見るべき指標

- **Self %** が高い関数（純粋ホットスポット）
- **Total %** が高い関数（配下含む支配領域）
- base/simd の差分で、どこが改善・悪化しているか

## 4. 取得後の開き方（重要）

### Native: samply の場合

- `scripts/profile-native.sh` は `samply record -o <json>` を実行する。
- 既定で `--save-only` を付けるため、record 後にローカルサーバは起動しない。
- record 後に `scripts/profile-native-summary.py` で self 上位の関数名サマリを出力する（`atos` 使用）。
- 出力ファイルを `samply` で開く:

```bash
samply load dsp/eval/profiles/native/<file>.json
```

- もし record 後に UI を自動で開きたい場合:

```bash
SAMPLY_SAVE_ONLY=0 npm run profile:native:dsss
```

### Native: perf の場合

- `scripts/profile-native.sh` は `perf record -g -o <file>.perf.data` を実行する。
- レポート表示:

```bash
perf report -i dsp/eval/profiles/native/<file>.perf.data
```

- 関数別の概況をテキストで見る:

```bash
perf report -i dsp/eval/profiles/native/<file>.perf.data --stdio | head -n 200
```

### WASM(Node): `--cpu-prof` の場合

- 出力: `dsp/eval/profiles/wasm-node/*.cpuprofile`
- Chrome DevTools で開く:
  1. Chrome で DevTools を開く
  2. `Performance` タブを開く
  3. 右上メニューから `Load profile...` を選ぶ
  4. `.cpuprofile` を読み込む

- もしくは Node Inspector の `Profiler` で読み込む

### base/simd 比較時のポイント

- 同じ入力条件（フレーム数・シード・`total_sim_sec`）で2回測る
- 各 run で上位 10 関数の Self/Total をメモする
- 差分が 5% 未満ならノイズの可能性があるため、反復回数を増やして再測定する

## 5. 注意点

- wasm 比較前に `build:wasm` を実行し、`pkg` / `pkg-simd` を最新化すること
- ばらつきが大きい場合は `BENCH_ITERS` を増やす（30 以上推奨）
- Node 側は `--expose-gc` 前提で実行する（既存 script は対応済み）
