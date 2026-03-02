# Mistcast

Mistcastは、ブラウザ上で動作する**音響データ通信（Data-over-Sound）ウェブアプリケーション**です。
スピーカーとマイクを使用し、音響空間を介したデータ伝送を実装しています。

## 特徴

- **ファウンテンコード（消失訂正符号）によるデータ復元**: RLNC（ランダム線形ネットワークコーディング）を採用しており、送信側は符号化パケットを継続的に射出します。受信側は、そのうち線形独立な任意の $n$ 個のパケットを受信した段階で元のデータを復元可能です。
- **ストリームの任意位置からの復号**: 各バーストが独立した同期情報（プリアンブル）を持つ設計となっており、送信が開始されているストリームのどのタイミングから受信を開始しても、必要なパケット数が揃い次第デコードを完了できます。
- **ブラウザ標準機能のみを利用**: Web Audio API および WebAssembly を使用し、特別なプラグインなしで現代的なブラウザ上で動作します。

## 技術仕様

### 1. 物理レイヤー (変調・復調)
- **対称リサンプリング**: 入出力サンプリングレート（44.1kHz/48kHz等）を、リサンプラによって内部処理レート（$3 \times R_c$）へ変換。サンプリングレートの不一致に起因するタイミング累積誤差を抑制する設計を採用。
- **DSSS (Direct Sequence Spread Spectrum)**: M系列（次数4, 長さ15）を用いた直接拡散。
- **DQPSK (Differential Quadrature Phase Shift Keying)**: 差動四位相偏移変調。
- **RRC (Root-Raised Cosine) フィルタ**: ロールオフ係数 0.30。
- **既定パラメータ**:
    - **キャリア周波数 ($f_c$)**: 15,000 Hz
    - **チップレート ($R_c$)**: 8,000 Hz
    - **占有帯域**: 約 10.4 kHz ($15,000 \pm 4,000 \times 1.3$)

### 2. 同期捕捉 (Synchronization)
- **短フレーム同期**: 1バーストあたりのパケット数を1〜2に制限し、プリアンブルを高頻度（バースト単位）で挿入。
- **同期ワード**: 16ビット（DBPSK）。

### 3. 誤り訂正と消失訂正
- **FEC**: 畳み込み符号（K=7, R=1/2）とViterbi復号によるビット誤り訂正。
- **CRC-16**: パケット単位（26バイト）の誤り検出。
- **RLNC (Random Linear Network Coding)**: GF(256) 上の消失訂正符号。ラットレス通信（Fountain Code）の性質を利用し、受信側で線形独立なパケットが必要数に達した段階でデータを復元。

## パケット構造

### 1. 物理層フレーム構造 (Burst Frame)
送信される1つのバーストは以下の構造を持つ。

```text
| Preamble (2 symbols) | Sync Word (16 bits) | Packet 1 (Coded) | Packet 2 (Coded) | Margin |
```

| セクション | 構成 | 役割 |
| :--- | :--- | :--- |
| **Preamble** | M系列(15ch) × 2回 `[M, -M]` | キャリア同期、タイミング抽出。 |
| **Sync Word** | 16 bits (DBPSK) | フレーム同期。 |
| **Packet n** | FEC済みビット列 (DQPSK) | 符号化されたデータパケット。 |

### 2. データリンク層 (Packet Format)
消失訂正と誤り検出を統合した、固定 **26バイト** (Payload 16B時) のパケット形式。

## 開発と評価

### クイックスタート
1. **依存関係のインストール**: `npm install`
2. **WASM コアのビルド**: `npm run build:wasm`
3. **開発サーバーの起動**: `npm run dev` (`http://localhost:5173`)

### テストとベンチマーク
- **DSP Core ユニットテスト**: `cd dsp && cargo test --release`
- **物理層ベンチマーク**: `scripts/phy_eval/` のスクリプトを使用し、シミュレートされたノイズ、マルチパス、CFO（周波数偏差）環境下での完遂率（$p_{complete}$）および BER を計測。

## ディレクトリ構造

```text
.
├── src/                # Webフロントエンド (Vue 3 / TypeScript)
│   ├── components/     # Sender/Receiver UIコンポーネント
│   ├── worker.ts       # DSP処理を実行する Web Worker
│   └── audio-processors.ts # Web Audio API と Worker のブリッジ
├── dsp/                # DSPコアエンジン (Rust)
│   ├── src/
│   │   ├── lib.rs              # 全体設定・パブリックAPI
│   │   ├── encoder.rs          # 送信処理
│   │   ├── decoder.rs          # 受信処理
│   │   ├── common/             # 数学的プリミティブ (M系列, CRC, RRC, NCO, Resample)
│   │   ├── phy/                # 物理層 (Modulator, Demodulator, Sync)
│   │   ├── coding/             # 符号化層 (FEC, Fountain, Interleaver)
│   │   └── frame/              # データリンク層 (Packet)
│   └── tests/          # 通信シミュレーションテスト
└── scripts/            # WASMビルド・評価用スクリプト
```

## 技術スタック
- **Frontend**: Vue 3, Vite, TypeScript
- **DSP Core**: Rust, WebAssembly (SIMD)
- **Communication**: Web Audio API, Web Workers (Comlink)

## ライセンス

MIT License
