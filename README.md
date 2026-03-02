# Mistcast

Mistcastは、音響環境下でのデータ伝送を目的としたDSP（デジタル信号処理）コアライブラリです。
スペクトラム拡散通信、前方誤り訂正、および消失訂正符号を組み合わせ、音響空間特有の物理的制約（ノイズ、反射、サンプリングレートの不一致）に対処するアーキテクチャを採用しています。

## 概要

本プロジェクトは、マイクとスピーカーを用いた音響データ通信（Data-over-Sound）の実装を提供します。WebAudio API 等のストリーミング環境での利用を想定し、サンプルチャンクの境界条件においても状態を維持するデコーダパイプラインを特徴としています。

## ディレクトリ構造

```text
dsp/src/
├── lib.rs              # 全体設定・パブリックAPI
├── encoder.rs          # 送信処理
├── decoder.rs          # 受信処理
├── common/             # 数学的プリミティブ
│   ├── msequence.rs    # M系列生成
│   ├── crc.rs          # 巡回冗長検査 (CRC-16)
│   ├── rrc_filter.rs   # RRCフィルタ
│   ├── nco.rs          # 数値制御発振器
│   ├── resample.rs     # 対称リサンプリング
│   └── decimator.rs    # (Legacy)
├── phy/                # 物理層
│   ├── modulator.rs    # 変調 (DQPSK/DSSS)
│   ├── demodulator.rs  # 復調
│   └── sync.rs         # 同期捕捉
├── coding/             # 符号化層
│   ├── fec.rs          # 誤り訂正 (Convolutional/Viterbi)
│   ├── fountain.rs     # 消失訂正 (RLNC over GF(256))
│   └── interleaver.rs  # ブロックインターリーバー
└── frame/              # データリンク層
    └── packet.rs       # パケット構造
```

## 技術的特徴

### 1. 物理レイヤー (変調・復調)
- **対称リサンプリング**: ハードウェアサンプルレート（44.1kHz/48kHz等）と、DSPコアの処理レート（$3 \times R_c$）をリサンプラによって分離しています。これにより、非整数倍のサンプルレートに起因する理論的なタイミング累積誤差の抑制を図っています。
- **DSSS (Direct Sequence Spread Spectrum)**: M系列（次数4, 長さ15）を用いた直接拡散。
- **DQPSK (Differential Quadrature Phase Shift Keying)**: 差動四位相偏移変調。
- **RRC (Root-Raised Cosine) フィルタ**: ロールオフ係数0.30。
- **デフォルトパラメータ (室内環境向け実測に基づく)**:
    - **キャリア周波数 ($f_c$)**: 15,000 Hz
    - **チップレート ($R_c$)**: 8,000 Hz

### 2. 同期捕捉 (Synchronization)
- **短フレーム同期**: 1バーストを短く構成し、プリアンブルを高頻度に挿入する設計としています。
- **動的しきい値調整**: 拡散率（SF）の変化に応じた、相関スコア判定しきい値の外部設定に対応しています。
- **同期ワード**: 16ビット（デフォルト）。

### 3. 誤り訂正と消失訂正
- **FEC**: 畳み込み符号（K=7, R=1/2）とViterbi復号。
- **CRC-16**: パケット単位の誤り検出。
- **RLNC (Random Linear Network Coding)**: GF(256) 上の消失訂正符号。受信側で必要な数の線形独立なパケットが得られた段階でデータを復元する、ラットレス通信の性質を利用しています。

## パケット構造

### 1. 物理層フレーム構造 (Burst Frame)
送信される1つのバーストは、以下の構造を持ちます。

```text
| Preamble (2 symbols) | Sync Word (16) | Packet 1 (Coded) | Packet 2 (Coded) | Margin |
```

| セクション | 構成 | 役割 |
| :--- | :--- | :--- |
| **Preamble** | M系列(15ch) × 2回 `[M, -M]` | キャリア同期、タイミング抽出。 |
| Sync Word | 16 bits (DBPSK) | フレーム同期。 |
| **Packet n** | FEC済みビット列 (DQPSK) | 実データ。通常、1フレームに1パケットを格納。 |


### 2. データリンク層 (Packet Format)
消失訂正（Fountain Code）と誤り検出（CRC）を統合した、固定 **26バイト** (Payload 16B時) のパケットです。

## テストと評価

### ユニットテスト

```bash
cd dsp
cargo test --release
```

### 物理層ベンチマーク
`scripts/phy_eval/` に、物理層の性能を評価するためのスクリプトを備えています。

1. **論理探索範囲の算出**: 物理制約に基づき、理論的に成立可能なパラメータ候補を算出します。
   ```bash
   ./.venv/bin/python scripts/phy_eval/calc_smoke_range.py
   ```
2. **実測ベンチマーク**: 算出した候補に対し、シミュレートされたノイズ、マルチパス、CFO環境下での完遂率を計測します。
   ```bash
   ./.venv/bin/python scripts/phy_eval/phy_bench.py
   ```

## 設計上の留意点
- **サンプリングレートへの対応**: 対称リサンプリングにより、異なるサンプルレート間での相互運用性を高める設計を行っています。
- **数学的一貫性**: フィルタ遅延や相関位置について、理論値に基づく実装を試みています。
- **ストリーミング処理**: `Decoder` は入力データの長さに依存せず、内部状態を維持しながら継続的に処理を行うように設計されています。
