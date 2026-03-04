# Baseline: Mary PHY 受信性能（等化器なし）

等化器実装前のベースラインを記録する。

## 測定条件

- **コミット**: `839dfd6` (feat(eval): add raw_ber measurement via LlrCallback in dsss_e2e_eval)
- **PHY**: Mary DQPSK
- **プリアンブル SF**: 63
- **マルチパスプロファイル**: harsh (`(0, 1.0), (9, 0.55), (23, 0.35), (49, 0.25), (87, 0.18)`)
- **trials**: 40
- **payload**: 64 bytes
- **chunk_samples**: 16384 (デフォルト)
- **等化器**: なし

## コマンド

```sh
cargo run --release --bin dsss_e2e_eval -- \
  --phy mary \
  --mode sweep-awgn \
  --multipath harsh \
  --preamble-sf 63 \
  --trials 40 \
  --sweep-awgn "0,0.05,0.1,0.2,0.3,0.5,0.7,1.0"
```

## 結果

`raw_ber` は `LlrCallback` によって計測した FEC 符号化ビットのハード判定 BER。
Fountain 符号の誤り訂正をバイパスしたチャネル BER そのもの。

| sigma | SNR (dB) | raw_ber   | p_complete |
|-------|----------|-----------|------------|
| 0.00  | ∞        | 0.000162  | 100%       |
| 0.05  | 17.8     | 0.000180  | 100%       |
| 0.10  | 11.8     | 0.000287  | 100%       |
| 0.20  | 5.8      | 0.000521  | 100%       |
| 0.30  | 2.2      | 0.001347  | 100%       |
| 0.50  | -2.2     | 0.017906  | 100%       |
| 0.70  | -5.1     | 0.124851  | 100%       |
| 1.00  | -8.2     | 0.435824  | 0%         |

## 観察

- sigma=0.0 でも raw_ber ≈ 0.016% のフロアが存在 → harsh マルチパスによる ISI（符号間干渉）に起因する。これが等化器で改善できる対象。
- Fountain 符号の強力な誤り訂正により p_complete は sigma=0.7 まで 100% を維持するが、raw_ber はすでに 12% に達している。
- sigma=0.5 (SNR ≈ -2 dB) が実用上の限界付近。等化器実装後に同じ条件で比較する。
