//! 畳み込み符号 (Convolutional Code) + Viterbiデコーダ
//!
//! # パラメータ
//! - 拘束長 K=7 (NASA Voyager 標準)
//! - 符号化率 R=1/2
//! - 生成多項式 G1=0o171 (109), G2=0o133 (91) (標準NASA多項式)
//!
//! # 動作
//! - エンコーダ: 1ビット入力 → 2ビット出力
//! - デコーダ: Viterbiアルゴリズム (ハード判定)
//!
//! # Viterbiアルゴリズムの状態数
//! 2^(K-1) = 2^6 = 64 状態

const CONSTRAINT_LEN: usize = 7;
const NUM_STATES: usize = 1 << (CONSTRAINT_LEN - 1); // 64
const G1: u8 = 0o171; // 0b1111001 = 121
const G2: u8 = 0o133; // 0b1011011 = 91

/// 生成多項式と状態からパリティビットを計算 (偶数パリティ)
#[inline]
fn parity(x: u8) -> u8 {
    x.count_ones() as u8 & 1
}

/// 現在の状態 `state` にビット `bit` を入力したときの出力シンボル (2ビット)
/// state は K-1=6 ビット幅
#[inline]
fn conv_output(state: u8, bit: u8) -> (u8, u8) {
    // 新しいシフトレジスタ値 (上位 K-1 ビット + 入力ビット)
    let reg = (state >> 1) | (bit << (CONSTRAINT_LEN as u8 - 2));
    // G1, G2との積のパリティ
    let v1 = parity(reg & G1);
    let v2 = parity(reg & G2);
    (v1, v2)
}

/// 次の状態を計算する
#[inline]
fn next_state(state: u8, bit: u8) -> u8 {
    ((state >> 1) | (bit << (CONSTRAINT_LEN as u8 - 2))) & (NUM_STATES as u8 - 1)
}

/// 畳み込み符号エンコーダ
///
/// ビット列を符号化する。出力は入力の2倍長。
/// 末尾にtailbits (K-1個のゼロ) を付加してトレリスをフラッシュする。
pub fn encode(bits: &[u8]) -> Vec<u8> {
    let mut state: u8 = 0;
    let mut out = Vec::with_capacity((bits.len() + CONSTRAINT_LEN - 1) * 2);

    // データビット
    for &bit in bits {
        let (v1, v2) = conv_output(state, bit);
        out.push(v1);
        out.push(v2);
        state = next_state(state, bit);
    }

    // テールビット: K-1 個の 0 を入力してトレリスをリセット
    for _ in 0..(CONSTRAINT_LEN - 1) {
        let (v1, v2) = conv_output(state, 0);
        out.push(v1);
        out.push(v2);
        state = next_state(state, 0);
    }

    out
}

/// Viterbiデコーダ (ハード判定)
///
/// 符号化ビット列を復号する。入力の約半分長のビット列を返す。
/// テールビットを考慮する。
pub fn decode(coded_bits: &[u8]) -> Vec<u8> {
    assert!(coded_bits.len().is_multiple_of(2), "符号化ビット列は偶数長であること");
    let num_symbols = coded_bits.len() / 2;

    // トレリス: path_metric[state] = 累積ハミング距離
    const INF: u32 = u32::MAX / 2;
    let mut path_metrics = vec![INF; NUM_STATES];
    path_metrics[0] = 0;

    // サバイバーパス: survivor[time][state] = 前の状態
    let mut survivors: Vec<Vec<u8>> = Vec::with_capacity(num_symbols);

    for sym_idx in 0..num_symbols {
        let r0 = coded_bits[sym_idx * 2];
        let r1 = coded_bits[sym_idx * 2 + 1];

        let mut new_metrics = vec![INF; NUM_STATES];
        let mut survivor = vec![0u8; NUM_STATES];

        // 各状態への遷移を評価
        for (prev_state, &metric) in path_metrics.iter().enumerate().take(NUM_STATES) {
            if metric == INF {
                continue;
            }
            for bit in 0u8..2 {
                let (v1, v2) = conv_output(prev_state as u8, bit);
                // ハミング距離
                let branch_metric = (v1 ^ r0) as u32 + (v2 ^ r1) as u32;
                let ns = next_state(prev_state as u8, bit) as usize;
                let total = metric + branch_metric;
                if total < new_metrics[ns] {
                    new_metrics[ns] = total;
                    survivor[ns] = prev_state as u8;
                }
            }
        }

        path_metrics = new_metrics;
        survivors.push(survivor);
    }

    // 最良の終端状態を選択 (テールビットにより状態0に収束するはず)
    let best_end_state = path_metrics
        .iter()
        .enumerate()
        .min_by_key(|&(_, &m)| m)
        .map(|(s, _)| s)
        .unwrap_or(0);

    // トレースバック
    let data_len = num_symbols.saturating_sub(CONSTRAINT_LEN - 1);
    let mut decoded = vec![0u8; data_len];
    let mut state = best_end_state as u8;

    for t in (0..num_symbols).rev() {
        let prev = survivors[t][state as usize];
        // このステップの入力ビットを復元
        // next_state(prev, bit) == state を満たすbitを見つける
        let bit = if (prev >> 1) | (1u8 << (CONSTRAINT_LEN as u8 - 2)) == state {
            1u8
        } else {
            0u8
        };
        if t < data_len {
            decoded[t] = bit;
        }
        state = prev;
    }

    decoded
}

/// ビット列をバイト列に変換 (MSB first)
pub fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    bits.chunks(8)
        .map(|chunk| {
            chunk.iter().enumerate().fold(0u8, |acc, (i, &b)| {
                acc | (b << (7 - i))
            })
        })
        .collect()
}

/// バイト列をビット列に変換 (MSB first)
pub fn bytes_to_bits(bytes: &[u8]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(bytes.len() * 8);
    for &byte in bytes {
        for i in (0..8).rev() {
            bits.push((byte >> i) & 1);
        }
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    /// エンコード→デコードの往復テスト (エラーなし)
    #[test]
    fn test_encode_decode_no_error() {
        let original: Vec<u8> = (0..8).map(|_| 1u8).chain(vec![0u8; 8]).collect();
        let coded = encode(&original);
        // coded.len() = (original.len() + K-1) * 2
        assert_eq!(coded.len(), (original.len() + CONSTRAINT_LEN - 1) * 2);

        let decoded = decode(&coded);
        assert_eq!(decoded.len(), original.len());
        assert_eq!(decoded, original, "エラーなし時にデコード結果が一致すること");
    }

    /// ランダムデータのエンコード→デコード
    #[test]
    fn test_encode_decode_random() {
        // 疑似乱数的なビット列 (u32で演算してからu8に変換)
        let original: Vec<u8> = (0..64u32)
            .map(|i| ((i * 37 + 11) % 2) as u8)
            .collect();
        let coded = encode(&original);
        let decoded = decode(&coded);
        assert_eq!(decoded, original, "ランダムビット列のデコード結果が一致すること");
    }

    /// 1ビットエラーの訂正確認
    #[test]
    fn test_single_bit_error_correction() {
        let original: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0];
        let mut coded = encode(&original);
        // 中央付近の1ビットを反転
        let flip_pos = coded.len() / 2;
        coded[flip_pos] ^= 1;
        let decoded = decode(&coded);
        assert_eq!(
            decoded, original,
            "1ビットエラーが訂正されること"
        );
    }

    /// エンコード長の確認
    #[test]
    fn test_encoded_length() {
        for input_len in [1, 8, 16, 32] {
            let bits = vec![1u8; input_len];
            let coded = encode(&bits);
            let expected_len = (input_len + CONSTRAINT_LEN - 1) * 2;
            assert_eq!(
                coded.len(),
                expected_len,
                "input_len={} の符号化後長が正しいこと",
                input_len
            );
        }
    }

    /// bytes_to_bits / bits_to_bytes 往復テスト
    #[test]
    fn test_byte_bit_conversion() {
        let original = vec![0xABu8, 0xCD, 0xEF];
        let bits = bytes_to_bits(&original);
        assert_eq!(bits.len(), 24);
        let recovered = bits_to_bytes(&bits);
        assert_eq!(recovered, original);
    }
}
