//! M系列 (Maximum Length Sequence) 生成器
//!
//! GF(2)上の線形帰還シフトレジスタ (LFSR) により、
//! 次数nのM系列(長さ 2^n - 1)を生成する。
//!
//! ## 実装方針
//! 参照実装 (WebAudio-Modem/msequence.ts) に準拠した設計:
//! - **左シフトLFSR**: `state = ((state << 1) | feedback) & mask`
//! - **MSB出力**: `output = (state >> (order-1)) & 1` → 0/1値 → {-1, +1}に変換
//! - **初期状態**: `1 << (order-1)` (MSBのみセット)
//!
//! ## 生成多項式 (原始多項式)
//! | 次数 | 多項式       | フィードバック計算                        |
//! |------|-------------|------------------------------------------|
//! | 5    | x⁵+x³+1   | `(state >> 4) ^ (state >> 2)`            |
//! | 6    | x⁶+x⁵+1   | `(state >> 5) ^ (state >> 4)`            |
//! | 7    | x⁷+x⁶+1   | `(state >> 6) ^ (state >> 5)`            |
//!
//! ## M系列の自己相関特性:
//! ```text
//! R(0) = N          (ピーク)
//! R(k) = -1  (k ≠ 0 mod N)
//! ```
//! この性質がDSSSの同期捕捉・逆拡散に使われる。

/// M系列生成器
///
/// # 例
/// ```
/// use dsp::msequence::MSequence;
/// let mut mseq = MSequence::new(5); // 次数5, 長さ31
/// let chips: Vec<i8> = mseq.generate(31);
/// assert_eq!(chips.len(), 31);
/// assert!(chips.iter().all(|&c| c == 1 || c == -1));
/// ```
pub struct MSequence {
    state: u32,
    initial_state: u32,
    order: usize,
}

impl MSequence {
    /// 次数 `order` のM系列生成器を作成する
    pub fn new(order: usize) -> Self {
        let initial_state = 1u32 << (order - 1); // MSBのみセット
        MSequence { state: initial_state, initial_state, order }
    }

    /// 初期状態にリセットする
    pub fn reset(&mut self) {
        self.state = self.initial_state;
    }

    /// 次の1チップを生成する
    ///
    /// LFSRの動作 (左シフト, MSB出力):
    ///   1. 出力 = MSB = (state >> (order-1)) & 1
    ///   2. フィードバック = 各タップXOR
    ///   3. state = ((state << 1) | feedback) & mask
    fn next_chip(&mut self) -> i8 {
        // MSBを出力 (0 → -1, 1 → +1)
        let out_bit = (self.state >> (self.order as u32 - 1)) & 1;

        // フィードバック計算 (次数に対応する原始多項式)
        let feedback = match self.order {
            5 => ((self.state >> 4) ^ (self.state >> 2)) & 1, // x⁵+x³+1
            6 => ((self.state >> 5) ^ (self.state >> 4)) & 1, // x⁶+x⁵+1
            7 => ((self.state >> 6) ^ (self.state >> 5)) & 1, // x⁷+x⁶+1
            _ => panic!("未対応のM系列次数: {}", self.order),
        };

        let mask = (1u32 << self.order) - 1;
        self.state = ((self.state << 1) | feedback) & mask;

        if out_bit == 1 { 1 } else { -1 }
    }

    /// `len` チップ分のM系列を生成する ({-1, +1} 値)
    pub fn generate(&mut self, len: usize) -> Vec<i8> {
        (0..len).map(|_| self.next_chip()).collect()
    }

    /// 1周期 (2^order - 1 チップ) を生成してリセットする
    pub fn one_period(&mut self) -> Vec<i8> {
        let period = (1 << self.order) - 1;
        let chips = self.generate(period);
        self.reset();
        chips
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// M系列の長さが 2^n - 1 であることを確認
    #[test]
    fn test_period_length() {
        let mut mseq = MSequence::new(5);
        let chips = mseq.one_period();
        assert_eq!(chips.len(), 31, "次数5のM系列長は31であること");
    }

    /// 出力値が {-1, +1} のみであることを確認
    #[test]
    fn test_chip_values() {
        let mut mseq = MSequence::new(5);
        let chips = mseq.one_period();
        assert!(
            chips.iter().all(|&c| c == 1 || c == -1),
            "チップ値は +1 または -1 のみであること"
        );
    }

    /// バランス特性: 1周期中の +1 と -1 の数の差が1であること
    /// M系列は2^(n-1)個の1と(2^(n-1)-1)個の0を含む
    #[test]
    fn test_balance() {
        for order in [5usize, 6, 7] {
            let mut mseq = MSequence::new(order);
            let chips = mseq.one_period();
            let ones = chips.iter().filter(|&&c| c == 1).count();
            let neg_ones = chips.iter().filter(|&&c| c == -1).count();
            assert_eq!(
                ones as i32 - neg_ones as i32,
                1,
                "次数{}: +1の数 - (-1)の数 = 1 であること (ones={}, neg_ones={})",
                order, ones, neg_ones
            );
        }
    }

    /// 自己相関の確認:
    ///   R(0) = N = 31
    ///   R(k) = -1 for k = 1..N-1 (周期的自己相関)
    #[test]
    fn test_autocorrelation() {
        let mut mseq = MSequence::new(5);
        let chips = mseq.one_period();
        let n = chips.len();

        // R(0) の確認
        let r0: i32 = chips.iter().map(|&c| c as i32 * c as i32).sum();
        assert_eq!(r0, 31, "自己相関ピーク R(0) = N = 31 であること");

        // R(k) for k = 1..N-1: 周期的自己相関
        for k in 1..n {
            let rk: i32 = chips
                .iter()
                .enumerate()
                .map(|(i, &c)| c as i32 * chips[(i + k) % n] as i32)
                .sum();
            assert_eq!(
                rk, -1,
                "周期的自己相関 R({}) = -1 であること (実際: {})",
                k, rk
            );
        }
    }

    /// 同じシードで2回生成すると同じ系列になることを確認
    #[test]
    fn test_deterministic() {
        let mut mseq1 = MSequence::new(5);
        let chips1 = mseq1.one_period();
        let mut mseq2 = MSequence::new(5);
        let chips2 = mseq2.one_period();
        assert_eq!(chips1, chips2, "同じシードで同じM系列が生成されること");
    }

    /// リセット後に同じ系列が生成されることを確認
    #[test]
    fn test_reset() {
        let mut mseq = MSequence::new(5);
        let first = mseq.generate(31);
        mseq.reset();
        let second = mseq.generate(31);
        assert_eq!(first, second, "reset後に同じ系列が生成されること");
    }

    /// 次数6, 7でも正しく動作することを確認
    #[test]
    fn test_order_6_and_7() {
        for order in [6usize, 7usize] {
            let mut mseq = MSequence::new(order);
            let chips = mseq.one_period();
            let n = (1usize << order) - 1;
            assert_eq!(chips.len(), n, "次数{}の長さが{}であること", order, n);

            let r0: i32 = chips.iter().map(|&c| c as i32 * c as i32).sum();
            assert_eq!(r0, n as i32, "次数{}のR(0)={}であること", order, n);

            for k in 1..n {
                let rk: i32 = chips
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| c as i32 * chips[(i + k) % n] as i32)
                    .sum();
                assert_eq!(rk, -1, "次数{} R({})=-1であること (実際:{})", order, k, rk);
            }
        }
    }

    /// LFSRが正しく1周期で元の状態に戻ることを確認
    #[test]
    fn test_period_returns_to_initial() {
        for order in [5usize, 6, 7] {
            let mut mseq = MSequence::new(order);
            let period = (1 << order) - 1;
            let _ = mseq.generate(period);
            // period チップ後、次のchipは最初と同じになるはず
            mseq.reset();
            let first_chip = mseq.next_chip();
            mseq.reset();
            let _ = mseq.generate(period);
            let wrapped_chip = mseq.next_chip();
            assert_eq!(
                first_chip, wrapped_chip,
                "次数{}: {}チップ後に元の状態に戻ること",
                order, period
            );
        }
    }
}
