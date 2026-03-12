//! MaryDQPSK (16-ary + DQPSK) 変復調モジュール
//!
//! 完全に独立したMaryDQPSK実装。既存のDBPSK/DQPSKコードとは
//! 一切依存関係がない。

pub mod decoder;
pub mod demodulator;
pub mod encoder;
pub mod modulator;
pub mod params;
pub mod sync;

/// インターリーバ設定モジュール
///
/// FECエンコーディング後のビット列に対するインターリーバのサイズ計算を提供する。
/// エンコーダとデコーダで同じ計算方法を保証するため、計算ロジックを一箇所に集約している。
pub mod interleaver_config {
    use crate::frame::packet::PACKET_BYTES;

    /// テールビット数（畳み込み符号の終了処理）
    pub const TAIL_BITS: usize = 6;

    /// 生ビット数（パケットバイト数 + テールビット）
    #[inline]
    pub const fn raw_bits() -> usize {
        PACKET_BYTES * 8 + TAIL_BITS
    }

    /// FECエンコーディング後のビット数（畳み込み符号で2倍）
    #[inline]
    pub const fn fec_bits() -> usize {
        raw_bits() * 2
    }

    /// インターリーバ行数（12）
    pub const INTERLEAVER_ROWS: usize = 12;

    /// インターリーバ列数
    ///
    /// `rows * cols == fec_bits` を満たすよう、payload/CRC変更時も無パディングを維持する。
    pub const INTERLEAVER_COLS: usize = fec_bits() / INTERLEAVER_ROWS;

    /// インターリーバ処理後のビット数
    ///
    /// INTERLEAVER_ROWS × INTERLEAVER_COLS = fec_bits()
    /// fec_bits() と完全一致するため、無駄なパディングが発生しない
    #[inline]
    pub const fn interleaved_bits() -> usize {
        INTERLEAVER_ROWS * INTERLEAVER_COLS
    }

    /// Maryシンボル境界（6ビット単位）に揃えたビット数
    ///
    /// interleaved_bits は 6 で割り切れるため、追加のパディングは不要
    #[inline]
    pub const fn mary_aligned_bits() -> usize {
        interleaved_bits()
    }

    /// Maryシンボル数
    #[inline]
    pub const fn mary_symbols() -> usize {
        interleaved_bits() / 6
    }
}

#[cfg(test)]
mod integration_tests;

#[cfg(test)]
mod tests {
    use super::interleaver_config;

    #[test]
    fn test_interleaver_config_values() {
        // 基本的な定数値の検証
        assert_eq!(
            interleaver_config::raw_bits(),
            246,
            "raw_bits should be 246"
        );
        assert_eq!(
            interleaver_config::fec_bits(),
            492,
            "fec_bits should be 492"
        );
        assert_eq!(
            interleaver_config::INTERLEAVER_ROWS,
            12,
            "INTERLEAVER_ROWS should be 12"
        );
        assert_eq!(
            interleaver_config::INTERLEAVER_COLS,
            41,
            "INTERLEAVER_COLS should be 41"
        );
        assert_eq!(
            interleaver_config::interleaved_bits(),
            492,
            "interleaved_bits should be 492"
        );
        assert_eq!(
            interleaver_config::mary_aligned_bits(),
            492,
            "mary_aligned_bits should be 492 (no padding)"
        );

        // 492は6で割り切れることを確認
        assert_eq!(
            interleaver_config::interleaved_bits() % 6,
            0,
            "492 should be divisible by 6"
        );

        // 82 symbolsであることを確認
        assert_eq!(
            interleaver_config::mary_symbols(),
            82,
            "Should have 82 Mary symbols"
        );
    }

    #[test]
    fn test_interleaver_matches_fec_bits() {
        // インターリーバサイズがFECビット数と完全一致することを確認
        assert_eq!(
            interleaver_config::interleaved_bits(),
            interleaver_config::fec_bits(),
            "インターリーバサイズはFECビット数と一致すべき"
        );
    }

    #[test]
    fn test_interleaver_rows_cols_product() {
        // 41 × 12 = 492 であることを確認
        assert_eq!(
            interleaver_config::INTERLEAVER_ROWS * interleaver_config::INTERLEAVER_COLS,
            492,
            "41 × 12 should equal 492"
        );
    }

    #[test]
    fn test_no_padding_needed() {
        // Maryシンボル境界にパディングが不要であることを確認
        assert_eq!(
            interleaver_config::interleaved_bits(),
            interleaver_config::mary_aligned_bits(),
            "Mary alignment should not require padding"
        );
    }
}
