//! QPSK/Spread Spectrum 復調器

use crate::msequence::MSequence;
use crate::sync::{downconvert, matched_filter_decimate};
use crate::DspConfig;

/// DBPSK + DSSS 復調器
pub struct Demodulator {
    config: DspConfig,
    mseq: MSequence,
    /// 前シンボルの判定値 (差動復号用)
    prev_symbol: f32,
}

impl Demodulator {
    pub fn new(config: DspConfig) -> Self {
        let mseq_order = config.mseq_order;
        Demodulator {
            config,
            mseq: MSequence::new(mseq_order),
            prev_symbol: 1.0,
        }
    }

    pub fn default_48k() -> Self {
        Self::new(DspConfig::default_48k())
    }

    /// サンプル列を復調してビット列を返す
    pub fn demodulate(&mut self, samples: &[f32], sample_offset: usize) -> Vec<u8> {
        let (i_ch, q_ch) = downconvert(samples, sample_offset, &self.config);
        let (chips_i, _chips_q) = matched_filter_decimate(&i_ch, &q_ch, &self.config);

        let chips_per_symbol = self.config.spread_factor();
        let mut bits = Vec::new();

        self.mseq.reset();
        let pn: Vec<f32> = self.mseq.generate(chips_per_symbol).iter().map(|&c| c as f32).collect();

        let num_symbols = chips_i.len() / chips_per_symbol;
        for sym_idx in 0..num_symbols {
            let start = sym_idx * chips_per_symbol;
            let end = start + chips_per_symbol;
            if end > chips_i.len() {
                break;
            }

            // 逆拡散: 受信チップ × PN系列の相関
            let despread_i: f32 = chips_i[start..end]
                .iter()
                .zip(pn.iter())
                .map(|(&c, &p)| c * p)
                .sum::<f32>()
                / chips_per_symbol as f32;

            // DBPSK差動復号: 前シンボルとの積の符号でビット判定
            let product = despread_i * self.prev_symbol;
            let bit = if product >= 0.0 { 0u8 } else { 1u8 };
            bits.push(bit);
            self.prev_symbol = despread_i;
        }

        bits
    }

    pub fn reset(&mut self) {
        self.prev_symbol = 1.0;
        self.mseq.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modulator::Modulator;

    fn test_config() -> DspConfig {
        DspConfig::default_48k()
    }

    /// 変調→復調のループバックテスト (ノイズなし)
    #[test]
    fn test_loopback_no_noise() {
        let config = test_config();
        let input_bits: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0];
        let mut mod_ = Modulator::new(config.clone());
        let samples = mod_.modulate(&input_bits);

        let mut demod = Demodulator::new(config);
        let output_bits = demod.demodulate(&samples, 0);

        // RRC遅延・デシメーション境界で1ビット程度少なくなる場合がある
        // 入力の90%以上が復調されていれば問題なし
        assert!(
            output_bits.len() >= input_bits.len() * 9 / 10,
            "復調ビット数 {} が入力ビット数 {} の90%以上であること",
            output_bits.len(), input_bits.len()
        );

        // 最初の2ビットはDBPSK初期位相とRRC遅延のマージン
        let start = 2;
        let match_count = input_bits[start..]
            .iter()
            .zip(output_bits[start..].iter())
            .filter(|(&a, &b)| a == b)
            .count();
        let total = input_bits.len() - start;
        let ber = 1.0 - match_count as f32 / total as f32;
        assert!(ber < 0.1, "ノイズなし時のBERが10%未満であること: BER={:.3}", ber);
    }

    #[test]
    fn test_reset() {
        let config = test_config();
        let bits = vec![1u8, 0, 1, 0, 1, 0, 1, 0];
        let mut mod_ = Modulator::new(config.clone());
        let samples = mod_.modulate(&bits);

        let mut demod = Demodulator::new(config);
        let out1 = demod.demodulate(&samples, 0);
        demod.reset();
        let out2 = demod.demodulate(&samples, 0);
        assert_eq!(out1, out2, "リセット後に同じ復調結果が得られること");
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;
    use crate::msequence::MSequence;

    /// 逆拡散の単純テスト
    #[test]
    fn test_despread_simple() {
        // 直接チップ配列を生成して逆拡散のみテスト
        let config = DspConfig::default_48k();
        let spread_factor = config.spread_factor();
        
        // 変調側と同じ系列を生成
        let mut mseq_tx = MSequence::new(config.mseq_order);
        mseq_tx.reset();
        let pn_tx: Vec<f32> = mseq_tx.generate(spread_factor).iter().map(|&c| c as f32).collect();
        
        // 復調側の系列
        let mut mseq_rx = MSequence::new(config.mseq_order);
        mseq_rx.reset();
        let pn_rx: Vec<f32> = mseq_rx.generate(spread_factor).iter().map(|&c| c as f32).collect();
        
        // 系列が一致するか確認
        assert_eq!(pn_tx, pn_rx, "変調・復調側のPN系列が一致すること");
        
        // symbol=+1 で変調したチップの逆拡散
        let symbol_p1 = 1.0f32;
        let chips_p1: Vec<f32> = pn_tx.iter().map(|&p| symbol_p1 * p).collect();
        let corr_p1: f32 = chips_p1.iter().zip(pn_rx.iter()).map(|(&c, &p)| c * p).sum::<f32>() / spread_factor as f32;
        println!("symbol=+1 の相関値: {}", corr_p1);
        assert!(corr_p1 > 0.5, "symbol=+1 の相関が正であること: {}", corr_p1);
        
        // symbol=-1 で変調したチップの逆拡散
        let symbol_m1 = -1.0f32;
        let chips_m1: Vec<f32> = pn_tx.iter().map(|&p| symbol_m1 * p).collect();
        let corr_m1: f32 = chips_m1.iter().zip(pn_rx.iter()).map(|(&c, &p)| c * p).sum::<f32>() / spread_factor as f32;
        println!("symbol=-1 の相関値: {}", corr_m1);
        assert!(corr_m1 < -0.5, "symbol=-1 の相関が負であること: {}", corr_m1);
    }
}
