//! QPSK/Spread Spectrum 復調器

use crate::msequence::MSequence;
use crate::sync::{downconvert, matched_filter_decimate};
use crate::DspConfig;

/// DBPSK + DSSS 復調器
pub struct Demodulator {
    config: DspConfig,
    mseq: MSequence,
    /// 前シンボルの判定値 (I, Q) (差動復号用)
    prev_symbol: (f32, f32),
}

impl Demodulator {
    pub fn new(config: DspConfig) -> Self {
        let mseq_order = config.mseq_order;
        Demodulator {
            config,
            mseq: MSequence::new(mseq_order),
            prev_symbol: (1.0, 0.0),
        }
    }

    pub fn default_48k() -> Self {
        Self::new(DspConfig::default_48k())
    }

    /// サンプル列を復調してビット列を返す
    pub fn demodulate(&mut self, samples: &[f32]) -> Vec<u8> {
        let (i_ch, q_ch) = downconvert(samples, 0, &self.config);
        let (chips_i, chips_q) = matched_filter_decimate(&i_ch, &q_ch, &self.config);
        self.demodulate_chips(&chips_i, &chips_q)
    }

    /// デシメーション済みのチップ列を復調してビット列を返す
    pub fn demodulate_chips(&mut self, chips_i: &[f32], chips_q: &[f32]) -> Vec<u8> {
        let chips_per_symbol = self.config.spread_factor();
        let mut bits = Vec::new();

        self.mseq.reset();
        let pn: Vec<f32> = self.mseq.generate(chips_per_symbol).iter().map(|&c| c as f32).collect();

        // 復調に使えるシンボル数はI/Qで短い方に合わせる
        let num_symbols = chips_i.len().min(chips_q.len()) / chips_per_symbol;
        for sym_idx in 0..num_symbols {
            let start = sym_idx * chips_per_symbol;

            // 逆拡散: 受信チップ × PN系列の相関
            let mut despread_i = 0.0;
            let mut despread_q = 0.0;
            for i in 0..chips_per_symbol {
                let p = pn[i];
                despread_i += chips_i[start + i] * p;
                despread_q += chips_q[start + i] * p;
            }
            despread_i /= chips_per_symbol as f32;
            despread_q /= chips_per_symbol as f32;

            // DBPSK差動復号: 前シンボルとの積の実部 I_k * I_{k-1} + Q_k * Q_{k-1}
            let (prev_i, prev_q) = self.prev_symbol;
            let dot_product = despread_i * prev_i + despread_q * prev_q;
            
            // 実部の符号でビット判定
            let bit = if dot_product >= 0.0 { 0u8 } else { 1u8 };
            bits.push(bit);
            
            self.prev_symbol = (despread_i, despread_q);
        }

        bits
    }

    pub fn reset(&mut self) {
        self.prev_symbol = (1.0, 0.0);
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
        let mut samples = mod_.modulate(&input_bits);

        // 送受信のRRCフィルタの遅延（合計 num_taps - 1 サンプル）により、
        // 最後のシンボルのエネルギーがフィルタ内に残ってしまう。
        // これを出し切る（フラッシュする）ために遅延分以上の無音サンプルを末尾に追加する。
        let total_delay = config.rrc_num_taps() - 1;
        samples.extend(vec![0.0; total_delay + config.samples_per_symbol()]);

        let mut demod = Demodulator::new(config);
        let output_bits = demod.demodulate(&samples);

        // 最初のビットはDBPSKの初期位相（差動符号化の最初の差分）に依存するため、
        // 実質的なデータビットは遅延を考慮して比較する必要がある。
        // 現在のRRCフィルタとデシメーションの遅延計算が正しければ、
        // 最初の1ビット目（あるいは2ビット目）以降は完全に入力と一致しなければならない。
        
        let start = 1; // 最初の1ビットは基準シンボルに対する差分
        let compare_len = input_bits.len() - start;
        
        assert!(
            output_bits.len() >= input_bits.len(),
            "復調ビット数 {} が入力ビット数 {} 以上であること",
            output_bits.len(), input_bits.len()
        );

        let mut match_count = 0;
        for i in 0..compare_len {
            if input_bits[start + i] == output_bits[start + i] {
                match_count += 1;
            } else {
                println!("Mismatch at index {}: expected {}, got {}", start + i, input_bits[start + i], output_bits[start + i]);
            }
        }
        
        assert_eq!(
            match_count, compare_len,
            "ノイズなしの環境では、遅延補正後の復調ビット列が完全に入力と一致しなければならない"
        );
    }

    #[test]
    fn test_reset() {
        let config = test_config();
        let bits = vec![1u8, 0, 1, 0, 1, 0, 1, 0];
        let mut mod_ = Modulator::new(config.clone());
        let samples = mod_.modulate(&bits);

        let mut demod = Demodulator::new(config);
        let out1 = demod.demodulate(&samples);
        demod.reset();
        let out2 = demod.demodulate(&samples);
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
