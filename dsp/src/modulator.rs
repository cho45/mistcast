//! DBPSK + DSSS 変調器
//!
//! # 変調パイプライン
//! 1. DBPSK差動符号化: bit[n] → phase[n] = phase[n-1] XOR bit[n]
//! 2. DSSS拡散: 各シンボルにM系列を掛ける (チップ展開)
//! 3. RRCパルス整形: 各チップをRRCフィルタで成形
//! 4. 帯域シフト: キャリアfc=8kHzで変調 (実信号)

use crate::msequence::MSequence;
use crate::params::SYNC_WORD;
use crate::rrc_filter::RrcFilter;
use crate::DspConfig;

/// 変調器
pub struct Modulator {
    config: DspConfig,
    mseq: MSequence,
    rrc: RrcFilter,
    /// 差動符号化の前シンボル位相 (0 or 1)
    prev_phase: u8,
    /// 現在のサンプルインデックス (キャリア位相計算用)
    sample_idx: usize,
}

impl Modulator {
    /// `DspConfig` を指定して変調器を作成する
    pub fn new(config: DspConfig) -> Self {
        let rrc = RrcFilter::from_config(&config);
        Modulator {
            mseq: MSequence::new(config.mseq_order),
            rrc,
            config,
            prev_phase: 0,
            sample_idx: 0,
        }
    }

    /// デフォルト設定 (48kHz) で変調器を作成する
    pub fn default_48k() -> Self {
        Self::new(DspConfig::default_48k())
    }

    /// プリアンブル (M系列 × PREAMBLE_REPEAT 周期) を生成する
    ///
    /// 受信側の同期捕捉に使用する。
    pub fn generate_preamble(&mut self) -> Vec<f32> {
        let period = self.config.spread_factor();
        let total_chips = period * self.config.preamble_repeat;

        self.mseq.reset();
        let chips = self.mseq.generate(total_chips);

        self.chips_to_samples(&chips)
    }

    /// ビット列を変調してサンプル列を返す
    ///
    /// 差動符号化 → DSSS拡散 → RRC整形 → キャリア変調
    pub fn modulate(&mut self, bits: &[u8]) -> Vec<f32> {
        let spread_factor = self.config.spread_factor();
        let mut all_chips: Vec<i8> = Vec::with_capacity(bits.len() * spread_factor);

        for &bit in bits {
            // DBPSK差動符号化
            self.prev_phase ^= bit;
            let symbol: i8 = if self.prev_phase == 0 { 1 } else { -1 };

            // DSSS拡散: 1シンボル = N個のチップ
            self.mseq.reset();
            let pn = self.mseq.generate(spread_factor);
            for chip in pn {
                all_chips.push(symbol * chip);
            }
        }

        self.chips_to_samples(&all_chips)
    }

    /// チップ列をRRC整形 + キャリア変調してサンプル列に変換
    fn chips_to_samples(&mut self, chips: &[i8]) -> Vec<f32> {
        let spc = self.config.samples_per_chip();
        let two_pi = 2.0 * std::f32::consts::PI;
        let fs = self.config.sample_rate;
        let fc = self.config.carrier_freq;
        let mut out = Vec::with_capacity(chips.len() * spc);

        for &chip in chips {
            for k in 0..spc {
                let baseband = if k == 0 { chip as f32 } else { 0.0 };
                let filtered = self.rrc.process(baseband);
                let t = self.sample_idx as f32 / fs;
                let carrier = (two_pi * fc * t).cos();
                out.push(filtered * carrier);
                self.sample_idx += 1;
            }
        }
        out
    }

    /// 送信フレーム全体を生成する (プリアンブル + 同期ワード + データ)
    pub fn encode_frame(&mut self, bits: &[u8]) -> Vec<f32> {
        let mut frame = self.generate_preamble();

        let sync_bits: Vec<u8> = (0..32).rev().map(|i| ((SYNC_WORD >> i) & 1) as u8).collect();
        frame.extend(self.modulate(&sync_bits));
        frame.extend(self.modulate(bits));

        frame
    }

    /// 変調器の状態をリセット
    pub fn reset(&mut self) {
        self.prev_phase = 0;
        self.sample_idx = 0;
        self.mseq.reset();
        self.rrc.reset();
    }

    pub fn config(&self) -> &DspConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_modulator() -> Modulator {
        Modulator::default_48k()
    }

    /// プリアンブル長の確認
    #[test]
    fn test_preamble_length() {
        let mut mod_ = make_modulator();
        let preamble = mod_.generate_preamble();
        let config = DspConfig::default_48k();
        let expected_samples = config.spread_factor() * config.preamble_repeat * config.samples_per_chip();
        assert_eq!(preamble.len(), expected_samples,
            "プリアンブル長が正しいこと: expected={}", expected_samples);
    }

    /// サンプル値が有限値であること
    #[test]
    fn test_samples_finite() {
        let mut mod_ = make_modulator();
        let bits = vec![0u8, 1, 0, 1, 1, 0, 1, 0];
        let samples = mod_.modulate(&bits);
        assert!(samples.iter().all(|&s| s.is_finite()), "全サンプルが有限値であること");
    }

    /// 変調出力の長さ確認
    #[test]
    fn test_modulate_length() {
        let mut mod_ = make_modulator();
        let config = DspConfig::default_48k();
        let bits = vec![0u8; 8];
        let samples = mod_.modulate(&bits);
        let expected = bits.len() * config.spread_factor() * config.samples_per_chip();
        assert_eq!(samples.len(), expected, "変調後サンプル数が正しいこと: {}", expected);
    }

    /// 変調出力の振幅が概ね±2以内であること
    #[test]
    fn test_amplitude_range() {
        let mut mod_ = make_modulator();
        let bits: Vec<u8> = (0..32).map(|i| i % 2).collect();
        let samples = mod_.modulate(&bits);
        let max_amp = samples.iter().cloned().fold(0.0f32, |a, s| a.max(s.abs()));
        assert!(max_amp < 2.0, "振幅が過度に大きくないこと: max={}", max_amp);
    }

    /// リセット後に同じ出力が得られること
    #[test]
    fn test_reset_deterministic() {
        let bits = vec![1u8, 0, 1, 1, 0, 0, 1, 0];
        let mut mod_ = make_modulator();
        let s1 = mod_.modulate(&bits);
        mod_.reset();
        let s2 = mod_.modulate(&bits);
        assert_eq!(s1, s2, "リセット後に同じ変調結果が得られること");
    }

    /// 44.1kHzでも動作すること
    #[test]
    fn test_44k_modulation() {
        let mut mod_ = Modulator::new(DspConfig::default_44k());
        let bits = vec![0u8, 1, 0, 1];
        let samples = mod_.modulate(&bits);
        assert!(!samples.is_empty());
        assert!(samples.iter().all(|&s| s.is_finite()));
    }
}
