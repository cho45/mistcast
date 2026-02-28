//! DBPSK + DSSS 変調器
//!
//! # 変調パイプライン
//! 1. DBPSK差動符号化: bit[n] → phase[n] = phase[n-1] XOR bit[n]
//! 2. DSSS拡散: 各シンボルにM系列を掛ける (チップ展開)
//! 3. RRCパルス整形: 各チップをRRCフィルタで成形
//! 4. 帯域シフト: キャリアfc=8kHzで変調 (実信号)

use crate::common::msequence::MSequence;
use crate::common::rrc_filter::RrcFilter;
use crate::params::{SYNC_WORD, SYNC_WORD_BITS};
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

    /// プリアンブル (M系列の [M, M, M, -M] パターン) を生成する
    ///
    /// 最後のシンボルを反転させることで同期の曖昧さを排除する。
    pub fn generate_preamble(&mut self) -> Vec<f32> {
        let sf = self.config.spread_factor();
        let repeat = self.config.preamble_repeat;
        let mut all_chips = Vec::with_capacity(sf * repeat);

        self.mseq.reset();
        let pn = self.mseq.generate(sf);

        for i in 0..repeat {
            let sign: i8 = if i == repeat - 1 { -1 } else { 1 };
            for &c in &pn {
                all_chips.push(sign * c);
            }
        }

        self.chips_to_samples(&all_chips)
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

        // 外部から modulate() が単発で呼ばれた場合のためのフラッシュ
        // encode_frame 内では末尾に 0 チップが追加されるため、そこでのフィルタリングがフラッシュの役割を果たす。
        let delay = self.rrc.delay();
        for _ in 0..delay {
            let filtered = self.rrc.process(0.0);
            let t = self.sample_idx as f32 / fs;
            let carrier = (two_pi * fc * t).cos();
            out.push(filtered * carrier);
            self.sample_idx += 1;
        }

        out
    }

    /// 送信フレーム全体を生成する (プリアンブル + 同期ワード + データ)
    pub fn encode_frame(&mut self, bits: &[u8]) -> Vec<f32> {
        self.reset();
        let sf = self.config.spread_factor();
        let preamble_repeat = self.config.preamble_repeat;

        let mut all_chips = Vec::new();

        // 1. プリアンブル
        self.mseq.reset();
        let pn = self.mseq.generate(sf);
        for rep in 0..preamble_repeat {
            let sign: i8 = if rep == preamble_repeat - 1 { -1 } else { 1 };
            for &chip in &pn {
                all_chips.push(sign * chip);
            }
        }

        // 2. 同期ワード
        let sync_bits: Vec<u8> = (0..SYNC_WORD_BITS)
            .rev()
            .map(|i| ((SYNC_WORD >> i) & 1) as u8)
            .collect();
        for &bit in &sync_bits {
            self.prev_phase ^= bit;
            let symbol: i8 = if self.prev_phase == 0 { 1 } else { -1 };
            self.mseq.reset();
            let pn = self.mseq.generate(sf);
            for chip in pn {
                all_chips.push(symbol * chip);
            }
        }

        // 3. データ
        for &bit in bits {
            self.prev_phase ^= bit;
            let symbol: i8 = if self.prev_phase == 0 { 1 } else { -1 };
            self.mseq.reset();
            let pn = self.mseq.generate(sf);
            for chip in pn {
                all_chips.push(symbol * chip);
            }
        }

        // 4. マージン (1シンボル分の無音チップ)
        all_chips.extend(vec![0; sf]);

        self.chips_to_samples(&all_chips)
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
        // チップ数 = SF * PREAMBLE_REPEAT
        // サンプル数 = チップ数 * SPC + フラッシュ(delay)
        let expected_samples =
            config.spread_factor() * config.preamble_repeat * config.samples_per_chip()
                + mod_.rrc.delay();
        assert_eq!(
            preamble.len(),
            expected_samples,
            "プリアンブル長が正しいこと: expected={}",
            expected_samples
        );
    }

    /// サンプル値が有限値であること
    #[test]
    fn test_samples_finite() {
        let mut mod_ = make_modulator();
        let bits = vec![0u8, 1, 0, 1, 1, 0, 1, 0];
        let samples = mod_.modulate(&bits);
        assert!(
            samples.iter().all(|&s| s.is_finite()),
            "全サンプルが有限値であること"
        );
    }

    /// 変調出力の長さ確認
    #[test]
    fn test_modulate_length() {
        let mut mod_ = make_modulator();
        let config = DspConfig::default_48k();
        let bits = vec![0u8; 8];
        let samples = mod_.modulate(&bits);
        let expected =
            bits.len() * config.spread_factor() * config.samples_per_chip() + mod_.rrc.delay();
        assert_eq!(
            samples.len(),
            expected,
            "変調後サンプル数が正しいこと: {}",
            expected
        );
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

    /// DBPSKとDSSS拡散の数学的・論理的正しさを検証する
    #[test]
    fn test_math_dbpsk_dsss() {
        let mut mod_ = make_modulator();
        let bits = vec![1u8, 0, 1];
        let fs = mod_.config.sample_rate;
        let fc = mod_.config.carrier_freq;
        let sf = mod_.config.spread_factor();
        let spc = mod_.config.samples_per_chip();
        let two_pi = 2.0 * std::f32::consts::PI;

        // --- 検証: encode_frame が生成する波形を理想的な受信機で復号する ---
        let frame = mod_.encode_frame(&bits);

        let mut demod_rrc = RrcFilter::from_config(&mod_.config);
        let mut i_ch = Vec::new();
        let mut q_ch = Vec::new();
        for (i, &s) in frame.iter().enumerate() {
            let t = i as f32 / fs;
            let (sin_v, cos_v) = (two_pi * fc * t).sin_cos();
            i_ch.push(demod_rrc.process(s * cos_v * 2.0));
            q_ch.push(demod_rrc.process(s * (-sin_v) * 2.0));
        }

        let total_delay = mod_.rrc.delay() + demod_rrc.delay();
        let mut mseq = MSequence::new(mod_.config.mseq_order);
        let preamble_repeat = mod_.config.preamble_repeat;

        // 1. プリアンブルの最後で位相基準を得る
        let mut prev_i = 0.0f32;
        let mut prev_q = 0.0f32;
        let ref_symbol_start = total_delay + (preamble_repeat - 1) * sf * spc;
        mseq.reset();
        let pn = mseq.generate(sf);
        for (c_idx, &pn_val) in pn.iter().enumerate() {
            let p = ref_symbol_start + c_idx * spc;
            prev_i += i_ch[p] * pn_val as f32;
            prev_q += q_ch[p] * pn_val as f32;
        }
        let mag = (prev_i * prev_i + prev_q * prev_q).sqrt().max(1e-6);
        prev_i /= mag;
        prev_q /= mag;

        // 2. 同期ワード + データ(bits)
        let data_start_chips = sf * preamble_repeat;
        let total_symbols = SYNC_WORD_BITS + bits.len();
        let mut recovered_bits = Vec::new();

        for s_idx in 0..total_symbols {
            let symbol_start = total_delay + (data_start_chips + s_idx * sf) * spc;
            let mut cur_i = 0.0;
            let mut cur_q = 0.0;

            mseq.reset();
            let pn = mseq.generate(sf);
            for (c_idx, &pn_val) in pn.iter().enumerate() {
                let p = symbol_start + c_idx * spc;
                cur_i += i_ch[p] * pn_val as f32;
                cur_q += q_ch[p] * pn_val as f32;
            }

            let dot = cur_i * prev_i + cur_q * prev_q;
            let bit = if dot > 0.0 { 0 } else { 1 };

            if s_idx >= SYNC_WORD_BITS {
                recovered_bits.push(bit);
            }

            let mag = (cur_i * cur_i + cur_q * cur_q).sqrt().max(1e-6);
            prev_i = cur_i / mag;
            prev_q = cur_q / mag;
        }

        assert_eq!(
            recovered_bits, bits,
            "プリアンブル・同期ワードを通過してデータが正しく復号されること"
        );
    }
}
