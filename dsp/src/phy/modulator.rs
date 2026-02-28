//! DBPSK + DSSS 変調器
//!
//! # 変調パイプライン
//! 1. DBPSK差動符号化: bit[n] → phase[n] = phase[n-1] XOR bit[n]
//! 2. DSSS拡散: 各シンボルにM系列を掛ける (チップ展開)
//! 3. RRCパルス整形: 各チップをRRCフィルタで成形
//! 4. 帯域シフト: キャリアfc=12kHzで変調 (実信号)

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
        let all_chips = self.bits_to_chips(bits);
        self.chips_to_samples(&all_chips)
    }

    fn append_symbol_chips(&mut self, symbol: i8, out: &mut Vec<i8>) {
        self.mseq.reset();
        for chip in self.mseq.generate(self.config.spread_factor()) {
            out.push(symbol * chip);
        }
    }

    fn bits_to_chips(&mut self, bits: &[u8]) -> Vec<i8> {
        let spread_factor = self.config.spread_factor();
        let mut all_chips: Vec<i8> = Vec::with_capacity(bits.len() * spread_factor);
        for &bit in bits {
            self.prev_phase ^= bit;
            let symbol: i8 = if self.prev_phase == 0 { 1 } else { -1 };
            self.append_symbol_chips(symbol, &mut all_chips);
        }
        all_chips
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
            self.append_symbol_chips(symbol, &mut all_chips);
        }

        // 3. データ
        for &bit in bits {
            self.prev_phase ^= bit;
            let symbol: i8 = if self.prev_phase == 0 { 1 } else { -1 };
            self.append_symbol_chips(symbol, &mut all_chips);
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
    use crate::common::rrc_filter::RrcFilter;

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
        let bits = vec![1u8, 0, 1, 1, 0, 1, 0];
        let sf = mod_.config.spread_factor();
        let chips = mod_.bits_to_chips(&bits);

        assert_eq!(
            chips.len(),
            bits.len() * sf,
            "1bitあたりSF個のチップに展開されること"
        );

        let mut mseq = MSequence::new(mod_.config.mseq_order);
        let pn = mseq.generate(sf);

        let mut expected_phase = 0u8;
        for (sym_idx, &bit) in bits.iter().enumerate() {
            expected_phase ^= bit;
            let expected_symbol = if expected_phase == 0 { 1i8 } else { -1i8 };
            let start = sym_idx * sf;
            let end = start + sf;
            for (chip_idx, &chip) in chips[start..end].iter().enumerate() {
                assert_eq!(
                    chip,
                    expected_symbol * pn[chip_idx],
                    "sym={} chip={} でDBPSK差動符号化とDSSS拡散が正しいこと",
                    sym_idx,
                    chip_idx
                );
            }
        }
    }

    /// 同期位置が既知の前提で、encode_frame波形からDBPSK遷移を復元できること
    #[test]
    fn test_known_sync_position_dbpsk_roundtrip_without_demodulator() {
        let mut mod_ = make_modulator();
        let bits = vec![1u8, 0, 1, 1, 0, 1, 0, 0, 1];
        let config = mod_.config().clone();
        let frame = mod_.encode_frame(&bits);

        let fs = config.sample_rate;
        let fc = config.carrier_freq;
        let two_pi = 2.0 * std::f32::consts::PI;
        let mut rrc_i = RrcFilter::from_config(&config);
        let mut rrc_q = RrcFilter::from_config(&config);
        let mut i_ch = Vec::with_capacity(frame.len());
        let mut q_ch = Vec::with_capacity(frame.len());
        for (idx, &s) in frame.iter().enumerate() {
            let t = idx as f32 / fs;
            let (sin_v, cos_v) = (two_pi * fc * t).sin_cos();
            i_ch.push(rrc_i.process(s * cos_v * 2.0));
            q_ch.push(rrc_q.process(s * (-sin_v) * 2.0));
        }

        let sf = config.spread_factor();
        let spc = config.samples_per_chip();
        let sym_len = sf * spc;
        let total_delay = mod_.rrc.delay() + rrc_i.delay();

        let mut mseq = MSequence::new(config.mseq_order);
        let pn = mseq.generate(sf);

        let correlate_symbol = |sym_idx: usize, phase: usize, i_ch: &[f32], q_ch: &[f32]| {
            let base = total_delay + phase + sym_idx * sym_len;
            let mut ci = 0.0f32;
            let mut cq = 0.0f32;
            for (chip_idx, &pn_val) in pn.iter().enumerate() {
                let p = base + chip_idx * spc;
                ci += i_ch[p] * pn_val as f32;
                cq += q_ch[p] * pn_val as f32;
            }
            (ci, cq)
        };

        let mut best_phase = 0usize;
        let mut best_mag = -1.0f32;
        for phase in 0..spc {
            let (ci, cq) = correlate_symbol(0, phase, &i_ch, &q_ch);
            let mag = ci * ci + cq * cq;
            if mag > best_mag {
                best_mag = mag;
                best_phase = phase;
            }
        }

        let sync_bits: Vec<u8> = (0..SYNC_WORD_BITS)
            .rev()
            .map(|i| ((SYNC_WORD >> i) & 1) as u8)
            .collect();
        let mut expected_bits = Vec::with_capacity(sync_bits.len() + bits.len());
        expected_bits.extend_from_slice(&sync_bits);
        expected_bits.extend_from_slice(&bits);

        // DBPSKの初期位相は +M (prev_phase=0) なので、+M側のプリアンブルを基準に
        // sync+payload全ビットを復元して一致を確認する。
        let mut prev = correlate_symbol(0, best_phase, &i_ch, &q_ch);
        for (idx, &expected_bit) in expected_bits.iter().enumerate() {
            let cur = correlate_symbol(config.preamble_repeat + idx, best_phase, &i_ch, &q_ch);
            let dot = prev.0 * cur.0 + prev.1 * cur.1;
            let decoded_bit = if dot >= 0.0 { 0 } else { 1 };
            assert_eq!(
                decoded_bit, expected_bit,
                "sym={} の差動ビット復元が一致すること",
                idx
            );
            prev = cur;
        }
    }
}
