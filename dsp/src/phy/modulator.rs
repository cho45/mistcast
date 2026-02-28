//! 差動PSK + DSSS 変調器
//!
//! # 変調パイプライン
//! 1. 差動符号化: `DBPSK` または `DQPSK` で位相遷移を決定
//! 2. DSSS拡散: 各シンボルにM系列を掛ける (チップ展開)
//! 3. RRCパルス整形: I/QチップをそれぞれRRCフィルタで成形
//! 4. 帯域シフト: キャリアfcで実信号へアップコンバート

use crate::common::msequence::MSequence;
use crate::common::rrc_filter::RrcFilter;
use crate::params::{MODULATION, SYNC_WORD, SYNC_WORD_BITS};
use crate::{DifferentialModulation, DspConfig};

#[inline]
fn phase_to_iq(phase: u8) -> (f32, f32) {
    match phase & 0x03 {
        0 => (1.0, 0.0),
        1 => (0.0, 1.0),
        2 => (-1.0, 0.0),
        _ => (0.0, -1.0),
    }
}

#[inline]
fn dbpsk_delta(bit: u8) -> u8 {
    if bit == 0 {
        0
    } else {
        2
    }
}

#[inline]
fn dqpsk_delta(b0: u8, b1: u8) -> u8 {
    match (b0 & 1, b1 & 1) {
        (0, 0) => 0,
        (0, 1) => 1,
        (1, 1) => 2,
        (1, 0) => 3,
        _ => unreachable!(),
    }
}

/// 変調器
pub struct Modulator {
    config: DspConfig,
    mseq: MSequence,
    rrc_i: RrcFilter,
    rrc_q: RrcFilter,
    /// 差動符号化の累積位相 (0,1,2,3) = (0,90,180,270度)
    prev_phase: u8,
    /// 現在のサンプルインデックス (キャリア位相計算用)
    sample_idx: usize,
}

impl Modulator {
    /// `DspConfig` を指定して変調器を作成する
    pub fn new(config: DspConfig) -> Self {
        let rrc_i = RrcFilter::from_config(&config);
        let rrc_q = RrcFilter::from_config(&config);
        Modulator {
            mseq: MSequence::new(config.mseq_order),
            rrc_i,
            rrc_q,
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
        let mut chips_i = Vec::with_capacity(sf * repeat);
        let mut chips_q = Vec::with_capacity(sf * repeat);

        self.mseq.reset();
        let pn = self.mseq.generate(sf);

        for i in 0..repeat {
            let sign = if i == repeat - 1 { -1.0 } else { 1.0 };
            for &c in &pn {
                chips_i.push(sign * c as f32);
                chips_q.push(0.0);
            }
        }

        self.chips_to_samples(&chips_i, &chips_q)
    }

    /// ビット列を変調してサンプル列を返す
    pub fn modulate(&mut self, bits: &[u8]) -> Vec<f32> {
        let (chips_i, chips_q) = self.bits_to_chips(bits);
        self.chips_to_samples(&chips_i, &chips_q)
    }

    fn append_symbol_chips(
        &mut self,
        symbol_i: f32,
        symbol_q: f32,
        out_i: &mut Vec<f32>,
        out_q: &mut Vec<f32>,
    ) {
        self.mseq.reset();
        for chip in self.mseq.generate(self.config.spread_factor()) {
            let c = chip as f32;
            out_i.push(symbol_i * c);
            out_q.push(symbol_q * c);
        }
    }

    fn append_bits_chips_with_mode(
        &mut self,
        mode: DifferentialModulation,
        bits: &[u8],
        out_i: &mut Vec<f32>,
        out_q: &mut Vec<f32>,
    ) {
        let mut idx = 0usize;
        while idx < bits.len() {
            let delta = match mode {
                DifferentialModulation::Dbpsk => {
                    let d = dbpsk_delta(bits[idx] & 1);
                    idx += 1;
                    d
                }
                DifferentialModulation::Dqpsk => {
                    let b0 = bits[idx] & 1;
                    let b1 = bits.get(idx + 1).copied().unwrap_or(0) & 1;
                    idx += 2;
                    dqpsk_delta(b0, b1)
                }
            };
            self.prev_phase = (self.prev_phase + delta) & 0x03;
            let (si, sq) = phase_to_iq(self.prev_phase);
            self.append_symbol_chips(si, sq, out_i, out_q);
        }
    }

    fn append_bits_chips(&mut self, bits: &[u8], out_i: &mut Vec<f32>, out_q: &mut Vec<f32>) {
        self.append_bits_chips_with_mode(MODULATION, bits, out_i, out_q);
    }

    fn bits_to_chips(&mut self, bits: &[u8]) -> (Vec<f32>, Vec<f32>) {
        let sf = self.config.spread_factor();
        let symbols = bits.len().div_ceil(MODULATION.bits_per_symbol());
        let mut chips_i = Vec::with_capacity(symbols * sf);
        let mut chips_q = Vec::with_capacity(symbols * sf);
        self.append_bits_chips(bits, &mut chips_i, &mut chips_q);
        (chips_i, chips_q)
    }

    /// チップ列をRRC整形 + キャリア変調してサンプル列に変換
    fn chips_to_samples(&mut self, chips_i: &[f32], chips_q: &[f32]) -> Vec<f32> {
        debug_assert_eq!(chips_i.len(), chips_q.len());
        let spc = self.config.samples_per_chip();
        let two_pi = 2.0 * std::f32::consts::PI;
        let fs = self.config.sample_rate;
        let fc = self.config.carrier_freq;
        let mut out = Vec::with_capacity(chips_i.len() * spc);

        for (&ci, &cq) in chips_i.iter().zip(chips_q.iter()) {
            for k in 0..spc {
                let i_imp = if k == 0 { ci } else { 0.0 };
                let q_imp = if k == 0 { cq } else { 0.0 };
                let i_f = self.rrc_i.process(i_imp);
                let q_f = self.rrc_q.process(q_imp);
                let t = self.sample_idx as f32 / fs;
                let (sin_v, cos_v) = (two_pi * fc * t).sin_cos();
                out.push(i_f * cos_v - q_f * sin_v);
                self.sample_idx += 1;
            }
        }

        let delay = self.rrc_i.delay().max(self.rrc_q.delay());
        for _ in 0..delay {
            let i_f = self.rrc_i.process(0.0);
            let q_f = self.rrc_q.process(0.0);
            let t = self.sample_idx as f32 / fs;
            let (sin_v, cos_v) = (two_pi * fc * t).sin_cos();
            out.push(i_f * cos_v - q_f * sin_v);
            self.sample_idx += 1;
        }

        out
    }

    /// 送信フレーム全体を生成する (プリアンブル + 同期ワード + データ)
    pub fn encode_frame(&mut self, bits: &[u8]) -> Vec<f32> {
        self.reset();
        let sf = self.config.spread_factor();
        let preamble_repeat = self.config.preamble_repeat;

        let sync_bits: Vec<u8> = (0..SYNC_WORD_BITS)
            .rev()
            .map(|i| ((SYNC_WORD >> i) & 1) as u8)
            .collect();
        let total_symbols = preamble_repeat
            + sync_bits.len()
            + bits.len().div_ceil(MODULATION.bits_per_symbol())
            + 1;
        let mut chips_i = Vec::with_capacity(total_symbols * sf);
        let mut chips_q = Vec::with_capacity(total_symbols * sf);

        // 1. プリアンブル
        self.mseq.reset();
        let pn = self.mseq.generate(sf);
        for rep in 0..preamble_repeat {
            let sign = if rep == preamble_repeat - 1 {
                -1.0
            } else {
                1.0
            };
            for &chip in &pn {
                chips_i.push(sign * chip as f32);
                chips_q.push(0.0);
            }
        }

        // 2. 同期ワード
        self.append_bits_chips_with_mode(
            DifferentialModulation::Dbpsk,
            &sync_bits,
            &mut chips_i,
            &mut chips_q,
        );

        // 3. データ
        self.append_bits_chips(bits, &mut chips_i, &mut chips_q);

        // 4. マージン (1シンボル分の無音チップ)
        chips_i.extend(vec![0.0; sf]);
        chips_q.extend(vec![0.0; sf]);

        self.chips_to_samples(&chips_i, &chips_q)
    }

    /// 変調器の状態をリセット
    pub fn reset(&mut self) {
        self.prev_phase = 0;
        self.sample_idx = 0;
        self.mseq.reset();
        self.rrc_i.reset();
        self.rrc_q.reset();
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

    fn symbols_for_bits(bits_len: usize) -> usize {
        bits_len.div_ceil(MODULATION.bits_per_symbol())
    }

    fn decode_diff_to_bits(
        mode: DifferentialModulation,
        diff_re: f32,
        diff_im: f32,
        out: &mut Vec<u8>,
    ) {
        match mode {
            DifferentialModulation::Dbpsk => {
                out.push(if diff_re >= 0.0 { 0 } else { 1 });
            }
            DifferentialModulation::Dqpsk => {
                if diff_re.abs() >= diff_im.abs() {
                    if diff_re >= 0.0 {
                        out.extend_from_slice(&[0, 0]);
                    } else {
                        out.extend_from_slice(&[1, 1]);
                    }
                } else if diff_im >= 0.0 {
                    out.extend_from_slice(&[0, 1]);
                } else {
                    out.extend_from_slice(&[1, 0]);
                }
            }
        }
    }

    /// プリアンブル長の確認
    #[test]
    fn test_preamble_length() {
        let mut mod_ = make_modulator();
        let preamble = mod_.generate_preamble();
        let config = DspConfig::default_48k();
        let expected_samples =
            config.spread_factor() * config.preamble_repeat * config.samples_per_chip()
                + mod_.rrc_i.delay();
        assert_eq!(preamble.len(), expected_samples);
    }

    /// サンプル値が有限値であること
    #[test]
    fn test_samples_finite() {
        let mut mod_ = make_modulator();
        let bits = vec![0u8, 1, 0, 1, 1, 0, 1, 0];
        let samples = mod_.modulate(&bits);
        assert!(samples.iter().all(|&s| s.is_finite()));
    }

    /// 変調出力の長さ確認
    #[test]
    fn test_modulate_length() {
        let mut mod_ = make_modulator();
        let config = DspConfig::default_48k();
        let bits = vec![0u8; 8];
        let samples = mod_.modulate(&bits);
        let expected =
            symbols_for_bits(bits.len()) * config.spread_factor() * config.samples_per_chip()
                + mod_.rrc_i.delay();
        assert_eq!(samples.len(), expected);
    }

    /// 変調出力の振幅が概ね±2以内であること
    #[test]
    fn test_amplitude_range() {
        let mut mod_ = make_modulator();
        let bits: Vec<u8> = (0..32).map(|i| i % 2).collect();
        let samples = mod_.modulate(&bits);
        let max_amp = samples.iter().cloned().fold(0.0f32, |a, s| a.max(s.abs()));
        assert!(max_amp < 2.0, "max={max_amp}");
    }

    /// リセット後に同じ出力が得られること
    #[test]
    fn test_reset_deterministic() {
        let bits = vec![1u8, 0, 1, 1, 0, 0, 1, 0];
        let mut mod_ = make_modulator();
        let s1 = mod_.modulate(&bits);
        mod_.reset();
        let s2 = mod_.modulate(&bits);
        assert_eq!(s1, s2);
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

    /// 差動変調とDSSS拡散の数学的正しさを検証する
    #[test]
    fn test_math_differential_dsss() {
        let mut mod_ = make_modulator();
        let bits = vec![1u8, 0, 1, 1, 0, 1, 0];
        let sf = mod_.config.spread_factor();
        let (chips_i, chips_q) = mod_.bits_to_chips(&bits);
        let symbols = symbols_for_bits(bits.len());

        assert_eq!(chips_i.len(), symbols * sf);
        assert_eq!(chips_q.len(), symbols * sf);

        let mut mseq = MSequence::new(mod_.config.mseq_order);
        let pn = mseq.generate(sf);

        let mut expected_phase = 0u8;
        let mut bit_idx = 0usize;
        for sym_idx in 0..symbols {
            let delta = match MODULATION {
                DifferentialModulation::Dbpsk => {
                    let d = dbpsk_delta(bits[bit_idx]);
                    bit_idx += 1;
                    d
                }
                DifferentialModulation::Dqpsk => {
                    let b0 = bits[bit_idx];
                    let b1 = bits.get(bit_idx + 1).copied().unwrap_or(0);
                    bit_idx += 2;
                    dqpsk_delta(b0, b1)
                }
            };
            expected_phase = (expected_phase + delta) & 0x03;
            let (si, sq) = phase_to_iq(expected_phase);
            let start = sym_idx * sf;
            let end = start + sf;
            for (chip_idx, (&ci, &cq)) in chips_i[start..end]
                .iter()
                .zip(chips_q[start..end].iter())
                .enumerate()
            {
                let p = pn[chip_idx] as f32;
                assert!((ci - si * p).abs() < 1e-6);
                assert!((cq - sq * p).abs() < 1e-6);
            }
        }
    }

    /// 同期位置が既知の前提で、encode_frame波形から差動ビットを復元できること
    #[test]
    fn test_known_sync_position_roundtrip_without_demodulator() {
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
        let total_delay = mod_.rrc_i.delay() + rrc_i.delay();

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

        let sync_symbols = sync_bits.len();
        let payload_symbols = symbols_for_bits(bits.len());
        let mut prev = correlate_symbol(0, best_phase, &i_ch, &q_ch);
        let mut decoded_bits = Vec::with_capacity(expected_bits.len() + 1);
        for sym_idx in 0..sync_symbols {
            let cur = correlate_symbol(config.preamble_repeat + sym_idx, best_phase, &i_ch, &q_ch);
            let diff_re = prev.0 * cur.0 + prev.1 * cur.1;
            let diff_im = prev.0 * cur.1 - prev.1 * cur.0;
            decode_diff_to_bits(
                DifferentialModulation::Dbpsk,
                diff_re,
                diff_im,
                &mut decoded_bits,
            );
            prev = cur;
        }
        for sym_idx in 0..payload_symbols {
            let cur = correlate_symbol(
                config.preamble_repeat + sync_symbols + sym_idx,
                best_phase,
                &i_ch,
                &q_ch,
            );
            let diff_re = prev.0 * cur.0 + prev.1 * cur.1;
            let diff_im = prev.0 * cur.1 - prev.1 * cur.0;
            decode_diff_to_bits(MODULATION, diff_re, diff_im, &mut decoded_bits);
            prev = cur;
        }
        decoded_bits.truncate(expected_bits.len());
        assert_eq!(decoded_bits, expected_bits);
    }
}
