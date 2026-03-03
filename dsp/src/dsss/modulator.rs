//! 差動PSK + DSSS 変調器
//!
//! # 変調パイプライン
//! 1. 差動符号化: `DBPSK` または `DQPSK` で位相遷移を決定
//! 2. DSSS拡散: 各シンボルにM系列を掛ける (チップ展開)
//! 3. RRCパルス整形: I/QチップをそれぞれRRCフィルタで成形
//! 4. 帯域シフト: キャリアfcで実信号へアップコンバート

use crate::common::msequence::MSequence;
use crate::common::nco::Nco;
use crate::common::resample::Resampler;
use crate::common::rrc_filter::RrcFilter;
use crate::params::{INTERNAL_SPC, MODULATION, SYNC_WORD};
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
    proc_config: DspConfig,
    mseq: MSequence,
    resampler_i: Resampler,
    resampler_q: Resampler,
    rrc_i: RrcFilter,
    rrc_q: RrcFilter,
    /// 差動符号化の累積位相 (0,1,2,3) = (0,90,180,270度)
    prev_phase: u8,
    /// キャリア波生成用NCO
    nco: Nco,
}

impl Modulator {
    /// `DspConfig` を指定して変調器を作成する
    pub fn new(config: DspConfig) -> Self {
        let proc_config = DspConfig::new_for_processing_from(&config);
        let rrc_i = RrcFilter::from_config(&proc_config);
        let rrc_q = RrcFilter::from_config(&proc_config);
        let nco = Nco::new(config.carrier_freq, config.sample_rate);

        // リサンプラのカットオフ設定: 送信側RRCの全帯域を通過させる
        let rrc_bw = proc_config.chip_rate * (1.0 + proc_config.rrc_alpha) * 0.5;
        let cutoff = Some(rrc_bw);

        Modulator {
            resampler_i: Resampler::new_with_cutoff(
                proc_config.sample_rate as u32,
                config.sample_rate as u32,
                cutoff,
            ),
            resampler_q: Resampler::new_with_cutoff(
                proc_config.sample_rate as u32,
                config.sample_rate as u32,
                cutoff,
            ),
            mseq: MSequence::new(config.mseq_order),
            rrc_i,
            rrc_q,
            config,
            proc_config,
            prev_phase: 0,
            nco,
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
        let spc = INTERNAL_SPC;

        // 1. 内部レート (fs_proc) でのベースバンド信号生成
        let mut bb_i = Vec::with_capacity(chips_i.len() * spc);
        let mut bb_q = Vec::with_capacity(chips_i.len() * spc);
        for (&ci, &cq) in chips_i.iter().zip(chips_q.iter()) {
            for k in 0..spc {
                let i_imp = if k == 0 { ci } else { 0.0 };
                let q_imp = if k == 0 { cq } else { 0.0 };
                bb_i.push(self.rrc_i.process(i_imp));
                bb_q.push(self.rrc_q.process(q_imp));
            }
        }

        // 2. 出力レート (fs_out) へのリサンプリング
        let mut resampled_i = Vec::new();
        let mut resampled_q = Vec::new();
        self.resampler_i.process(&bb_i, &mut resampled_i);
        self.resampler_q.process(&bb_q, &mut resampled_q);

        // 3. 出力レートでのキャリア混合 (Mix)
        let mut out = Vec::with_capacity(resampled_i.len());
        for (&i_f, &q_f) in resampled_i.iter().zip(resampled_q.iter()) {
            let lo = self.nco.step();
            out.push(i_f * lo.re - q_f * lo.im);
        }

        out
    }

    /// RRCフィルタに残っている遅延分のサンプルをゼロで押し出して出力する
    pub fn flush(&mut self) -> Vec<f32> {
        // RRCフィルタの応答全体（全タップ分）を押し出すために必要な無音サンプルを計算。
        // delay() ではなく num_taps() を使うのが物理的に正しい（テールの終わりまで出すため）。
        let rrc_taps_bb = self.rrc_i.num_taps().max(self.rrc_q.num_taps());
        let ratio = self.config.sample_rate / self.proc_config.sample_rate;
        let rrc_push_out = (rrc_taps_bb as f32 * ratio).ceil() as usize;

        let mut out = self.modulate_silence(rrc_push_out);

        let mut res_i = Vec::new();
        let mut res_q = Vec::new();
        self.resampler_i.flush(&mut res_i);
        self.resampler_q.flush(&mut res_q);

        for (&si, &sq) in res_i.iter().zip(res_q.iter()) {
            let lo = self.nco.step();
            out.push(si * lo.re - sq * lo.im);
        }

        out
    }

    /// 指定されたサンプル数分だけ無音 (0.0) を入力して Modulator を進める
    ///
    /// これにより、無音期間中も NCO が回転し、RRC フィルタのテールが自然に出力される。
    pub fn modulate_silence(&mut self, samples: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(samples);

        // 出力サンプル数が samples に達するまで内部レートで無音を生成し、
        // リサンプルとミキシングを行う。
        let ratio = self.config.sample_rate / self.proc_config.sample_rate;
        let needed_bb = (samples as f32 / ratio).ceil() as usize + 1;

        for _ in 0..needed_bb {
            let i_f = self.rrc_i.process(0.0);
            let q_f = self.rrc_q.process(0.0);

            let mut res_i = Vec::new();
            let mut res_q = Vec::new();
            self.resampler_i.process(&[i_f], &mut res_i);
            self.resampler_q.process(&[q_f], &mut res_q);

            for (&si, &sq) in res_i.iter().zip(res_q.iter()) {
                let lo = self.nco.step();
                out.push(si * lo.re - sq * lo.im);
            }
            if out.len() >= samples {
                break;
            }
        }

        out.truncate(samples);
        out
    }

    /// 送信フレーム全体を生成する (プリアンブル + 同期ワード + データ)
    pub fn encode_frame(&mut self, bits: &[u8]) -> Vec<f32> {
        self.prev_phase = 0;
        let sf = self.config.spread_factor();
        let preamble_repeat = self.config.preamble_repeat;

        let sync_bits: Vec<u8> = (0..self.config.sync_word_bits)
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
        self.nco.reset();
        self.mseq.reset();
        self.rrc_i.reset();
        self.rrc_q.reset();
        let rrc_bw = self.proc_config.chip_rate * (1.0 + self.proc_config.rrc_alpha) * 0.5;
        self.resampler_i.reconfigure(
            self.proc_config.sample_rate as u32,
            self.config.sample_rate as u32,
            Some(rrc_bw),
        );
        self.resampler_q.reconfigure(
            self.proc_config.sample_rate as u32,
            self.config.sample_rate as u32,
            Some(rrc_bw),
        );
    }

    pub fn config(&self) -> &DspConfig {
        &self.config
    }

    /// 変調器の物理的な群遅延（出力サンプル数単位でのピーク位置）を返す
    pub fn delay(&self) -> usize {
        // 1. RRCフィルタの群遅延 (ベースバンドレート)
        let rrc_delay_bb = self.rrc_i.delay(); // (num_taps - 1) / 2
        // 2. リサンプラの群遅延 (ベースバンドレート)
        let resampler_delay_bb = self.resampler_i.delay() as f64 
            / (self.config.sample_rate as f64 / self.proc_config.sample_rate as f64);
        
        // 合計遅延をターゲットレートへ換算
        let ratio = self.config.sample_rate as f64 / self.proc_config.sample_rate as f64;
        ((rrc_delay_bb as f64 + resampler_delay_bb) * ratio).round() as usize
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
        let mut preamble = mod_.generate_preamble();
        // ストリームの末尾を出し切る
        preamble.extend(mod_.flush());

        let config = DspConfig::default_48k();
        let expected_base =
            config.spread_factor() * config.preamble_repeat * config.samples_per_chip();
        
        // 理論的な合計長 = ベース信号長 + 物理的テール長 (130サンプル)
        // 130 = (RRC応答長 49 + リサンプラ応答長 17 - 1) * 2
        let expected_total = expected_base + 130; 
        
        let diff = (preamble.len() as i32 - expected_total as i32).abs();
        assert!(
            diff <= 2,
            "len={}, expected_total={}, diff={}",
            preamble.len(),
            expected_total,
            diff
        );
    }

    /// サンプル値が有限値であること
    #[test]
    fn test_samples_finite() {
        let mut mod_ = make_modulator();
        let bits = vec![0u8, 1, 0, 1, 1, 0, 1, 0];
        let samples = mod_.modulate(&bits);
        assert!(samples.iter().all(|&s| s.is_finite()));
    }

    #[test]
    fn test_carrier_phase_precision_loss() {
        let mut mod_ = make_modulator();
        let bits = vec![0xAA; 10]; // Short payload

        // Generate normal frame
        let frame1 = mod_.encode_frame(&bits);

        // Fast forward NCO to simulate 10 minutes of audio running
        // 48000 samples/sec * 600 sec = 28,800,000 samples
        for _ in 0..28_800_000 {
            mod_.nco.step();
        }

        // Generate frame after a long time
        let frame2 = mod_.encode_frame(&bits);

        // We expect the amplitude envelope to be essentially the same
        // (no high frequency distortion/attenuation from f32 precision loss)

        let rms1 = (frame1.iter().map(|&x| x * x).sum::<f32>() / frame1.len() as f32).sqrt();
        let rms2 = (frame2.iter().map(|&x| x * x).sum::<f32>() / frame2.len() as f32).sqrt();

        // RMS should not change more than 1%
        assert!(
            (rms1 - rms2).abs() < rms1 * 0.01,
            "Carrier phase precision loss detected! rms1: {}, rms2: {}",
            rms1,
            rms2
        );
    }

    /// 変調出力の長さ確認
    #[test]
    fn test_modulate_length() {
        let mut mod_ = make_modulator();
        let config = DspConfig::default_48k();
        let bits = vec![0u8; 8];
        let mut samples = mod_.modulate(&bits);
        // ストリームの末尾を出し切る
        samples.extend(mod_.flush());

        let expected_base =
            symbols_for_bits(bits.len()) * config.spread_factor() * config.samples_per_chip();
        
        // 理論的な合計長 = ベース信号長 + 物理的テール長 (130サンプル)
        let expected_total = expected_base + 130; 
        
        let diff = (samples.len() as i32 - expected_total as i32).abs();
        assert!(
            diff <= 2,
            "len={}, expected_total={}, diff={}",
            samples.len(),
            expected_total,
            diff
        );
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

    /// 変調器の物理的な遅延（delay()メソッド）が実際のピーク位置と一致することを検証する
    #[test]
    fn test_modulator_delay_method_matches_physical_peak() {
        let mut mod_ = make_modulator();
        let expected_delay = mod_.delay();

        // 1チップ（1.0）のインパルスを入力
        let mut samples = mod_.chips_to_samples(&[1.0], &[0.0]);
        samples.extend(mod_.flush());

        // ピーク位置を特定
        let (peak_idx, &peak_val) = samples.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .unwrap();

        println!("Modulator Latency: peak_idx={}, delay()={}, peak_val={:.4}", peak_idx, expected_delay, peak_val);

        // delay() の戻り値が物理現象（インパルス応答のピーク）と完全に一致することを保証
        assert_eq!(peak_idx, expected_delay, "Modulator::delay() mismatch with physical peak position");
        assert!(peak_val > 0.5, "Peak is too weak");
    }

    /// 変調器全体の物理的な遅延と応答長を検証する
    ///
    /// インパルス（1チップ）を入力した際のピーク位置と全体長を測定し、
    /// RRCフィルタおよびリサンプラの群遅延の合計と一致することを検証する。
    #[test]
    fn test_modulator_total_physical_latency() {
        let mut mod_ = make_modulator();

        // 1チップ（1インパルス）を入力
        let mut samples = mod_.chips_to_samples(&[1.0], &[0.0]);
        samples.extend(mod_.flush());

        // 1. 物理的なピーク位置（Group Delay）の検証
        let (peak_idx, &peak_val) = samples.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .unwrap();

        // 理論値の計算:
        // RRCフィルタ群遅延: (16 symbols * 3 spc + 1 taps - 1) / 2 = 24 サンプル (@24kHz)
        // リサンプラ群遅延: (17 taps - 1) / 2 = 8 サンプル (@24kHz)
        // 合計群遅延: 24 + 8 = 32 サンプル (@24kHz)
        // 出力レート換算: 32 * (48000 / 24000) = 64 サンプル (@48kHz)
        let expected_peak_idx = 64; 

        println!("Modulator Physical Peak: idx={}, value={:.4}, expected={}", peak_idx, peak_val, expected_peak_idx);
        assert_eq!(peak_idx, expected_peak_idx, "Modulator group delay (peak position) mismatch");
        assert!(peak_val > 0.5, "Peak value is too weak, signal may be distorted");

        // 2. 全応答長（Total Response Length）の検証
        // 理論値: (RRC応答長 49 + リサンプラ応答長 17 - 1) * 2 = 130 サンプル付近
        // 実際には flush 分が含まれ、136 サンプル程度になる。
        println!("Modulator Total Response Length: {}", samples.len());
        assert!(samples.len() >= 130, "Total response is too short, samples are being lost");
        assert!(samples.len() <= 150, "Total response is unexpectedly long");
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

    #[test]
    fn test_gapless_continuity() {
        let mut mod_ = make_modulator();
        let bits = vec![0x55; 4];

        // 1回で2フレーム分のビットをまとめて変調
        let mut bits2 = bits.clone();
        bits2.extend_from_slice(&bits);
        let samples_all = mod_.modulate(&bits2);

        mod_.reset();

        // 分割して変調
        let samples_part1 = mod_.modulate(&bits);
        let samples_part2 = mod_.modulate(&bits);

        let mut samples_combined = samples_part1;
        samples_combined.extend_from_slice(&samples_part2);

        assert_eq!(samples_all.len(), samples_combined.len());
        // 浮動小数点の誤差を考慮して比較
        for (&a, &b) in samples_all.iter().zip(samples_combined.iter()) {
            assert!((a - b).abs() < 1e-6f32, "a: {}, b: {}", a, b);
        }
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
        let mut frame = mod_.encode_frame(&bits);
        // RRCフィルタの遅延を吸収するためのパディングを追加
        frame.extend(mod_.flush());

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
        // リサンプラの遅延等により位相が数チップ分ずれる可能性があるため、
        // 1シンボル分まるごと探索してピーク位置（シンボルの境界）を特定する。
        for phase in 0..sym_len {
            let (ci, cq) = correlate_symbol(0, phase, &i_ch, &q_ch);
            let mag = ci * ci + cq * cq;
            if mag > best_mag {
                best_mag = mag;
                best_phase = phase;
            }
        }

        let sync_bits: Vec<u8> = (0..config.sync_word_bits)
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
