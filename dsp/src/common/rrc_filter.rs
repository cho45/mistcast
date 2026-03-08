//! ルートレイズドコサイン (RRC) FIRフィルタ
//!
//! RRCフィルタはシンボル間干渉 (ISI) を最小化するための波形整形フィルタ。
//! 送信側・受信側の両方に同じRRCフィルタを適用すると、
//! 合計でレイズドコサイン応答となり、ナイキスト基準を満たす。
//!
//! フィルタ係数の計算式:
//!   t ≠ 0, t ≠ ±Ts/(4α) の場合:
//!     h(t) = [sin(π·t/Ts·(1-α)) + 4α·t/Ts·cos(π·t/Ts·(1+α))]
//!            / [π·t/Ts·(1-(4α·t/Ts)^2)]
//!   t = 0:
//!     h(0) = (1-α) + 4α/π
//!   t = ±Ts/(4α):
//!     h(t) = α/√2·[(1+2/π)·sin(π/(4α)) + (1-2/π)·cos(π/(4α))]

use crate::DspConfig;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use std::arch::wasm32::{f32x4, f32x4_add, f32x4_extract_lane, f32x4_mul};

/// RRCフィルタ係数を計算する
///
/// # 引数
/// - `num_taps`: フィルタのタップ数 (奇数推奨)
/// - `alpha`: ロールオフ係数 (0 < alpha <= 1)
/// - `chip_rate`: チップレート (Hz)
/// - `sample_rate`: サンプリングレート (Hz)
///
/// # 戻り値
/// 正規化された係数列 (L2ノルムで正規化)
pub fn rrc_coeffs(num_taps: usize, alpha: f32, chip_rate: f32, sample_rate: f32) -> Vec<f32> {
    let ts = 1.0 / chip_rate; // チップ周期
    let dt = 1.0 / sample_rate; // サンプル間隔

    let center = (num_taps / 2) as i32;
    let mut coeffs: Vec<f32> = (0..num_taps as i32)
        .map(|i| {
            let n = i - center;
            let t = n as f32 * dt;
            rrc_at(t, ts, alpha)
        })
        .collect();

    // 正規化: 係数の二乗和の平方根で割る (単位ゲイン)
    let norm: f32 = coeffs.iter().map(|&c| c * c).sum::<f32>().sqrt();
    if norm > 0.0 {
        coeffs.iter_mut().for_each(|c| *c /= norm);
    }
    coeffs
}

/// RRCフィルタのインパルス応答値を時刻 t で評価する
fn rrc_at(t: f32, ts: f32, alpha: f32) -> f32 {
    let eps = 1e-6 * ts;

    if t.abs() < eps {
        return (1.0 - alpha) + 4.0 * alpha / std::f32::consts::PI;
    }

    let critical = ts / (4.0 * alpha);
    if (t.abs() - critical).abs() < eps {
        let pi = std::f32::consts::PI;
        return alpha / (2.0_f32).sqrt()
            * ((1.0 + 2.0 / pi) * (pi / (4.0 * alpha)).sin()
                + (1.0 - 2.0 / pi) * (pi / (4.0 * alpha)).cos());
    }

    let pi = std::f32::consts::PI;
    let x = t / ts;
    let numer = (pi * x * (1.0 - alpha)).sin() + 4.0 * alpha * x * (pi * x * (1.0 + alpha)).cos();
    let denom = pi * x * (1.0 - (4.0 * alpha * x).powi(2));
    numer / denom
}

/// 状態を保持するRRCフィルタ (逐次サンプル処理用)
pub struct RrcFilter {
    coeffs: Vec<f32>,
    coeffs_rev: Vec<f32>,
    // 2N バッファに同じ値を二重書きし、常に連続窓を取得できるようにする
    history: Vec<f32>,
    pos: usize,
}

/// RRCマッチドフィルタ + デシメーションを統合した逐次フィルタ。
///
/// 入力ごとに状態は更新するが、出力は `decimation` サンプルに1回だけ生成する。
/// これにより「RRC -> LPF -> decimation」の2段を1段化できる。
pub struct DecimatingRrcFilter {
    coeffs: Vec<f32>,
    buffer: Vec<f32>,
    pos: usize,
    decimation: usize,
    phase: usize,
}

impl RrcFilter {
    /// `DspConfig` からフィルタを作成する
    pub fn from_config(config: &DspConfig) -> Self {
        let coeffs = rrc_coeffs(
            config.rrc_num_taps(),
            config.rrc_alpha,
            config.chip_rate,
            config.proc_sample_rate(),
        );
        let num_taps = coeffs.len();
        let mut coeffs_rev = coeffs.clone();
        coeffs_rev.reverse();
        RrcFilter {
            coeffs,
            coeffs_rev,
            history: vec![0.0f32; num_taps * 2],
            pos: 0,
        }
    }

    /// カスタムパラメータでフィルタを作成する
    pub fn with_params(num_taps: usize, alpha: f32, chip_rate: f32, sample_rate: f32) -> Self {
        let coeffs = rrc_coeffs(num_taps, alpha, chip_rate, sample_rate);
        let mut coeffs_rev = coeffs.clone();
        coeffs_rev.reverse();
        RrcFilter {
            coeffs,
            coeffs_rev,
            history: vec![0.0f32; num_taps * 2],
            pos: 0,
        }
    }

    #[inline]
    fn push_and_window_start(&mut self, sample: f32) -> usize {
        let n = self.coeffs.len();
        let write = self.pos;
        self.history[write] = sample;
        self.history[write + n] = sample;
        self.pos += 1;
        if self.pos >= n {
            self.pos = 0;
        }
        write + 1
    }

    /// 1サンプルを処理する (循環バッファによる畳み込み)
    #[cfg_attr(
        all(target_arch = "wasm32", target_feature = "simd128"),
        allow(dead_code)
    )]
    fn process_scalar_path(&mut self, sample: f32) -> f32 {
        let n = self.coeffs.len();
        let start = self.push_and_window_start(sample);
        let mut out = 0.0f32;
        for k in 0..n {
            out += self.coeffs_rev[k] * self.history[start + k];
        }
        out
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[inline]
    fn hsum4(v: std::arch::wasm32::v128) -> f32 {
        f32x4_extract_lane::<0>(v)
            + f32x4_extract_lane::<1>(v)
            + f32x4_extract_lane::<2>(v)
            + f32x4_extract_lane::<3>(v)
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn process_simd_path(&mut self, sample: f32) -> f32 {
        let n = self.coeffs.len();
        let start = self.push_and_window_start(sample);
        let mut sum4 = f32x4(0.0, 0.0, 0.0, 0.0);
        let mut k = 0usize;
        while k + 4 <= n {
            let x = f32x4(
                self.history[start + k],
                self.history[start + k + 1],
                self.history[start + k + 2],
                self.history[start + k + 3],
            );
            let h = f32x4(
                self.coeffs_rev[k],
                self.coeffs_rev[k + 1],
                self.coeffs_rev[k + 2],
                self.coeffs_rev[k + 3],
            );
            sum4 = f32x4_add(sum4, f32x4_mul(h, x));
            k += 4;
        }

        let mut out = Self::hsum4(sum4);
        for kk in k..n {
            out += self.coeffs_rev[kk] * self.history[start + kk];
        }
        out
    }

    pub fn process(&mut self, sample: f32) -> f32 {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            return self.process_simd_path(sample);
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            self.process_scalar_path(sample)
        }
    }

    /// サンプル列をまとめて処理する
    pub fn process_block(&mut self, samples: &[f32]) -> Vec<f32> {
        samples.iter().map(|&s| self.process(s)).collect()
    }

    /// サンプル列をインプレースで処理する（ゼロアロケーション）
    ///
    /// # 引数
    /// - `samples`: 処理するサンプル列（入出力兼用）
    ///
    /// # 注意
    /// 入力サンプルは上書きされます。元の値を保持する必要がある場合は、
    /// 事前にコピーを作成してください。
    pub fn process_block_in_place(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.process(*s);
        }
    }

    /// フィルタ状態をリセットする
    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.pos = 0;
    }

    /// フィルタの群遅延 (サンプリング間隔単位) を返す
    /// FIRフィルタの場合、(タップ数 - 1) / 2 となる。
    pub fn delay(&self) -> usize {
        (self.coeffs.len() - 1) / 2
    }

    pub fn num_taps(&self) -> usize {
        self.coeffs.len()
    }
}

impl DecimatingRrcFilter {
    pub fn from_config(config: &DspConfig, decimation: usize) -> Self {
        assert!(decimation > 0, "decimation must be > 0");
        let coeffs = rrc_coeffs(
            config.rrc_num_taps(),
            config.rrc_alpha,
            config.chip_rate,
            config.proc_sample_rate(),
        );
        let taps = coeffs.len();
        Self {
            coeffs,
            buffer: vec![0.0f32; taps],
            pos: 0,
            decimation,
            phase: 0,
        }
    }

    #[inline]
    fn push_and_maybe_output(&mut self, sample: f32) -> Option<f32> {
        self.buffer[self.pos] = sample;
        self.pos += 1;
        if self.pos >= self.buffer.len() {
            self.pos = 0;
        }

        if self.phase != 0 {
            self.phase += 1;
            if self.phase >= self.decimation {
                self.phase = 0;
            }
            return None;
        }

        let n = self.coeffs.len();
        let mut out = 0.0f32;
        let mut idx = self.pos;
        for k in 0..n {
            idx = if idx == 0 { n - 1 } else { idx - 1 };
            out += self.coeffs[k] * self.buffer[idx];
        }

        self.phase += 1;
        if self.phase >= self.decimation {
            self.phase = 0;
        }

        Some(out)
    }

    pub fn process_block(&mut self, input: &[f32], output: &mut Vec<f32>) {
        output.clear();
        if input.is_empty() {
            return;
        }
        output.reserve(input.len() / self.decimation + 1);
        for &x in input {
            if let Some(y) = self.push_and_maybe_output(x) {
                output.push(y);
            }
        }
    }

    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.pos = 0;
        self.phase = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    use wasm_bindgen_test::wasm_bindgen_test;

    fn test_config() -> DspConfig {
        DspConfig::default_48k()
    }

    /// インパルス応答のピーク位置が理論的な遅延と一致することを確認
    #[test]
    fn test_impulse_response_delay() {
        let config = test_config();
        let mut filter = RrcFilter::from_config(&config);
        let delay = filter.delay();

        let mut impulse = vec![0.0f32; delay * 2 + 1];
        impulse[0] = 1.0;

        let output = filter.process_block(&impulse);
        let (peak_idx, &peak_val) = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        assert_eq!(
            peak_idx, delay,
            "ピーク位置が理論的遅延 {} と一致すること",
            delay
        );
        assert!(peak_val > 0.0, "ピーク値が正であること");
    }

    /// ナイキスト基準の検証:
    /// 2つのRRCフィルタを縦続 (= レイズドコサイン) した場合、
    /// サンプリング点 t=k*Ts (k≠0) での応答がほぼ0になる
    #[test]
    fn test_nyquist_isi_zero() {
        let config = test_config();
        let sps = config.samples_per_chip();
        let h = rrc_coeffs(
            config.rrc_num_taps(),
            config.rrc_alpha,
            config.chip_rate,
            config.sample_rate,
        );

        // 2つのRRCを畳み込み → レイズドコサイン応答
        let rc_len = 2 * h.len() - 1;
        let mut rc = vec![0.0f32; rc_len];
        for i in 0..h.len() {
            for j in 0..h.len() {
                rc[i + j] += h[i] * h[j];
            }
        }

        let center = rc_len / 2;
        let peak = rc[center];
        assert!(peak > 0.0, "ピークが正であること");

        // 整数比 (48000/8000 = 6) の場合、サンプリング点でのISIは極めて低くなるはず
        // 許容閾値: 2%以内 (実信号レベルでの干渉抑制)
        let tolerance = 0.02f32;
        for k in 1..=4 {
            if center + k * sps < rc_len {
                let isi = rc[center + k * sps].abs() / peak;
                assert!(
                    isi < tolerance,
                    "t=+{}*Ts での ISI = {:.4} >= {:.4} (ナイキスト条件違反)",
                    k,
                    isi,
                    tolerance
                );
            }
            if center >= k * sps {
                let isi = rc[center - k * sps].abs() / peak;
                assert!(
                    isi < tolerance,
                    "t=-{}*Ts での ISI = {:.4} >= {:.4} (ナイキスト条件違反)",
                    k,
                    isi,
                    tolerance
                );
            }
        }
    }

    /// フィルタが対称であることを確認 (線形位相フィルタの必要条件)
    #[test]
    fn test_symmetry() {
        let config = test_config();
        let h = rrc_coeffs(
            config.rrc_num_taps(),
            config.rrc_alpha,
            config.chip_rate,
            config.sample_rate,
        );
        let n = h.len();
        for i in 0..n / 2 {
            let diff = (h[i] - h[n - 1 - i]).abs();
            assert!(
                diff < 1e-6,
                "係数が対称でない: h[{}]={} vs h[{}]={}",
                i,
                h[i],
                n - 1 - i,
                h[n - 1 - i]
            );
        }
    }

    /// 44.1kHz でも正しく動作すること
    #[test]
    fn test_44k_config() {
        let config = DspConfig::default_44k();
        let h = rrc_coeffs(
            config.rrc_num_taps(),
            config.rrc_alpha,
            config.chip_rate,
            config.sample_rate,
        );
        assert!(!h.is_empty());
        assert!(h.iter().all(|&c| c.is_finite()));
        // 対称性確認
        let n = h.len();
        for i in 0..n / 2 {
            assert!((h[i] - h[n - 1 - i]).abs() < 1e-6);
        }
    }

    /// process_block の動作確認
    #[test]
    fn test_process_block() {
        let config = test_config();
        let mut filter = RrcFilter::from_config(&config);
        let input: Vec<f32> = (0..200).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let output = filter.process_block(&input);
        assert_eq!(output.len(), input.len());
        let max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max > 0.0);
    }

    /// process_block_in_place の動作確認
    #[test]
    fn test_process_block_in_place() {
        let config = test_config();
        let mut filter = RrcFilter::from_config(&config);

        // 入力をコピー
        let input: Vec<f32> = (0..200).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let mut buffer = input.clone();

        // インプレース処理
        filter.process_block_in_place(&mut buffer);

        // 通常の処理と比較
        let expected = filter.process_block(&input);

        assert_eq!(buffer.len(), expected.len());
        for (i, (&actual, &exp)) in buffer.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-6,
                "index {}: {} != {}",
                i,
                actual,
                exp
            );
        }
    }

    /// process_block_in_place が process_block と同じ結果を出すこと
    #[test]
    fn test_process_block_in_place_matches_process_block() {
        let config = test_config();
        let mut filter1 = RrcFilter::from_config(&config.clone());
        let mut filter2 = RrcFilter::from_config(&config);

        let input: Vec<f32> = (0..1000)
            .map(|i| {
                let t = i as f32 / config.sample_rate;
                0.7 * (2.0 * std::f32::consts::PI * 800.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 2300.0 * t).cos()
            })
            .collect();

        // 通常の処理
        let expected = filter1.process_block(&input);

        // インプレース処理
        let mut buffer = input.clone();
        filter2.process_block_in_place(&mut buffer);

        assert_eq!(buffer.len(), expected.len());
        for (i, (&actual, &exp)) in buffer.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-5,
                "index {}: {} != {}",
                i,
                actual,
                exp
            );
        }
    }

    #[test]
    fn test_decimating_rrc_matches_subsampled_rrc() {
        let config = test_config();
        let decimation = 3usize;
        let mut full = RrcFilter::from_config(&config);
        let mut decim = DecimatingRrcFilter::from_config(&config, decimation);

        let mut input = Vec::with_capacity(4096);
        for i in 0..4096 {
            let t = i as f32 / config.sample_rate;
            input.push(
                0.7 * (2.0 * std::f32::consts::PI * 800.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 2300.0 * t).cos(),
            );
        }

        let full_out = full.process_block(&input);
        let expected: Vec<f32> = full_out.into_iter().step_by(decimation).collect();

        let mut got = Vec::new();
        decim.process_block(&input, &mut got);

        assert_eq!(got.len(), expected.len());
        let max_err = got
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_err < 1e-5, "max_err={}", max_err);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[wasm_bindgen_test]
    fn test_rrc_filter_simd_matches_scalar_path() {
        let config = test_config();
        let mut scalar = RrcFilter::from_config(&config);
        let mut simd = RrcFilter::from_config(&config);

        let input: Vec<f32> = (0..2048)
            .map(|i| {
                let t = i as f32 / config.sample_rate;
                0.73 * (2.0 * std::f32::consts::PI * 700.0 * t).sin()
                    + 0.19 * (2.0 * std::f32::consts::PI * 1900.0 * t).cos()
                    + 0.08 * (2.0 * std::f32::consts::PI * 3200.0 * t).sin()
            })
            .collect();

        for (idx, &x) in input.iter().enumerate() {
            let y_scalar = scalar.process_scalar_path(x);
            let y_simd = simd.process_simd_path(x);
            assert!(
                (y_scalar - y_simd).abs() < 1e-5,
                "idx={} scalar={} simd={}",
                idx,
                y_scalar,
                y_simd
            );
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[wasm_bindgen_test]
    fn test_rrc_filter_simd_tap_remainder_boundaries() {
        // taps=17 (mod4=1), taps=19 (mod4=3) の端数経路を検証
        for &taps in &[17usize, 19usize] {
            let mut scalar = RrcFilter::with_params(taps, 0.3, 8_000.0, 24_000.0);
            let mut simd = RrcFilter::with_params(taps, 0.3, 8_000.0, 24_000.0);
            for i in 0..(taps * 8 + 5) {
                let t = i as f32 / 24_000.0;
                let x = 0.61 * (2.0 * std::f32::consts::PI * 900.0 * t).sin()
                    + 0.27 * (2.0 * std::f32::consts::PI * 2100.0 * t).cos();
                let y_scalar = scalar.process_scalar_path(x);
                let y_simd = simd.process_simd_path(x);
                assert!(
                    (y_scalar - y_simd).abs() < 1e-5,
                    "taps={} i={} scalar={} simd={}",
                    taps,
                    i,
                    y_scalar,
                    y_simd
                );
            }
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[wasm_bindgen_test]
    fn test_rrc_filter_simd_ring_wrap_boundary() {
        let taps = 17usize;
        let mut scalar = RrcFilter::with_params(taps, 0.25, 8_000.0, 24_000.0);
        let mut simd = RrcFilter::with_params(taps, 0.25, 8_000.0, 24_000.0);

        // リングバッファ位置が何度も折り返す長さ
        for i in 0..(taps * 20) {
            let x = ((i % 11) as f32 - 5.0) * 0.07;
            let y_scalar = scalar.process_scalar_path(x);
            let y_simd = simd.process_simd_path(x);
            assert!(
                (y_scalar - y_simd).abs() < 1e-5,
                "i={} scalar={} simd={}",
                i,
                y_scalar,
                y_simd
            );
        }
    }
}
