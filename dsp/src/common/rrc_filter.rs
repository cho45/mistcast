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
use std::arch::wasm32::{f32x4, f32x4_add, f32x4_extract_lane, f32x4_mul, v128, v128_load};

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

/// I/Q 2チャネルを同一係数・同一位相で同時に処理するRRCフィルタ。
///
/// 2本の `RrcFilter` を別々に回す場合と比べ、係数参照とループ制御の
/// オーバーヘッドを削減する。
pub struct IqRrcFilter {
    coeffs_rev: Vec<f32>,
    history_i: Vec<f32>,
    history_q: Vec<f32>,
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
        let input_ptr = self.history[start..start + n].as_ptr();
        let coeff_ptr = self.coeffs_rev.as_ptr();
        while k + 8 <= n {
            // SAFETY: `k + 8 <= n` の範囲でのみ進むため、
            // `input_ptr[k..k+8]` と `coeff_ptr[k..k+8]` は有効領域内。
            let x0 = unsafe { v128_load(input_ptr.add(k) as *const v128) };
            let h0 = unsafe { v128_load(coeff_ptr.add(k) as *const v128) };
            let x1 = unsafe { v128_load(input_ptr.add(k + 4) as *const v128) };
            let h1 = unsafe { v128_load(coeff_ptr.add(k + 4) as *const v128) };
            sum4 = f32x4_add(sum4, f32x4_mul(h0, x0));
            sum4 = f32x4_add(sum4, f32x4_mul(h1, x1));
            k += 8;
        }

        while k + 4 <= n {
            // SAFETY: `k + 4 <= n` の条件下でのみ読み込むため、
            // `input_ptr[k..k+4]` と `coeff_ptr[k..k+4]` は有効範囲内。
            let x = unsafe { v128_load(input_ptr.add(k) as *const v128) };
            let h = unsafe { v128_load(coeff_ptr.add(k) as *const v128) };
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
        let mut output = Vec::with_capacity(samples.len());
        self.process_block_into(samples, &mut output);
        output
    }

    /// サンプル列を出力バッファへ処理する（バッファ再利用）
    pub fn process_block_into(&mut self, samples: &[f32], output: &mut Vec<f32>) {
        output.clear();
        if samples.is_empty() {
            return;
        }
        output.reserve(samples.len());
        for &s in samples {
            output.push(self.process(s));
        }
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

impl IqRrcFilter {
    /// `DspConfig` から I/Q 同時RRCフィルタを作成する
    pub fn from_config(config: &DspConfig) -> Self {
        Self::with_params(
            config.rrc_num_taps(),
            config.rrc_alpha,
            config.chip_rate,
            config.proc_sample_rate(),
        )
    }

    /// カスタムパラメータで I/Q 同時RRCフィルタを作成する
    pub fn with_params(num_taps: usize, alpha: f32, chip_rate: f32, sample_rate: f32) -> Self {
        let coeffs = rrc_coeffs(num_taps, alpha, chip_rate, sample_rate);
        let mut coeffs_rev = coeffs;
        coeffs_rev.reverse();
        let n = coeffs_rev.len();
        Self {
            coeffs_rev,
            history_i: vec![0.0f32; n * 2],
            history_q: vec![0.0f32; n * 2],
            pos: 0,
        }
    }

    #[inline]
    fn push_pair_and_window_start(&mut self, sample_i: f32, sample_q: f32) -> usize {
        let n = self.coeffs_rev.len();
        let write = self.pos;
        self.history_i[write] = sample_i;
        self.history_q[write] = sample_q;
        self.history_i[write + n] = sample_i;
        self.history_q[write + n] = sample_q;
        self.pos += 1;
        if self.pos >= n {
            self.pos = 0;
        }
        write + 1
    }

    #[cfg_attr(
        all(target_arch = "wasm32", target_feature = "simd128"),
        allow(dead_code)
    )]
    fn process_scalar_pair_path(&mut self, sample_i: f32, sample_q: f32) -> (f32, f32) {
        let n = self.coeffs_rev.len();
        let start = self.push_pair_and_window_start(sample_i, sample_q);
        let mut out_i = 0.0f32;
        let mut out_q = 0.0f32;
        for k in 0..n {
            let h = self.coeffs_rev[k];
            out_i += h * self.history_i[start + k];
            out_q += h * self.history_q[start + k];
        }
        (out_i, out_q)
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn process_simd_pair_path(&mut self, sample_i: f32, sample_q: f32) -> (f32, f32) {
        let n = self.coeffs_rev.len();
        let start = self.push_pair_and_window_start(sample_i, sample_q);
        let input_i_ptr = self.history_i[start..start + n].as_ptr();
        let input_q_ptr = self.history_q[start..start + n].as_ptr();
        let coeff_ptr = self.coeffs_rev.as_ptr();
        let mut sum4_i = f32x4(0.0, 0.0, 0.0, 0.0);
        let mut sum4_q = f32x4(0.0, 0.0, 0.0, 0.0);
        let mut k = 0usize;

        while k + 8 <= n {
            // SAFETY: `k + 8 <= n` を満たす間のみ進むため、
            // `input_{i,q}_ptr[k..k+8]` と `coeff_ptr[k..k+8]` は
            // 連続した有効な `f32` 領域内。
            let xi0 = unsafe { v128_load(input_i_ptr.add(k) as *const v128) };
            let xi1 = unsafe { v128_load(input_i_ptr.add(k + 4) as *const v128) };
            let xq0 = unsafe { v128_load(input_q_ptr.add(k) as *const v128) };
            let xq1 = unsafe { v128_load(input_q_ptr.add(k + 4) as *const v128) };
            let h0 = unsafe { v128_load(coeff_ptr.add(k) as *const v128) };
            let h1 = unsafe { v128_load(coeff_ptr.add(k + 4) as *const v128) };
            sum4_i = f32x4_add(sum4_i, f32x4_mul(h0, xi0));
            sum4_i = f32x4_add(sum4_i, f32x4_mul(h1, xi1));
            sum4_q = f32x4_add(sum4_q, f32x4_mul(h0, xq0));
            sum4_q = f32x4_add(sum4_q, f32x4_mul(h1, xq1));
            k += 8;
        }

        while k + 4 <= n {
            // SAFETY: `k + 4 <= n` の条件下でのみ読み込むため、
            // 各ポインタは有効範囲内。
            let xi = unsafe { v128_load(input_i_ptr.add(k) as *const v128) };
            let xq = unsafe { v128_load(input_q_ptr.add(k) as *const v128) };
            let h = unsafe { v128_load(coeff_ptr.add(k) as *const v128) };
            sum4_i = f32x4_add(sum4_i, f32x4_mul(h, xi));
            sum4_q = f32x4_add(sum4_q, f32x4_mul(h, xq));
            k += 4;
        }

        let mut out_i = RrcFilter::hsum4(sum4_i);
        let mut out_q = RrcFilter::hsum4(sum4_q);
        for kk in k..n {
            let h = self.coeffs_rev[kk];
            out_i += h * self.history_i[start + kk];
            out_q += h * self.history_q[start + kk];
        }
        (out_i, out_q)
    }

    pub fn process_pair(&mut self, sample_i: f32, sample_q: f32) -> (f32, f32) {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            return self.process_simd_pair_path(sample_i, sample_q);
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            self.process_scalar_pair_path(sample_i, sample_q)
        }
    }

    pub fn process_block_into(
        &mut self,
        input_i: &[f32],
        input_q: &[f32],
        output_i: &mut Vec<f32>,
        output_q: &mut Vec<f32>,
    ) {
        assert_eq!(input_i.len(), input_q.len(), "I/Q input length must match");
        output_i.clear();
        output_q.clear();
        if input_i.is_empty() {
            return;
        }
        output_i.reserve(input_i.len());
        output_q.reserve(input_q.len());
        for (&si, &sq) in input_i.iter().zip(input_q.iter()) {
            let (yi, yq) = self.process_pair(si, sq);
            output_i.push(yi);
            output_q.push(yq);
        }
    }

    pub fn process_block_in_place(&mut self, input_i: &mut [f32], input_q: &mut [f32]) {
        assert_eq!(input_i.len(), input_q.len(), "I/Q input length must match");
        for idx in 0..input_i.len() {
            let (yi, yq) = self.process_pair(input_i[idx], input_q[idx]);
            input_i[idx] = yi;
            input_q[idx] = yq;
        }
    }

    pub fn reset(&mut self) {
        self.history_i.fill(0.0);
        self.history_q.fill(0.0);
        self.pos = 0;
    }

    pub fn delay(&self) -> usize {
        (self.coeffs_rev.len() - 1) / 2
    }

    pub fn num_taps(&self) -> usize {
        self.coeffs_rev.len()
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
    fn test_process_block_into_matches_process_block() {
        let config = test_config();
        let mut filter1 = RrcFilter::from_config(&config.clone());
        let mut filter2 = RrcFilter::from_config(&config);

        let input: Vec<f32> = (0..1500)
            .map(|i| {
                let t = i as f32 / config.sample_rate;
                0.55 * (2.0 * std::f32::consts::PI * 1200.0 * t).sin()
                    + 0.33 * (2.0 * std::f32::consts::PI * 2600.0 * t).cos()
            })
            .collect();

        let expected = filter1.process_block(&input);

        let mut output = vec![123.0f32; 8];
        filter2.process_block_into(&input, &mut output);

        assert_eq!(output.len(), expected.len());
        for (i, (&actual, &exp)) in output.iter().zip(expected.iter()).enumerate() {
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

    #[test]
    fn test_iq_rrc_filter_matches_two_mono_filters_whole_input() {
        let config = test_config();
        let mut mono_i = RrcFilter::from_config(&config);
        let mut mono_q = RrcFilter::from_config(&config);
        let mut iq = IqRrcFilter::from_config(&config);

        let input_i: Vec<f32> = (0..4096)
            .map(|i| {
                let t = i as f32 / config.proc_sample_rate();
                0.73 * (2.0 * std::f32::consts::PI * 700.0 * t).sin()
                    + 0.19 * (2.0 * std::f32::consts::PI * 1900.0 * t).cos()
                    + 0.08 * (2.0 * std::f32::consts::PI * 3200.0 * t).sin()
            })
            .collect();
        let input_q: Vec<f32> = (0..4096)
            .map(|i| {
                let t = i as f32 / config.proc_sample_rate();
                0.61 * (2.0 * std::f32::consts::PI * 900.0 * t).cos()
                    + 0.27 * (2.0 * std::f32::consts::PI * 2100.0 * t).sin()
                    + 0.12 * (2.0 * std::f32::consts::PI * 3700.0 * t).cos()
            })
            .collect();

        let expected_i = mono_i.process_block(&input_i);
        let expected_q = mono_q.process_block(&input_q);
        let mut out_i = Vec::new();
        let mut out_q = Vec::new();
        iq.process_block_into(&input_i, &input_q, &mut out_i, &mut out_q);

        assert_eq!(out_i.len(), expected_i.len());
        assert_eq!(out_q.len(), expected_q.len());
        for (idx, (&a, &e)) in out_i.iter().zip(expected_i.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "I idx={} actual={} expected={}",
                idx,
                a,
                e
            );
        }
        for (idx, (&a, &e)) in out_q.iter().zip(expected_q.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "Q idx={} actual={} expected={}",
                idx,
                a,
                e
            );
        }
    }

    #[test]
    fn test_iq_rrc_filter_matches_two_mono_filters_chunked() {
        let config = test_config();
        let mut mono_i = RrcFilter::from_config(&config);
        let mut mono_q = RrcFilter::from_config(&config);
        let mut iq = IqRrcFilter::from_config(&config);

        let input_i: Vec<f32> = (0..6000)
            .map(|i| {
                let t = i as f32 / config.proc_sample_rate();
                0.66 * (2.0 * std::f32::consts::PI * 830.0 * t).sin()
                    + 0.24 * (2.0 * std::f32::consts::PI * 2600.0 * t).cos()
            })
            .collect();
        let input_q: Vec<f32> = (0..6000)
            .map(|i| {
                let t = i as f32 / config.proc_sample_rate();
                0.58 * (2.0 * std::f32::consts::PI * 910.0 * t).cos()
                    + 0.31 * (2.0 * std::f32::consts::PI * 2400.0 * t).sin()
            })
            .collect();

        let mut expected_i = Vec::new();
        let mut expected_q = Vec::new();
        let mut out_i = Vec::new();
        let mut out_q = Vec::new();
        let mut pos = 0usize;
        for &chunk_len in &[1usize, 3, 31, 127, 5, 211, 64, 1024, 77, 4096] {
            if pos >= input_i.len() {
                break;
            }
            let end = (pos + chunk_len).min(input_i.len());
            expected_i.extend(mono_i.process_block(&input_i[pos..end]));
            expected_q.extend(mono_q.process_block(&input_q[pos..end]));

            let mut tmp_i = Vec::new();
            let mut tmp_q = Vec::new();
            iq.process_block_into(
                &input_i[pos..end],
                &input_q[pos..end],
                &mut tmp_i,
                &mut tmp_q,
            );
            out_i.extend(tmp_i);
            out_q.extend(tmp_q);
            pos = end;
        }
        if pos < input_i.len() {
            expected_i.extend(mono_i.process_block(&input_i[pos..]));
            expected_q.extend(mono_q.process_block(&input_q[pos..]));
            let mut tmp_i = Vec::new();
            let mut tmp_q = Vec::new();
            iq.process_block_into(&input_i[pos..], &input_q[pos..], &mut tmp_i, &mut tmp_q);
            out_i.extend(tmp_i);
            out_q.extend(tmp_q);
        }

        assert_eq!(out_i.len(), expected_i.len());
        assert_eq!(out_q.len(), expected_q.len());
        for (idx, (&a, &e)) in out_i.iter().zip(expected_i.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "I idx={} actual={} expected={}",
                idx,
                a,
                e
            );
        }
        for (idx, (&a, &e)) in out_q.iter().zip(expected_q.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "Q idx={} actual={} expected={}",
                idx,
                a,
                e
            );
        }
    }

    #[test]
    #[should_panic(expected = "I/Q input length must match")]
    fn test_iq_rrc_filter_panics_on_mismatched_input_lengths() {
        let config = test_config();
        let mut iq = IqRrcFilter::from_config(&config);
        let mut out_i = Vec::new();
        let mut out_q = Vec::new();
        iq.process_block_into(&[0.0f32; 8], &[0.0f32; 7], &mut out_i, &mut out_q);
    }

    #[test]
    #[should_panic(expected = "I/Q input length must match")]
    fn test_iq_rrc_filter_in_place_panics_on_mismatched_input_lengths() {
        let config = test_config();
        let mut iq = IqRrcFilter::from_config(&config);
        let mut in_i = [0.0f32; 8];
        let mut in_q = [0.0f32; 7];
        iq.process_block_in_place(&mut in_i, &mut in_q);
    }

    #[test]
    fn test_iq_rrc_filter_reset_restarts_state_equivalent_to_fresh_instance() {
        let config = test_config();
        let mut under_test = IqRrcFilter::from_config(&config);
        let mut fresh = IqRrcFilter::from_config(&config);

        let input_i: Vec<f32> = (0..4096)
            .map(|i| {
                let t = i as f32 / config.proc_sample_rate();
                0.72 * (2.0 * std::f32::consts::PI * 820.0 * t).sin()
                    + 0.22 * (2.0 * std::f32::consts::PI * 1730.0 * t).cos()
            })
            .collect();
        let input_q: Vec<f32> = (0..4096)
            .map(|i| {
                let t = i as f32 / config.proc_sample_rate();
                0.64 * (2.0 * std::f32::consts::PI * 910.0 * t).cos()
                    + 0.28 * (2.0 * std::f32::consts::PI * 2010.0 * t).sin()
            })
            .collect();

        let mut throwaway_i = Vec::new();
        let mut throwaway_q = Vec::new();
        under_test.process_block_into(
            &input_i[..333],
            &input_q[..333],
            &mut throwaway_i,
            &mut throwaway_q,
        );
        under_test.reset();

        let mut out_i_reset = Vec::new();
        let mut out_q_reset = Vec::new();
        under_test.process_block_into(&input_i, &input_q, &mut out_i_reset, &mut out_q_reset);

        let mut out_i_fresh = Vec::new();
        let mut out_q_fresh = Vec::new();
        fresh.process_block_into(&input_i, &input_q, &mut out_i_fresh, &mut out_q_fresh);

        assert_eq!(out_i_reset.len(), out_i_fresh.len());
        assert_eq!(out_q_reset.len(), out_q_fresh.len());
        for (idx, (&a, &e)) in out_i_reset.iter().zip(out_i_fresh.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "I idx={} reset={} fresh={}",
                idx,
                a,
                e
            );
        }
        for (idx, (&a, &e)) in out_q_reset.iter().zip(out_q_fresh.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "Q idx={} reset={} fresh={}",
                idx,
                a,
                e
            );
        }
    }

    #[test]
    fn test_iq_rrc_filter_process_block_in_place_matches_block_into() {
        let config = test_config();
        let mut filter1 = IqRrcFilter::from_config(&config);
        let mut filter2 = IqRrcFilter::from_config(&config);

        let mut input_i: Vec<f32> = (0..2000)
            .map(|i| {
                let t = i as f32 / config.proc_sample_rate();
                0.7 * (2.0 * std::f32::consts::PI * 800.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 2300.0 * t).cos()
            })
            .collect();
        let mut input_q: Vec<f32> = (0..2000)
            .map(|i| {
                let t = i as f32 / config.proc_sample_rate();
                0.55 * (2.0 * std::f32::consts::PI * 1100.0 * t).cos()
                    + 0.3 * (2.0 * std::f32::consts::PI * 1700.0 * t).sin()
            })
            .collect();

        let mut expected_i = Vec::new();
        let mut expected_q = Vec::new();
        filter1.process_block_into(&input_i, &input_q, &mut expected_i, &mut expected_q);

        filter2.process_block_in_place(&mut input_i, &mut input_q);
        assert_eq!(input_i.len(), expected_i.len());
        assert_eq!(input_q.len(), expected_q.len());
        for (idx, (&a, &e)) in input_i.iter().zip(expected_i.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "I idx={} actual={} expected={}",
                idx,
                a,
                e
            );
        }
        for (idx, (&a, &e)) in input_q.iter().zip(expected_q.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "Q idx={} actual={} expected={}",
                idx,
                a,
                e
            );
        }
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
    fn test_rrc_filter_simd_tap_remainder_paths_after_unroll8() {
        // taps=21 (mod8=5), taps=23 (mod8=7) で
        // 8要素ループ -> 4要素ループ -> 端数ループの全経路を通す。
        for &taps in &[21usize, 23usize] {
            let mut scalar = RrcFilter::with_params(taps, 0.35, 8_000.0, 24_000.0);
            let mut simd = RrcFilter::with_params(taps, 0.35, 8_000.0, 24_000.0);
            for i in 0..(taps * 10 + 3) {
                let t = i as f32 / 24_000.0;
                let x = 0.57 * (2.0 * std::f32::consts::PI * 1100.0 * t).sin()
                    + 0.29 * (2.0 * std::f32::consts::PI * 2900.0 * t).cos()
                    + 0.14 * (2.0 * std::f32::consts::PI * 3700.0 * t).sin();
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

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[wasm_bindgen_test]
    fn test_iq_rrc_filter_simd_matches_scalar_path() {
        let config = test_config();
        let mut scalar = IqRrcFilter::from_config(&config);
        let mut simd = IqRrcFilter::from_config(&config);

        for i in 0..2048 {
            let t = i as f32 / config.proc_sample_rate();
            let x_i = 0.73 * (2.0 * std::f32::consts::PI * 700.0 * t).sin()
                + 0.19 * (2.0 * std::f32::consts::PI * 1900.0 * t).cos()
                + 0.08 * (2.0 * std::f32::consts::PI * 3200.0 * t).sin();
            let x_q = 0.61 * (2.0 * std::f32::consts::PI * 900.0 * t).cos()
                + 0.27 * (2.0 * std::f32::consts::PI * 2100.0 * t).sin()
                + 0.12 * (2.0 * std::f32::consts::PI * 3700.0 * t).cos();
            let (y_i_scalar, y_q_scalar) = scalar.process_scalar_pair_path(x_i, x_q);
            let (y_i_simd, y_q_simd) = simd.process_simd_pair_path(x_i, x_q);
            assert!(
                (y_i_scalar - y_i_simd).abs() < 1e-5,
                "I i={} scalar={} simd={}",
                i,
                y_i_scalar,
                y_i_simd
            );
            assert!(
                (y_q_scalar - y_q_simd).abs() < 1e-5,
                "Q i={} scalar={} simd={}",
                i,
                y_q_scalar,
                y_q_simd
            );
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[wasm_bindgen_test]
    fn test_iq_rrc_filter_simd_tap_remainder_boundaries() {
        for &taps in &[17usize, 19usize] {
            let mut scalar = IqRrcFilter::with_params(taps, 0.3, 8_000.0, 24_000.0);
            let mut simd = IqRrcFilter::with_params(taps, 0.3, 8_000.0, 24_000.0);
            for i in 0..(taps * 8 + 5) {
                let t = i as f32 / 24_000.0;
                let x_i = 0.61 * (2.0 * std::f32::consts::PI * 900.0 * t).sin()
                    + 0.27 * (2.0 * std::f32::consts::PI * 2100.0 * t).cos();
                let x_q = 0.55 * (2.0 * std::f32::consts::PI * 700.0 * t).cos()
                    + 0.31 * (2.0 * std::f32::consts::PI * 1700.0 * t).sin();
                let (y_i_scalar, y_q_scalar) = scalar.process_scalar_pair_path(x_i, x_q);
                let (y_i_simd, y_q_simd) = simd.process_simd_pair_path(x_i, x_q);
                assert!(
                    (y_i_scalar - y_i_simd).abs() < 1e-5,
                    "I taps={} i={} scalar={} simd={}",
                    taps,
                    i,
                    y_i_scalar,
                    y_i_simd
                );
                assert!(
                    (y_q_scalar - y_q_simd).abs() < 1e-5,
                    "Q taps={} i={} scalar={} simd={}",
                    taps,
                    i,
                    y_q_scalar,
                    y_q_simd
                );
            }
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[wasm_bindgen_test]
    fn test_iq_rrc_filter_simd_tap_remainder_paths_after_unroll8() {
        for &taps in &[21usize, 23usize] {
            let mut scalar = IqRrcFilter::with_params(taps, 0.35, 8_000.0, 24_000.0);
            let mut simd = IqRrcFilter::with_params(taps, 0.35, 8_000.0, 24_000.0);
            for i in 0..(taps * 10 + 3) {
                let t = i as f32 / 24_000.0;
                let x_i = 0.57 * (2.0 * std::f32::consts::PI * 1100.0 * t).sin()
                    + 0.29 * (2.0 * std::f32::consts::PI * 2900.0 * t).cos()
                    + 0.14 * (2.0 * std::f32::consts::PI * 3700.0 * t).sin();
                let x_q = 0.52 * (2.0 * std::f32::consts::PI * 1000.0 * t).cos()
                    + 0.33 * (2.0 * std::f32::consts::PI * 2500.0 * t).sin()
                    + 0.15 * (2.0 * std::f32::consts::PI * 3400.0 * t).cos();
                let (y_i_scalar, y_q_scalar) = scalar.process_scalar_pair_path(x_i, x_q);
                let (y_i_simd, y_q_simd) = simd.process_simd_pair_path(x_i, x_q);
                assert!(
                    (y_i_scalar - y_i_simd).abs() < 1e-5,
                    "I taps={} i={} scalar={} simd={}",
                    taps,
                    i,
                    y_i_scalar,
                    y_i_simd
                );
                assert!(
                    (y_q_scalar - y_q_simd).abs() < 1e-5,
                    "Q taps={} i={} scalar={} simd={}",
                    taps,
                    i,
                    y_q_scalar,
                    y_q_simd
                );
            }
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[wasm_bindgen_test]
    fn test_iq_rrc_filter_simd_ring_wrap_boundary() {
        let taps = 17usize;
        let mut scalar = IqRrcFilter::with_params(taps, 0.25, 8_000.0, 24_000.0);
        let mut simd = IqRrcFilter::with_params(taps, 0.25, 8_000.0, 24_000.0);

        for i in 0..(taps * 20) {
            let x_i = ((i % 11) as f32 - 5.0) * 0.07;
            let x_q = (((i + 3) % 13) as f32 - 6.0) * 0.05;
            let (y_i_scalar, y_q_scalar) = scalar.process_scalar_pair_path(x_i, x_q);
            let (y_i_simd, y_q_simd) = simd.process_simd_pair_path(x_i, x_q);
            assert!(
                (y_i_scalar - y_i_simd).abs() < 1e-5,
                "I i={} scalar={} simd={}",
                i,
                y_i_scalar,
                y_i_simd
            );
            assert!(
                (y_q_scalar - y_q_simd).abs() < 1e-5,
                "Q i={} scalar={} simd={}",
                i,
                y_q_scalar,
                y_q_simd
            );
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[wasm_bindgen_test]
    fn test_iq_rrc_filter_simd_block_into_chunk_and_empty_boundaries() {
        let config = test_config();
        let mut scalar = IqRrcFilter::from_config(&config);
        let mut simd = IqRrcFilter::from_config(&config);

        let input_i: Vec<f32> = (0..4096)
            .map(|i| {
                let t = i as f32 / config.proc_sample_rate();
                0.68 * (2.0 * std::f32::consts::PI * 1200.0 * t).sin()
                    + 0.22 * (2.0 * std::f32::consts::PI * 2800.0 * t).cos()
            })
            .collect();
        let input_q: Vec<f32> = (0..4096)
            .map(|i| {
                let t = i as f32 / config.proc_sample_rate();
                0.57 * (2.0 * std::f32::consts::PI * 900.0 * t).cos()
                    + 0.31 * (2.0 * std::f32::consts::PI * 2300.0 * t).sin()
            })
            .collect();

        let mut out_i_scalar = Vec::new();
        let mut out_q_scalar = Vec::new();
        let mut out_i_simd = Vec::new();
        let mut out_q_simd = Vec::new();

        // 空入力
        scalar.process_block_into(&[], &[], &mut out_i_scalar, &mut out_q_scalar);
        simd.process_block_into(&[], &[], &mut out_i_simd, &mut out_q_simd);
        assert!(out_i_scalar.is_empty());
        assert!(out_q_scalar.is_empty());
        assert!(out_i_simd.is_empty());
        assert!(out_q_simd.is_empty());

        let mut pos = 0usize;
        for &chunk_len in &[1usize, 2, 17, 33, 5, 257, 19, 1024, 11, 2048] {
            if pos >= input_i.len() {
                break;
            }
            let end = (pos + chunk_len).min(input_i.len());
            let mut tmp_i_scalar = Vec::new();
            let mut tmp_q_scalar = Vec::new();
            let mut tmp_i_simd = Vec::new();
            let mut tmp_q_simd = Vec::new();
            scalar.process_block_into(
                &input_i[pos..end],
                &input_q[pos..end],
                &mut tmp_i_scalar,
                &mut tmp_q_scalar,
            );
            simd.process_block_into(
                &input_i[pos..end],
                &input_q[pos..end],
                &mut tmp_i_simd,
                &mut tmp_q_simd,
            );
            out_i_scalar.extend(tmp_i_scalar);
            out_q_scalar.extend(tmp_q_scalar);
            out_i_simd.extend(tmp_i_simd);
            out_q_simd.extend(tmp_q_simd);
            pos = end;
        }
        if pos < input_i.len() {
            let mut tmp_i_scalar = Vec::new();
            let mut tmp_q_scalar = Vec::new();
            let mut tmp_i_simd = Vec::new();
            let mut tmp_q_simd = Vec::new();
            scalar.process_block_into(
                &input_i[pos..],
                &input_q[pos..],
                &mut tmp_i_scalar,
                &mut tmp_q_scalar,
            );
            simd.process_block_into(
                &input_i[pos..],
                &input_q[pos..],
                &mut tmp_i_simd,
                &mut tmp_q_simd,
            );
            out_i_scalar.extend(tmp_i_scalar);
            out_q_scalar.extend(tmp_q_scalar);
            out_i_simd.extend(tmp_i_simd);
            out_q_simd.extend(tmp_q_simd);
        }

        assert_eq!(out_i_scalar.len(), out_i_simd.len());
        assert_eq!(out_q_scalar.len(), out_q_simd.len());
        for (idx, (&a, &b)) in out_i_scalar.iter().zip(out_i_simd.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "I idx={} scalar={} simd={}",
                idx,
                a,
                b
            );
        }
        for (idx, (&a, &b)) in out_q_scalar.iter().zip(out_q_simd.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "Q idx={} scalar={} simd={}",
                idx,
                a,
                b
            );
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[wasm_bindgen_test]
    fn test_iq_rrc_filter_simd_in_place_chunk_and_empty_boundaries() {
        let config = test_config();
        let mut scalar = IqRrcFilter::from_config(&config);
        let mut simd = IqRrcFilter::from_config(&config);

        let input_i: Vec<f32> = (0..3000)
            .map(|i| {
                let t = i as f32 / config.proc_sample_rate();
                0.63 * (2.0 * std::f32::consts::PI * 1300.0 * t).sin()
                    + 0.27 * (2.0 * std::f32::consts::PI * 2600.0 * t).cos()
            })
            .collect();
        let input_q: Vec<f32> = (0..3000)
            .map(|i| {
                let t = i as f32 / config.proc_sample_rate();
                0.59 * (2.0 * std::f32::consts::PI * 1000.0 * t).cos()
                    + 0.29 * (2.0 * std::f32::consts::PI * 2200.0 * t).sin()
            })
            .collect();

        // 空入力
        let mut empty_i: [f32; 0] = [];
        let mut empty_q: [f32; 0] = [];
        scalar.process_block_in_place(&mut empty_i, &mut empty_q);
        simd.process_block_in_place(&mut empty_i, &mut empty_q);

        let mut out_i_scalar = Vec::new();
        let mut out_q_scalar = Vec::new();
        let mut out_i_simd = Vec::new();
        let mut out_q_simd = Vec::new();
        let mut pos = 0usize;
        for &chunk_len in &[1usize, 7, 29, 3, 101, 5, 511, 13, 2048] {
            if pos >= input_i.len() {
                break;
            }
            let end = (pos + chunk_len).min(input_i.len());

            let mut chunk_i_scalar = input_i[pos..end].to_vec();
            let mut chunk_q_scalar = input_q[pos..end].to_vec();
            scalar.process_block_in_place(&mut chunk_i_scalar, &mut chunk_q_scalar);
            out_i_scalar.extend(chunk_i_scalar);
            out_q_scalar.extend(chunk_q_scalar);

            let mut chunk_i_simd = input_i[pos..end].to_vec();
            let mut chunk_q_simd = input_q[pos..end].to_vec();
            simd.process_block_in_place(&mut chunk_i_simd, &mut chunk_q_simd);
            out_i_simd.extend(chunk_i_simd);
            out_q_simd.extend(chunk_q_simd);
            pos = end;
        }
        if pos < input_i.len() {
            let mut chunk_i_scalar = input_i[pos..].to_vec();
            let mut chunk_q_scalar = input_q[pos..].to_vec();
            scalar.process_block_in_place(&mut chunk_i_scalar, &mut chunk_q_scalar);
            out_i_scalar.extend(chunk_i_scalar);
            out_q_scalar.extend(chunk_q_scalar);

            let mut chunk_i_simd = input_i[pos..].to_vec();
            let mut chunk_q_simd = input_q[pos..].to_vec();
            simd.process_block_in_place(&mut chunk_i_simd, &mut chunk_q_simd);
            out_i_simd.extend(chunk_i_simd);
            out_q_simd.extend(chunk_q_simd);
        }

        assert_eq!(out_i_scalar.len(), out_i_simd.len());
        assert_eq!(out_q_scalar.len(), out_q_simd.len());
        for (idx, (&a, &b)) in out_i_scalar.iter().zip(out_i_simd.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "I idx={} scalar={} simd={}",
                idx,
                a,
                b
            );
        }
        for (idx, (&a, &b)) in out_q_scalar.iter().zip(out_q_simd.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "Q idx={} scalar={} simd={}",
                idx,
                a,
                b
            );
        }
    }
}
