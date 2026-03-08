#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use std::arch::wasm32::{f32x4, f32x4_add, f32x4_extract_lane, f32x4_mul};

#[derive(Clone, Copy)]
#[cfg_attr(
    all(target_arch = "wasm32", target_feature = "simd128"),
    allow(dead_code)
)]
enum ResampleBackend {
    Scalar,
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Simd,
}

pub struct Resampler {
    pub source_rate: u32,
    pub target_rate: u32,
    step: f64,
    phase: f64,
    num_phases: usize,
    taps_per_phase: usize,
    coeffs: Vec<f32>,
    history: Vec<f32>,
    history_start: usize,
}

impl Resampler {
    const MIN_TAPS_PER_PHASE: usize = 17;
    const MAX_TAPS_PER_PHASE: usize = 257;

    fn normalize_taps_per_phase(taps: usize) -> usize {
        taps.clamp(Self::MIN_TAPS_PER_PHASE, Self::MAX_TAPS_PER_PHASE) | 1
    }

    fn design_taps_per_phase(source_rate: u32, target_rate: u32, cutoff_hz: f64) -> usize {
        let source_nyquist = 0.5 * source_rate as f64;
        let available_transition_hz = (source_nyquist - cutoff_hz).max(50.0);

        // passband の 35% を基準に遷移帯を置き、実現可能範囲にクランプ。
        let desired_transition_hz = (cutoff_hz * 0.35).max(200.0);
        let transition_hz = desired_transition_hz
            .min(available_transition_hz * 0.95)
            .max(50.0);
        let delta_f = (transition_hz / source_rate as f64).max(1e-6);

        // Blackman窓の経験式: N ≈ 5.5 / Δf（cycles/sample）
        let base_taps = (5.5 / delta_f).ceil() as usize;

        // 強いデシメーション時は stopband 抑圧を優先して係数を増やす。
        let decimation_factor = (source_rate as f64 / target_rate as f64).max(1.0);
        let scale = decimation_factor.sqrt().ceil() as usize;
        base_taps.saturating_mul(scale.max(1))
    }

    #[inline]
    fn default_backend() -> ResampleBackend {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            return ResampleBackend::Simd;
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            ResampleBackend::Scalar
        }
    }

    #[inline]
    fn phase_coeffs(&self, phase: usize) -> &[f32] {
        let start = phase * self.taps_per_phase;
        &self.coeffs[start..start + self.taps_per_phase]
    }

    #[inline]
    fn convolve_scalar(&self, base: usize, coeffs: &[f32]) -> f32 {
        let start = self.history_start + base;
        let input = &self.history[start..start + coeffs.len()];
        let mut sum = 0.0f32;
        for (&x, &h) in input.iter().zip(coeffs.iter()) {
            sum += x * h;
        }
        sum
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
    #[inline]
    fn convolve_simd(&self, base: usize, coeffs: &[f32]) -> f32 {
        let start = self.history_start + base;
        let input = &self.history[start..start + coeffs.len()];
        let mut sum4 = f32x4(0.0, 0.0, 0.0, 0.0);
        let mut tap = 0usize;
        while tap + 4 <= coeffs.len() {
            let x = f32x4(input[tap], input[tap + 1], input[tap + 2], input[tap + 3]);
            let h = f32x4(
                coeffs[tap],
                coeffs[tap + 1],
                coeffs[tap + 2],
                coeffs[tap + 3],
            );
            sum4 = f32x4_add(sum4, f32x4_mul(x, h));
            tap += 4;
        }

        let mut sum = Self::hsum4(sum4);
        for (i, &h) in coeffs[tap..].iter().enumerate() {
            sum += input[tap + i] * h;
        }
        sum
    }

    #[inline]
    fn maybe_compact_history(&mut self) {
        if self.history_start == 0 {
            return;
        }

        let remaining = self.history.len().saturating_sub(self.history_start);
        let should_compact = self.history_start >= self.taps_per_phase * 2
            && (self.history_start * 2 >= self.history.len()
                || self.history_start >= 16_384
                || remaining < self.taps_per_phase + 8);

        if should_compact {
            self.history.drain(0..self.history_start);
            self.history_start = 0;
        }
    }

    /// `cutoff_hz` を指定した場合、リサンプラ内LPFのカットオフを明示できる。
    /// 指定しない場合は従来どおり target_rate ベースの自動値を使う。
    /// `taps_per_phase` を指定した場合は固定タップ数として使用する。
    pub fn new_with_cutoff(
        source_rate: u32,
        target_rate: u32,
        cutoff_hz: Option<f32>,
        taps_per_phase: Option<usize>,
    ) -> Self {
        assert!(source_rate > 0, "source_rate must be > 0");
        assert!(target_rate > 0, "target_rate must be > 0");

        // 入力時間軸上での出力サンプル刻み
        let step = source_rate as f64 / target_rate as f64;

        let num_phases = 256;

        // source 側正規化周波数（Nyquist=0.5）
        // - 指定なし: target_rate に合わせた従来値
        // - 指定あり: min(source/2, target/2, cutoff_hz) にクランプして利用
        let default_cutoff_hz = 0.45f64 * source_rate.min(target_rate) as f64;
        let max_cutoff_hz = 0.49f64 * (source_rate.min(target_rate) as f64);
        let cutoff_hz = cutoff_hz
            .map(|v| v as f64)
            .unwrap_or(default_cutoff_hz)
            .min(max_cutoff_hz)
            .max(1.0);
        let default_taps = Self::design_taps_per_phase(source_rate, target_rate, cutoff_hz);
        let taps_per_phase = Self::normalize_taps_per_phase(taps_per_phase.unwrap_or(default_taps));
        let mut coeffs = vec![0.0f32; num_phases * taps_per_phase];
        let cutoff = cutoff_hz / source_rate as f64;
        let center = (taps_per_phase - 1) as f64 / 2.0;

        for p in 0..num_phases {
            let frac = p as f64 / num_phases as f64;
            let mut sum = 0.0f64;
            let phase_start = p * taps_per_phase;
            let phase_coeffs = &mut coeffs[phase_start..phase_start + taps_per_phase];

            for (i, coeff) in phase_coeffs.iter_mut().enumerate() {
                let x = i as f64 - center - frac;
                let sinc_arg = 2.0 * cutoff * x;
                let sinc = if sinc_arg.abs() < 1e-12 {
                    1.0
                } else {
                    let pix = std::f64::consts::PI * sinc_arg;
                    pix.sin() / pix
                };

                // Blackman window
                let w = 0.42
                    - 0.5
                        * (2.0 * std::f64::consts::PI * i as f64 / (taps_per_phase - 1) as f64)
                            .cos()
                    + 0.08
                        * (4.0 * std::f64::consts::PI * i as f64 / (taps_per_phase - 1) as f64)
                            .cos();

                let h = 2.0 * cutoff * sinc * w;
                *coeff = h as f32;
                sum += h;
            }

            // DCゲインを1に正規化
            let inv_sum = 1.0f64 / sum;
            for coeff in phase_coeffs {
                *coeff = (*coeff as f64 * inv_sum) as f32;
            }
        }

        Self {
            source_rate,
            target_rate,
            step,
            phase: 0.0,
            num_phases,
            taps_per_phase,
            coeffs,
            history: vec![0.0; taps_per_phase - 1], // 立ち上がり分の遅延をゼロで埋める
            history_start: 0,
        }
    }

    fn process_with_backend(
        &mut self,
        input: &[f32],
        output: &mut Vec<f32>,
        backend: ResampleBackend,
    ) {
        if input.is_empty() {
            return;
        }

        // 入力をすべて履歴に追加
        self.history.extend_from_slice(input);

        // 処理に必要な最小の長さはフィルタの全タップ数
        let available = self.history.len().saturating_sub(self.history_start);
        if available < self.taps_per_phase {
            return;
        }

        // 常に history[base .. base + taps_per_phase] の窓で演算する。
        let limit = (available as isize - self.taps_per_phase as isize + 1).max(0) as f64;

        // 追加される可能性がある出力数を概算して、pushの再確保を減らす。
        if self.phase < limit {
            let est = ((limit - self.phase) / self.step).ceil().max(0.0) as usize;
            output.reserve(est.saturating_add(1));
        }

        while self.phase < limit {
            let base = self.phase.floor() as usize;
            let frac = self.phase - base as f64;

            let mut phase_idx = (frac * self.num_phases as f64).floor() as usize;
            if phase_idx >= self.num_phases {
                phase_idx = self.num_phases - 1;
            }

            let coeffs = self.phase_coeffs(phase_idx);

            let sum = match backend {
                ResampleBackend::Scalar => self.convolve_scalar(base, coeffs),
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                ResampleBackend::Simd => self.convolve_simd(base, coeffs),
            };

            output.push(sum);
            self.phase += self.step;
        }

        // 整数サンプル分だけ history を消費
        let consumed = self.phase.floor() as usize;
        if consumed > 0 {
            self.history_start = (self.history_start + consumed).min(self.history.len());
            self.phase -= consumed as f64;
            self.maybe_compact_history();
        }

        if self.phase < 0.0 {
            self.phase = 0.0;
        }
    }

    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        self.process_with_backend(input, output, Self::default_backend());
    }

    pub fn reset(&mut self) {
        self.history = vec![0.0; self.taps_per_phase - 1];
        self.history_start = 0;
        self.phase = 0.0;
    }

    pub fn reconfigure(
        &mut self,
        source_rate: u32,
        target_rate: u32,
        cutoff_hz: Option<f32>,
        taps_per_phase: Option<usize>,
    ) {
        *self = Self::new_with_cutoff(source_rate, target_rate, cutoff_hz, taps_per_phase);
    }

    /// 残っている履歴をすべて出力し、内部状態をリセットする。
    /// バーストの最後で呼び出す。
    fn flush_with_backend(&mut self, output: &mut Vec<f32>, backend: ResampleBackend) {
        if self.history.len().saturating_sub(self.history_start) == 0 {
            return;
        }

        // history の中にある有効なサンプルをすべて窓に通すために、必要な分のゼロを追加
        let padding_len = self.taps_per_phase - 1;
        let padding = vec![0.0f32; padding_len];
        self.process_with_backend(&padding, output, backend);

        // 状態リセット
        self.history = vec![0.0; self.taps_per_phase - 1];
        self.history_start = 0;
        self.phase = 0.0;
    }

    pub fn flush(&mut self, output: &mut Vec<f32>) {
        self.flush_with_backend(output, Self::default_backend());
    }

    /// リサンプラの物理的遅延（ターゲットレート換算のサンプル数）を返す
    pub fn delay(&self) -> usize {
        let center = (self.taps_per_phase as f64 - 1.0) / 2.0;
        let ratio = self.target_rate as f64 / self.source_rate as f64;
        (center * ratio).ceil() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;
    use rustfft::FftPlanner;
    use std::f32::consts::PI;
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    use wasm_bindgen_test::wasm_bindgen_test;

    fn dominant_frequency(samples: &[f32], sample_rate: u32) -> f32 {
        let n = samples.len();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);

        let mut complex_buffer: Vec<Complex<f32>> =
            samples.iter().map(|&val| Complex::new(val, 0.0)).collect();
        fft.process(&mut complex_buffer);

        let mut max_magnitude = 0.0;
        let mut peak_index = 0;
        for (i, val) in complex_buffer.iter().enumerate().take(n / 2).skip(1) {
            let magnitude = val.norm();
            if magnitude > max_magnitude {
                max_magnitude = magnitude;
                peak_index = i;
            }
        }

        (peak_index as f32 * sample_rate as f32) / n as f32
    }

    // リサンプリング前後で特定の正弦波の周波数が維持されているかをテストする関数
    fn test_resampling_sine_wave(
        source_rate: u32,
        target_rate: u32,
        test_freq: f32,
        duration_sec: f32,
    ) {
        let num_samples_in = (source_rate as f32 * duration_sec).ceil() as usize;

        // 指定周波数の正弦波を生成 (入力)
        let mut input = Vec::with_capacity(num_samples_in);
        for i in 0..num_samples_in {
            let t = i as f32 / source_rate as f32;
            let val = (2.0 * PI * test_freq * t).sin();
            input.push(val);
        }

        // リサンプラに通す
        let mut resampler = Resampler::new_with_cutoff(source_rate, target_rate, None, None);
        let mut output = Vec::new();
        resampler.process(&input, &mut output);
        resampler.flush(&mut output);

        // 出力のFFTを行い、ピーク周波数を探す
        let output_len = output.len();
        assert!(output_len > 0, "Output should not be empty");

        let detected_freq = dominant_frequency(&output, target_rate);

        // 分解能の許容範囲 (bin size) を計算
        let freq_resolution = target_rate as f32 / output_len as f32;
        let tolerance = freq_resolution; // ±1bin 程度の誤差を許容

        assert!(
            (detected_freq - test_freq).abs() <= tolerance,
            "Frequency mismatch! Expected ~{} Hz, detected ~{} Hz (Source rate: {}, Target rate: {}, Resolution: {})",
            test_freq,
            detected_freq,
            source_rate,
            target_rate,
            freq_resolution
        );
    }

    #[test]
    fn test_downsampling_preserves_frequency() {
        // 50kHz から 48kHz へのダウンサンプリング
        test_resampling_sine_wave(50_000, 48_000, 1_000.0, 0.5);
    }

    #[test]
    fn test_upsampling_preserves_frequency() {
        // 44.1kHz から 48kHz へのアップサンプリング
        test_resampling_sine_wave(44_100, 48_000, 4_000.0, 0.5);
    }

    #[test]
    fn test_continuous_processing() {
        // 連続的にバッファを渡した場合に、履歴(history)と位相が正しく接続されるかを検証
        let source_rate = 10_000;
        let target_rate = 8_000;
        let mut resampler_chunks = Resampler::new_with_cutoff(source_rate, target_rate, None, None);
        let mut resampler_whole = Resampler::new_with_cutoff(source_rate, target_rate, None, None);

        let input: Vec<f32> = (0..4_000)
            .map(|i| {
                let t = i as f32 / source_rate as f32;
                (2.0 * PI * 410.0 * t).sin() + 0.3 * (2.0 * PI * 1200.0 * t).sin()
            })
            .collect();

        // チャンクに分割して処理
        let mut out_chunks = Vec::new();
        for chunk in input.chunks(137) {
            resampler_chunks.process(chunk, &mut out_chunks);
        }

        // 一括で処理
        let mut out_whole = Vec::new();
        resampler_whole.process(&input, &mut out_whole);

        assert!((out_chunks.len() as isize - out_whole.len() as isize).abs() <= 1);

        let min_len = out_chunks.len().min(out_whole.len());
        let rmse = (out_chunks[..min_len]
            .iter()
            .zip(out_whole[..min_len].iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f32>()
            / min_len as f32)
            .sqrt();

        assert!(
            rmse < 1e-4,
            "Chunked output diverged from whole output: rmse={}",
            rmse
        );
    }

    #[test]
    fn test_high_ratio_downsampling_stopband() {
        // 200kHz → 48kHz ダウンサンプリング時、ナイキスト (24kHz) 超の信号が
        // 十分に減衰されることを確認する。
        let source_rate = 200_000u32;
        let target_rate = 48_000u32;
        let mut resampler = Resampler::new_with_cutoff(source_rate, target_rate, None, None);

        // 30kHz (ナイキスト超) の正弦波を入力
        let len = 50_000;
        let input: Vec<f32> = (0..len)
            .map(|i| {
                let t = i as f32 / source_rate as f32;
                (2.0 * PI * 30_000.0 * t).sin()
            })
            .collect();

        let mut output = Vec::new();
        resampler.process(&input, &mut output);

        // FIR 過渡を除いてパワーを計算
        let skip = 100.min(output.len().saturating_sub(1));
        let power =
            output[skip..].iter().map(|v| v * v).sum::<f32>() / (output.len() - skip) as f32;

        // 入力パワーは 0.5 (sin^2平均)。少なくとも -30dB (0.001) 以下に減衰。
        assert!(
            power < 0.001,
            "Stopband signal not attenuated: power={}",
            power
        );
    }

    #[test]
    fn test_long_run_output_count_tracks_ratio_50k_to_48k() {
        let source_rate = 50_000u32;
        let target_rate = 48_000u32;
        let mut resampler = Resampler::new_with_cutoff(source_rate, target_rate, None, None);

        let total_input = 500_000usize; // 10秒相当
        let input: Vec<f32> = (0..total_input)
            .map(|i| {
                let t = i as f32 / source_rate as f32;
                (2.0 * PI * 1_200.0 * t).sin() + 0.2 * (2.0 * PI * 7_500.0 * t).sin()
            })
            .collect();

        let chunk_pattern = [127usize, 509, 1021, 4093];
        let mut pos = 0usize;
        let mut chunk_idx = 0usize;
        let mut output = Vec::new();
        while pos < input.len() {
            let chunk_len = chunk_pattern[chunk_idx % chunk_pattern.len()];
            let end = (pos + chunk_len).min(input.len());
            resampler.process(&input[pos..end], &mut output);
            pos = end;
            chunk_idx += 1;
        }

        let expected =
            ((total_input as f64) * (target_rate as f64) / (source_rate as f64)).round() as isize;
        let actual = output.len() as isize;
        let err = (actual - expected).abs();

        // 有限長処理のため端の遅延分は残るが、誤差はフィルタ長オーダーに収まるべき。
        let tolerance = resampler.taps_per_phase as isize + 4;
        assert!(
            err <= tolerance,
            "Long-run count drift too large: actual={} expected={} err={} tol={}",
            actual,
            expected,
            err,
            tolerance
        );
    }

    #[test]
    fn test_long_run_output_count_tracks_ratio_200k_to_48k() {
        let source_rate = 200_000u32;
        let target_rate = 48_000u32;
        let mut resampler = Resampler::new_with_cutoff(source_rate, target_rate, None, None);

        let total_input = 2_000_000usize; // 10秒相当
        let input: Vec<f32> = (0..total_input)
            .map(|i| {
                let t = i as f32 / source_rate as f32;
                (2.0 * PI * 900.0 * t).sin() + 0.2 * (2.0 * PI * 18_000.0 * t).sin()
            })
            .collect();

        let chunk_pattern = [113usize, 701, 4096, 8191];
        let mut pos = 0usize;
        let mut chunk_idx = 0usize;
        let mut output = Vec::new();
        while pos < input.len() {
            let chunk_len = chunk_pattern[chunk_idx % chunk_pattern.len()];
            let end = (pos + chunk_len).min(input.len());
            resampler.process(&input[pos..end], &mut output);
            pos = end;
            chunk_idx += 1;
        }

        let expected =
            ((total_input as f64) * (target_rate as f64) / (source_rate as f64)).round() as isize;
        let actual = output.len() as isize;
        let err = (actual - expected).abs();

        let tolerance = resampler.taps_per_phase as isize + 4;
        assert!(
            err <= tolerance,
            "Long-run count drift too large: actual={} expected={} err={} tol={}",
            actual,
            expected,
            err,
            tolerance
        );
    }

    #[test]
    fn test_custom_cutoff_reduces_high_audio_band() {
        let source_rate = 50_000u32;
        let target_rate = 48_000u32;
        let tone_hz = 10_000.0f32;
        let len = 120_000usize;

        let input: Vec<f32> = (0..len)
            .map(|i| {
                let t = i as f32 / source_rate as f32;
                (2.0 * PI * tone_hz * t).sin()
            })
            .collect();

        let mut default_rs = Resampler::new_with_cutoff(source_rate, target_rate, None, None);
        let mut low_cut_rs =
            Resampler::new_with_cutoff(source_rate, target_rate, Some(5_000.0), None);
        let mut out_default = Vec::new();
        let mut out_low_cut = Vec::new();
        default_rs.process(&input, &mut out_default);
        low_cut_rs.process(&input, &mut out_low_cut);

        let skip_default = out_default.len().min(2_000);
        let skip_low = out_low_cut.len().min(2_000);
        let rms_default = (out_default[skip_default..]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            / (out_default.len() - skip_default).max(1) as f32)
            .sqrt();
        let rms_low_cut = (out_low_cut[skip_low..].iter().map(|v| v * v).sum::<f32>()
            / (out_low_cut.len() - skip_low).max(1) as f32)
            .sqrt();

        assert!(
            rms_low_cut < rms_default * 0.3,
            "Custom cutoff did not attenuate high tone enough: default_rms={} low_cut_rms={}",
            rms_default,
            rms_low_cut
        );
    }

    #[test]
    fn test_negative_phase_drift_repro() {
        let mut rs = Resampler::new_with_cutoff(24000, 48000, None, None);
        println!("\nInitial phase: {}", rs.phase);

        let mut total_output = 0;
        // 1サンプルずつの入力を20回繰り返す
        for i in 0..20 {
            let mut out = Vec::new();
            rs.process(&[0.0], &mut out);
            total_output += out.len();
            println!("Iter {}: phase={}, output_len={}", i, rs.phase, out.len());
        }

        // 期待される挙動: 24k->48k (比率2.0) なので、20個入れたら約40個出るべき
        // バグがある場合、phase が負に沈み続け、出力が全く出ないか、あるいは負のインデックスでクラッシュする
        println!("Total output samples: {}", total_output);
        assert!(
            total_output > 0,
            "Resampler produced NO output due to negative phase drift!"
        );
    }

    #[test]
    fn test_resampler_latency_and_timing_accuracy() {
        // 24kHz -> 48kHz (ratio 2.0)
        let mut rs = Resampler::new_with_cutoff(24000, 48000, None, None);
        let expected_delay_out = rs.delay();

        // 0番目にインパルスを入力
        let mut input = vec![0.0f32; 64];
        input[0] = 1.0;

        let mut output = Vec::new();
        rs.process(&input, &mut output);
        rs.flush(&mut output);

        // 出力の中で最大の振幅（ピーク）を持つ位置を探す
        let (peak_idx, &peak_val) = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .unwrap();

        println!(
            "Resampler Delay Verification: Peak at idx={}, value={:.4}, delay()={}",
            peak_idx, peak_val, expected_delay_out
        );

        // 物理的整合性の検証: 0番目の入力がピークに達するまでの時間は delay() と一致しなければならない
        assert_eq!(
            peak_idx, expected_delay_out,
            "Resampler peak position mismatch with delay()"
        );
        assert!(peak_val > 0.5, "Impulse response peak is too weak");
    }

    #[test]
    fn test_explicit_taps_override_is_applied() {
        let auto = Resampler::new_with_cutoff(24_000, 48_000, None, None);
        let explicit = Resampler::new_with_cutoff(24_000, 48_000, None, Some(73));

        assert_eq!(explicit.taps_per_phase, 73);
        assert_ne!(auto.taps_per_phase, explicit.taps_per_phase);
    }

    #[test]
    fn test_explicit_taps_are_normalized_to_odd_and_clamped() {
        let min_norm = Resampler::new_with_cutoff(24_000, 48_000, None, Some(16));
        let odd_norm = Resampler::new_with_cutoff(24_000, 48_000, None, Some(40));
        let max_norm = Resampler::new_with_cutoff(24_000, 48_000, None, Some(400));

        assert_eq!(min_norm.taps_per_phase, 17);
        assert_eq!(odd_norm.taps_per_phase, 41);
        assert_eq!(max_norm.taps_per_phase, 257);
    }

    #[test]
    fn test_48k_to_24k_taps_and_aliasing() {
        // 48kHz -> 24kHz ダウンサンプリング時のタップ数とエイリアzing特性を確認
        let source_rate = 48_000u32;
        let target_rate = 24_000u32;

        // 現在の実装で自動計算されるタップ数を確認
        let mut resampler = Resampler::new_with_cutoff(source_rate, target_rate, None, None);

        println!("\n=== 48kHz -> 24kHz リサンプラ評価 ===");
        println!("  source_rate: {} Hz", source_rate);
        println!("  target_rate: {} Hz", target_rate);
        println!(
            "  decimation_factor: {:.2}",
            source_rate as f64 / target_rate as f64
        );
        println!("  taps_per_phase: {}", resampler.taps_per_phase);
        println!("  delay (output samples): {}", resampler.delay());

        // 設計パラメータの詳細を計算して表示
        let source_nyquist = 0.5 * source_rate as f64;
        let default_cutoff_hz = 0.45f64 * source_rate.min(target_rate) as f64;
        let cutoff_hz = default_cutoff_hz
            .min(0.49 * (source_rate.min(target_rate) as f64))
            .max(1.0);
        let available_transition_hz = (source_nyquist - cutoff_hz).max(50.0);
        let desired_transition_hz = (cutoff_hz * 0.35).max(200.0);
        let transition_hz = desired_transition_hz
            .min(available_transition_hz * 0.95)
            .max(50.0);
        let delta_f = (transition_hz / source_rate as f64).max(1e-6);
        let base_taps = (5.5 / delta_f).ceil() as usize;
        let decimation_factor = source_rate as f64 / target_rate as f64;
        let scale = decimation_factor.sqrt().ceil() as usize;

        println!("\n設計パラメータ:");
        println!("  cutoff_hz: {:.1} Hz", cutoff_hz);
        println!("  source_nyquist: {:.1} Hz", source_nyquist);
        println!("  target_nyquist: {:.1} Hz", target_rate as f64 / 2.0);
        println!("  transition_hz: {:.1} Hz", transition_hz);
        println!("  passband edge: {:.1} Hz", cutoff_hz);
        println!("  stopband start: {:.1} Hz", cutoff_hz + transition_hz);
        println!("  delta_f: {:.6}", delta_f);
        println!("  base_taps (5.5/Δf): {}", base_taps);
        println!("  scale (sqrt(decimation)): {}", scale);

        // エイリアシング解析
        println!("\nエイリアシング解析:");
        let stopband_start = cutoff_hz + transition_hz;
        println!("  stopband開始: {:.1} Hz", stopband_start);
        println!(
            "  エイリアzing源帯域: {:.1} Hz - {:.1} Hz",
            stopband_start, source_nyquist
        );

        let alias_fold_to_start =
            (target_rate as f64 / 2.0) - (source_nyquist - target_rate as f64 / 2.0);
        let alias_fold_to_end =
            (target_rate as f64 / 2.0) - (stopband_start - target_rate as f64 / 2.0);
        println!(
            "  折り返し先帯域: {:.1} Hz - {:.1} Hz",
            alias_fold_to_start.max(0.0),
            alias_fold_to_end
        );

        // 高域信号のエイリアzing抑制を確認
        // 15kHz (stopband内) の信号が十分に減衰されるかテスト
        let test_freq = 15_000.0f32;
        let len = source_rate as usize; // 1秒
        let input: Vec<f32> = (0..len)
            .map(|i| {
                let t = i as f32 / source_rate as f32;
                (2.0 * PI * test_freq * t).sin()
            })
            .collect();

        let mut output = Vec::new();
        resampler.process(&input, &mut output);

        // FIR 過渡を除いてパワーを計算
        let skip = resampler.taps_per_phase * 2;
        let skip = skip.min(output.len().saturating_sub(1));
        let power = if output.len() > skip {
            output[skip..].iter().map(|v| v * v).sum::<f32>() / (output.len() - skip) as f32
        } else {
            0.0
        };

        // 入力パワーは 0.5 (sin^2平均)
        let attenuation_db = 10.0 * (power / 0.5).log10();
        println!("\n{} Hz 信号のエイリアzing抑制:", test_freq);
        println!("  入力パワー: {:.3} (0.5 期待)", 0.5);
        println!("  出力パワー: {:.6}", power);
        println!("  減衰量: {:.1} dB", attenuation_db);

        // Blackman窓の理論上のサイドローブレベルは約-58dBから-74dB
        // 実際のフィルタでは-60dB以下が望ましい
        let attenuation_acceptable = power < 0.001; // -30dB = 0.001
        let attenuation_good = power < 0.0001; // -40dB = 0.0001

        if attenuation_good {
            println!("  ✓ エイリアzing抑制: 良好 (< -40dB)");
        } else if attenuation_acceptable {
            println!("  △ エイリアzing抑制: 許容範囲 (< -30dB)");
        } else {
            println!("  ✗ エイリアzing抑制: 不十分 (> -30dB)");
        }

        // タップ数の妥当性を検証
        println!("\nタップ数評価:");
        println!("  現在のタップ数: {}", resampler.taps_per_phase);

        // 計算上の最小タップ数 (エイリアzingを避けるための理論値)
        // Nyquistの基準では、過渡帯幅 Δf と必要な減衰量 A からタップ数が決まる
        // Blackman窓では約 -74dB のサイドローブ
        // Δf = 3780/48000 = 0.07875 の場合
        // N ≈ 5.5/0.07875 ≈ 70 がベース
        // これに decimation factor の sqrt を掛けて 140 -> 141

        // 遷移帯を狭くした場合の試算 (より厳密なエイリアzing抑制)
        let tighter_transition = transition_hz * 0.7; // 70%に
        let delta_f_tight = (tighter_transition / source_rate as f64).max(1e-6);
        let base_taps_tight = (5.5 / delta_f_tight).ceil() as usize;
        let taps_tight = base_taps_tight.saturating_mul(scale.max(1));
        let taps_tight_norm = if taps_tight.is_multiple_of(2) {
            taps_tight + 1
        } else {
            taps_tight
        };

        println!("  遷移帯を70%にした場合: {} taps (推定)", taps_tight_norm);

        // 遷移帯を広くした場合の試算 (計算量削減)
        let wider_transition = (transition_hz * 1.3).min(available_transition_hz);
        let delta_f_wide = (wider_transition / source_rate as f64).max(1e-6);
        let base_taps_wide = (5.5 / delta_f_wide).ceil() as usize;
        let taps_wide = base_taps_wide.saturating_mul(scale.max(1));
        let taps_wide_norm = if taps_wide.is_multiple_of(2) {
            taps_wide + 1
        } else {
            taps_wide
        };

        println!("  遷移帯を130%にした場合: {} taps (推定)", taps_wide_norm);

        assert!(
            attenuation_acceptable,
            "15kHz signal should be attenuated sufficiently"
        );
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[wasm_bindgen_test]
    fn test_resampler_simd_matches_scalar_whole_input() {
        let source_rate = 50_000u32;
        let target_rate = 48_000u32;
        let mut scalar =
            Resampler::new_with_cutoff(source_rate, target_rate, Some(20_000.0), Some(73));
        let mut simd =
            Resampler::new_with_cutoff(source_rate, target_rate, Some(20_000.0), Some(73));

        let input: Vec<f32> = (0..4096)
            .map(|i| {
                let t = i as f32 / source_rate as f32;
                0.8 * (2.0 * PI * 1000.0 * t).sin()
                    + 0.2 * (2.0 * PI * 4300.0 * t).cos()
                    + 0.1 * (2.0 * PI * 9000.0 * t).sin()
            })
            .collect();

        let mut out_scalar = Vec::new();
        scalar.process_with_backend(&input, &mut out_scalar, ResampleBackend::Scalar);
        scalar.flush_with_backend(&mut out_scalar, ResampleBackend::Scalar);

        let mut out_simd = Vec::new();
        simd.process_with_backend(&input, &mut out_simd, ResampleBackend::Simd);
        simd.flush_with_backend(&mut out_simd, ResampleBackend::Simd);

        assert_eq!(out_scalar.len(), out_simd.len());
        for (idx, (&a, &b)) in out_scalar.iter().zip(out_simd.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "idx={} scalar={} simd={}", idx, a, b);
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[wasm_bindgen_test]
    fn test_resampler_simd_matches_scalar_chunked_input() {
        let source_rate = 44_100u32;
        let target_rate = 48_000u32;
        let mut scalar = Resampler::new_with_cutoff(source_rate, target_rate, Some(18_000.0), None);
        let mut simd = Resampler::new_with_cutoff(source_rate, target_rate, Some(18_000.0), None);

        let input: Vec<f32> = (0..6000)
            .map(|i| {
                let t = i as f32 / source_rate as f32;
                0.9 * (2.0 * PI * 800.0 * t).sin() + 0.15 * (2.0 * PI * 6500.0 * t).cos()
            })
            .collect();

        let mut out_scalar = Vec::new();
        let mut out_simd = Vec::new();
        for chunk in input.chunks(137) {
            scalar.process_with_backend(chunk, &mut out_scalar, ResampleBackend::Scalar);
            simd.process_with_backend(chunk, &mut out_simd, ResampleBackend::Simd);
        }
        scalar.flush_with_backend(&mut out_scalar, ResampleBackend::Scalar);
        simd.flush_with_backend(&mut out_simd, ResampleBackend::Simd);

        assert_eq!(out_scalar.len(), out_simd.len());
        for (idx, (&a, &b)) in out_scalar.iter().zip(out_simd.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "idx={} scalar={} simd={}", idx, a, b);
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[wasm_bindgen_test]
    fn test_resampler_simd_tap_remainder_boundaries() {
        // taps=17 (mod4=1), taps=19 (mod4=3) の端数畳み込み経路を検証
        for &taps in &[17usize, 19usize] {
            let mut scalar = Resampler::new_with_cutoff(48_000, 44_100, Some(17_000.0), Some(taps));
            let mut simd = Resampler::new_with_cutoff(48_000, 44_100, Some(17_000.0), Some(taps));

            let input: Vec<f32> = (0..5000)
                .map(|i| {
                    let t = i as f32 / 48_000.0;
                    0.7 * (2.0 * PI * 1200.0 * t).sin() + 0.25 * (2.0 * PI * 5100.0 * t).cos()
                })
                .collect();

            let mut out_scalar = Vec::new();
            scalar.process_with_backend(&input, &mut out_scalar, ResampleBackend::Scalar);
            scalar.flush_with_backend(&mut out_scalar, ResampleBackend::Scalar);

            let mut out_simd = Vec::new();
            simd.process_with_backend(&input, &mut out_simd, ResampleBackend::Simd);
            simd.flush_with_backend(&mut out_simd, ResampleBackend::Simd);

            assert_eq!(out_scalar.len(), out_simd.len(), "taps={}", taps);
            for (idx, (&a, &b)) in out_scalar.iter().zip(out_simd.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5,
                    "taps={} idx={} scalar={} simd={}",
                    taps,
                    idx,
                    a,
                    b
                );
            }
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[wasm_bindgen_test]
    fn test_resampler_simd_chunk_and_empty_input_boundaries() {
        let mut scalar = Resampler::new_with_cutoff(50_000, 48_000, Some(20_000.0), Some(73));
        let mut simd = Resampler::new_with_cutoff(50_000, 48_000, Some(20_000.0), Some(73));

        let input: Vec<f32> = (0..4096)
            .map(|i| {
                let t = i as f32 / 50_000.0;
                0.8 * (2.0 * PI * 1000.0 * t).sin() + 0.2 * (2.0 * PI * 4700.0 * t).sin()
            })
            .collect();

        let mut out_scalar = Vec::new();
        let mut out_simd = Vec::new();

        // 空入力と細切れ入力を混在させる
        scalar.process_with_backend(&[], &mut out_scalar, ResampleBackend::Scalar);
        simd.process_with_backend(&[], &mut out_simd, ResampleBackend::Simd);
        for chunk in input.chunks(1).take(32) {
            scalar.process_with_backend(chunk, &mut out_scalar, ResampleBackend::Scalar);
            simd.process_with_backend(chunk, &mut out_simd, ResampleBackend::Simd);
        }
        for chunk in input[32..].chunks(257) {
            scalar.process_with_backend(chunk, &mut out_scalar, ResampleBackend::Scalar);
            simd.process_with_backend(chunk, &mut out_simd, ResampleBackend::Simd);
        }
        scalar.flush_with_backend(&mut out_scalar, ResampleBackend::Scalar);
        simd.flush_with_backend(&mut out_simd, ResampleBackend::Simd);

        assert_eq!(out_scalar.len(), out_simd.len());
        for (idx, (&a, &b)) in out_scalar.iter().zip(out_simd.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "idx={} scalar={} simd={}", idx, a, b);
        }
    }
}
