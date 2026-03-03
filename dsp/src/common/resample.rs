pub struct Resampler {
    pub source_rate: u32,
    pub target_rate: u32,
    step: f64,
    phase: f64,
    num_phases: usize,
    taps_per_phase: usize,
    coeffs: Vec<Vec<f32>>,
    history: Vec<f32>,
}

impl Resampler {
    /// `cutoff_hz` を指定した場合、リサンプラ内LPFのカットオフを明示できる。
    /// 指定しない場合は従来どおり target_rate ベースの自動値を使う。
    pub fn new_with_cutoff(source_rate: u32, target_rate: u32, cutoff_hz: Option<f32>) -> Self {
        assert!(source_rate > 0, "source_rate must be > 0");
        assert!(target_rate > 0, "target_rate must be > 0");

        // 入力時間軸上での出力サンプル刻み
        let step = source_rate as f64 / target_rate as f64;

        let num_phases = 256;
        // step ≈ 1 で 17 タップ、step ≈ 4.2 で ~85 タップ。
        let taps_per_phase = {
            let raw = (step.ceil() as usize * 17).max(17);
            raw | 1 // 奇数保証
        };
        let mut coeffs = vec![vec![0.0; taps_per_phase]; num_phases];

        // source 側正規化周波数（Nyquist=0.5）
        // - 指定なし: target_rate に合わせた従来値
        // - 指定あり: min(source/2, target/2, cutoff_hz) にクランプして利用
        let default_cutoff_hz = 0.5f64 * target_rate as f64 * 0.95;
        let max_cutoff_hz = 0.49f64 * (source_rate.min(target_rate) as f64);
        let cutoff_hz = cutoff_hz
            .map(|v| v as f64)
            .unwrap_or(default_cutoff_hz)
            .min(max_cutoff_hz)
            .max(1.0);
        let cutoff = cutoff_hz / source_rate as f64;
        let center = (taps_per_phase - 1) as f64 / 2.0;

        for (p, phase_coeffs) in coeffs.iter_mut().enumerate() {
            let frac = p as f64 / num_phases as f64;
            let mut sum = 0.0f64;

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
        }
    }

    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        if input.is_empty() {
            return;
        }

        // 入力をすべて履歴に追加
        self.history.extend_from_slice(input);

        // 処理に必要な最小の長さはフィルタの全タップ数
        if self.history.len() < self.taps_per_phase {
            return;
        }

        // 常に history[base .. base + taps_per_phase] の窓で演算する。
        let limit = (self.history.len() as isize - self.taps_per_phase as isize + 1).max(0) as f64;

        while self.phase < limit {
            let base = self.phase.floor() as usize;
            let frac = self.phase - base as f64;

            let mut phase_idx = (frac * self.num_phases as f64).floor() as usize;
            if phase_idx >= self.num_phases {
                phase_idx = self.num_phases - 1;
            }

            let coeffs = &self.coeffs[phase_idx];

            let mut sum = 0.0f32;
            for (tap, &h) in coeffs.iter().enumerate() {
                // history の中から現在の位相に対応する窓を畳み込む
                sum += self.history[base + tap] * h;
            }

            output.push(sum);
            self.phase += self.step;
        }

        // 整数サンプル分だけ history を消費
        let consumed = self.phase.floor() as usize;
        if consumed > 0 {
            self.history.drain(0..consumed);
            self.phase -= consumed as f64;
        }

        if self.phase < 0.0 {
            self.phase = 0.0;
        }
    }

    pub fn reset(&mut self) {
        self.history = vec![0.0; self.taps_per_phase - 1];
        self.phase = 0.0;
    }

    pub fn reconfigure(&mut self, source_rate: u32, target_rate: u32, cutoff_hz: Option<f32>) {
        *self = Self::new_with_cutoff(source_rate, target_rate, cutoff_hz);
    }

    /// 残っている履歴をすべて出力し、内部状態をリセットする。
    /// バーストの最後で呼び出す。
    pub fn flush(&mut self, output: &mut Vec<f32>) {
        if self.history.is_empty() {
            return;
        }

        // history の中にある有効なサンプルをすべて窓に通すために、必要な分のゼロを追加
        let padding_len = self.taps_per_phase - 1;
        let padding = vec![0.0f32; padding_len];
        self.process(&padding, output);

        // 状態リセット
        self.history = vec![0.0; self.taps_per_phase - 1];
        self.phase = 0.0;
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
        let mut resampler = Resampler::new_with_cutoff(source_rate, target_rate, None);
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
        let mut resampler_chunks = Resampler::new_with_cutoff(source_rate, target_rate, None);
        let mut resampler_whole = Resampler::new_with_cutoff(source_rate, target_rate, None);

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
        let mut resampler = Resampler::new_with_cutoff(source_rate, target_rate, None);

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
        let mut resampler = Resampler::new_with_cutoff(source_rate, target_rate, None);

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
        let mut resampler = Resampler::new_with_cutoff(source_rate, target_rate, None);

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

        let mut default_rs = Resampler::new_with_cutoff(source_rate, target_rate, None);
        let mut low_cut_rs = Resampler::new_with_cutoff(source_rate, target_rate, Some(5_000.0));
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
        let mut rs = Resampler::new_with_cutoff(24000, 48000, None);
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
        let mut rs = Resampler::new_with_cutoff(24000, 48000, None);
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
}
