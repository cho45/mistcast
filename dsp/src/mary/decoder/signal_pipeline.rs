//! 信号処理パイプラインモジュール
//!
//! Real→IQ変換、リサンプリング、RRCフィルタリングなどの信号処理フロントエンドを管理する。

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::common::nco::complex_mul_interleaved2_simd;
use crate::common::nco::Nco;
use crate::common::resample::Resampler;
use crate::common::rrc_filter::RrcFilter;
use crate::DspConfig;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use std::arch::wasm32::{f32x4, v128, v128_store};

/// 信号処理パイプライン
pub struct SignalPipeline {
    pub resampler_i: Resampler,
    pub resampler_q: Resampler,
    pub rrc_filter_i: RrcFilter,
    pub rrc_filter_q: RrcFilter,
    pub lo_nco: Nco,
    pub sample_buffer_i: Vec<f32>,
    pub sample_buffer_q: Vec<f32>,
    // ゼロアロケーション用バッファ
    pub(crate) mix_buffer_i: Vec<f32>,
    pub(crate) mix_buffer_q: Vec<f32>,
    pub(crate) resample_buffer_i: Vec<f32>,
    pub(crate) resample_buffer_q: Vec<f32>,
    pub(crate) rrc_filtered_i: Vec<f32>,
    pub(crate) rrc_filtered_q: Vec<f32>,
}

impl SignalPipeline {
    pub fn new(config: &DspConfig) -> Self {
        let proc_sample_rate = config.proc_sample_rate();
        let lo_nco = Nco::new(-config.carrier_freq, config.sample_rate);

        let rrc_bw = config.chip_rate * (1.0 + config.rrc_alpha) * 0.5;
        let cutoff = Some(rrc_bw);

        Self {
            resampler_i: Resampler::new_with_cutoff(
                config.sample_rate as u32,
                proc_sample_rate as u32,
                cutoff,
                Some(config.rx_resampler_taps),
            ),
            resampler_q: Resampler::new_with_cutoff(
                config.sample_rate as u32,
                proc_sample_rate as u32,
                cutoff,
                Some(config.rx_resampler_taps),
            ),
            rrc_filter_i: RrcFilter::from_config(config),
            rrc_filter_q: RrcFilter::from_config(config),
            lo_nco,
            sample_buffer_i: Vec::new(),
            sample_buffer_q: Vec::new(),
            // ゼロアロケーションバッファ初期化
            mix_buffer_i: Vec::with_capacity(4096),
            mix_buffer_q: Vec::with_capacity(4096),
            resample_buffer_i: Vec::with_capacity(6144),
            resample_buffer_q: Vec::with_capacity(6144),
            rrc_filtered_i: Vec::with_capacity(6144),
            rrc_filtered_q: Vec::with_capacity(6144),
        }
    }

    pub fn reset(&mut self) {
        self.rrc_filter_i.reset();
        self.rrc_filter_q.reset();
        self.lo_nco.reset();
        self.sample_buffer_i.clear();
        self.sample_buffer_q.clear();
        self.mix_buffer_i.clear();
        self.mix_buffer_q.clear();
        self.resample_buffer_i.clear();
        self.resample_buffer_q.clear();
        self.rrc_filtered_i.clear();
        self.rrc_filtered_q.clear();
    }

    /// Real→IQ変換（ゼロアロケーション）
    pub fn mix_real_to_iq_zero_alloc(&mut self, samples: &[f32]) {
        self.mix_buffer_i.clear();
        self.mix_buffer_q.clear();

        // キャパシティ確認（必要なら拡張）
        if self.mix_buffer_i.capacity() < samples.len() {
            self.mix_buffer_i.reserve(samples.len());
            self.mix_buffer_q.reserve(samples.len());
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            let mut idx = 0usize;
            let mut interleaved = [0.0f32; 16];
            while idx + 8 <= samples.len() {
                let s0 = samples[idx] * 2.0;
                let s1 = samples[idx + 1] * 2.0;
                let s2 = samples[idx + 2] * 2.0;
                let s3 = samples[idx + 3] * 2.0;
                let s4 = samples[idx + 4] * 2.0;
                let s5 = samples[idx + 5] * 2.0;
                let s6 = samples[idx + 6] * 2.0;
                let s7 = samples[idx + 7] * 2.0;

                let x0 = f32x4(s0, 0.0, s1, 0.0);
                let x1 = f32x4(s2, 0.0, s3, 0.0);
                let x2 = f32x4(s4, 0.0, s5, 0.0);
                let x3 = f32x4(s6, 0.0, s7, 0.0);
                let (n0, n1, n2, n3) = self.lo_nco.step8_interleaved();
                let y0 = complex_mul_interleaved2_simd(x0, n0);
                let y1 = complex_mul_interleaved2_simd(x1, n1);
                let y2 = complex_mul_interleaved2_simd(x2, n2);
                let y3 = complex_mul_interleaved2_simd(x3, n3);

                unsafe {
                    v128_store(interleaved.as_mut_ptr() as *mut v128, y0);
                    v128_store(interleaved.as_mut_ptr().add(4) as *mut v128, y1);
                    v128_store(interleaved.as_mut_ptr().add(8) as *mut v128, y2);
                    v128_store(interleaved.as_mut_ptr().add(12) as *mut v128, y3);
                }
                for pair in interleaved.chunks_exact(2) {
                    self.mix_buffer_i.push(pair[0]);
                    self.mix_buffer_q.push(pair[1]);
                }
                idx += 8;
            }

            for &s in &samples[idx..] {
                let lo = self.lo_nco.step();
                self.mix_buffer_i.push(s * lo.re * 2.0);
                self.mix_buffer_q.push(s * lo.im * 2.0);
            }
        }

        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        // 安全なパターン: pushを使用（unsafe set_lenの回避）
        for &s in samples {
            let lo = self.lo_nco.step();
            self.mix_buffer_i.push(s * lo.re * 2.0);
            self.mix_buffer_q.push(s * lo.im * 2.0);
        }
    }

    /// サンプルを処理してバッファに格納
    pub fn process_samples(&mut self, samples: &[f32]) -> usize {
        // 1. Real → IQ 変換（事前確保バッファ使用）
        self.mix_real_to_iq_zero_alloc(samples);

        // 2. リサンプリング（事前確保バッファ使用）
        self.resample_buffer_i.clear();
        self.resample_buffer_q.clear();

        self.resampler_i
            .process(&self.mix_buffer_i, &mut self.resample_buffer_i);
        self.resampler_q
            .process(&self.mix_buffer_q, &mut self.resample_buffer_q);

        // 3. RRCフィルタ（インプレースAPI使用）
        self.rrc_filtered_i
            .resize(self.resample_buffer_i.len(), 0.0);
        self.rrc_filtered_q
            .resize(self.resample_buffer_q.len(), 0.0);
        self.rrc_filtered_i.copy_from_slice(&self.resample_buffer_i);
        self.rrc_filtered_q.copy_from_slice(&self.resample_buffer_q);
        self.rrc_filter_i
            .process_block_in_place(&mut self.rrc_filtered_i);
        self.rrc_filter_q
            .process_block_in_place(&mut self.rrc_filtered_q);

        // 4. 出力バッファに追加
        self.sample_buffer_i.extend_from_slice(&self.rrc_filtered_i);
        self.sample_buffer_q.extend_from_slice(&self.rrc_filtered_q);

        self.sample_buffer_i.len()
    }

    /// バッファから指定数のサンプルを消費
    #[cfg(test)]
    pub fn drain(&mut self, count: usize) {
        if count >= self.sample_buffer_i.len() {
            self.sample_buffer_i.clear();
            self.sample_buffer_q.clear();
        } else {
            self.sample_buffer_i.drain(0..count);
            self.sample_buffer_q.drain(0..count);
        }
    }

    /// 利用可能なサンプル数を取得
    #[cfg(test)]
    pub fn available_samples(&self) -> usize {
        self.sample_buffer_i.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    use crate::common::nco::Nco;
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn scalar_mix_reference(samples: &[f32], nco: &mut Nco) -> (Vec<f32>, Vec<f32>) {
        let mut i_out = Vec::with_capacity(samples.len());
        let mut q_out = Vec::with_capacity(samples.len());
        for &s in samples {
            let lo = nco.step();
            i_out.push(s * lo.re * 2.0);
            q_out.push(s * lo.im * 2.0);
        }
        (i_out, q_out)
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn assert_vec_close(actual: &[f32], expected: &[f32], eps: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= eps,
                "idx={} actual={} expected={} eps={}",
                idx,
                a,
                e,
                eps
            );
        }
    }

    #[test]
    fn test_signal_pipeline_creation() {
        let config = DspConfig::default_48k();
        let pipeline = SignalPipeline::new(&config);
        assert_eq!(pipeline.available_samples(), 0);
    }

    #[test]
    fn test_signal_pipeline_reset() {
        let config = DspConfig::default_48k();
        let mut pipeline = SignalPipeline::new(&config);
        let samples = vec![0.0f32; 100];
        pipeline.process_samples(&samples);
        assert!(pipeline.available_samples() > 0);
        pipeline.reset();
        assert_eq!(pipeline.available_samples(), 0);
    }

    #[test]
    fn test_signal_pipeline_process_samples() {
        let config = DspConfig::default_48k();
        let mut pipeline = SignalPipeline::new(&config);
        let samples = vec![0.0f32; 1000];
        let count = pipeline.process_samples(&samples);
        assert_eq!(count, pipeline.available_samples());
    }

    #[test]
    fn test_signal_pipeline_drain() {
        let config = DspConfig::default_48k();
        let mut pipeline = SignalPipeline::new(&config);
        let samples = vec![0.0f32; 1000];
        pipeline.process_samples(&samples);
        let available = pipeline.available_samples();
        pipeline.drain(available / 2);
        assert_eq!(pipeline.available_samples(), available - available / 2);
    }

    #[test]
    fn test_mix_real_to_iq_zero_alloc_generates_expected_baseband_for_first_sample() {
        let config = DspConfig::default_48k();
        let mut pipeline = SignalPipeline::new(&config);

        pipeline.mix_real_to_iq_zero_alloc(&[1.0]);

        assert_eq!(pipeline.mix_buffer_i.len(), 1);
        assert_eq!(pipeline.mix_buffer_q.len(), 1);
        assert!((pipeline.mix_buffer_i[0] - 2.0).abs() < 1e-6);
        assert!(pipeline.mix_buffer_q[0].abs() < 1e-6);
    }

    #[test]
    fn test_mix_real_to_iq_zero_alloc_clears_previous_mix_buffers() {
        let config = DspConfig::default_48k();
        let mut pipeline = SignalPipeline::new(&config);

        pipeline.mix_real_to_iq_zero_alloc(&[1.0, 2.0, 3.0]);
        assert_eq!(pipeline.mix_buffer_i.len(), 3);

        pipeline.mix_real_to_iq_zero_alloc(&[4.0]);

        assert_eq!(pipeline.mix_buffer_i.len(), 1);
        assert_eq!(pipeline.mix_buffer_q.len(), 1);
    }

    #[test]
    fn test_signal_pipeline_process_samples_empty_input_keeps_buffers_empty() {
        let config = DspConfig::default_48k();
        let mut pipeline = SignalPipeline::new(&config);

        let count = pipeline.process_samples(&[]);

        assert_eq!(count, 0);
        assert_eq!(pipeline.available_samples(), 0);
        assert!(pipeline.sample_buffer_q.is_empty());
    }

    #[test]
    fn test_signal_pipeline_drain_overflow_clears_all_samples() {
        let config = DspConfig::default_48k();
        let mut pipeline = SignalPipeline::new(&config);
        let samples = vec![0.0f32; 1000];
        pipeline.process_samples(&samples);

        pipeline.drain(usize::MAX);

        assert_eq!(pipeline.available_samples(), 0);
        assert!(pipeline.sample_buffer_q.is_empty());
    }

    #[test]
    fn test_signal_pipeline_reset_clears_internal_scratch_buffers() {
        let config = DspConfig::default_48k();
        let mut pipeline = SignalPipeline::new(&config);
        pipeline.mix_buffer_i.push(1.0);
        pipeline.mix_buffer_q.push(2.0);
        pipeline.resample_buffer_i.push(3.0);
        pipeline.resample_buffer_q.push(4.0);
        pipeline.rrc_filtered_i.push(5.0);
        pipeline.rrc_filtered_q.push(6.0);
        pipeline.sample_buffer_i.push(7.0);
        pipeline.sample_buffer_q.push(8.0);
        let _ = pipeline.lo_nco.step();

        pipeline.reset();

        assert!(pipeline.mix_buffer_i.is_empty());
        assert!(pipeline.mix_buffer_q.is_empty());
        assert!(pipeline.resample_buffer_i.is_empty());
        assert!(pipeline.resample_buffer_q.is_empty());
        assert!(pipeline.rrc_filtered_i.is_empty());
        assert!(pipeline.rrc_filtered_q.is_empty());
        assert!(pipeline.sample_buffer_i.is_empty());
        assert!(pipeline.sample_buffer_q.is_empty());

        pipeline.mix_real_to_iq_zero_alloc(&[1.0]);
        assert!((pipeline.mix_buffer_i[0] - 2.0).abs() < 1e-6);
        assert!(pipeline.mix_buffer_q[0].abs() < 1e-6);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[wasm_bindgen_test]
    fn test_mix_real_to_iq_zero_alloc_simd_matches_scalar_reference() {
        let config = DspConfig::default_48k();
        let mut pipeline = SignalPipeline::new(&config);
        let mut ref_nco = Nco::new(-config.carrier_freq, config.sample_rate);
        let samples = vec![
            0.25, -0.5, 0.125, 0.75, -0.8, 0.33, -0.22, 0.9, -0.4, 0.1, 0.0, -1.0, 0.6,
        ];

        pipeline.mix_real_to_iq_zero_alloc(&samples);
        let (expected_i, expected_q) = scalar_mix_reference(&samples, &mut ref_nco);

        assert_vec_close(&pipeline.mix_buffer_i, &expected_i, 1e-6);
        assert_vec_close(&pipeline.mix_buffer_q, &expected_q, 1e-6);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[wasm_bindgen_test]
    fn test_mix_real_to_iq_zero_alloc_simd_matches_scalar_across_calls() {
        let config = DspConfig::default_48k();
        let mut pipeline = SignalPipeline::new(&config);
        let mut ref_nco = Nco::new(-config.carrier_freq, config.sample_rate);

        let first = vec![0.1, -0.3, 0.5, -0.7, 0.9, -0.2, 0.4, -0.6];
        pipeline.mix_real_to_iq_zero_alloc(&first);
        let (expected_i_first, expected_q_first) = scalar_mix_reference(&first, &mut ref_nco);
        assert_vec_close(&pipeline.mix_buffer_i, &expected_i_first, 1e-6);
        assert_vec_close(&pipeline.mix_buffer_q, &expected_q_first, 1e-6);

        let second = vec![0.11, 0.22, -0.33, -0.44, 0.55, 0.66, -0.77, 0.88, -0.99];
        pipeline.mix_real_to_iq_zero_alloc(&second);
        let (expected_i_second, expected_q_second) = scalar_mix_reference(&second, &mut ref_nco);
        assert_vec_close(&pipeline.mix_buffer_i, &expected_i_second, 1e-6);
        assert_vec_close(&pipeline.mix_buffer_q, &expected_q_second, 1e-6);
    }
}
