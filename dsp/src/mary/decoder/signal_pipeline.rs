//! 信号処理パイプラインモジュール
//!
//! Real→IQ変換、リサンプリング、RRCフィルタリングなどの信号処理フロントエンドを管理する。

use crate::common::nco::Nco;
use crate::common::resample::Resampler;
use crate::common::rrc_filter::RrcFilter;
use crate::DspConfig;

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
        self.sample_buffer_i
            .extend_from_slice(&self.rrc_filtered_i);
        self.sample_buffer_q
            .extend_from_slice(&self.rrc_filtered_q);

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
}
