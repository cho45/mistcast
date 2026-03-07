//! 等化制御モジュール
//!
//! 周波数領域等化器（FDE）の管理、CIR処理、パス選択ロジックを提供する。

use crate::common::equalization::{FrequencyDomainEqualizer, MmseSettings};
use crate::mary::sync::MarySyncDetector;
use crate::DspConfig;
use num_complex::Complex32;

/// CIR正規化モード
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CirNormalizationMode {
    None,
    UnitEnergy,
    Peak,
}

/// CIR後処理
pub fn postprocess_cir(
    cir: &mut [Complex32],
    cir_normalization_mode: CirNormalizationMode,
    cir_tap_threshold_alpha: f32,
) {
    if cir.is_empty() {
        return;
    }

    let max_mag = cir.iter().map(|c| c.norm()).fold(0.0f32, |a, b| a.max(b));

    if max_mag > 0.0 && cir_tap_threshold_alpha > 0.0 {
        let threshold = cir_tap_threshold_alpha * max_mag;
        for tap in cir.iter_mut() {
            if tap.norm() < threshold {
                *tap = Complex32::new(0.0, 0.0);
            }
        }
    }

    match cir_normalization_mode {
        CirNormalizationMode::None => {}
        CirNormalizationMode::UnitEnergy => {
            let energy = cir.iter().map(|c| c.norm_sqr()).sum::<f32>();
            let scale = (energy + 1e-12).sqrt();
            for tap in cir.iter_mut() {
                *tap /= scale;
            }
        }
        CirNormalizationMode::Peak => {
            let peak = cir
                .iter()
                .map(|c| c.norm())
                .fold(0.0f32, |a, b| a.max(b))
                .max(1e-12);
            for tap in cir.iter_mut() {
                *tap /= peak;
            }
        }
    }
}

/// 等化制御コントローラ
pub struct EqualizationController {
    equalizer: Option<FrequencyDomainEqualizer>,
    equalized_buffer: Vec<Complex32>,
    equalizer_input_offset: usize,
    fde_enabled: bool,
    current_frame_use_fde: bool,
    fde_auto_path_select: bool,
    fde_mmse_settings: MmseSettings,
    cir_normalization_mode: CirNormalizationMode,
    cir_tap_threshold_alpha: f32,
}

pub struct SyncWordPathMseInput<'a> {
    pub sync_detector: &'a MarySyncDetector,
    pub cir: &'a [Complex32],
    pub sample_buffer_i: &'a [f32],
    pub sample_buffer_q: &'a [f32],
    pub sync_start_idx: usize,
    pub sync_end_idx: usize,
    pub cir_len: usize,
    pub mmse: MmseSettings,
    pub cfo_rad_per_sample: f32,
}

impl EqualizationController {
    pub fn new(config: &DspConfig, fde_enabled: bool) -> Self {
        let equalizer = if fde_enabled {
            Some(Self::build_default_equalizer(config))
        } else {
            None
        };

        Self {
            equalizer,
            equalized_buffer: Vec::new(),
            equalizer_input_offset: 0,
            fde_enabled,
            current_frame_use_fde: fde_enabled,
            fde_auto_path_select: false,
            fde_mmse_settings: MmseSettings::default(),
            cir_normalization_mode: CirNormalizationMode::None,
            cir_tap_threshold_alpha: 0.0,
        }
    }

    fn build_default_equalizer(dsp_config: &DspConfig) -> FrequencyDomainEqualizer {
        let spc = dsp_config.proc_samples_per_chip();
        let cir_samples = dsp_config.preamble_sf * spc;
        let fft_size = Self::default_fde_fft_size(dsp_config);
        let mut initial_cir = vec![Complex32::new(0.0, 0.0); cir_samples];
        initial_cir[0] = Complex32::new(1.0, 0.0);
        FrequencyDomainEqualizer::new(&initial_cir, fft_size, 15.0)
    }

    fn default_fde_fft_size(dsp_config: &DspConfig) -> usize {
        let spc = dsp_config.proc_samples_per_chip();
        let cir_samples = dsp_config.preamble_sf * spc;
        (cir_samples * 2).next_power_of_two().max(1024)
    }

    pub fn reset(&mut self, config: &DspConfig) {
        if self.fde_enabled {
            self.equalizer = Some(Self::build_default_equalizer(config));
        }
        self.reset_frame_buffers();
    }

    /// FDE有効/無効を設定
    pub fn set_fde_enabled(&mut self, enabled: bool, config: &DspConfig) {
        self.fde_enabled = enabled;
        if enabled {
            if self.equalizer.is_none() {
                self.equalizer = Some(Self::build_default_equalizer(config));
            }
            self.current_frame_use_fde = true;
        } else {
            self.equalizer = None;
            self.current_frame_use_fde = false;
            self.fde_auto_path_select = false;
        }
        self.reset_frame_buffers();
    }

    /// 自動パス選択を設定
    pub fn set_fde_auto_path_select(&mut self, enabled: bool) {
        self.fde_auto_path_select = enabled;
        if !enabled {
            self.current_frame_use_fde = self.equalizer.is_some();
        }
    }

    /// MMSE設定を更新
    pub fn set_fde_mmse_settings(
        &mut self,
        snr_db: f32,
        lambda_scale: f32,
        lambda_floor: f32,
        max_inv_gain: Option<f32>,
    ) {
        self.fde_mmse_settings =
            MmseSettings::new(snr_db, lambda_scale, lambda_floor, max_inv_gain);
    }

    /// CIR後処理設定を更新
    pub fn set_cir_postprocess(
        &mut self,
        normalization_mode: CirNormalizationMode,
        tap_threshold_alpha: f32,
    ) {
        self.cir_normalization_mode = normalization_mode;
        self.cir_tap_threshold_alpha = tap_threshold_alpha.max(0.0);
    }

    /// live equalizer を一時的に再構成して、既知区間(sync word)の実測パスMSEを評価する。
    ///
    /// ここで返す `Pred MSE` は理論モデル予測ではなく、フレーム開始時に
    /// 既知区間である sync word を raw/FDE の両経路に replay して得た実測値。
    ///
    /// FDE 側は sync word 直前の `overlap_len()` サンプルを warmup として使う。
    /// 実デコーダはフレーム開始時に zero-initialized な equalizer state から始まるため、
    /// replay でも履歴不足分は 0 埋めして同じ初期条件を再現する。
    pub fn evaluate_sync_word_path_mse_with_live_equalizer(
        &mut self,
        input: SyncWordPathMseInput<'_>,
    ) -> (f32, f32) {
        let SyncWordPathMseInput {
            sync_detector,
            cir,
            sample_buffer_i,
            sample_buffer_q,
            sync_start_idx,
            sync_end_idx,
            cir_len,
            mmse,
            cfo_rad_per_sample,
        } = input;
        if sync_end_idx > sample_buffer_i.len() || sync_end_idx > sample_buffer_q.len() {
            return (f32::NAN, f32::NAN);
        }

        let mse_raw = sync_detector
            .sync_word_mse_iq(
                sample_buffer_i,
                sample_buffer_q,
                sync_start_idx,
                cfo_rad_per_sample,
            )
            .unwrap_or(f32::NAN);

        let Some(eq) = self.equalizer.as_mut() else {
            return (mse_raw, f32::NAN);
        };

        let overlap = eq.overlap_len();
        let available_history = sync_start_idx.min(overlap);
        let missing_history = overlap - available_history;
        let analysis_start = sync_start_idx - available_history;
        let input_len = sync_end_idx.saturating_sub(analysis_start) + missing_history;
        if input_len == 0 {
            return (mse_raw, f32::NAN);
        }

        let mut eval_input = Vec::with_capacity(input_len);
        eval_input.resize(missing_history, Complex32::new(0.0, 0.0));
        for idx in analysis_start..sync_end_idx {
            eval_input.push(Complex32::new(sample_buffer_i[idx], sample_buffer_q[idx]));
        }

        let mut eval_output = Vec::with_capacity(input_len);
        eq.set_cir_with_mmse(&cir[..cir_len], mmse);
        eq.reset();
        eq.process(&eval_input, &mut eval_output);
        eq.flush(&mut eval_output);

        let skip = overlap;
        let mse_fde = sync_detector
            .sync_word_mse_complex(&eval_output, skip, cfo_rad_per_sample)
            .unwrap_or(f32::NAN);

        (mse_raw, mse_fde)
    }

    /// パス選択を実行
    pub fn select_path(&mut self, mse_raw: f32, mse_fde: f32, auto_path_select: bool) {
        let use_fde = if auto_path_select {
            if mse_fde.is_nan() || !mse_fde.is_finite() {
                false
            } else if mse_raw.is_nan() || !mse_raw.is_finite() {
                true
            } else {
                mse_fde < mse_raw
            }
        } else {
            self.equalizer.is_some()
        };

        self.current_frame_use_fde = use_fde;
    }

    /// 等化器処理を実行（warmup drain 含む）
    pub fn process_equalizer_with_warmup_drain(
        &mut self,
        input: &[Complex32],
        pending_warmup_samples: usize,
    ) -> usize {
        if input.is_empty() {
            return 0;
        }

        if !self.current_frame_use_fde {
            // FDE無効時はパススルー
            self.equalized_buffer.extend_from_slice(input);
            return input.len();
        }

        let Some(eq) = self.equalizer.as_mut() else {
            return 0;
        };

        // 等化実行
        let added = eq.process(input, &mut self.equalized_buffer);
        if added == 0 {
            return 0;
        }

        // ウォームアップ由来の中間状態を先頭から捨てる
        let warmup_to_drain = added.min(pending_warmup_samples);
        if warmup_to_drain > 0 {
            self.consume_equalized_prefix(warmup_to_drain);
        }

        added
    }

    /// 等化器のCIRとMMSEパラメータを設定
    pub fn setup_equalizer(&mut self, cir: &[Complex32], mmse: MmseSettings) {
        if let Some(eq) = self.equalizer.as_mut() {
            eq.set_cir_with_mmse(cir, mmse);
            eq.reset();
        }
    }

    pub fn equalizer_ref(&self) -> Option<&FrequencyDomainEqualizer> {
        self.equalizer.as_ref()
    }

    pub fn equalized_buffer(&self) -> &[Complex32] {
        &self.equalized_buffer
    }

    pub fn equalized_len(&self) -> usize {
        self.equalized_buffer.len()
    }

    pub fn input_offset(&self) -> usize {
        self.equalizer_input_offset
    }

    pub fn current_frame_use_fde(&self) -> bool {
        self.current_frame_use_fde
    }

    pub fn fde_auto_path_select(&self) -> bool {
        self.fde_auto_path_select
    }

    pub fn fde_mmse_settings(&self) -> MmseSettings {
        self.fde_mmse_settings
    }

    pub fn cir_normalization_mode(&self) -> CirNormalizationMode {
        self.cir_normalization_mode
    }

    pub fn cir_tap_threshold_alpha(&self) -> f32 {
        self.cir_tap_threshold_alpha
    }

    pub fn reset_frame_buffers(&mut self) {
        self.equalized_buffer.clear();
        self.equalizer_input_offset = 0;
    }

    pub fn advance_input_offset(&mut self, consumed: usize) {
        self.equalizer_input_offset = self.equalizer_input_offset.saturating_add(consumed);
    }

    pub fn rewind_input_offset(&mut self, drained: usize) {
        self.equalizer_input_offset = self.equalizer_input_offset.saturating_sub(drained);
    }

    pub fn consume_equalized_prefix(&mut self, count: usize) -> usize {
        let drained = count.min(self.equalized_buffer.len());
        if drained == self.equalized_buffer.len() {
            self.equalized_buffer.clear();
        } else if drained > 0 {
            self.equalized_buffer.drain(0..drained);
        }
        drained
    }

    #[cfg(test)]
    pub(crate) fn extend_test_equalized_buffer(&mut self, samples: &[Complex32]) {
        self.equalized_buffer.extend_from_slice(samples);
    }

    #[cfg(test)]
    pub(crate) fn replace_test_equalized_buffer(&mut self, samples: Vec<Complex32>) {
        self.equalized_buffer = samples;
        self.equalizer_input_offset = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mary::sync::MarySyncDetector;
    use crate::DspConfig;

    #[test]
    fn test_postprocess_cir_empty() {
        let mut cir = vec![];
        postprocess_cir(&mut cir, CirNormalizationMode::UnitEnergy, 0.1);
        assert_eq!(cir.len(), 0);
    }

    #[test]
    fn test_postprocess_cir_unit_energy() {
        let mut cir = vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 1.0)];
        postprocess_cir(&mut cir, CirNormalizationMode::UnitEnergy, 0.0);
        let energy = cir.iter().map(|c| c.norm_sqr()).sum::<f32>();
        assert!((energy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_postprocess_cir_peak_normalization() {
        let mut cir = vec![Complex32::new(1.0, 0.0), Complex32::new(0.5, 0.0)];
        postprocess_cir(&mut cir, CirNormalizationMode::Peak, 0.0);
        let peak = cir.iter().map(|c| c.norm()).fold(0.0f32, |a, b| a.max(b));
        assert!((peak - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_postprocess_cir_threshold() {
        let mut cir = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(0.05, 0.0),
            Complex32::new(0.8, 0.0),
        ];
        postprocess_cir(&mut cir, CirNormalizationMode::None, 0.1);
        // 0.05 < 0.1 * 1.0 なので削除されるべき
        assert!(cir[1].norm() < 1e-6);
        assert!(cir[0].norm() > 0.9);
        assert!(cir[2].norm() > 0.7);
    }

    #[test]
    fn test_select_path_prefers_lower_fde_mse_when_auto_enabled() {
        let config = DspConfig::default_48k();
        let mut controller = EqualizationController::new(&config, true);
        controller.set_fde_auto_path_select(true);

        controller.select_path(0.20, 0.05, controller.fde_auto_path_select());
        assert!(controller.current_frame_use_fde());

        controller.select_path(0.05, 0.20, controller.fde_auto_path_select());
        assert!(!controller.current_frame_use_fde());
    }

    #[test]
    fn test_select_path_with_auto_disabled_tracks_equalizer_presence() {
        let config = DspConfig::default_48k();
        let mut controller = EqualizationController::new(&config, true);

        controller.select_path(0.01, 10.0, false);
        assert!(controller.current_frame_use_fde());

        controller.set_fde_enabled(false, &config);
        controller.select_path(10.0, 0.01, false);
        assert!(!controller.current_frame_use_fde());
    }

    #[test]
    fn test_process_equalizer_with_warmup_drain_passthrough_without_fde() {
        let config = DspConfig::default_48k();
        let mut controller = EqualizationController::new(&config, false);
        let input = vec![Complex32::new(1.0, 0.0), Complex32::new(0.5, -0.25)];

        let added = controller.process_equalizer_with_warmup_drain(&input, 1);

        assert_eq!(added, input.len());
        assert_eq!(controller.equalized_buffer(), input.as_slice());
    }

    #[test]
    fn test_frame_buffer_accounting_helpers_update_offset_and_prefix() {
        let config = DspConfig::default_48k();
        let mut controller = EqualizationController::new(&config, false);
        controller.extend_test_equalized_buffer(&[
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
        ]);

        controller.advance_input_offset(10);
        controller.rewind_input_offset(4);
        let drained = controller.consume_equalized_prefix(2);

        assert_eq!(controller.input_offset(), 6);
        assert_eq!(drained, 2);
        assert_eq!(controller.equalized_len(), 1);
        assert_eq!(controller.equalized_buffer()[0], Complex32::new(3.0, 0.0));
    }

    #[test]
    fn test_consume_equalized_prefix_handles_zero_and_overflow() {
        let config = DspConfig::default_48k();
        let mut controller = EqualizationController::new(&config, false);
        controller.extend_test_equalized_buffer(&[
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
        ]);

        assert_eq!(controller.consume_equalized_prefix(0), 0);
        assert_eq!(controller.equalized_len(), 2);

        assert_eq!(controller.consume_equalized_prefix(10), 2);
        assert_eq!(controller.equalized_len(), 0);
    }

    #[test]
    fn test_reset_frame_buffers_clears_buffer_and_offset() {
        let config = DspConfig::default_48k();
        let mut controller = EqualizationController::new(&config, false);
        controller.extend_test_equalized_buffer(&[Complex32::new(1.0, 0.0)]);
        controller.advance_input_offset(12);

        controller.reset_frame_buffers();

        assert_eq!(controller.equalized_len(), 0);
        assert_eq!(controller.input_offset(), 0);
    }

    #[test]
    fn test_disabling_fde_clears_auto_path_select_and_current_frame_use_fde() {
        let config = DspConfig::default_48k();
        let mut controller = EqualizationController::new(&config, true);
        controller.set_fde_auto_path_select(true);

        controller.set_fde_enabled(false, &config);

        assert!(!controller.fde_auto_path_select());
        assert!(!controller.current_frame_use_fde());
        assert!(controller.equalizer_ref().is_none());
    }

    #[test]
    fn test_enabling_fde_recreates_equalizer_and_resets_frame_buffers() {
        let config = DspConfig::default_48k();
        let mut controller = EqualizationController::new(&config, false);
        controller.extend_test_equalized_buffer(&[Complex32::new(1.0, 0.0)]);
        controller.advance_input_offset(7);

        controller.set_fde_enabled(true, &config);

        assert!(controller.equalizer_ref().is_some());
        assert!(controller.current_frame_use_fde());
        assert_eq!(controller.equalized_len(), 0);
        assert_eq!(controller.input_offset(), 0);
    }

    #[test]
    fn test_reset_rebuilds_default_equalizer_and_clears_frame_state() {
        let config = DspConfig::default_48k();
        let mut controller = EqualizationController::new(&config, true);
        controller.extend_test_equalized_buffer(&[Complex32::new(1.0, 0.0)]);
        controller.advance_input_offset(9);

        let cir_len = config.preamble_sf * config.proc_samples_per_chip();
        let mut cir = vec![Complex32::new(0.0, 0.0); cir_len];
        cir[0] = Complex32::new(0.5, 0.0);
        controller.setup_equalizer(&cir, MmseSettings::new(3.0, 2.0, 0.5, Some(4.0)));

        controller.reset(&config);

        assert!(controller.equalizer_ref().is_some());
        assert_eq!(controller.equalized_len(), 0);
        assert_eq!(controller.input_offset(), 0);
    }

    #[test]
    fn test_evaluate_sync_word_path_mse_returns_nan_when_bounds_invalid() {
        let config = DspConfig::default_48k();
        let detector = MarySyncDetector::new(
            config.clone(),
            MarySyncDetector::THRESHOLD_COARSE_DEFAULT,
            MarySyncDetector::THRESHOLD_FINE_DEFAULT,
        );
        let mut controller = EqualizationController::new(&config, true);
        let cir_len = config.preamble_sf * config.proc_samples_per_chip();
        let cir = vec![Complex32::new(1.0, 0.0); cir_len];
        let i = vec![0.0; 8];
        let q = vec![0.0; 8];

        let (mse_raw, mse_fde) =
            controller.evaluate_sync_word_path_mse_with_live_equalizer(SyncWordPathMseInput {
                sync_detector: &detector,
                cir: &cir,
                sample_buffer_i: &i,
                sample_buffer_q: &q,
                sync_start_idx: 0,
                sync_end_idx: 16,
                cir_len,
                mmse: MmseSettings::default(),
                cfo_rad_per_sample: 0.0,
            });

        assert!(mse_raw.is_nan());
        assert!(mse_fde.is_nan());
    }

    #[test]
    fn test_evaluate_sync_word_path_mse_without_equalizer_returns_raw_and_nan_fde() {
        let config = DspConfig::default_48k();
        let detector = MarySyncDetector::new(
            config.clone(),
            MarySyncDetector::THRESHOLD_COARSE_DEFAULT,
            MarySyncDetector::THRESHOLD_FINE_DEFAULT,
        );
        let mut controller = EqualizationController::new(&config, false);
        let cir_len = config.preamble_sf * config.proc_samples_per_chip();
        let cir = vec![Complex32::new(1.0, 0.0); cir_len];
        let total = detector.known_interval_len_samples();
        let sync_start = total - detector.sync_word_len_samples();
        let i = vec![1.0; total];
        let q = vec![0.0; total];

        let (mse_raw, mse_fde) =
            controller.evaluate_sync_word_path_mse_with_live_equalizer(SyncWordPathMseInput {
                sync_detector: &detector,
                cir: &cir,
                sample_buffer_i: &i,
                sample_buffer_q: &q,
                sync_start_idx: sync_start,
                sync_end_idx: total,
                cir_len,
                mmse: MmseSettings::default(),
                cfo_rad_per_sample: 0.0,
            });

        let expected_raw = detector
            .sync_word_mse_iq(&i, &q, sync_start, 0.0)
            .expect("non-zero known interval should produce a raw MSE estimate");
        assert_eq!(mse_raw, expected_raw);
        assert!(mse_fde.is_nan());
    }

    #[test]
    fn test_evaluate_sync_word_path_mse_zero_pads_when_sync_warmup_history_insufficient() {
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 13;
        config.preamble_repeat = 1;
        config.sync_word_bits = 8;
        let detector = MarySyncDetector::new(
            config.clone(),
            MarySyncDetector::THRESHOLD_COARSE_DEFAULT,
            MarySyncDetector::THRESHOLD_FINE_DEFAULT,
        );
        let mut controller = EqualizationController::new(&config, true);
        let cir_len = config.preamble_sf * config.proc_samples_per_chip();
        let cir = vec![Complex32::new(1.0, 0.0); cir_len];
        let total = detector.known_interval_len_samples();
        let sync_start = total - detector.sync_word_len_samples();
        let i = vec![1.0; total];
        let q = vec![0.0; total];

        let overlap = controller
            .equalizer_ref()
            .expect("FDE enabled")
            .overlap_len();
        assert!(sync_start < overlap);

        let (mse_raw, mse_fde) =
            controller.evaluate_sync_word_path_mse_with_live_equalizer(SyncWordPathMseInput {
                sync_detector: &detector,
                cir: &cir,
                sample_buffer_i: &i,
                sample_buffer_q: &q,
                sync_start_idx: sync_start,
                sync_end_idx: total,
                cir_len,
                mmse: MmseSettings::default(),
                cfo_rad_per_sample: 0.0,
            });

        assert!(mse_raw.is_finite());
        assert!(mse_fde.is_finite(), "mse_fde={}", mse_fde);
    }

    #[test]
    fn test_evaluate_sync_word_path_mse_fde_is_finite_with_sufficient_warmup() {
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 127;
        config.preamble_repeat = 2;
        config.sync_word_bits = 8;
        let detector = MarySyncDetector::new(config.clone(), 0.0, 0.0);
        let mut controller = EqualizationController::new(&config, true);
        let cir_len = config.preamble_sf * config.proc_samples_per_chip();
        let mut cir = vec![Complex32::new(0.0, 0.0); cir_len];
        cir[0] = Complex32::new(1.0, 0.0);

        let total = detector.known_interval_len_samples();
        let sync_start = total - detector.sync_word_len_samples();
        let i = vec![1.0; total];
        let q = vec![0.0; total];

        let overlap = controller
            .equalizer_ref()
            .expect("FDE enabled")
            .overlap_len();
        assert!(sync_start >= overlap);

        let (mse_raw, mse_fde) =
            controller.evaluate_sync_word_path_mse_with_live_equalizer(SyncWordPathMseInput {
                sync_detector: &detector,
                cir: &cir,
                sample_buffer_i: &i,
                sample_buffer_q: &q,
                sync_start_idx: sync_start,
                sync_end_idx: total,
                cir_len,
                mmse: MmseSettings::default(),
                cfo_rad_per_sample: 0.0,
            });

        assert!(mse_raw.is_finite() || mse_raw.is_nan());
        assert!(mse_fde.is_finite(), "mse_fde={}", mse_fde);
    }
}
