//! MaryDQPSKデコーダ
//!
//! # 復調パイプライン
//! 1. プリアンブル検出（Walsh[0]、DBPSK）
//! 2. Sync Word検出
//! 3. 16系列並列相復調によるMaryDQPSK復調
//! 4. Max-Log-MAP LLR計算
//! 5. Fountainデコーディング

use crate::coding::fountain::{FountainDecoder, FountainParams};
use crate::common::equalization::MmseSettings;
use crate::common::walsh::WalshCorrelator;
use crate::frame::packet::Packet;
use crate::mary::decoder_stats::DecoderStats;
use crate::mary::demodulator::Demodulator;
use crate::mary::equalization::{EqualizationController, postprocess_cir};
use crate::mary::fountain_receiver;
use crate::mary::interleaver_config;
use crate::mary::packet_decoder::{
    self, LlrCallback, PacketDecodeBuffers, PacketDecodeOptions,
};
use crate::mary::params::{PAYLOAD_SPREAD_FACTOR, SYNC_SPREAD_FACTOR};
use crate::mary::signal_pipeline::SignalPipeline;
use crate::mary::sync::{ChannelQualityEstimate, MarySyncDetector};
use crate::mary::tracking::{self, TrackingState};
use crate::params::PAYLOAD_SIZE;
use crate::DspConfig;
use num_complex::Complex32;

const TRACKING_EARLY_LATE_DELTA_CHIP: f32 = 0.5;
const LLR_ERASURE_QUANTILE_DEFAULT: f32 = 0.20;
const LLR_ERASURE_LIST_SIZE_DEFAULT: usize = 8;

// DecodeProgressとCirNormalizationModeは各モジュールから再エクスポート
pub use crate::mary::decoder_stats::DecodeProgress;
pub use crate::mary::equalization::CirNormalizationMode;

/// デコーダの状態
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecoderState {
    Searching,
    EqualizedDecoding,
}

#[derive(Default)]
struct SearchState {
    last_search_idx: usize,
}

struct FrameSession {
    tracking_state: TrackingState,
    packets_processed_in_burst: usize,
    consecutive_crc_errors: usize,
    pending_warmup_samples: usize,
    pending_warmup_input_samples: usize,
    remaining_samples_in_frame: isize,
}

impl FrameSession {
    fn new(warmup_samples: usize, frame_samples: usize) -> Self {
        Self {
            tracking_state: TrackingState::new(),
            packets_processed_in_burst: 0,
            consecutive_crc_errors: 0,
            pending_warmup_samples: warmup_samples,
            pending_warmup_input_samples: warmup_samples,
            remaining_samples_in_frame: frame_samples as isize,
        }
    }

    fn packets_decoded_in_burst(&self) -> usize {
        self.packets_processed_in_burst.saturating_sub(1)
    }

    fn sync_handoff_completed(&self) -> bool {
        self.packets_processed_in_burst != 0
    }

    fn complete_sync_handoff(&mut self, tracking_state: TrackingState) {
        self.tracking_state = tracking_state;
        self.packets_processed_in_burst = 1;
    }

    fn record_packet_result(&mut self, success: bool) {
        if success {
            self.consecutive_crc_errors = 0;
        } else {
            self.consecutive_crc_errors += 1;
        }
        self.packets_processed_in_burst += 1;
    }

    fn input_fully_consumed(&self) -> bool {
        self.pending_warmup_input_samples == 0 && self.remaining_samples_in_frame <= 0
    }

    fn raw_drain_limit(&self) -> usize {
        self.pending_warmup_input_samples + self.remaining_samples_in_frame.max(0) as usize
    }

    fn consume_raw_input(&mut self, drained_raw_samples: usize) {
        let warmup_drain = drained_raw_samples.min(self.pending_warmup_input_samples);
        self.pending_warmup_input_samples -= warmup_drain;
        let payload_drain = drained_raw_samples - warmup_drain;
        self.remaining_samples_in_frame -= payload_drain as isize;
    }

    fn consume_equalized_warmup(&mut self, added_samples: usize) {
        let warmup_to_drain = added_samples.min(self.pending_warmup_samples);
        self.pending_warmup_samples -= warmup_to_drain;
    }
}

struct FrameStartPlan {
    use_fde_this_frame: bool,
    mmse: MmseSettings,
    cir_len: usize,
    sync_start: usize,
    overlap: usize,
    frame_samples: usize,
}

/// MaryDQPSKデコーダ
pub struct Decoder {
    pub config: DspConfig,
    pipeline: SignalPipeline,
    equalization: EqualizationController,
    demodulator: Demodulator,
    fountain_decoder: FountainDecoder,
    pub recovered_data: Option<Vec<u8>>,
    sync_detector: MarySyncDetector,

    // 同期・追従状態
    state: DecoderState,
    search: SearchState,
    frame_session: Option<FrameSession>,
    viterbi_list_size: usize,
    llr_erasure_second_pass_enabled: bool,
    llr_erasure_quantile: f32,
    llr_erasure_list_size: usize,

    // 統計
    stats: DecoderStats,

    /// デバッグ観測用コールバック: デインターリーブ・デスクランブル後のLLRをパススルーする
    pub llr_callback: Option<LlrCallback>,

    // ゼロアロケーション用バッファプール
    cir_buffer: Vec<Complex32>,
    complex_buffer: Vec<Complex32>,
    packet_decode_buffers: PacketDecodeBuffers,
}

impl Decoder {
    fn frame_session(&self) -> &FrameSession {
        self.frame_session
            .as_ref()
            .expect("frame session must be initialized")
    }

    fn frame_session_mut(&mut self) -> &mut FrameSession {
        self.frame_session
            .as_mut()
            .expect("frame session must be initialized")
    }

    /// 新しいデコーダを作成する
    pub fn new(_data_size: usize, fountain_k: usize, dsp_config: DspConfig) -> Self {
        let tc = MarySyncDetector::THRESHOLD_COARSE_DEFAULT;
        let tf = MarySyncDetector::THRESHOLD_FINE_DEFAULT;

        // ゼロアロケーションバッファの初期化
        let spc = dsp_config.proc_samples_per_chip();
        let cir_buffer_size = dsp_config.preamble_sf * spc;

        Decoder {
            pipeline: SignalPipeline::new(&dsp_config),
            equalization: EqualizationController::new(&dsp_config, true),
            demodulator: Demodulator::new(),
            fountain_decoder: FountainDecoder::new(FountainParams::new(fountain_k, PAYLOAD_SIZE)),
            recovered_data: None,
            sync_detector: MarySyncDetector::new(dsp_config.clone(), tc, tf),
            config: dsp_config.clone(),
            state: DecoderState::Searching,
            search: SearchState::default(),
            frame_session: None,
            viterbi_list_size: 1,
            llr_erasure_second_pass_enabled: false,
            llr_erasure_quantile: LLR_ERASURE_QUANTILE_DEFAULT,
            llr_erasure_list_size: LLR_ERASURE_LIST_SIZE_DEFAULT,
            stats: DecoderStats::new(),
            llr_callback: None,
            cir_buffer: vec![Complex32::new(0.0, 0.0); cir_buffer_size],
            complex_buffer: Vec::with_capacity(16_000),
            packet_decode_buffers: PacketDecodeBuffers::new(),
        }
    }

    pub fn set_fde_enabled(&mut self, enabled: bool) {
        self.equalization.set_fde_enabled(enabled, &self.config);
    }

    pub fn set_fde_auto_path_select(&mut self, enabled: bool) {
        self.equalization.set_fde_auto_path_select(enabled);
    }

    pub fn set_fde_mmse_settings(
        &mut self,
        snr_db: f32,
        lambda_scale: f32,
        lambda_floor: f32,
        max_inv_gain: Option<f32>,
    ) {
        self.equalization.set_fde_mmse_settings(snr_db, lambda_scale, lambda_floor, max_inv_gain);
    }

    pub fn set_cir_postprocess(
        &mut self,
        normalization_mode: CirNormalizationMode,
        tap_threshold_alpha: f32,
    ) {
        self.equalization.set_cir_postprocess(normalization_mode, tap_threshold_alpha);
    }

    pub fn set_viterbi_list_size(&mut self, list_size: usize) {
        self.viterbi_list_size = list_size.max(1);
    }

    pub fn set_llr_erasure_second_pass(&mut self, enabled: bool, quantile: f32, list_size: usize) {
        self.llr_erasure_second_pass_enabled = enabled;
        self.llr_erasure_quantile = quantile.clamp(0.0, 1.0);
        self.llr_erasure_list_size = list_size.max(1);
    }

    pub fn process_samples(&mut self, samples: &[f32]) -> DecodeProgress {
        if self.recovered_data.is_some() {
            return self.progress();
        }
        self.stats.stats_total_samples += samples.len();

        // 信号処理パイプラインで処理
        self.pipeline.process_samples(samples);

        self.detect_and_process_frames()
    }

    fn detect_and_process_frames(&mut self) -> DecodeProgress {
        loop {
            if self.recovered_data.is_some() {
                break;
            }

            match self.state {
                DecoderState::Searching => {
                    if !self.handle_searching() {
                        break;
                    }
                }
                DecoderState::EqualizedDecoding => {
                    if !self.handle_decoding() {
                        break;
                    }
                }
            }
        }
        self.progress()
    }

    fn fill_complex_buffer_from_pipeline(&mut self, start: usize, len: usize) {
        self.complex_buffer.clear();
        if self.complex_buffer.capacity() < len {
            self.complex_buffer.reserve(len);
        }
        unsafe {
            self.complex_buffer.set_len(len);
        }
        for (dst_idx, src_idx) in (start..start + len).enumerate() {
            self.complex_buffer[dst_idx] = Complex32::new(
                self.pipeline.sample_buffer_i[src_idx],
                self.pipeline.sample_buffer_q[src_idx],
            );
        }
    }

    fn packets_decoded_in_burst(&self) -> usize {
        self.frame_session
            .as_ref()
            .map_or(0, FrameSession::packets_decoded_in_burst)
    }

    fn frame_input_fully_consumed(&self) -> bool {
        self.frame_session
            .as_ref()
            .is_some_and(FrameSession::input_fully_consumed)
    }

    fn frame_is_complete(&self, max_packets: usize) -> bool {
        self.frame_input_fully_consumed() && self.packets_decoded_in_burst() >= max_packets
    }

    fn transition_to_searching_after_frame_end(&mut self) {
        self.search.last_search_idx = 0;
        self.equalization.set_equalizer_input_offset(0);
        self.equalization.clear_equalized_buffer();
        self.state = DecoderState::Searching;
        self.frame_session = None;
    }

    fn abort_current_frame(&mut self) {
        self.search.last_search_idx = 0;
        self.equalization.set_equalizer_input_offset(0);
        self.equalization.clear_equalized_buffer();
        self.state = DecoderState::Searching;
        self.frame_session = None;
    }

    fn trim_search_buffers_if_needed(&mut self) {
        let max_buffer_len = 100_000;
        let keep_len = 50_000;
        if self.pipeline.sample_buffer_i.len() > max_buffer_len {
            let drain_len = self.pipeline.sample_buffer_i.len() - keep_len;
            self.pipeline.sample_buffer_i.drain(0..drain_len);
            self.pipeline.sample_buffer_q.drain(0..drain_len);
            self.search.last_search_idx = self.search.last_search_idx.saturating_sub(drain_len);
            let new_offset = self
                .equalization
                .equalizer_input_offset()
                .saturating_sub(drain_len);
            self.equalization.set_equalizer_input_offset(new_offset);
        }
    }

    fn begin_frame_from_sync(&mut self, plan: FrameStartPlan) {
        let FrameStartPlan {
            use_fde_this_frame,
            mmse,
            cir_len,
            sync_start,
            overlap,
            frame_samples,
        } = plan;

        let initial_drain_len = sync_start.saturating_sub(overlap);
        if initial_drain_len > 0 {
            self.pipeline.sample_buffer_i.drain(0..initial_drain_len);
            self.pipeline.sample_buffer_q.drain(0..initial_drain_len);
        }

        self.equalization.clear_equalized_buffer();
        self.search.last_search_idx = 0;
        self.equalization.set_equalizer_input_offset(0);

        if use_fde_this_frame {
            self.equalization
                .setup_equalizer(&self.cir_buffer[..cir_len], mmse);
        }

        let warmup_real_len = sync_start
            .saturating_sub(initial_drain_len)
            .min(self.pipeline.sample_buffer_i.len());
        self.fill_complex_buffer_from_pipeline(0, warmup_real_len);

        self.frame_session = Some(FrameSession::new(warmup_real_len, frame_samples));

        self.equalize_from_complex_buffer(warmup_real_len, warmup_real_len);

        self.frame_session_mut().tracking_state = TrackingState::new();
        self.demodulator.set_prev_phase(Complex32::new(1.0, 0.0));
        self.state = DecoderState::EqualizedDecoding;
    }

    fn feed_unprocessed_samples_to_equalizer(&mut self) -> usize {
        let start = self.equalization.equalizer_input_offset();
        let to_process = self.pipeline.sample_buffer_i.len().saturating_sub(start);
        if to_process == 0 {
            return 0;
        }

        self.fill_complex_buffer_from_pipeline(start, to_process);
        self.equalize_from_complex_buffer(to_process, to_process);
        to_process
    }

    fn perform_sync_handoff_if_needed(&mut self, spc: usize, sync_word_bits: usize) -> bool {
        if self.frame_session().sync_handoff_completed() {
            return true;
        }

        let required_sync_samples = sync_word_bits * SYNC_SPREAD_FACTOR * spc;
        if self.equalization.equalized_buffer().len() < required_sync_samples {
            return false;
        }

        let repeat = self.config.preamble_repeat;
        let early_late_delta = (spc as f32 * TRACKING_EARLY_LATE_DELTA_CHIP).max(1.0);

        let mut best_timing_offset = 0usize;
        let mut best_sync_score = f32::NEG_INFINITY;
        for t_offset in 0..spc {
            let mut total_energy = 0.0f32;
            let mut used = 0usize;
            for i in 0..sync_word_bits {
                let symbol_start = t_offset + i * SYNC_SPREAD_FACTOR * spc;
                let corr = if let Some(c) =
                    self.despread_symbol_inner(symbol_start, 0.0, SYNC_SPREAD_FACTOR, 0)
                {
                    c
                } else {
                    break;
                };
                total_energy += corr[0].norm_sqr();
                used += 1;
            }
            if used == sync_word_bits {
                let score = total_energy / used as f32;
                if score > best_sync_score {
                    best_sync_score = score;
                    best_timing_offset = t_offset;
                }
            }
        }
        if best_timing_offset > 0 {
            self.equalization.drain_equalized_buffer(best_timing_offset);
        }

        let mut st = self.frame_session().tracking_state;

        self.complex_buffer.clear();
        if self.complex_buffer.capacity() < sync_word_bits {
            self.complex_buffer.reserve(sync_word_bits);
        }
        let mut prev_y: Option<Complex32> = None;
        let mut sum_diff = Complex32::new(0.0, 0.0);
        let mut sw = 0.0f32;
        let mut sx = 0.0f32;
        let mut sy = 0.0f32;
        let mut sxx = 0.0f32;
        let mut sxy = 0.0f32;
        for i in 0..sync_word_bits {
            let symbol_start = i * SYNC_SPREAD_FACTOR * spc;
            let on = if let Some(c) =
                self.despread_symbol_inner(symbol_start, 0.0, SYNC_SPREAD_FACTOR, 0)
            {
                c[0]
            } else {
                break;
            };

            let expected_sign = self.sync_detector.sync_symbols()[repeat + i];
            let y = on * Complex32::new(expected_sign, 0.0);
            if let Some(prev) = prev_y {
                sum_diff += y * prev.conj();
            }
            prev_y = Some(y);
            self.complex_buffer.push(y);

            let early =
                self.despread_symbol_inner(symbol_start, -early_late_delta, SYNC_SPREAD_FACTOR, 0);
            let late =
                self.despread_symbol_inner(symbol_start, early_late_delta, SYNC_SPREAD_FACTOR, 0);
            if let (Some(e), Some(l)) = (early, late) {
                let err = tracking::timing_error_from_early_late(e[0].norm(), l[0].norm());
                let w = on.norm_sqr().max(1e-6);
                let x = i as f32;
                sw += w;
                sx += w * x;
                sy += w * err;
                sxx += w * x * x;
                sxy += w * x * err;
            }
        }

        if self.complex_buffer.is_empty() {
            return false;
        }

        let omega_sync = if sum_diff.norm_sqr() > 1e-12 {
            sum_diff.arg()
        } else {
            0.0
        };

        let mut sum_base = Complex32::new(0.0, 0.0);
        for (i, &y) in self.complex_buffer.iter().enumerate() {
            let ang = -omega_sync * i as f32;
            let (s, c) = ang.sin_cos();
            sum_base += y * Complex32::new(c, s);
        }
        let phi0 = if sum_base.norm_sqr() > 1e-12 {
            sum_base.arg()
        } else {
            0.0
        };

        let sync_to_payload_scale = PAYLOAD_SPREAD_FACTOR as f32 / SYNC_SPREAD_FACTOR as f32;
        st.phase_rate = (omega_sync * sync_to_payload_scale).clamp(
            -tracking::TRACKING_PHASE_RATE_LIMIT_RAD,
            tracking::TRACKING_PHASE_RATE_LIMIT_RAD,
        );

        let last_sync_idx = (self.complex_buffer.len() - 1) as f32;
        let phi_last = phi0 + omega_sync * last_sync_idx;
        let phi_payload0 = phi_last + st.phase_rate;
        let (s_ref, c_ref) = phi_payload0.sin_cos();
        st.phase_ref = Complex32::new(c_ref, s_ref);

        if let Some(&last_y) = self.complex_buffer.last() {
            let last_sign = self.sync_detector.sync_symbols()[repeat + self.complex_buffer.len() - 1];
            let last_corr = last_y * Complex32::new(last_sign, 0.0);
            let (s_last, c_last) = (-phi_last).sin_cos();
            let prev = last_corr * Complex32::new(c_last, s_last);
            let norm = prev.norm().max(1e-6);
            self.demodulator.set_prev_phase(prev / norm);
        } else {
            self.demodulator.set_prev_phase(Complex32::new(1.0, 0.0));
        }

        let timing_limit = spc as f32 * tracking::TRACKING_TIMING_LIMIT_CHIP;
        let timing_rate_limit = spc as f32 * tracking::TRACKING_TIMING_RATE_LIMIT_CHIP;
        if sw > 1e-9 {
            let denom = sw * sxx - sx * sx;
            if denom.abs() > 1e-9 {
                let slope = (sw * sxy - sx * sy) / denom;
                let intercept = (sy - slope * sx) / sw;
                st.timing_offset = (intercept * early_late_delta).clamp(-timing_limit, timing_limit);
                st.timing_rate = (slope * early_late_delta * sync_to_payload_scale)
                    .clamp(-timing_rate_limit, timing_rate_limit);
            } else {
                let intercept = sy / sw;
                st.timing_offset = (intercept * early_late_delta).clamp(-timing_limit, timing_limit);
                st.timing_rate = 0.0;
            }
        } else {
            st.timing_offset = 0.0;
            st.timing_rate = 0.0;
        }

        self.frame_session_mut().complete_sync_handoff(st);
        self.equalization
            .drain_equalized_buffer(required_sync_samples.saturating_sub(spc));
        true
    }

    fn handle_searching(&mut self) -> bool {
        let spc = self.config.proc_samples_per_chip().max(1);
        let sf_preamble = self.config.preamble_sf;
        let sf_sync = SYNC_SPREAD_FACTOR;
        let repeat = self.config.preamble_repeat;
        let sync_word_bits = self.config.sync_word_bits;

        let required_len = (sf_preamble * repeat + sf_sync * sync_word_bits) * spc;
        if self.pipeline.sample_buffer_i.len() < self.search.last_search_idx + required_len {
            return false;
        }

        let (sync_opt, next_search_idx) = self.sync_detector.detect(
            &self.pipeline.sample_buffer_i,
            &self.pipeline.sample_buffer_q,
            self.search.last_search_idx,
        );

        if let Some(s) = sync_opt {
            let spc = self.config.proc_samples_per_chip().max(1);
            let sf_preamble = self.config.preamble_sf;
            let sf_sync = SYNC_SPREAD_FACTOR;
            let sf_payload = PAYLOAD_SPREAD_FACTOR;
            let repeat = self.config.preamble_repeat;
            let sync_word_bits = self.config.sync_word_bits;
            let packets_per_frame = self.config.packets_per_burst;
            let expected_symbols = interleaver_config::mary_symbols();

            let sync_start = s.peak_sample_idx.saturating_sub(spc / 2);
            let preamble_len = sf_preamble * repeat * spc;
            let preamble_start_idx = s
                .peak_sample_idx
                .saturating_sub(preamble_len)
                .saturating_sub(spc / 2);

            // 1. 新しい CIR を推定 (絶対にバッファを削る前に行う)
            // ゼロアロケーション: 事前確保バッファ使用
            let cir_len = sf_preamble * spc;
            self.cir_buffer.fill(Complex32::new(0.0, 0.0));
            let mut chq = ChannelQualityEstimate::default();
            {
                let cir_slice = &mut self.cir_buffer[..cir_len];
                self.sync_detector.estimate_channel_quality(
                    &self.pipeline.sample_buffer_i,
                    &self.pipeline.sample_buffer_q,
                    preamble_start_idx,
                    cir_slice,
                    &mut chq,
                );
                postprocess_cir(
                    cir_slice,
                    self.equalization.cir_normalization_mode(),
                    self.equalization.cir_tap_threshold_alpha(),
                );
            }
            let mut mmse = self.equalization.fde_mmse_settings();
            if let Some(snr_db) = chq.snr_db {
                mmse.snr_db = snr_db.clamp(-20.0, 40.0);
            }

            let mut pred_mse_fde = f32::NAN;
            let mut pred_mse_raw = f32::NAN;
            if self.equalization.equalizer_ref().is_some() {
                let known_end_idx =
                    preamble_start_idx + self.sync_detector.known_interval_len_samples();
                let (mse_raw, mse_fde) = self.equalization.known_interval_path_mse(
                    &self.sync_detector,
                    &self.cir_buffer,
                    &self.pipeline.sample_buffer_i,
                    &self.pipeline.sample_buffer_q,
                    preamble_start_idx,
                    known_end_idx,
                    cir_len,
                    mmse,
                    chq.cfo_rad_per_sample,
                );
                pred_mse_fde = mse_fde;
                pred_mse_raw = mse_raw;
                self.equalization.select_path(mse_raw, mse_fde, self.equalization.fde_auto_path_select());
            }
            let use_fde_this_frame = self.equalization.current_frame_use_fde();
            self.stats.last_pred_mse_fde = pred_mse_fde;
            self.stats.last_pred_mse_raw = pred_mse_raw;
            self.stats.last_est_snr_db = chq.snr_db.unwrap_or(f32::NAN);
            if use_fde_this_frame {
                self.stats.fde_selected_frames += 1;
                self.stats.last_path_used = 1;
            } else {
                self.stats.raw_selected_frames += 1;
                self.stats.last_path_used = 0;
            }

            let overlap = if use_fde_this_frame {
                self.equalization.equalizer_ref().map_or(0, |eq| eq.overlap_len())
            } else {
                0
            };

            let frame_samples = (sync_word_bits * sf_sync
                + packets_per_frame * (expected_symbols * sf_payload))
                * spc;
            self.begin_frame_from_sync(FrameStartPlan {
                use_fde_this_frame,
                mmse,
                cir_len,
                sync_start,
                overlap,
                frame_samples,
            });
            true
        } else {
            self.search.last_search_idx = next_search_idx;
            self.trim_search_buffers_if_needed();
            false
        }
    }

    fn handle_decoding(&mut self) -> bool {
        let spc = self.config.proc_samples_per_chip().max(1);
        let sf_payload = PAYLOAD_SPREAD_FACTOR;
        let sync_word_bits = self.config.sync_word_bits;

        let to_process = self.feed_unprocessed_samples_to_equalizer();

        if !self.perform_sync_handoff_if_needed(spc, sync_word_bits) {
            return false;
        }

        let expected_symbols = interleaver_config::mary_symbols();
        let packet_samples = expected_symbols * sf_payload * spc;
        let max_packets = self.config.packets_per_burst;
        let packets_decoded = self.packets_decoded_in_burst();

        if packets_decoded >= max_packets {
            if self.frame_is_complete(max_packets) {
                self.transition_to_searching_after_frame_end();
                return true;
            }
            return to_process > 0;
        }

        if self.equalization.equalized_buffer_mut().len() < packet_samples + spc {
            return false;
        }

        let result = self.process_packet_core();
        if !result.processed {
            self.abort_current_frame();
            return false;
        }
        let mut st = self.frame_session().tracking_state;
        let offset_int = st.timing_offset.round().clamp(-1.0, 1.0) as i32;
        let actual_drain_len = (packet_samples as i32 + offset_int).max(0) as usize;
        st.timing_offset -= offset_int as f32;
        self.frame_session_mut().tracking_state = st;

        let buffer_len = self.equalization.equalized_buffer().len();
        self.equalization.equalized_buffer_mut()
            .drain(0..actual_drain_len.min(buffer_len));

        if let Some(packet) = result.packet {
            self.receive_decoded_packet(packet);
        }

        self.frame_session_mut().record_packet_result(result.success);

        if self.frame_is_complete(max_packets) {
            self.transition_to_searching_after_frame_end();
        }

        true
    }

    fn process_packet_core(&mut self) -> packet_decoder::PacketProcessResult {
        let spc = self.config.proc_samples_per_chip().max(1);
        let mut tracking_state = self.frame_session().tracking_state;
        let mut prev_phase = self.demodulator.prev_phase();
        let equalized_buffer = self.equalization.equalized_buffer();
        let correlators = self.demodulator.correlators();
        let options = PacketDecodeOptions {
            spc,
            early_late_delta: (spc as f32 * TRACKING_EARLY_LATE_DELTA_CHIP).max(1.0),
            viterbi_list_size: self.viterbi_list_size,
            llr_erasure_second_pass_enabled: self.llr_erasure_second_pass_enabled,
            llr_erasure_quantile: self.llr_erasure_quantile,
            llr_erasure_list_size: self.llr_erasure_list_size,
        };

        let result = packet_decoder::process_packet_core(
            &self.demodulator,
            &mut prev_phase,
            &mut tracking_state,
            &mut self.stats,
            &mut self.packet_decode_buffers,
            &mut self.llr_callback,
            &options,
            |symbol_start, timing_offset, sample_shift| {
                despread_symbol_with_timing_from(
                    equalized_buffer,
                    correlators,
                    spc,
                    symbol_start,
                    timing_offset,
                    sample_shift,
                )
            },
        );
        self.frame_session_mut().tracking_state = tracking_state;
        self.demodulator.set_prev_phase(prev_phase);
        result
    }

    fn receive_decoded_packet(&mut self, packet: Packet) {
        let pkt_k = packet.lt_k as usize;
        if pkt_k != self.fountain_decoder.params().k {
            self.rebuild_fountain_decoder(pkt_k);
        }

        let result = fountain_receiver::receive_packet(
            &mut self.fountain_decoder,
            &mut self.stats,
            packet,
        );
        if let Some(data) = result.recovered_data {
            self.recovered_data = Some(data);
        }
    }

    fn rebuild_fountain_decoder(&mut self, fountain_k: usize) {
        let params = FountainParams::new(fountain_k, PAYLOAD_SIZE);
        self.fountain_decoder = FountainDecoder::new(params);
        self.recovered_data = None;
        self.stats.reset();
    }

    fn progress(&self) -> DecodeProgress {
        self.stats.to_progress(&self.fountain_decoder, &self.config, self.recovered_data.is_some())
    }

    fn equalize_from_complex_buffer(&mut self, valid_len: usize, raw_consumed: usize) {
        if valid_len == 0 {
            return;
        }
        let mut input_vec = std::mem::take(&mut self.complex_buffer);
        self.equalize_with_raw_consumed(&input_vec[..valid_len], raw_consumed);
        input_vec.clear();
        self.complex_buffer = input_vec;
    }

    fn equalize_with_raw_consumed(&mut self, input: &[Complex32], raw_consumed: usize) {
        if input.is_empty() {
            return;
        }

        // A. 等化器へ投入済みオフセットを先に進める（drainで前方を削った分は後で戻す）
        let new_offset = self.equalization.equalizer_input_offset().saturating_add(raw_consumed);
        self.equalization.set_equalizer_input_offset(new_offset);

        // B. 生バッファの物理削除 (投入時に実行)
        // フレーム境界を壊さないよう、ウォームアップ + remaining を上限にする。
        let limit = self
            .frame_session
            .as_ref()
            .map_or(0, FrameSession::raw_drain_limit);
        let to_drain_raw = raw_consumed.min(limit).min(self.pipeline.sample_buffer_i.len());
        if to_drain_raw > 0 {
            self.pipeline.sample_buffer_i.drain(0..to_drain_raw);
            self.pipeline.sample_buffer_q.drain(0..to_drain_raw);
            let new_offset = self.equalization.equalizer_input_offset().saturating_sub(to_drain_raw);
            self.equalization.set_equalizer_input_offset(new_offset);

            if let Some(frame_session) = self.frame_session.as_mut() {
                frame_session.consume_raw_input(to_drain_raw);
            }
        }

        // C. 等化実行
        let pending_warmup_samples = self
            .frame_session
            .as_ref()
            .map_or(0, |frame_session| frame_session.pending_warmup_samples);
        let added = self
            .equalization
            .process_equalizer_with_warmup_drain(input, pending_warmup_samples);
        if added == 0 {
            return;
        }

        // D. ウォームアップ由来の中間状態を先頭から捨てる
        if let Some(frame_session) = self.frame_session.as_mut() {
            frame_session.consume_equalized_warmup(added);
        }
    }

    pub fn recovered_data(&self) -> Option<&[u8]> {
        self.recovered_data.as_deref()
    }

    fn despread_symbol_inner(
        &self,
        symbol_start: usize,
        timing_offset: f32,
        sf: usize,
        walsh_idx: usize,
    ) -> Option<[num_complex::Complex32; 1]> {
        let spc = self.config.proc_samples_per_chip().max(1);

        // round() 後の添字がバッファ内に収まるかを事前確認する。
        // 負値を usize へキャストした際の 0 への飽和を防ぐ。
        let min_p = symbol_start as f32 + timing_offset + ((spc as f32 - 1.0) / 2.0);
        let max_p = symbol_start as f32
            + ((sf - 1) * spc) as f32
            + timing_offset
            + ((spc as f32 - 1.0) / 2.0);
        let min_idx = min_p.round() as isize;
        let max_idx = max_p.round() as isize;
        let len = self.equalization.equalized_buffer().len() as isize;
        if min_idx < 0 || max_idx >= len {
            return None;
        }

        let mut results = [num_complex::Complex32::new(0.0, 0.0); 1];
        for chip_idx in 0..sf {
            let p = symbol_start as f32
                + (chip_idx * spc) as f32
                + timing_offset
                + ((spc as f32 - 1.0) / 2.0);
            let i_idx = p.round() as isize;
            if i_idx < 0 || i_idx >= len {
                return None;
            }
            let sample = self.equalization.equalized_buffer()[i_idx as usize];
            let walsh_val = self.demodulator.correlators()[walsh_idx].sequence()[chip_idx] as f32;
            results[0] += sample * walsh_val;
        }

        Some(results)
    }

    #[cfg(test)]
    fn despread_symbol_with_timing(
        &self,
        symbol_start: usize,
        timing_offset: f32,
        sample_shift: f32,
    ) -> Option<[Complex32; 16]> {
        despread_symbol_with_timing_from(
            self.equalization.equalized_buffer(),
            self.demodulator.correlators(),
            self.config.proc_samples_per_chip().max(1),
            symbol_start,
            timing_offset,
            sample_shift,
        )
    }

    pub fn reset(&mut self) {
        let params = self.fountain_decoder.params().clone();
        self.pipeline.reset();
        self.equalization.reset(&self.config);
        self.demodulator.reset();
        self.fountain_decoder = FountainDecoder::new(params);
        self.recovered_data = None;
        self.state = DecoderState::Searching;
        self.search = SearchState::default();
        self.frame_session = None;
        self.stats.reset();
        self.packet_decode_buffers.clear();
    }
}

fn despread_symbol_with_timing_from(
    equalized_buffer: &[Complex32],
    correlators: &[WalshCorrelator],
    spc: usize,
    symbol_start: usize,
    timing_offset: f32,
    sample_shift: f32,
) -> Option<[Complex32; 16]> {
    let sf = PAYLOAD_SPREAD_FACTOR;

    let min_p = symbol_start as f32 + timing_offset + sample_shift + ((spc as f32 - 1.0) / 2.0);
    let max_p = symbol_start as f32
        + ((sf - 1) * spc) as f32
        + timing_offset
        + sample_shift
        + ((spc as f32 - 1.0) / 2.0);
    let min_idx = min_p.round() as isize;
    let max_idx = max_p.round() as isize;
    let len = equalized_buffer.len() as isize;
    if min_idx < 0 || max_idx >= len {
        return None;
    }

    let mut results = [Complex32::new(0.0, 0.0); 16];
    for chip_idx in 0..sf {
        let p = symbol_start as f32
            + (chip_idx * spc) as f32
            + timing_offset
            + sample_shift
            + ((spc as f32 - 1.0) / 2.0);
        let i_idx = p.round() as isize;
        if i_idx < 0 || i_idx >= len {
            return None;
        }
        let sample = equalized_buffer[i_idx as usize];

        for (idx, correlator) in correlators.iter().enumerate() {
            let walsh_val = correlator.sequence()[chip_idx] as f32;
            results[idx] += sample * walsh_val;
        }
    }

    Some(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::walsh::WalshDictionary;

    #[inline]
    fn apply_llr_erasure_quantile(llrs: &mut [f32], quantile: f32) {
        packet_decoder::apply_llr_erasure_quantile(llrs, quantile);
    }

    fn make_decoder() -> Decoder {
        let config = DspConfig::default_48k();
        Decoder::new(160, 10, config)
    }

    #[test]
    fn test_decoder_creation_and_reset() {
        let mut decoder = make_decoder();
        decoder.reset();
    }

    #[test]
    fn test_silence_input_does_not_complete() {
        let mut decoder = make_decoder();
        let silence = vec![0.0f32; 4800];
        decoder.process_samples(&silence);
        assert!(!decoder.progress().complete);
    }

    #[test]
    fn test_progress_before_completion() {
        let decoder = make_decoder();
        let progress = decoder.progress();
        assert_eq!(progress.received_packets, 0);
        assert!(!progress.complete);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut decoder = make_decoder();
        let silence = vec![0.0f32; 1000];
        decoder.process_samples(&silence);
        assert!(decoder.stats.stats_total_samples > 0);
        decoder.reset();
        assert_eq!(decoder.stats.stats_total_samples, 0);
        assert!(decoder.recovered_data.is_none());
        assert!(decoder.pipeline.sample_buffer_i.is_empty());
        assert!(decoder.pipeline.sample_buffer_q.is_empty());
    }

    #[test]
    fn test_next_phase_gate_enabled_uses_2_of_3_and_hysteresis() {
        // OFF→ON: ON閾値で2条件満たせば有効化する
        assert!(tracking::next_phase_gate_enabled(false, 1.2, 0.15, 1.0));

        // ON維持: ON閾値は割ってもOFF閾値で2条件満たせば維持する
        assert!(tracking::next_phase_gate_enabled(true, 0.8, 0.1, 1.0));

        // ON→OFF: OFF閾値で1条件以下なら無効化する
        assert!(!tracking::next_phase_gate_enabled(true, 0.10, 0.01, 1.1));
    }

    #[test]
    fn test_apply_llr_erasure_quantile_zeroes_small_abs_values() {
        let mut llrs = [0.10, -0.20, 0.30, -0.90];
        apply_llr_erasure_quantile(&mut llrs, 0.5);
        assert_eq!(llrs, [0.0, 0.0, 0.30, -0.90]);
    }

    #[test]
    fn test_continuous_processing() {
        let mut decoder = make_decoder();
        let chunk = vec![0.0f32; 100];
        for _ in 0..10 {
            decoder.process_samples(&chunk);
        }
        let progress = decoder.progress();
        assert!(!progress.complete);
    }

    #[test]
    fn test_large_sample_buffer() {
        let mut decoder = make_decoder();
        let large_buffer = vec![0.0f32; 48000];
        decoder.process_samples(&large_buffer);
        assert!(!decoder.progress().complete);
    }

    #[test]
    fn test_sync_to_payload_handoff_basic() {
        use crate::mary::encoder::Encoder;
        let config = DspConfig::default_48k();
        let mut encoder = Encoder::new(config.clone());
        let mut decoder = Decoder::new(160, 10, config);
        let data = vec![0x12u8; 16];
        encoder.set_data(&data);
        let frame = encoder.encode_frame();
        assert!(frame.is_some(), "Should encode a frame");
        let frame_samples = frame.unwrap();
        decoder.process_samples(&frame_samples);
        let progress = decoder.progress();
        let _ = progress;
    }

    #[test]
    fn test_spread_factor_config_consistency() {
        let sf_sync = SYNC_SPREAD_FACTOR;
        let sf_payload = PAYLOAD_SPREAD_FACTOR;
        assert!(
            sf_sync > 0 && sf_payload > 0,
            "Spread factors must be positive"
        );
    }

    #[test]
    fn test_payload_walsh_index_detection() {
        use crate::mary::encoder::Encoder;
        let config = DspConfig::default_48k();
        let mut encoder = Encoder::new(config.clone());
        let mut decoder = Decoder::new(160, 10, config);
        let data = vec![0xABu8; 32];
        encoder.set_data(&data);
        let frame = encoder.encode_frame();
        assert!(frame.is_some(), "Should encode a frame");
        decoder.process_samples(&frame.unwrap());
        let progress = decoder.progress();
        let _ = progress;
    }

    #[test]
    fn test_preamble_walsh0_correlation() {
        use crate::mary::encoder::Encoder;
        let config = DspConfig::default_48k();
        let mut encoder = Encoder::new(config.clone());
        let mut decoder = Decoder::new(160, 10, config);
        let data = vec![0x34u8; 16];
        encoder.set_data(&data);
        let frame = encoder.encode_frame();
        assert!(frame.is_some(), "Should encode a frame");
        let frame_samples = frame.unwrap();
        decoder.process_samples(&frame_samples);
        let progress = decoder.progress();
        let _ = progress;
    }

    #[test]
    fn test_sync_word_bit_pattern() {
        let sync_word = crate::params::SYNC_WORD;
        let sync_word_bits = crate::params::SYNC_WORD_BITS;
        let sync_bits: Vec<u8> = (0..sync_word_bits)
            .map(|i| ((sync_word >> (sync_word_bits - 1 - i)) & 1) as u8)
            .collect();
        assert!(sync_bits.iter().all(|&b| b == 0 || b == 1));
        assert_eq!(sync_bits[0], 1);
    }

    #[test]
    fn test_encoder_decoder_small_data_roundtrip() {
        use crate::mary::encoder::Encoder;
        let config = DspConfig::default_48k();
        let mut encoder = Encoder::new(config.clone());
        let mut decoder = Decoder::new(160, 10, config);
        let data = vec![0xABu8; 16];
        encoder.set_data(&data);
        for _ in 0..20 {
            if let Some(frame) = encoder.encode_frame() {
                decoder.process_samples(&frame);
            }
            if decoder.recovered_data().is_some() {
                break;
            }
        }
        let recovered = decoder.recovered_data().expect("Should recover data");
        assert_eq!(
            &recovered[..data.len()],
            &data[..],
            "Recovered data mismatch"
        );
    }

    #[test]
    fn test_encoder_decoder_frame_structure() {
        use crate::mary::encoder::Encoder;
        let config = DspConfig::default_48k();
        let mut encoder = Encoder::new(config.clone());
        let mut decoder = Decoder::new(160, 10, config);
        let data = vec![0x12u8; 16];
        encoder.set_data(&data);
        let frame = encoder.encode_frame().unwrap();
        assert!(frame.len() > 2000, "Frame should be long enough");
        assert!(frame.iter().all(|&s| s.is_finite()));
        let progress = decoder.process_samples(&frame);
        assert!(
            !progress.complete,
            "Single frame should not complete decoding"
        );
    }

    #[test]
    fn test_encoder_decoder_preamble_detection() {
        use crate::mary::encoder::Encoder;
        let config = DspConfig::default_48k();
        let mut encoder = Encoder::new(config.clone());
        let mut decoder = Decoder::new(160, 10, config);
        let data = vec![0x34u8; 16];
        encoder.set_data(&data);
        let frame = encoder.encode_frame().unwrap();
        let progress = decoder.process_samples(&frame);
        let _ = progress.received_packets;
    }

    #[test]
    fn test_sync_ground_truth_alignment() {
        use crate::mary::modulator::Modulator;
        let config = DspConfig::default_48k();
        let spc = config.proc_samples_per_chip(); // 3

        // 1. 各セクションのサンプル数を送信レート(48k)で実測
        let mut modulator = Modulator::new(config.clone());
        let mut preamble_part = Vec::new();
        modulator.generate_preamble(&mut preamble_part);
        let preamble_len_48k = preamble_part.len();

        // 信号全体を生成 (新しいインスタンスで)
        let mut modulator2 = Modulator::new(config.clone());
        let data = vec![0x55u8; 16];
        let mut frame_48k = Vec::new();
        modulator2.encode_frame(&data, &mut frame_48k);

        println!("[GT] Preamble len (48k): {}", preamble_len_48k);

        // 2. Decoder の前処理パイプラインのみ通過させる
        // process_samples は同期確定後に内部バッファを消費するため、
        // GT検証では detect 前の生データを直接使う。
        let mut decoder = Decoder::new(160, 10, config.clone());

        // 内部バッファを直接操作してテスト用にデータを準備
        decoder.pipeline.mix_real_to_iq_zero_alloc(&frame_48k);

        // リサンプリング
        let mut i_resampled = Vec::new();
        let mut q_resampled = Vec::new();
        decoder
            .pipeline
            .resampler_i
            .process(&decoder.pipeline.mix_buffer_i, &mut i_resampled);
        decoder
            .pipeline
            .resampler_q
            .process(&decoder.pipeline.mix_buffer_q, &mut q_resampled);

        // RRCフィルタ（インプレースAPI使用）
        let mut i_filtered = i_resampled.clone();
        let mut q_filtered = q_resampled.clone();
        decoder
            .pipeline
            .rrc_filter_i
            .process_block_in_place(&mut i_filtered);
        decoder
            .pipeline
            .rrc_filter_q
            .process_block_in_place(&mut q_filtered);

        // 48k -> 24k (proc_rate) へのリサンプリング後の長さを算出
        let preamble_len_24k = preamble_len_48k / 2;

        // 3. SyncDetector を実行
        let (sync_opt, _) = decoder.sync_detector.detect(&i_filtered, &q_filtered, 0);

        if let Some(s) = sync_opt {
            // peak_sample_idx は同期語 0 番目の最初のチップの中央
            // 期待値 = (プリアンブル終了点) + (チップ 0 の半分) + (受信側遅延) + (送信側遅延)
            let rx_delay = decoder.pipeline.resampler_i.delay()
                + decoder.pipeline.rrc_filter_i.delay();
            let detector_delay = decoder.sync_detector.filter_delay();
            // 同期位置は「送信側RRC群遅延」ではなく、TX resampler の遅延寄与で整合する。
            let tx_rrc_bw = config.chip_rate * (1.0 + config.rrc_alpha) * 0.5;
            let tx_resampler = crate::common::resample::Resampler::new_with_cutoff(
                config.proc_sample_rate() as u32,
                config.sample_rate as u32,
                Some(tx_rrc_bw),
                Some(config.tx_resampler_taps),
            );
            let tx_delay_24k = (tx_resampler.delay() as f32
                / (config.sample_rate / config.proc_sample_rate()))
            .round() as usize;

            let expected_peak =
                preamble_len_24k + (spc / 2) + rx_delay + tx_delay_24k + detector_delay;
            let diff = s.peak_sample_idx as i32 - expected_peak as i32;
            println!(
                "[GT] Detected peak: {}, Expected peak: {}, Diff: {}, Delay: rx={}, tx_adj={}",
                s.peak_sample_idx,
                expected_peak,
                diff,
                rx_delay + detector_delay,
                tx_delay_24k
            );

            // 許容誤差範囲内で整合性を確認する
            assert!(
                diff.abs() <= 2,
                "Sync peak must match Ground Truth (diff={})",
                diff
            );
        } else {
            panic!(
                "Sync not detected in GT test! Frame len: {}, Preamble len: {}",
                frame_48k.len(),
                preamble_len_48k
            );
        }
    }

    #[test]
    fn test_sync_with_different_configurations() {
        let config_48k = DspConfig::default_48k();
        let mut decoder_48k = Decoder::new(160, 10, config_48k);
        let config_44k = DspConfig::default_44k();
        let mut decoder_44k = Decoder::new(160, 10, config_44k);
        let silence_48k = vec![0.0f32; 4800];
        let silence_44k = vec![0.0f32; 4410];
        decoder_48k.process_samples(&silence_48k);
        decoder_44k.process_samples(&silence_44k);
        assert!(!decoder_48k.progress().complete);
        assert!(!decoder_44k.progress().complete);
    }

    #[test]
    fn test_preamble_correlation_math() {
        let wdict = WalshDictionary::default_w16();
        let walsh0_sf15: Vec<i8> = wdict.w16[0].iter().take(15).copied().collect();
        let sf = 15;
        let signal_i: Vec<f32> = walsh0_sf15.iter().map(|&w| w as f32).collect();
        let signal_q: Vec<f32> = vec![0.0; sf];
        let mut correlation_i = 0.0f32;
        let mut correlation_q = 0.0f32;
        for idx in 0..sf {
            correlation_i += signal_i[idx] * walsh0_sf15[idx] as f32;
            correlation_q += signal_q[idx] * walsh0_sf15[idx] as f32;
        }
        let magnitude = (correlation_i * correlation_i + correlation_q * correlation_q).sqrt();
        assert!(magnitude > sf as f32 * 0.9);
    }

    #[test]
    fn test_sync_detection_with_noise() {
        use crate::mary::encoder::Encoder;
        use rand::prelude::*;
        let config = DspConfig::default_48k();
        let mut encoder = Encoder::new(config.clone());
        let mut decoder = Decoder::new(160, 10, config);
        let data = vec![0xCDu8; 16];
        encoder.set_data(&data);
        let frame = encoder.encode_frame().unwrap();
        let mut rng = thread_rng();
        let noisy_frame: Vec<f32> = frame
            .iter()
            .map(|&s| s + (rng.gen::<f32>() - 0.5) * 0.02)
            .collect();
        decoder.process_samples(&noisy_frame);
        assert!(!decoder.progress().complete);
    }

    #[test]
    fn test_sync_insufficient_buffer() {
        let mut decoder = make_decoder();
        let short_buffer = vec![0.0f32; 100];
        let progress = decoder.process_samples(&short_buffer);
        assert!(!progress.complete);
        assert_eq!(progress.received_packets, 0);
    }

    #[test]
    fn test_sync_maintenance_across_frames() {
        use crate::mary::encoder::Encoder;
        let config = DspConfig::default_48k();
        let mut encoder = Encoder::new(config.clone());
        let mut decoder = Decoder::new(160, 10, config.clone());
        let data = vec![0x12u8; 32];
        encoder.set_data(&data);
        let mut frame_count = 0;
        for _ in 0..5 {
            let frame = encoder.encode_frame();
            if let Some(samples) = frame {
                decoder.process_samples(&samples);
                frame_count += 1;
                let progress = decoder.progress();
                if progress.complete {
                    break;
                }
            }
        }
        assert!(frame_count >= 1);
    }

    #[test]
    fn test_preamble_correlation_with_perfect_signal() {
        use crate::mary::encoder::Encoder;
        let config = DspConfig::default_48k();
        let mut encoder = Encoder::new(config.clone());
        let mut decoder = Decoder::new(160, 10, config);
        let data = vec![0xABu8; 16];
        encoder.set_data(&data);
        let frame = encoder.encode_frame().unwrap();
        decoder.process_samples(&frame);
        assert!(decoder.stats.stats_total_samples >= frame.len());
    }

    #[test]
    fn test_preamble_correlation_with_noise_only() {
        let config = DspConfig::default_48k();
        let mut decoder = Decoder::new(160, 10, config);
        let noise: Vec<f32> = (0..5000)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.1)
            .collect();
        decoder.process_samples(&noise);
        assert_eq!(decoder.progress().received_packets, 0);
        assert!(!decoder.progress().complete);
    }

    #[test]
    fn test_preamble_last_symbol_inversion_pattern() {
        use crate::mary::encoder::Encoder;
        let config = DspConfig::default_48k();
        let mut encoder = Encoder::new(config.clone());
        let mut decoder = Decoder::new(160, 10, config);
        let data = vec![0x12u8; 16];
        encoder.set_data(&data);
        let frame = encoder.encode_frame().unwrap();
        decoder.process_samples(&frame);
        assert!(decoder.stats.stats_total_samples >= frame.len());
    }

    #[test]
    fn test_encoder_decoder_basic_functionality() {
        use crate::mary::encoder::Encoder;
        let config = DspConfig::default_48k();
        let mut encoder = Encoder::new(config.clone());
        let data = vec![0x12u8; 16];
        encoder.set_data(&data);
        let mut decoder = Decoder::new(160, 10, config);
        let mut total_frames = 0;
        for _ in 0..30 {
            let frame = encoder.encode_frame();
            if let Some(samples) = frame {
                decoder.process_samples(&samples);
                total_frames += 1;
                if decoder.recovered_data().is_some() {
                    break;
                }
            }
        }
        let recovered = decoder.recovered_data().expect("Should recover data");
        assert_eq!(
            &recovered[..data.len()],
            &data[..],
            "Recovered data mismatch after {} frames",
            total_frames
        );
    }

    fn apply_clock_drift_ppm(input: &[f32], ppm: f32) -> Vec<f32> {
        if input.is_empty() || ppm.abs() < 1.0 {
            return input.to_vec();
        }
        let out_len = input.len();
        let mut out = Vec::with_capacity(out_len);
        let mut time = 0.0f32;
        let time_step = 1.0 + ppm / 1_000_000.0;
        for _ in 0..out_len {
            let i0 = time.floor() as usize;
            let frac = (time - i0 as f32).clamp(0.0, 1.0);
            if i0 + 1 < input.len() {
                let a = input[i0];
                let b = input[i0 + 1];
                out.push(a + (b - a) * frac);
            } else if i0 < input.len() {
                out.push(input[i0]);
            } else {
                out.push(input[input.len() - 1]);
            }
            time += time_step;
        }
        out
    }

    #[test]
    fn test_decoder_incremental_chunk_processing() {
        use crate::mary::encoder::Encoder;
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 127;
        let mut encoder = Encoder::new(config.clone());
        let mut decoder = Decoder::new(160, 10, config);
        decoder.config.packets_per_burst = 5;
        let data = vec![0x12u8; 80];
        encoder.set_data(&data);
        let mut signal = Vec::new();
        let mut packets = Vec::new();
        for _ in 0..5 {
            let p = encoder.fountain_encoder_mut().unwrap().next_packet();
            packets.push(p);
        }
        signal.extend(encoder.encode_burst(&packets));
        signal.extend(encoder.flush());
        signal.extend(vec![0.0; 10000]);
        for chunk in signal.chunks(100) {
            decoder.process_samples(chunk);
            if decoder.recovered_data().is_some() {
                break;
            }
        }
        let recovered = decoder
            .recovered_data()
            .expect("Decoder should recover data even with small incremental chunks");
        assert_eq!(&recovered[..data.len()], &data[..]);
    }

    #[test]
    fn test_decoder_tracking_tolerates_clock_drift_ppm() {
        use crate::mary::encoder::Encoder;
        use std::sync::{Arc, Mutex};
        use std::time::Instant;
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 127;
        let mut encoder = Encoder::new(config.clone());
        let mut decoder = Decoder::new(640, 40, config);
        decoder.config.packets_per_burst = 40;
        let data = vec![0x55u8; 640];
        encoder.set_data(&data);
        let mut signal = Vec::new();
        let mut expected_fec_bits_list = Vec::new();
        let mut packets = Vec::new();
        for _ in 0..40 {
            let p = encoder.fountain_encoder_mut().unwrap().next_packet();
            let seq = (p.seq % (u32::from(u16::MAX) + 1)) as u16;
            let pkt = Packet::new(seq, 40, &p.data);
            let bits = crate::coding::fec::bytes_to_bits(&pkt.serialize());
            let fec_bits = crate::coding::fec::encode(&bits);
            expected_fec_bits_list.push(fec_bits);
            packets.push(p);
        }
        signal.extend(encoder.encode_burst(&packets));
        signal.extend(encoder.flush());
        signal.extend(vec![0.0; 10000]);
        let physical_duration_sec = signal.len() as f32 / 48000.0;
        let received_llrs = Arc::new(Mutex::new(Vec::new()));
        let received_llrs_clone = Arc::clone(&received_llrs);
        decoder.llr_callback = Some(Box::new(move |llrs: &[f32]| {
            received_llrs_clone.lock().unwrap().push(llrs.to_vec());
        }));
        let drifted = apply_clock_drift_ppm(&signal, 200.0);
        let start_time = Instant::now();
        for chunk in drifted.chunks(2048) {
            decoder.process_samples(chunk);
            if decoder.recovered_data().is_some() {
                break;
            }
        }
        let processing_duration = start_time.elapsed();
        let realtime_ratio = processing_duration.as_secs_f32() / physical_duration_sec;
        println!("--- Performance & BER Trend (Clock Drift 200ppm) ---");
        println!(
            "Processing Time: {:?}, Physical Time: {:.3}s, Ratio: {:.3}",
            processing_duration, physical_duration_sec, realtime_ratio
        );
        let llrs = received_llrs.lock().unwrap();
        for (i, p_llrs) in llrs.iter().enumerate() {
            let expected = &expected_fec_bits_list[i];
            let mut errors = 0;
            for (j, &bit) in expected.iter().enumerate() {
                let llr = p_llrs[j];
                if (bit == 0 && llr <= 0.0) || (bit == 1 && llr >= 0.0) {
                    errors += 1;
                }
            }
            if i % 10 == 0 || i == llrs.len() - 1 {
                println!("Packet {}: {} errors / {} bits", i, errors, expected.len());
            }
        }
        assert!(
            realtime_ratio < 1.0,
            "Processing too slow: ratio={}",
            realtime_ratio
        );
        let recovered = decoder
            .recovered_data()
            .expect("Decoder should recover data under clock drift (needs tracking)");
        assert_eq!(&recovered[..data.len()], &data[..]);
    }

    struct CarrierOffsetTrial {
        offset_hz: f32,
        realtime_ratio: f32,
        ber_errors: Vec<(usize, usize, usize)>,
        received_packets: usize,
        needed_packets: usize,
        crc_error_packets: usize,
        parse_error_packets: usize,
        recovered: bool,
    }

    fn run_carrier_offset_trial(offset_hz: f32) -> CarrierOffsetTrial {
        use crate::mary::encoder::Encoder;
        use std::sync::{Arc, Mutex};
        use std::time::Instant;
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 127;
        let mut encoder = Encoder::new(config.clone());
        let mut rx_config = config.clone();
        rx_config.carrier_freq += offset_hz;
        let mut decoder = Decoder::new(640, 40, rx_config);
        decoder.config.packets_per_burst = 40;
        let data = vec![0xAAu8; 640];
        encoder.set_data(&data);
        let mut signal = Vec::new();
        let mut expected_fec_bits_list = Vec::new();
        let mut packets = Vec::new();
        for _ in 0..40 {
            let p = encoder.fountain_encoder_mut().unwrap().next_packet();
            let seq = (p.seq % (u32::from(u16::MAX) + 1)) as u16;
            let pkt = Packet::new(seq, 40, &p.data);
            let bits = crate::coding::fec::bytes_to_bits(&pkt.serialize());
            let fec_bits = crate::coding::fec::encode(&bits);
            expected_fec_bits_list.push(fec_bits);
            packets.push(p);
        }
        signal.extend(encoder.encode_burst(&packets));
        signal.extend(encoder.flush());
        signal.extend(vec![0.0; 10000]);
        let physical_duration_sec = signal.len() as f32 / 48000.0;
        let received_llrs = Arc::new(Mutex::new(Vec::new()));
        let received_llrs_clone = Arc::clone(&received_llrs);
        decoder.llr_callback = Some(Box::new(move |llrs: &[f32]| {
            received_llrs_clone.lock().unwrap().push(llrs.to_vec());
        }));
        let start_time = Instant::now();
        for chunk in signal.chunks(2048) {
            decoder.process_samples(chunk);
            if decoder.recovered_data().is_some() {
                break;
            }
        }
        let processing_duration = start_time.elapsed();
        let realtime_ratio = processing_duration.as_secs_f32() / physical_duration_sec;
        let mut ber_errors = Vec::new();
        let llrs = received_llrs.lock().unwrap();
        for (i, p_llrs) in llrs.iter().enumerate() {
            let expected = &expected_fec_bits_list[i];
            let mut errors = 0;
            for (j, &bit) in expected.iter().enumerate() {
                let llr = p_llrs[j];
                if (bit == 0 && llr <= 0.0) || (bit == 1 && llr >= 0.0) {
                    errors += 1;
                }
            }
            ber_errors.push((i, errors, expected.len()));
        }
        let progress = decoder.progress();
        let recovered = decoder
            .recovered_data()
            .map(|recovered| &recovered[..data.len()] == data.as_slice())
            .unwrap_or(false);
        CarrierOffsetTrial {
            offset_hz,
            realtime_ratio,
            ber_errors,
            received_packets: progress.received_packets,
            needed_packets: progress.needed_packets,
            crc_error_packets: progress.crc_error_packets,
            parse_error_packets: progress.parse_error_packets,
            recovered,
        }
    }

    #[test]
    fn test_decoder_tracking_tolerates_carrier_offset() {
        let result = run_carrier_offset_trial(20.0);
        println!(
            "--- Performance & BER Trend (Carrier Offset {:.1}Hz) ---",
            result.offset_hz
        );
        println!("Ratio: {:.3}", result.realtime_ratio);
        for (i, errors, bits) in result.ber_errors.iter() {
            if i % 10 == 0 || *i + 1 == result.ber_errors.len() {
                println!("Packet {}: {} errors / {} bits", i, errors, bits);
            }
        }
        println!(
            "Test finished: received={}, needed={}, crc_errors={}, parse_errors={}",
            result.received_packets,
            result.needed_packets,
            result.crc_error_packets,
            result.parse_error_packets
        );
        assert!(
            result.realtime_ratio < 1.0,
            "Processing too slow: ratio={}",
            result.realtime_ratio
        );
        assert!(
            result.recovered,
            "Decoder should recover data under carrier offset (needs tracking)"
        );
    }

    /// 回帰テスト:
    /// Decoder経路では prev_phase が単位振幅に正規化される。
    /// このとき diff の振幅は |corr| 次元だが、denom に |corr|^2 (max_energy) を使うと
    /// DQPSK LLR が過小スケールになる。
    #[test]
    fn test_decoder_dqpsk_llr_scale_regression_with_normalized_prev_phase() {
        let mut decoder = make_decoder();

        // Decoder の Sync→Payload ハンドオーバーと同じく prev_phase は正規化される。
        decoder.demodulator.set_prev_phase(Complex32::new(1.0, 0.0));

        // 理想条件: sf=16 の完全相関（振幅16, 位相0）
        let best_corr = Complex32::new(16.0, 0.0);
        let phase_ref = Complex32::new(1.0, 0.0);
        let on_rot = best_corr * phase_ref.conj();
        let diff = on_rot * decoder.demodulator.prev_phase().conj();
        let max_energy = best_corr.norm_sqr(); // 256

        // Walsh LLR は max* 補正により 1.0 より僅かに小さくなる。
        // bit=0 側: [256, 0, ...] / bit=1 側: [0, 0, ...] の max* 差分を期待値とする。
        let mut energies = [0.0f32; 16];
        energies[0] = max_energy;
        let walsh_llr = decoder.demodulator.walsh_llr(&energies, max_energy);
        let walsh_avg_abs = walsh_llr.iter().map(|v| v.abs()).sum::<f32>() / walsh_llr.len() as f32;
        let max_star = |a: f32, b: f32| -> f32 { a.max(b) + (-(a - b).abs()).exp().ln_1p() };
        let expected_walsh_abs = (max_star(max_energy, 0.0) - max_star(0.0, 0.0)) / max_energy;
        assert!(
            (walsh_avg_abs - expected_walsh_abs).abs() < 1e-6,
            "Walsh LLR scale mismatch: got {}, expected {}",
            walsh_avg_abs,
            expected_walsh_abs
        );

        // 旧式（誤り）: diff は振幅次元なのにエネルギー次元で割ると過小化する。
        let dqpsk_llr_old = decoder.demodulator.dqpsk_llr(diff, max_energy);
        let dqpsk_old_avg_abs =
            dqpsk_llr_old.iter().map(|v| v.abs()).sum::<f32>() / dqpsk_llr_old.len() as f32;
        assert!(
            dqpsk_old_avg_abs < 0.2,
            "Old DQPSK normalization should be too small: avg_abs={} (llr={:?})",
            dqpsk_old_avg_abs,
            dqpsk_llr_old
        );

        // 新式（修正）: diff の振幅次元に合わせて |on_rot| で正規化する。
        let dqpsk_llr = decoder.demodulator.dqpsk_llr(diff, on_rot.norm());
        let dqpsk_avg_abs = dqpsk_llr.iter().map(|v| v.abs()).sum::<f32>() / dqpsk_llr.len() as f32;
        assert!(
            dqpsk_avg_abs > 0.5,
            "Fixed DQPSK normalization should keep enough soft information: avg_abs={} (llr={:?})",
            dqpsk_avg_abs,
            dqpsk_llr
        );
    }

    /// 回帰テスト:
    /// 負方向の timing/sample shift により先頭側へはみ出す場合は
    /// 相関計算を行わず None を返すべき。
    #[test]
    fn test_despread_symbol_with_timing_rejects_negative_index_underflow() {
        let mut decoder = make_decoder();
        let spc = decoder.config.proc_samples_per_chip().max(1);

        // 上側境界には十分余裕を持たせる
        decoder.equalization.equalized_buffer_mut().extend_from_slice(&vec![Complex32::new(0.0, 0.0); 256]);

        // 先頭付近のシンボルで、trackingの下限方向へ寄せる
        let symbol_start = spc;
        let timing_offset = -(spc as f32) * tracking::TRACKING_TIMING_LIMIT_CHIP;
        let sample_shift = -(spc as f32 * TRACKING_EARLY_LATE_DELTA_CHIP).max(1.0);

        let corrs = decoder.despread_symbol_with_timing(symbol_start, timing_offset, sample_shift);
        assert!(
            corrs.is_none(),
            "despread should reject negative sample index underflow: symbol_start={}, timing_offset={}, sample_shift={}",
            symbol_start,
            timing_offset,
            sample_shift
        );
    }

    /// 再現テスト:
    /// パケット復調中に despread が underflow で失敗した場合は、
    /// フレームを破棄して Searching へ戻るべき。
    ///
    /// 現状は `processed=false` で `handle_decoding()` が早期returnし、
    /// state が EqualizedDecoding のまま据え置かれるため、以降の再同期へ進めず固着する。
    #[test]
    fn test_decoder_should_fallback_to_searching_on_packet_despread_underflow() {
        let mut decoder = make_decoder();
        let spc = decoder.config.proc_samples_per_chip().max(1);
        let expected_symbols = interleaver_config::mary_symbols();
        let packet_samples = expected_symbols * PAYLOAD_SPREAD_FACTOR * spc;

        decoder.state = DecoderState::EqualizedDecoding;
        decoder.config.packets_per_burst = 2;
        decoder.frame_session = Some(FrameSession {
            tracking_state: TrackingState {
                phase_ref: Complex32::new(1.0, 0.0),
                phase_rate: 0.0,
                timing_offset: -(spc as f32) * tracking::TRACKING_TIMING_LIMIT_CHIP,
                timing_rate: 0.0,
                phase_gate_enabled: false,
            },
            packets_processed_in_burst: 1,
            consecutive_crc_errors: 0,
            pending_warmup_samples: 0,
            pending_warmup_input_samples: 0,
            remaining_samples_in_frame: packet_samples as isize,
        });
        decoder.equalization.clear_equalized_buffer();
        decoder.equalization.equalized_buffer_mut().extend_from_slice(&vec![Complex32::new(0.0, 0.0); packet_samples + spc + 8]);

        let advanced = decoder.handle_decoding();
        assert!(!advanced, "underflow path should stop current decode step");
        assert_eq!(
            decoder.state,
            DecoderState::Searching,
            "decoder should fallback to Searching after packet despread underflow"
        );
    }

    #[test]
    #[ignore = "diagnostic sweep for carrier tracking limit"]
    fn test_decoder_tracking_carrier_offset_sweep() {
        let offsets_hz = [
            0.0f32, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0,
        ];
        let mut max_pass_hz = 0.0f32;
        for &offset_hz in &offsets_hz {
            let result = run_carrier_offset_trial(offset_hz);
            println!(
                "[sweep] offset={:.1}Hz recovered={} ratio={:.3} received={}/{} crc_err={}",
                offset_hz,
                result.recovered,
                result.realtime_ratio,
                result.received_packets,
                result.needed_packets,
                result.crc_error_packets
            );
            if result.recovered {
                max_pass_hz = offset_hz;
            }
        }
        println!("[sweep] max recovered offset = {:.1}Hz", max_pass_hz);
    }

    #[test]
    #[ignore = "diagnostic estimate for required phase step/gain margin"]
    fn test_decoder_tracking_phase_gain_estimation() {
        let config = DspConfig::default_48k();
        let spc = config.proc_samples_per_chip().max(1) as f32;
        let sym_samples = PAYLOAD_SPREAD_FACTOR as f32 * spc;
        let sym_period_sec = sym_samples / config.proc_sample_rate();
        for offset_hz in [5.0f32, 10.0, 15.0, 20.0, 25.0, 30.0] {
            let phase_step_rad = 2.0 * std::f32::consts::PI * offset_hz * sym_period_sec;
            let margin = tracking::TRACKING_PHASE_RATE_LIMIT_RAD / phase_step_rad.max(1e-6);
            println!(
                "[gain-est] offset={:.1}Hz step={:.4}rad/sym clamp_margin={:.2}x",
                offset_hz, phase_step_rad, margin
            );
        }
    }

    #[test]
    fn test_decoder_continuous_multiple_bursts() {
        use crate::mary::encoder::Encoder;
        let config = DspConfig::default_48k();
        let mut encoder = Encoder::new(config.clone());

        let data_size = 160;
        let fountain_k = 10;
        let mut decoder = Decoder::new(data_size, fountain_k, config.clone());
        decoder.config.packets_per_burst = 1; // 1パケットごとに同期が必要な設定

        let data = vec![0x55u8; data_size];
        encoder.set_data(&data);

        let mut total_signal = Vec::new();
        // 20フレーム分生成（冗長性を持たせる）
        for _ in 0..20 {
            if let Some(frame) = encoder.encode_frame() {
                total_signal.extend(frame);
            }
        }
        total_signal.extend(encoder.flush());
        total_signal.extend(vec![0.0; 10000]);

        for chunk in total_signal.chunks(32768) {
            decoder.process_samples(chunk);
            if decoder.recovered_data().is_some() {
                break;
            }
        }

        let progress = decoder.progress();
        println!("Final received packets: {}", progress.received_packets);

        assert!(
            progress.complete,
            "Should recover data from continuous bursts. Received {}/{} packets",
            progress.received_packets, fountain_k
        );
        let recovered = decoder
            .recovered_data()
            .expect("Should have recovered data");
        assert_eq!(&recovered[..data.len()], &data[..]);
    }

    #[test]
    fn test_fde_multipath_recovery() {
        use crate::mary::encoder::Encoder;
        use rand::{Rng, SeedableRng};
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 127;

        let mut encoder = Encoder::new(config.clone());

        let data_size = 32; // 2 packets
        let fountain_k = 2;
        let mut decoder = Decoder::new(data_size, fountain_k, config.clone());
        decoder.config.packets_per_burst = 1;

        let data = vec![0x42u8; data_size];
        encoder.set_data(&data);

        // 1. 信号生成
        let mut original_signal = Vec::new();
        for _ in 0..5 {
            if let Some(frame) = encoder.encode_frame() {
                original_signal.extend(frame);
            }
        }
        original_signal.extend(encoder.flush());

        // 2. 2パス・マルチパスチャネルの適用: y(t) = x(t) + 0.7 * x(t - tau)
        // tau = 4 chips.
        let spc = config.proc_samples_per_chip();
        let delay_samples = 4 * spc * (config.sample_rate / config.proc_sample_rate()) as usize;
        let mut multipath_signal = original_signal.clone();
        let alpha = 0.7f32;
        for t in delay_samples..original_signal.len() {
            multipath_signal[t] += original_signal[t - delay_samples] * alpha;
        }

        // 押し出し用の無音を追加
        multipath_signal.extend(vec![0.0; 10000]);

        // 3. 軽微なノイズ重畳 (SNR ~25dB)
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED_FDE);
        for s in multipath_signal.iter_mut() {
            *s += (rng.gen::<f32>() - 0.5) * 0.05;
        }

        // --- Step A: 等化ありでの復元確認 ---
        decoder.process_samples(&multipath_signal);
        let progress = decoder.progress();

        assert!(
            progress.complete,
            "FDE should recover data under strong multipath. Received {}/{} packets",
            progress.received_packets, fountain_k
        );
        let recovered = decoder
            .recovered_data()
            .expect("Should have recovered data");
        assert_eq!(
            &recovered[..data.len()],
            &data[..],
            "Recovered data mismatch with FDE"
        );

        // --- Step B: 等化なし対照 ---
        let mut decoder_no_fde = Decoder::new(data_size, fountain_k, config.clone());
        decoder_no_fde.config.packets_per_burst = 1;
        decoder_no_fde.set_fde_enabled(false);

        decoder_no_fde.process_samples(&multipath_signal);

        let progress_no_fde = decoder_no_fde.progress();
        assert!(
            progress.received_packets >= progress_no_fde.received_packets,
            "FDE should not underperform no-FDE in received packets: with_fde={}, no_fde={}",
            progress.received_packets,
            progress_no_fde.received_packets
        );
        assert!(
            progress.crc_error_packets <= progress_no_fde.crc_error_packets,
            "FDE should not increase CRC errors: with_fde={}, no_fde={}",
            progress.crc_error_packets,
            progress_no_fde.crc_error_packets
        );
    }

    #[test]
    fn test_fde_auto_disabled_keeps_fde_path() {
        use crate::mary::encoder::Encoder;
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 127;
        let mut encoder = Encoder::new(config.clone());
        let mut decoder = Decoder::new(160, 10, config);
        decoder.config.packets_per_burst = 1;
        decoder.set_fde_enabled(true);
        decoder.set_fde_auto_path_select(false);
        decoder.set_fde_mmse_settings(15.0, 1.0, 10.0, None);

        let data = vec![0x23u8; 32];
        encoder.set_data(&data);
        let mut signal = encoder.encode_frame().expect("frame");
        signal.extend(vec![0.0; 8000]);
        decoder.process_samples(&signal);

        assert!(
            decoder.equalization.current_frame_use_fde(),
            "AUTO無効時は常にFDE経路を使うべき"
        );
    }

    #[test]
    fn test_fde_auto_selects_raw_on_flat_channel_with_large_lambda_floor() {
        use crate::mary::encoder::Encoder;
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 127;
        let mut encoder = Encoder::new(config.clone());
        let mut decoder = Decoder::new(160, 10, config);
        decoder.config.packets_per_burst = 1;
        decoder.set_fde_enabled(true);
        decoder.set_fde_auto_path_select(true);
        decoder.set_fde_mmse_settings(15.0, 1.0, 10.0, None);

        let data = vec![0x45u8; 32];
        encoder.set_data(&data);
        let mut signal = encoder.encode_frame().expect("frame");
        signal.extend(vec![0.0; 8000]);
        decoder.process_samples(&signal);

        assert!(
            !decoder.equalization.current_frame_use_fde(),
            "平坦チャネルかつ強正則化ではAUTOはRAWを選ぶべき"
        );
    }
}
