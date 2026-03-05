//! MaryDQPSKデコーダ
//!
//! # 復調パイプライン
//! 1. プリアンブル検出（Walsh[0]、DBPSK）
//! 2. Sync Word検出
//! 3. 16系列並列相復調によるMaryDQPSK復調
//! 4. Max-Log-MAP LLR計算
//! 5. Fountainデコーディング

use crate::coding::fec;
use crate::coding::fountain::{FountainDecoder, FountainPacket, FountainParams};
use crate::coding::interleaver::BlockInterleaver;
use crate::coding::scrambler::Scrambler;
use crate::common::equalization::{FrequencyDomainEqualizer, MmseSettings};
use crate::common::nco::Nco;
use crate::common::resample::Resampler;
use crate::common::rrc_filter::RrcFilter;
use crate::frame::packet::Packet;
use crate::mary::demodulator::Demodulator;
use crate::mary::interleaver_config;
use crate::mary::sync::{ChannelQualityEstimate, MarySyncDetector, SyncResult};
use crate::params::PAYLOAD_SIZE;
use crate::DspConfig;
use num_complex::Complex32;

const TRACKING_TIMING_PROP_GAIN: f32 = 0.18;
const TRACKING_TIMING_RATE_GAIN: f32 = 0.01;
// 位相追従ゲイン設計メモ (default_48k):
// - proc_fs = chip_rate * INTERNAL_SPC = 8k * 3 = 24kHz
// - payload 1symbol = PAYLOAD_SPREAD_FACTOR * spc = 16 * 3 = 48sample = 2.0ms
// - 目標 CFO 20Hz の位相回転は Δphi = 2π f Ts ≈ 2π*20*0.002 = 0.251rad/symbol
// - 目安:
//   - TRACKING_PHASE_PROP_GAIN は 0.25..0.45 程度
//   - TRACKING_PHASE_FREQ_GAIN は 0.03..0.08 程度
//     (|phase_err|≈0.25rad 時に 8..30 symbol で 0.25rad/symbol へ到達できる帯域)
const TRACKING_PHASE_PROP_GAIN: f32 = 0.35;
const TRACKING_PHASE_FREQ_GAIN: f32 = 0.05;
const TRACKING_TIMING_LIMIT_CHIP: f32 = 2.0;
const TRACKING_TIMING_RATE_LIMIT_CHIP: f32 = 0.25;
const TRACKING_EARLY_LATE_DELTA_CHIP: f32 = 0.5;
const TRACKING_PHASE_RATE_LIMIT_RAD: f32 = 2.6;
const TRACKING_PHASE_STEP_CLAMP: f32 = 2.8;
const SYNC_SPREAD_FACTOR: usize = crate::params::SPREAD_FACTOR;
const PAYLOAD_SPREAD_FACTOR: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CirNormalizationMode {
    None,
    UnitEnergy,
    Peak,
}

/// デコード進捗
#[derive(Debug, Clone)]
pub struct DecodeProgress {
    pub received_packets: usize,
    pub needed_packets: usize,
    pub rank_packets: usize,
    pub stalled_packets: usize,
    pub dependent_packets: usize,
    pub duplicate_packets: usize,
    pub crc_error_packets: usize,
    pub parse_error_packets: usize,
    pub invalid_neighbor_packets: usize,
    pub last_packet_seq: i32,
    pub last_rank_up_seq: i32,
    pub progress: f32,
    pub complete: bool,
    pub fde_selected_frames: usize,
    pub raw_selected_frames: usize,
    pub last_path_used: i32,
    pub last_pred_mse_fde: f32,
    pub last_pred_mse_raw: f32,
    pub last_est_snr_db: f32,
    pub basis_matrix: Vec<u8>,
}

/// LLR観測用コールバック型
pub type LlrCallback = Box<dyn FnMut(&[f32]) + Send>;

/// デコーダの状態
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecoderState {
    Searching,
    EqualizedDecoding,
}

/// MaryDQPSKデコーダ
pub struct Decoder {
    pub config: DspConfig,
    resampler_i: Resampler,
    resampler_q: Resampler,
    rrc_filter_i: RrcFilter,
    rrc_filter_q: RrcFilter,
    sample_buffer_i: Vec<f32>,
    sample_buffer_q: Vec<f32>,
    equalizer: Option<FrequencyDomainEqualizer>,
    equalized_buffer: Vec<Complex32>,
    equalizer_input_offset: usize,
    demodulator: Demodulator,
    fountain_decoder: FountainDecoder,
    pub recovered_data: Option<Vec<u8>>,
    lo_nco: Nco,
    sync_detector: MarySyncDetector,

    // 同期・追従状態
    state: DecoderState,
    last_search_idx: usize,
    current_sync: Option<SyncResult>,
    tracking_state: Option<TrackingState>,
    packets_processed_in_burst: usize,
    consecutive_crc_errors: usize,
    last_packet_seq: Option<u32>,
    last_rank_up_seq: Option<u32>,
    pending_warmup_samples: usize,
    pending_warmup_input_samples: usize,
    remaining_samples_in_frame: isize,
    fde_auto_path_select: bool,
    current_frame_use_fde: bool,
    fde_selected_frames: usize,
    raw_selected_frames: usize,
    last_path_used: i32,
    last_pred_mse_fde: f32,
    last_pred_mse_raw: f32,
    last_est_snr_db: f32,
    fde_mmse_settings: MmseSettings,
    cir_normalization_mode: CirNormalizationMode,
    cir_tap_threshold_alpha: f32,

    // 統計
    pub received_packets: usize,
    pub stalled_packets: usize,
    pub dependent_packets: usize,
    pub duplicate_packets: usize,
    pub crc_error_packets: usize,
    pub parse_error_packets: usize,
    pub invalid_neighbor_packets: usize,
    pub stats_total_samples: usize,

    /// デバッグ観測用コールバック: デインターリーブ・デスクランブル後のLLRをパススルーする
    pub llr_callback: Option<LlrCallback>,

    // ゼロアロケーション用バッファプール
    pub(crate) mix_buffer_i: Vec<f32>,
    pub(crate) mix_buffer_q: Vec<f32>,
    resample_buffer_i: Vec<f32>,
    resample_buffer_q: Vec<f32>,
    rrc_filtered_i: Vec<f32>,
    rrc_filtered_q: Vec<f32>,
    cir_buffer: Vec<Complex32>,
    complex_buffer: Vec<Complex32>,
    packet_llrs_buffer: Vec<f32>,
    deinterleave_buffer: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
struct TrackingState {
    phase_ref: Complex32,
    phase_rate: f32,
    timing_offset: f32,
    timing_rate: f32,
}

impl Decoder {
    fn default_fde_fft_size(dsp_config: &DspConfig) -> usize {
        let spc = dsp_config.proc_samples_per_chip();
        let cir_samples = dsp_config.preamble_sf * spc;
        (cir_samples * 2).next_power_of_two().max(1024)
    }

    fn build_default_equalizer(dsp_config: &DspConfig) -> FrequencyDomainEqualizer {
        let spc = dsp_config.proc_samples_per_chip();
        let cir_samples = dsp_config.preamble_sf * spc;
        let fft_size = Self::default_fde_fft_size(dsp_config);
        let mut initial_cir = vec![Complex32::new(0.0, 0.0); cir_samples];
        initial_cir[0] = Complex32::new(1.0, 0.0);
        FrequencyDomainEqualizer::new(&initial_cir, fft_size, 15.0)
    }

    /// 新しいデコーダを作成する
    pub fn new(_data_size: usize, fountain_k: usize, dsp_config: DspConfig) -> Self {
        let proc_sample_rate = dsp_config.proc_sample_rate();
        let lo_nco = Nco::new(-dsp_config.carrier_freq, dsp_config.sample_rate);

        let rrc_bw = dsp_config.chip_rate * (1.0 + dsp_config.rrc_alpha) * 0.5;
        let cutoff = Some(rrc_bw);

        let tc = MarySyncDetector::THRESHOLD_COARSE_DEFAULT;
        let tf = MarySyncDetector::THRESHOLD_FINE_DEFAULT;

        // ゼロアロケーションバッファの初期化
        let spc = dsp_config.proc_samples_per_chip();
        let cir_buffer_size = dsp_config.preamble_sf * spc;

        Decoder {
            resampler_i: Resampler::new_with_cutoff(
                dsp_config.sample_rate as u32,
                proc_sample_rate as u32,
                cutoff,
            ),
            resampler_q: Resampler::new_with_cutoff(
                dsp_config.sample_rate as u32,
                proc_sample_rate as u32,
                cutoff,
            ),
            rrc_filter_i: RrcFilter::from_config(&dsp_config),
            rrc_filter_q: RrcFilter::from_config(&dsp_config),
            sample_buffer_i: Vec::new(),
            sample_buffer_q: Vec::new(),
            demodulator: Demodulator::new(),
            fountain_decoder: FountainDecoder::new(FountainParams::new(fountain_k, PAYLOAD_SIZE)),
            recovered_data: None,
            sync_detector: MarySyncDetector::new(dsp_config.clone(), tc, tf),
            config: dsp_config.clone(),
            lo_nco,
            equalizer: Some(Self::build_default_equalizer(&dsp_config)),
            equalized_buffer: Vec::new(),
            equalizer_input_offset: 0,
            state: DecoderState::Searching,
            last_search_idx: 0,
            current_sync: None,
            tracking_state: None,
            packets_processed_in_burst: 0,
            consecutive_crc_errors: 0,
            last_packet_seq: None,
            last_rank_up_seq: None,
            pending_warmup_samples: 0,
            pending_warmup_input_samples: 0,
            remaining_samples_in_frame: 0,
            fde_auto_path_select: false,
            current_frame_use_fde: true,
            fde_selected_frames: 0,
            raw_selected_frames: 0,
            last_path_used: -1,
            last_pred_mse_fde: f32::NAN,
            last_pred_mse_raw: f32::NAN,
            last_est_snr_db: f32::NAN,
            fde_mmse_settings: MmseSettings::default(),
            cir_normalization_mode: CirNormalizationMode::None,
            cir_tap_threshold_alpha: 0.0,
            received_packets: 0,
            stalled_packets: 0,
            dependent_packets: 0,
            duplicate_packets: 0,
            crc_error_packets: 0,
            parse_error_packets: 0,
            invalid_neighbor_packets: 0,
            llr_callback: None,
            stats_total_samples: 0,
            // ゼロアロケーションバッファ初期化
            mix_buffer_i: Vec::with_capacity(4096),
            mix_buffer_q: Vec::with_capacity(4096),
            resample_buffer_i: Vec::with_capacity(6144),
            resample_buffer_q: Vec::with_capacity(6144),
            rrc_filtered_i: Vec::with_capacity(6144),
            rrc_filtered_q: Vec::with_capacity(6144),
            cir_buffer: vec![Complex32::new(0.0, 0.0); cir_buffer_size],
            complex_buffer: Vec::with_capacity(16_000),
            packet_llrs_buffer: Vec::with_capacity(interleaver_config::interleaved_bits()),
            deinterleave_buffer: Vec::with_capacity(interleaver_config::interleaved_bits()),
        }
    }

    pub fn set_fde_enabled(&mut self, enabled: bool) {
        if enabled {
            if self.equalizer.is_none() {
                self.equalizer = Some(Self::build_default_equalizer(&self.config));
            }
            self.current_frame_use_fde = true;
        } else {
            self.equalizer = None;
            self.current_frame_use_fde = false;
            self.fde_auto_path_select = false;
        }
        self.equalized_buffer.clear();
        self.equalizer_input_offset = 0;
    }

    pub fn set_fde_auto_path_select(&mut self, enabled: bool) {
        self.fde_auto_path_select = enabled;
        if !enabled {
            self.current_frame_use_fde = self.equalizer.is_some();
        }
    }

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

    pub fn set_cir_postprocess(
        &mut self,
        normalization_mode: CirNormalizationMode,
        tap_threshold_alpha: f32,
    ) {
        self.cir_normalization_mode = normalization_mode;
        self.cir_tap_threshold_alpha = tap_threshold_alpha.max(0.0);
    }

    pub fn process_samples(&mut self, samples: &[f32]) -> DecodeProgress {
        if self.recovered_data.is_some() {
            return self.progress();
        }
        self.stats_total_samples += samples.len();

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
        self.rrc_filtered_i.clear();
        self.rrc_filtered_q.clear();
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

        self.sample_buffer_i.extend_from_slice(&self.rrc_filtered_i);
        self.sample_buffer_q.extend_from_slice(&self.rrc_filtered_q);

        self.detect_and_process_frames()
    }

    fn mix_real_to_iq_zero_alloc(&mut self, samples: &[f32]) {
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

    fn handle_searching(&mut self) -> bool {
        let spc = self.config.proc_samples_per_chip().max(1);
        let sf_preamble = self.config.preamble_sf;
        let sf_sync = SYNC_SPREAD_FACTOR;
        let repeat = self.config.preamble_repeat;
        let sync_word_bits = self.config.sync_word_bits;

        let required_len = (sf_preamble * repeat + sf_sync * sync_word_bits) * spc;
        if self.sample_buffer_i.len() < self.last_search_idx + required_len {
            return false;
        }

        let (sync_opt, next_search_idx) = self.sync_detector.detect(
            &self.sample_buffer_i,
            &self.sample_buffer_q,
            self.last_search_idx,
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
                    &self.sample_buffer_i,
                    &self.sample_buffer_q,
                    preamble_start_idx,
                    cir_slice,
                    &mut chq,
                );
                Self::postprocess_cir(
                    cir_slice,
                    self.cir_normalization_mode,
                    self.cir_tap_threshold_alpha,
                );
            }
            let mut mmse = self.fde_mmse_settings;
            if let Some(snr_db) = chq.snr_db {
                mmse.snr_db = snr_db.clamp(-20.0, 40.0);
            }

            let mut use_fde_this_frame = self.equalizer.is_some();
            let mut pred_mse_fde = f32::NAN;
            let mut pred_mse_raw = f32::NAN;
            if self.equalizer.is_some() {
                let known_end_idx =
                    preamble_start_idx + self.sync_detector.known_interval_len_samples();
                let (mse_raw, mse_fde) = self.known_interval_path_mse(
                    preamble_start_idx,
                    known_end_idx,
                    cir_len,
                    mmse,
                    chq.cfo_rad_per_sample,
                );
                pred_mse_fde = mse_fde;
                pred_mse_raw = mse_raw;
                if self.fde_auto_path_select && mse_fde.is_finite() && mse_raw.is_finite() {
                    use_fde_this_frame = mse_fde < mse_raw;
                }
            }
            self.current_frame_use_fde = use_fde_this_frame;
            self.last_pred_mse_fde = pred_mse_fde;
            self.last_pred_mse_raw = pred_mse_raw;
            self.last_est_snr_db = chq.snr_db.unwrap_or(f32::NAN);
            if use_fde_this_frame {
                self.fde_selected_frames += 1;
                self.last_path_used = 1;
            } else {
                self.raw_selected_frames += 1;
                self.last_path_used = 0;
            }

            let overlap = if use_fde_this_frame {
                self.equalizer.as_ref().map_or(0, |eq| eq.overlap_len())
            } else {
                0
            };

            // 1. 同期検出直後の初期クリーニング
            // プリアンブル以前の不要なデータを即座に物理削除
            let initial_drain_len = sync_start.saturating_sub(overlap);
            if initial_drain_len > 0 {
                self.sample_buffer_i.drain(0..initial_drain_len);
                self.sample_buffer_q.drain(0..initial_drain_len);
            }

            // --- 新しいフレームの準備 ---
            self.equalized_buffer.clear();
            self.last_search_idx = 0;
            self.equalizer_input_offset = 0;

            // EQ リセット
            if use_fde_this_frame {
                if let Some(ref mut eq) = self.equalizer {
                    eq.set_cir_with_mmse(&self.cir_buffer[..cir_len], mmse);
                    eq.reset();
                }
            }

            // 2. ワームアップ投入
            // EQ.reset() は内部に overlap 分のゼロ履歴を持つため、ここで不足分をゼロ埋めすると
            // 履歴を二重に積んで時間軸を押し出してしまう。実サンプルのみ投入する。
            let warmup_real_len = sync_start
                .saturating_sub(initial_drain_len)
                .min(self.sample_buffer_i.len());

            // ゼロアロケーション: 事前確保バッファ使用
            self.complex_buffer.clear();
            if self.complex_buffer.capacity() < warmup_real_len {
                self.complex_buffer.reserve(warmup_real_len);
            }
            unsafe {
                self.complex_buffer.set_len(warmup_real_len);
            }
            for i in 0..warmup_real_len {
                self.complex_buffer[i] =
                    Complex32::new(self.sample_buffer_i[i], self.sample_buffer_q[i]);
            }
            // 3. カウンタ初期化
            self.pending_warmup_samples = warmup_real_len;
            self.pending_warmup_input_samples = warmup_real_len;
            let frame_samples = (sync_word_bits * sf_sync
                + packets_per_frame * (expected_symbols * sf_payload))
                * spc;
            self.remaining_samples_in_frame = frame_samples as isize;

            self.equalize_from_complex_buffer(warmup_real_len, warmup_real_len);

            // 4. 状態移行
            self.current_sync = Some(s.clone());
            self.packets_processed_in_burst = 0;
            self.tracking_state = Some(TrackingState {
                phase_ref: Complex32::new(1.0, 0.0),
                phase_rate: 0.0,
                timing_offset: 0.0,
                timing_rate: 0.0,
            });
            self.demodulator.set_prev_phase(Complex32::new(1.0, 0.0));
            self.state = DecoderState::EqualizedDecoding;

            true
        } else {
            // 検出されなかった
            self.last_search_idx = next_search_idx;

            let max_buffer_len = 100_000;
            let keep_len = 50_000;
            if self.sample_buffer_i.len() > max_buffer_len {
                let drain_len = self.sample_buffer_i.len() - keep_len;
                self.sample_buffer_i.drain(0..drain_len);
                self.sample_buffer_q.drain(0..drain_len);
                self.last_search_idx = self.last_search_idx.saturating_sub(drain_len);
                self.equalizer_input_offset = self.equalizer_input_offset.saturating_sub(drain_len);
            }
            false
        }
    }

    fn handle_decoding(&mut self) -> bool {
        let spc = self.config.proc_samples_per_chip().max(1);
        let sf_sync = SYNC_SPREAD_FACTOR;
        let sf_payload = PAYLOAD_SPREAD_FACTOR;
        let sync_word_bits = self.config.sync_word_bits;

        // 1. 等化器への投入 (統合口 equalize を使用)
        // 未投入領域のみを投入する。物理削除と remaining_samples_in_frame の減算は equalize 内で行う。
        let to_process = self
            .sample_buffer_i
            .len()
            .saturating_sub(self.equalizer_input_offset);
        if to_process > 0 {
            // ゼロアロケーション: 事前確保バッファ使用
            self.complex_buffer.clear();
            if self.complex_buffer.capacity() < to_process {
                self.complex_buffer.reserve(to_process);
            }
            unsafe {
                self.complex_buffer.set_len(to_process);
            }
            for (idx, i) in
                (self.equalizer_input_offset..self.equalizer_input_offset + to_process).enumerate()
            {
                self.complex_buffer[idx] =
                    Complex32::new(self.sample_buffer_i[i], self.sample_buffer_q[i]);
            }
            self.equalize_from_complex_buffer(to_process, to_process);
        }

        // 2. 同期語の再復調と位相基準の確立
        if self.packets_processed_in_burst == 0 {
            let required_sync_samples = sync_word_bits * sf_sync * spc;

            if self.equalized_buffer.len() < required_sync_samples {
                return false;
            }

            // 同期語の非コヒーレント(エネルギー)整合度で、chip内のサンプル位相(0..spc-1)を決定する。
            // CFO 下ではコヒーレント和が位相回転で打ち消されるため、タイミング決定は位相非依存にする。
            let mut best_timing_offset = 0usize;
            let mut best_sync_score = f32::NEG_INFINITY;
            for t_offset in 0..spc {
                let mut total_energy = 0.0f32;
                let mut used = 0usize;
                for i in 0..sync_word_bits {
                    let symbol_start = t_offset + i * sf_sync * spc;
                    let corr = if let Some(c) =
                        self.despread_symbol_inner(symbol_start, 0.0, sf_sync, 0)
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
                self.equalized_buffer.drain(0..best_timing_offset);
            }

            let mut st = self.tracking_state.expect("st must exist");
            let repeat = self.config.preamble_repeat;
            let early_late_delta = (spc as f32 * TRACKING_EARLY_LATE_DELTA_CHIP).max(1.0);

            // sync語全体を使った一括推定:
            // y_i = corr_i * expected_sign_i ≈ A * exp(j*(phi0 + i*omega_sync))
            self.complex_buffer.clear();
            if self.complex_buffer.capacity() < sync_word_bits {
                self.complex_buffer.reserve(sync_word_bits);
            }
            let mut prev_y: Option<Complex32> = None;
            let mut sum_diff = Complex32::new(0.0, 0.0);

            // timing LS 用の統計量（重み付き直線近似 e_i ≈ a + b*i）
            let mut sw = 0.0f32;
            let mut sx = 0.0f32;
            let mut sy = 0.0f32;
            let mut sxx = 0.0f32;
            let mut sxy = 0.0f32;
            for i in 0..sync_word_bits {
                let symbol_start = i * sf_sync * spc;
                let on = if let Some(c) = self.despread_symbol_inner(symbol_start, 0.0, sf_sync, 0)
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

                let early = self.despread_symbol_inner(symbol_start, -early_late_delta, sf_sync, 0);
                let late = self.despread_symbol_inner(symbol_start, early_late_delta, sf_sync, 0);
                if let (Some(e), Some(l)) = (early, late) {
                    let err = timing_error_from_early_late(e[0].norm(), l[0].norm());
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
                -TRACKING_PHASE_RATE_LIMIT_RAD,
                TRACKING_PHASE_RATE_LIMIT_RAD,
            );

            let last_sync_idx = (self.complex_buffer.len() - 1) as f32;
            let phi_last = phi0 + omega_sync * last_sync_idx;
            let phi_payload0 = phi_last + st.phase_rate;
            let (s_ref, c_ref) = phi_payload0.sin_cos();
            st.phase_ref = Complex32::new(c_ref, s_ref);

            if let Some(&last_y) = self.complex_buffer.last() {
                let last_sign =
                    self.sync_detector.sync_symbols()[repeat + self.complex_buffer.len() - 1];
                let last_corr = last_y * Complex32::new(last_sign, 0.0);
                let (s_last, c_last) = (-phi_last).sin_cos();
                let prev = last_corr * Complex32::new(c_last, s_last);
                let norm = prev.norm().max(1e-6);
                self.demodulator.set_prev_phase(prev / norm);
            } else {
                self.demodulator.set_prev_phase(Complex32::new(1.0, 0.0));
            }

            let timing_limit = spc as f32 * TRACKING_TIMING_LIMIT_CHIP;
            let timing_rate_limit = spc as f32 * TRACKING_TIMING_RATE_LIMIT_CHIP;
            if sw > 1e-9 {
                let denom = sw * sxx - sx * sx;
                if denom.abs() > 1e-9 {
                    let slope = (sw * sxy - sx * sy) / denom;
                    let intercept = (sy - slope * sx) / sw;
                    st.timing_offset =
                        (intercept * early_late_delta).clamp(-timing_limit, timing_limit);
                    st.timing_rate = (slope * early_late_delta * sync_to_payload_scale)
                        .clamp(-timing_rate_limit, timing_rate_limit);
                } else {
                    let intercept = sy / sw;
                    st.timing_offset =
                        (intercept * early_late_delta).clamp(-timing_limit, timing_limit);
                    st.timing_rate = 0.0;
                }
            } else {
                st.timing_offset = 0.0;
                st.timing_rate = 0.0;
            }

            self.tracking_state = Some(st);
            self.packets_processed_in_burst = 1;
            let drain_len = required_sync_samples.saturating_sub(spc);
            self.equalized_buffer.drain(0..drain_len);
        }

        // 3. パケット復調
        let expected_symbols = interleaver_config::mary_symbols();
        let packet_samples = expected_symbols * sf_payload * spc;
        let max_packets = self.config.packets_per_burst;
        let packets_decoded = self.packets_processed_in_burst.saturating_sub(1);

        if packets_decoded >= max_packets {
            if self.pending_warmup_input_samples == 0 && self.remaining_samples_in_frame <= 0 {
                self.last_search_idx = 0;
                self.equalizer_input_offset = 0;
                self.equalized_buffer.clear();
                self.state = DecoderState::Searching;
                self.current_sync = None;
                self.pending_warmup_samples = 0;
                self.pending_warmup_input_samples = 0;
                return true;
            }
            return to_process > 0;
        }

        if self.equalized_buffer.len() < packet_samples + spc {
            return false;
        }

        let (_avg_energy, success, processed) = self.process_packet_core();
        if !processed {
            return false;
        }
        let mut st = self.tracking_state.expect("st exists");
        let offset_int = st.timing_offset.round().clamp(-1.0, 1.0) as i32;
        let actual_drain_len = (packet_samples as i32 + offset_int).max(0) as usize;
        st.timing_offset -= offset_int as f32;
        self.tracking_state = Some(st);

        self.equalized_buffer
            .drain(0..actual_drain_len.min(self.equalized_buffer.len()));

        if success {
            self.consecutive_crc_errors = 0;
        } else {
            self.consecutive_crc_errors += 1;
        }
        self.packets_processed_in_burst += 1;

        // フレーム入力を規定量消費し、規定パケット数を処理し終えたら Searching へ戻る
        let packets_decoded = self.packets_processed_in_burst.saturating_sub(1);
        if self.pending_warmup_input_samples == 0
            && self.remaining_samples_in_frame <= 0
            && packets_decoded >= max_packets
        {
            self.last_search_idx = 0;
            self.equalizer_input_offset = 0;
            self.equalized_buffer.clear();
            self.state = DecoderState::Searching;
            self.current_sync = None;
            self.pending_warmup_samples = 0;
            self.pending_warmup_input_samples = 0;
        }

        true
    }

    fn process_packet_core(&mut self) -> (f32, bool, bool) {
        let spc = self.config.proc_samples_per_chip().max(1);
        let sf_payload = PAYLOAD_SPREAD_FACTOR;
        let interleaved_bits = interleaver_config::interleaved_bits();
        let expected_symbols = interleaver_config::mary_symbols();

        let mut st = self
            .tracking_state
            .expect("Tracking state must be initialized");
        let timing_limit = spc as f32 * TRACKING_TIMING_LIMIT_CHIP;
        let timing_rate_limit = spc as f32 * TRACKING_TIMING_RATE_LIMIT_CHIP;
        let early_late_delta = (spc as f32 * TRACKING_EARLY_LATE_DELTA_CHIP).max(1.0);

        // ゼロアロケーション: 事前確保バッファ使用
        self.packet_llrs_buffer.clear();
        let mut total_packet_energy = 0.0f32;

        for sym_idx in 0..expected_symbols {
            let symbol_start = spc + sym_idx * sf_payload * spc;

            let on_corrs = if let Some(c) =
                self.despread_symbol_with_timing(symbol_start, st.timing_offset, 0.0)
            {
                c
            } else {
                return (0.0, false, false);
            };
            let early_corrs = if let Some(c) =
                self.despread_symbol_with_timing(symbol_start, st.timing_offset, -early_late_delta)
            {
                c
            } else {
                return (0.0, false, false);
            };
            let late_corrs = if let Some(c) =
                self.despread_symbol_with_timing(symbol_start, st.timing_offset, early_late_delta)
            {
                c
            } else {
                return (0.0, false, false);
            };

            let mut max_energy = 0.0f32;
            let mut best_idx = 0usize;
            for (idx, corr) in on_corrs.iter().enumerate() {
                let energy = corr.norm_sqr();
                if energy > max_energy {
                    max_energy = energy;
                    best_idx = idx;
                }
            }
            total_packet_energy += max_energy;

            let best_corr = on_corrs[best_idx];
            let on_rot = best_corr * st.phase_ref.conj();
            let diff = on_rot * self.demodulator.prev_phase().conj();

            let energies: [f32; 16] = on_corrs.map(|c| (c * st.phase_ref.conj()).norm_sqr());
            let walsh_llr = self.demodulator.walsh_llr(&energies, max_energy);
            let dqpsk_llr = self.demodulator.dqpsk_llr(diff, max_energy);

            self.packet_llrs_buffer.extend_from_slice(&walsh_llr);
            self.packet_llrs_buffer.extend_from_slice(&dqpsk_llr);

            let decided = if dqpsk_llr[0] >= 0.0 && dqpsk_llr[1] >= 0.0 {
                Complex32::new(1.0, 0.0)
            } else if dqpsk_llr[0] >= 0.0 && dqpsk_llr[1] < 0.0 {
                Complex32::new(0.0, 1.0)
            } else if dqpsk_llr[0] < 0.0 && dqpsk_llr[1] < 0.0 {
                Complex32::new(-1.0, 0.0)
            } else {
                Complex32::new(0.0, -1.0)
            };

            let phase_err = phase_error_from_diff(diff, decided);
            st.phase_rate = update_phase_rate(st.phase_rate, phase_err);
            let (sin_dphi, cos_dphi) =
                phase_step_from_phase_error(phase_err, st.phase_rate).sin_cos();
            st.phase_ref *= Complex32::new(cos_dphi, sin_dphi);
            st.phase_ref /= st.phase_ref.norm().max(1e-6);

            let timing_err = timing_error_from_early_late(
                early_corrs[best_idx].norm(),
                late_corrs[best_idx].norm(),
            );
            st.timing_rate = update_timing_rate(st.timing_rate, timing_err, timing_rate_limit);
            st.timing_offset =
                update_timing_offset(st.timing_offset, st.timing_rate, timing_err, timing_limit);

            let on_norm = on_rot.norm().max(1e-6);
            self.demodulator.set_prev_phase(on_rot / on_norm);
        }

        self.tracking_state = Some(st);

        let avg_energy = total_packet_energy / expected_symbols as f32;
        // バッファの所有権を一時的に移して借用衝突を回避する（追加アロケーションなし）
        let mut llr_buf = std::mem::take(&mut self.packet_llrs_buffer);
        let packet_llrs_len = llr_buf.len().min(interleaved_bits);
        let success = if packet_llrs_len >= interleaved_bits {
            self.decode_llrs(&llr_buf[..packet_llrs_len]) > 0
        } else {
            false
        };
        llr_buf.clear();
        self.packet_llrs_buffer = llr_buf;

        (avg_energy, success, true)
    }

    fn decode_llrs(&mut self, llrs: &[f32]) -> usize {
        let p_bits_len = crate::frame::packet::PACKET_BYTES * 8;
        let fec_bits = interleaver_config::fec_bits();
        let rows = interleaver_config::INTERLEAVER_ROWS; // 29
        let cols = interleaver_config::INTERLEAVER_COLS; // 12
        let interleaved_bits = interleaver_config::interleaved_bits(); // 348

        let packet_chunk_bits = interleaver_config::mary_aligned_bits(); // 348（パディングなし）
        let mut success_count = 0;

        for packet_llrs in llrs.chunks(packet_chunk_bits) {
            if packet_llrs.len() < interleaved_bits {
                break;
            }

            let valid_llrs = &packet_llrs[..interleaved_bits];

            let interleaver = BlockInterleaver::new(rows, cols);
            // インプレースAPI使用
            self.deinterleave_buffer.resize(interleaved_bits, 0.0);
            interleaver.deinterleave_f32_in_place(
                valid_llrs,
                &mut self.deinterleave_buffer[..interleaved_bits],
            );

            let mut scrambler = Scrambler::default();
            for llr in self.deinterleave_buffer[..interleaved_bits].iter_mut() {
                if scrambler.next_bit() == 1 {
                    *llr = -*llr;
                }
            }

            if let Some(ref mut callback) = self.llr_callback {
                callback(&self.deinterleave_buffer[..interleaved_bits]);
            }

            let decoded_bits = fec::decode_soft(&self.deinterleave_buffer[..fec_bits]);
            if decoded_bits.len() < p_bits_len {
                continue;
            }

            let decoded_bytes = fec::bits_to_bytes(&decoded_bits[..p_bits_len]);

            match Packet::deserialize(&decoded_bytes) {
                Ok(packet) => {
                    let pkt_k = packet.lt_k as usize;
                    if pkt_k != self.fountain_decoder.params().k {
                        self.rebuild_fountain_decoder(pkt_k);
                    }

                    self.last_packet_seq = Some(packet.lt_seq as u32);

                    let fountain_packet = FountainPacket {
                        seq: packet.lt_seq as u32,
                        coefficients: crate::coding::fountain::reconstruct_packet_coefficients(
                            packet.lt_seq as u32,
                            self.fountain_decoder.params().k,
                        ),
                        data: packet.payload.to_vec(),
                    };

                    use crate::coding::fountain::ReceiveOutcome;
                    let outcome = self.fountain_decoder.receive_with_outcome(fountain_packet);
                    match outcome {
                        ReceiveOutcome::AcceptedRankUp => {
                            self.received_packets += 1;
                            self.last_rank_up_seq = Some(packet.lt_seq as u32);
                        }
                        ReceiveOutcome::AcceptedNoRankUp => {
                            self.received_packets += 1;
                            self.stalled_packets += 1;
                            self.dependent_packets += 1;
                        }
                        ReceiveOutcome::DuplicateSeq => {
                            self.duplicate_packets += 1;
                        }
                        ReceiveOutcome::InvalidPacket => {
                            self.parse_error_packets += 1;
                        }
                    }

                    if let Some(data) = self.fountain_decoder.decode() {
                        self.recovered_data = Some(data);
                    }
                    success_count += 1;
                }
                Err(e) => {
                    use crate::frame::packet::PacketParseError;
                    match e {
                        PacketParseError::CrcMismatch { .. } => self.crc_error_packets += 1,
                        _ => self.parse_error_packets += 1,
                    }
                }
            }
        }
        success_count
    }

    fn rebuild_fountain_decoder(&mut self, fountain_k: usize) {
        let params = FountainParams::new(fountain_k, PAYLOAD_SIZE);
        self.fountain_decoder = FountainDecoder::new(params);
        self.recovered_data = None;
        self.last_packet_seq = None;
        self.last_rank_up_seq = None;
        self.received_packets = 0;
        self.stalled_packets = 0;
        self.dependent_packets = 0;
        self.duplicate_packets = 0;
        self.crc_error_packets = 0;
        self.parse_error_packets = 0;
        self.invalid_neighbor_packets = 0;
        self.fde_selected_frames = 0;
        self.raw_selected_frames = 0;
        self.last_path_used = -1;
        self.last_pred_mse_fde = f32::NAN;
        self.last_pred_mse_raw = f32::NAN;
        self.last_est_snr_db = f32::NAN;
    }

    fn progress(&self) -> DecodeProgress {
        let needed = self.fountain_decoder.params().k;
        let progress = self.fountain_decoder.progress();

        DecodeProgress {
            received_packets: self.received_packets,
            needed_packets: needed,
            rank_packets: self.fountain_decoder.rank(),
            stalled_packets: self.stalled_packets,
            dependent_packets: self.dependent_packets,
            duplicate_packets: self.duplicate_packets,
            crc_error_packets: self.crc_error_packets,
            parse_error_packets: self.parse_error_packets,
            invalid_neighbor_packets: self.invalid_neighbor_packets,
            last_packet_seq: self.last_packet_seq.map(|s| s as i32).unwrap_or(-1),
            last_rank_up_seq: self.last_rank_up_seq.map(|s| s as i32).unwrap_or(-1),
            progress,
            complete: self.recovered_data.is_some(),
            fde_selected_frames: self.fde_selected_frames,
            raw_selected_frames: self.raw_selected_frames,
            last_path_used: self.last_path_used,
            last_pred_mse_fde: self.last_pred_mse_fde,
            last_pred_mse_raw: self.last_pred_mse_raw,
            last_est_snr_db: self.last_est_snr_db,
            basis_matrix: self.fountain_decoder.get_basis_matrix(),
        }
    }

    fn postprocess_cir(
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

    fn known_interval_path_mse(
        &mut self,
        preamble_start_idx: usize,
        known_end_idx: usize,
        cir_len: usize,
        mmse: MmseSettings,
        cfo_rad_per_sample: f32,
    ) -> (f32, f32) {
        if known_end_idx > self.sample_buffer_i.len() || known_end_idx > self.sample_buffer_q.len()
        {
            return (f32::NAN, f32::NAN);
        }

        let mse_raw = self
            .sync_detector
            .known_sequence_mse_iq(
                &self.sample_buffer_i,
                &self.sample_buffer_q,
                preamble_start_idx,
                cfo_rad_per_sample,
            )
            .unwrap_or(f32::NAN);

        let Some(eq) = self.equalizer.as_mut() else {
            return (mse_raw, f32::NAN);
        };

        let overlap = eq.overlap_len();
        let analysis_start = preamble_start_idx.saturating_sub(overlap);
        let input_len = known_end_idx.saturating_sub(analysis_start);
        if input_len == 0 {
            return (mse_raw, f32::NAN);
        }

        let mut eval_input = Vec::with_capacity(input_len);
        for idx in analysis_start..known_end_idx {
            eval_input.push(Complex32::new(
                self.sample_buffer_i[idx],
                self.sample_buffer_q[idx],
            ));
        }

        let mut eval_output = Vec::with_capacity(input_len);
        eq.set_cir_with_mmse(&self.cir_buffer[..cir_len], mmse);
        eq.reset();
        eq.process(&eval_input, &mut eval_output);
        eq.flush(&mut eval_output);

        let known_start_in_eval = preamble_start_idx.saturating_sub(analysis_start);
        let mse_fde = self
            .sync_detector
            .known_sequence_mse_complex(&eval_output, known_start_in_eval, cfo_rad_per_sample)
            .unwrap_or(f32::NAN);

        (mse_raw, mse_fde)
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
        self.equalizer_input_offset = self.equalizer_input_offset.saturating_add(raw_consumed);

        // B. 生バッファの物理削除 (投入時に実行)
        // フレーム境界を壊さないよう、ウォームアップ + remaining を上限にする。
        let limit =
            self.pending_warmup_input_samples + self.remaining_samples_in_frame.max(0) as usize;
        let to_drain_raw = raw_consumed.min(limit).min(self.sample_buffer_i.len());
        if to_drain_raw > 0 {
            self.sample_buffer_i.drain(0..to_drain_raw);
            self.sample_buffer_q.drain(0..to_drain_raw);
            self.equalizer_input_offset = self.equalizer_input_offset.saturating_sub(to_drain_raw);

            let warmup_drain = to_drain_raw.min(self.pending_warmup_input_samples);
            self.pending_warmup_input_samples -= warmup_drain;
            let payload_drain = to_drain_raw - warmup_drain;
            self.remaining_samples_in_frame -= payload_drain as isize;
        }

        if self.current_frame_use_fde {
            if let Some(ref mut eq) = self.equalizer {
                // C. 等化実行
                let added = eq.process(input, &mut self.equalized_buffer);
                if added == 0 {
                    return;
                }

                // D. 等化後バッファのアライメント (ゴミ捨て)
                // ウォームアップ由来の中間状態を先頭から捨てる。
                let warmup_to_drain = added.min(self.pending_warmup_samples);
                if warmup_to_drain > 0 {
                    self.equalized_buffer.drain(0..warmup_to_drain);
                    self.pending_warmup_samples -= warmup_to_drain;
                }
            }
        } else {
            // FDE無効時はパススルーで同じ管理ルールを適用する
            self.equalized_buffer.extend_from_slice(input);
            let warmup_to_drain = input.len().min(self.pending_warmup_samples);
            if warmup_to_drain > 0 {
                self.equalized_buffer.drain(0..warmup_to_drain);
                self.pending_warmup_samples -= warmup_to_drain;
            }
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

        // 境界チェック: 最後にアクセスする可能性のあるインデックスを ceil で見積もる
        let max_p = symbol_start as f32
            + ((sf - 1) * spc) as f32
            + timing_offset
            + ((spc as f32 - 1.0) / 2.0);
        let last_required_idx = max_p.ceil() as usize;
        if last_required_idx >= self.equalized_buffer.len() {
            return None;
        }

        let mut results = [num_complex::Complex32::new(0.0, 0.0); 1];
        for chip_idx in 0..sf {
            let p = symbol_start as f32
                + (chip_idx * spc) as f32
                + timing_offset
                + ((spc as f32 - 1.0) / 2.0);
            let i_idx = p.round() as usize;
            let sample = self.equalized_buffer[i_idx];
            let walsh_val = self.demodulator.correlators()[walsh_idx].sequence()[chip_idx] as f32;
            results[0] += sample * walsh_val;
        }

        Some(results)
    }

    fn despread_symbol_with_timing(
        &self,
        symbol_start: usize,
        timing_offset: f32,
        sample_shift: f32,
    ) -> Option<[Complex32; 16]> {
        let spc = self.config.proc_samples_per_chip().max(1);
        let sf = 16;

        // 境界チェック: 最後にアクセスする可能性のあるインデックスを ceil で見積もる
        let max_p = symbol_start as f32
            + ((sf - 1) * spc) as f32
            + timing_offset
            + sample_shift
            + ((spc as f32 - 1.0) / 2.0);
        let last_required_idx = max_p.ceil() as usize;
        if last_required_idx >= self.equalized_buffer.len() {
            return None;
        }

        let mut results = [Complex32::new(0.0, 0.0); 16];
        for chip_idx in 0..sf {
            let p = symbol_start as f32
                + (chip_idx * spc) as f32
                + timing_offset
                + sample_shift
                + ((spc as f32 - 1.0) / 2.0);
            let i_idx = p.round() as usize;
            let sample = self.equalized_buffer[i_idx];

            for (idx, correlator) in self.demodulator.correlators().iter().enumerate() {
                let walsh_val = correlator.sequence()[chip_idx] as f32;
                results[idx] += sample * walsh_val;
            }
        }

        Some(results)
    }

    pub fn reset(&mut self) {
        let params = self.fountain_decoder.params().clone();
        self.rrc_filter_i.reset();
        self.rrc_filter_q.reset();
        self.demodulator.reset();
        self.fountain_decoder = FountainDecoder::new(params);
        self.recovered_data = None;
        self.lo_nco.reset();
        self.state = DecoderState::Searching;
        self.last_search_idx = 0;
        self.current_sync = None;
        self.tracking_state = None;
        self.packets_processed_in_burst = 0;
        self.consecutive_crc_errors = 0;
        self.last_packet_seq = None;
        self.last_rank_up_seq = None;
        self.pending_warmup_samples = 0;
        self.pending_warmup_input_samples = 0;
        self.remaining_samples_in_frame = 0;
        self.current_frame_use_fde = self.equalizer.is_some();
        self.fde_selected_frames = 0;
        self.raw_selected_frames = 0;
        self.last_path_used = -1;
        self.last_pred_mse_fde = f32::NAN;
        self.last_pred_mse_raw = f32::NAN;
        self.last_est_snr_db = f32::NAN;
        self.sample_buffer_i.clear();
        self.sample_buffer_q.clear();
        self.equalized_buffer.clear();
        self.equalizer_input_offset = 0;
        self.received_packets = 0;
        self.stalled_packets = 0;
        self.dependent_packets = 0;
        self.duplicate_packets = 0;
        self.crc_error_packets = 0;
        self.parse_error_packets = 0;
        self.invalid_neighbor_packets = 0;
        self.stats_total_samples = 0;
    }
}

#[inline]
fn timing_error_from_early_late(early_mag: f32, late_mag: f32) -> f32 {
    (late_mag - early_mag) / (late_mag + early_mag + 1e-6)
}

#[inline]
fn update_timing_rate(timing_rate: f32, timing_err: f32, timing_rate_limit: f32) -> f32 {
    (timing_rate + TRACKING_TIMING_RATE_GAIN * timing_err)
        .clamp(-timing_rate_limit, timing_rate_limit)
}

#[inline]
fn update_timing_offset(
    timing_offset: f32,
    timing_rate: f32,
    timing_err: f32,
    timing_limit: f32,
) -> f32 {
    (timing_offset + timing_rate + TRACKING_TIMING_PROP_GAIN * timing_err)
        .clamp(-timing_limit, timing_limit)
}

#[inline]
fn phase_error_from_diff(diff: Complex32, decided_symbol: Complex32) -> f32 {
    let diff_data_removed = diff * decided_symbol.conj();
    diff_data_removed.im.atan2(diff_data_removed.re)
}

#[inline]
fn update_phase_rate(phase_rate: f32, phase_err: f32) -> f32 {
    (phase_rate + TRACKING_PHASE_FREQ_GAIN * phase_err).clamp(
        -TRACKING_PHASE_RATE_LIMIT_RAD,
        TRACKING_PHASE_RATE_LIMIT_RAD,
    )
}

#[inline]
fn phase_step_from_phase_error(phase_err: f32, phase_rate: f32) -> f32 {
    (phase_rate + TRACKING_PHASE_PROP_GAIN * phase_err)
        .clamp(-TRACKING_PHASE_STEP_CLAMP, TRACKING_PHASE_STEP_CLAMP)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::walsh::WalshDictionary;

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
        assert!(decoder.stats_total_samples > 0);
        decoder.reset();
        assert_eq!(decoder.stats_total_samples, 0);
        assert!(decoder.recovered_data.is_none());
        assert!(decoder.sample_buffer_i.is_empty());
        assert!(decoder.sample_buffer_q.is_empty());
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
    fn test_spread_factor_transition() {
        let sf_sync = SYNC_SPREAD_FACTOR;
        let sf_payload = PAYLOAD_SPREAD_FACTOR;
        assert_ne!(
            sf_sync, sf_payload,
            "Sync and Payload should have different SF"
        );
        assert_eq!(
            sf_payload - sf_sync,
            1,
            "Payload SF should be 1 more than Sync SF"
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
        decoder.mix_real_to_iq_zero_alloc(&frame_48k);

        // リサンプリング
        let mut i_resampled = Vec::new();
        let mut q_resampled = Vec::new();
        decoder
            .resampler_i
            .process(&decoder.mix_buffer_i, &mut i_resampled);
        decoder
            .resampler_q
            .process(&decoder.mix_buffer_q, &mut q_resampled);

        // RRCフィルタ（インプレースAPI使用）
        let mut i_filtered = i_resampled.clone();
        let mut q_filtered = q_resampled.clone();
        decoder.rrc_filter_i.process_block_in_place(&mut i_filtered);
        decoder.rrc_filter_q.process_block_in_place(&mut q_filtered);

        // 48k -> 24k (proc_rate) へのリサンプリング後の長さを算出
        let preamble_len_24k = preamble_len_48k / 2;

        // 3. SyncDetector を実行
        let (sync_opt, _) = decoder.sync_detector.detect(&i_filtered, &q_filtered, 0);

        if let Some(s) = sync_opt {
            // peak_sample_idx は同期語 0 番目の最初のチップの中央
            // 期待値 = (プリアンブル終了点) + (チップ 0 の半分) + (受信側遅延) + (送信側遅延)
            // 送信側 (Modulator) も内部で RRC フィルタを通しているため、その分 (16 samples) 遅延する。
            let rx_delay = decoder.resampler_i.delay() + decoder.rrc_filter_i.delay();
            let detector_delay = decoder.sync_detector.filter_delay();
            let tx_delay = 16; // Modulator's RRC filter delay at 48k? Wait, it is at 48k.

            // 48kレートでの遅延 16 は、24kレートでは 8 サンプル。
            let expected_peak =
                preamble_len_24k + (spc / 2) + rx_delay + (tx_delay / 2) + detector_delay;
            let diff = s.peak_sample_idx as i32 - expected_peak as i32;
            println!(
                "[GT] Detected peak: {}, Expected peak: {}, Diff: {}, Delay: rx={}, tx_adj={}",
                s.peak_sample_idx,
                expected_peak,
                diff,
                rx_delay + detector_delay,
                tx_delay / 2
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
        assert!(decoder.stats_total_samples >= frame.len());
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
        assert!(decoder.stats_total_samples >= frame.len());
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
            let margin = TRACKING_PHASE_RATE_LIMIT_RAD / phase_step_rad.max(1e-6);
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
            decoder.current_frame_use_fde,
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
            !decoder.current_frame_use_fde,
            "平坦チャネルかつ強正則化ではAUTOはRAWを選ぶべき"
        );
    }
}
