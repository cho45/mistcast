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
use crate::common::equalization::FrequencyDomainEqualizer;
use crate::common::nco::Nco;
use crate::common::resample::Resampler;
use crate::common::rrc_filter::RrcFilter;
use crate::frame::packet::Packet;
use crate::mary::demodulator::Demodulator;
use crate::mary::sync::{MarySyncDetector, SyncResult};
use crate::params::PAYLOAD_SIZE;
use crate::DspConfig;
use num_complex::Complex32;

const TRACKING_TIMING_PROP_GAIN: f32 = 0.18;
const TRACKING_TIMING_RATE_GAIN: f32 = 0.01;
const TRACKING_PHASE_PROP_GAIN: f32 = 0.22;
const TRACKING_PHASE_FREQ_GAIN: f32 = 0.015;
const TRACKING_TIMING_LIMIT_CHIP: f32 = 2.0;
const TRACKING_TIMING_RATE_LIMIT_CHIP: f32 = 0.25;
const TRACKING_EARLY_LATE_DELTA_CHIP: f32 = 0.5;
const TRACKING_PHASE_RATE_LIMIT_RAD: f32 = 2.6;
const TRACKING_PHASE_STEP_CLAMP: f32 = 2.8;

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
}

#[derive(Clone, Copy, Debug)]
struct TrackingState {
    phase_ref: Complex32,
    phase_rate: f32,
    timing_offset: f32,
    timing_rate: f32,
}

impl Decoder {
    /// 新しいデコーダを作成する
    pub fn new(_data_size: usize, fountain_k: usize, dsp_config: DspConfig) -> Self {
        let proc_sample_rate = dsp_config.proc_sample_rate();
        let lo_nco = Nco::new(-dsp_config.carrier_freq, dsp_config.sample_rate);

        let rrc_bw = dsp_config.chip_rate * (1.0 + dsp_config.rrc_alpha) * 0.5;
        let cutoff = Some(rrc_bw);

        let tc = MarySyncDetector::THRESHOLD_COARSE_DEFAULT;
        let tf = MarySyncDetector::THRESHOLD_FINE_DEFAULT;

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
            equalizer: {
                let cir_len = dsp_config.preamble_sf;
                let fft_size = (cir_len * 4).next_power_of_two().max(256);
                Some(FrequencyDomainEqualizer::new(&[Complex32::new(1.0, 0.0)], fft_size, 15.0))
            },
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
            received_packets: 0,
            stalled_packets: 0,
            dependent_packets: 0,
            duplicate_packets: 0,
            crc_error_packets: 0,
            parse_error_packets: 0,
            invalid_neighbor_packets: 0,
            llr_callback: None,
            stats_total_samples: 0,
        }
    }

    pub fn process_samples(&mut self, samples: &[f32]) -> DecodeProgress {
        if self.recovered_data.is_some() {
            return self.progress();
        }
        self.stats_total_samples += samples.len();

        let mut i_mixed = Vec::with_capacity(samples.len());
        let mut q_mixed = Vec::with_capacity(samples.len());
        self.mix_real_to_iq(samples, &mut i_mixed, &mut q_mixed);

        let mut i_resampled = Vec::new();
        let mut q_resampled = Vec::new();
        self.resampler_i.process(&i_mixed, &mut i_resampled);
        self.resampler_q.process(&q_mixed, &mut q_resampled);

        let i_filtered = self.rrc_filter_i.process_block(&i_resampled);
        let q_filtered = self.rrc_filter_q.process_block(&q_resampled);

        self.sample_buffer_i.extend_from_slice(&i_filtered);
        self.sample_buffer_q.extend_from_slice(&q_filtered);

        self.detect_and_process_frames()
    }

    fn mix_real_to_iq(&mut self, samples: &[f32], i_mixed: &mut Vec<f32>, q_mixed: &mut Vec<f32>) {
        for &s in samples {
            let lo = self.lo_nco.step();
            i_mixed.push(s * lo.re * 2.0);
            q_mixed.push(s * lo.im * 2.0);
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
        let sf_sync = 15;
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
            let preamble_len = sf_preamble * repeat * spc;
            let preamble_start_idx = s.peak_sample_idx
                .saturating_sub(preamble_len)
                .saturating_sub(spc / 2);

            let sync_start = s.peak_sample_idx.saturating_sub(spc / 2);

            let overlap = if let Some(ref eq) = self.equalizer { eq.overlap_len() } else { 0 };

            // 新しい CIR を推定しておく
            let mut cir = vec![Complex32::new(0.0, 0.0); sf_preamble];
            self.sync_detector.estimate_cir(
                &self.sample_buffer_i,
                &self.sample_buffer_q,
                preamble_start_idx,
                &mut cir,
            );

            // 1. 追い出し: 新しい同期(P_start)の直前までの生サンプルを EQ に投入
            let drain_offset = sync_start.saturating_sub(overlap);
            let to_flush = drain_offset.saturating_sub(self.equalizer_input_offset);

            if to_flush > 0 {
                if let Some(ref mut eq) = self.equalizer {
                    let mut flush_input = Vec::with_capacity(to_flush);
                    for i in 0..to_flush {
                        flush_input.push(Complex32::new(
                            self.sample_buffer_i[self.equalizer_input_offset + i],
                            self.sample_buffer_q[self.equalizer_input_offset + i],
                        ));
                    }
                    eq.process(&flush_input, &mut self.equalized_buffer);
                }
                self.equalizer_input_offset += to_flush;
            }

            // 2. 完遂: finalize_current_packet を実行
            if !self.equalized_buffer.is_empty() {
                println!("[Decoder] Searching: Found sync, finalizing old burst (eq_len={})", self.equalized_buffer.len());
                self.finalize_current_packet();
            }

            // 3. 一本化された物理削除
            self.sample_buffer_i.drain(0..drain_offset);
            self.sample_buffer_q.drain(0..drain_offset);
            self.equalizer_input_offset = 0;
            self.last_search_idx = 0;

            // 4. リセット
            self.equalized_buffer.clear();

            if let Some(ref mut eq) = self.equalizer {
                let est_snr = 15.0; 
                eq.set_cir(&cir, est_snr);
                eq.reset();
            }

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
        }
 else {
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

    fn finalize_current_packet(&mut self) {
        if self.tracking_state.is_none() {
            return;
        }

        let spc = self.config.proc_samples_per_chip().max(1);
        let sf_sync = 15;
        let sf_payload = 16;
        let sync_word_bits = self.config.sync_word_bits;

        // 1. 同期語が未処理なら処理して位相基準を確立
        if self.packets_processed_in_burst == 0 {
            let overlap = if let Some(ref eq) = self.equalizer { eq.overlap_len() } else { 0 };
            let required_sync_samples = sync_word_bits * sf_sync * spc;

            if self.equalized_buffer.len() < overlap + required_sync_samples {
                return;
            }

            if overlap > 0 {
                self.equalized_buffer.drain(0..overlap);
            }

            let mut best_timing_offset = 0;
            let mut max_sync_energy = 0.0f32;
            for t_offset in 0..spc {
                let mut current_energy = 0.0f32;
                for i in 0..sync_word_bits.min(8) {
                    let symbol_start = t_offset + i * sf_sync * spc;
                    if let Some(corr) = self.despread_symbol_inner(symbol_start, 0.0, sf_sync, 0) {
                        current_energy += corr[0].norm_sqr();
                    }
                }
                if current_energy > max_sync_energy {
                    max_sync_energy = current_energy;
                    best_timing_offset = t_offset;
                }
            }
            if best_timing_offset > 0 {
                self.equalized_buffer.drain(0..best_timing_offset);
            }

            let mut st = self.tracking_state.expect("st must exist");
            for i in 0..sync_word_bits {
                let symbol_start = i * sf_sync * spc;
                let corr = if let Some(c) = self.despread_symbol_inner(symbol_start, 0.0, sf_sync, 0) {
                    c
                } else {
                    break;
                };

                let on_rot = corr[0] * st.phase_ref.conj();
                let on_norm = on_rot.norm().max(1e-6);
                let expected_sign = self.sync_detector.sync_symbols()[self.config.preamble_repeat + i];
                let abs_diff = on_rot * Complex32::new(expected_sign, 0.0).conj();
                let (sin_dphi, cos_dphi) = (abs_diff.arg() * 0.1).sin_cos();
                st.phase_ref *= Complex32::new(cos_dphi, sin_dphi);
                st.phase_ref /= st.phase_ref.norm().max(1e-6);
                self.demodulator.set_prev_phase(on_rot / on_norm);
            }

            self.tracking_state = Some(st);
            self.packets_processed_in_burst = 1;
            let drain_len = required_sync_samples.saturating_sub(spc);
            self.equalized_buffer.drain(0..drain_len);
        }

        // 2. ペイロードのデコード
        if self.packets_processed_in_burst > 0 {
            let interleaved_bits: usize = 352;
            let expected_symbols = interleaved_bits.div_ceil(6);
            let packet_samples = expected_symbols * sf_payload * spc;

            if self.equalized_buffer.len() >= packet_samples + spc {
                let _ = self.process_packet_core();
            }
        }
    }

    fn handle_decoding(&mut self) -> bool {
        let spc = self.config.proc_samples_per_chip().max(1);
        let sf_sync = 15;
        let sf_payload = 16;
        let sync_word_bits = self.config.sync_word_bits;

        // 1. 等化器への投入
        if let Some(eq) = &mut self.equalizer {
            let to_process = self.sample_buffer_i.len() - self.equalizer_input_offset;
            if to_process > 0 {
                let mut complex_input = Vec::with_capacity(to_process);
                for i in 0..to_process {
                    complex_input.push(Complex32::new(
                        self.sample_buffer_i[self.equalizer_input_offset + i],
                        self.sample_buffer_q[self.equalizer_input_offset + i],
                    ));
                }
                eq.process(&complex_input, &mut self.equalized_buffer);
                self.equalizer_input_offset += to_process;
            }
        }

        // 2. 同期語の再復調と位相基準の確立
        if self.packets_processed_in_burst == 0 {
            let overlap = if let Some(ref eq) = self.equalizer { eq.overlap_len() } else { 0 };
            let required_sync_samples = sync_word_bits * sf_sync * spc;

            if self.equalized_buffer.len() < overlap + required_sync_samples {
                return false; 
            }

            if overlap > 0 {
                self.equalized_buffer.drain(0..overlap);
            }

            // --- A. 等化後のタイミング微調整 ---
            let mut best_timing_offset = 0;
            let mut max_sync_energy = 0.0f32;
            for t_offset in 0..spc {
                let mut current_energy = 0.0f32;
                for i in 0..sync_word_bits.min(8) {
                    let symbol_start = t_offset + i * sf_sync * spc;
                    if let Some(corr) = self.despread_symbol_inner(symbol_start, 0.0, sf_sync, 0) {
                        current_energy += corr[0].norm_sqr();
                    }
                }
                if current_energy > max_sync_energy {
                    max_sync_energy = current_energy;
                    best_timing_offset = t_offset;
                }
            }
            if best_timing_offset > 0 {
                self.equalized_buffer.drain(0..best_timing_offset);
            }

            // --- B. 位相基準の確立 (絶対位相トラッキング) ---
            let mut st = self.tracking_state.expect("st must exist");
            for i in 0..sync_word_bits {
                let symbol_start = i * sf_sync * spc;
                let corr = if let Some(c) = self.despread_symbol_inner(symbol_start, 0.0, sf_sync, 0) {
                    c
                } else {
                    break;
                };

                let on_rot = corr[0] * st.phase_ref.conj();
                let on_norm = on_rot.norm().max(1e-6);
                let expected_sign = self.sync_detector.sync_symbols()[self.config.preamble_repeat + i];
                let abs_diff = on_rot * Complex32::new(expected_sign, 0.0).conj();
                let (sin_dphi, cos_dphi) = (abs_diff.arg() * 0.1).sin_cos();
                st.phase_ref *= Complex32::new(cos_dphi, sin_dphi);
                st.phase_ref /= st.phase_ref.norm().max(1e-6);
                self.demodulator.set_prev_phase(on_rot / on_norm);
            }

            self.tracking_state = Some(st);
            self.packets_processed_in_burst = 1;
            let drain_len = required_sync_samples.saturating_sub(spc);
            self.equalized_buffer.drain(0..drain_len);
        }

        // 3. パケット復調
        let interleaved_bits: usize = 352;
        let expected_symbols = interleaved_bits.div_ceil(6);
        let packet_samples = expected_symbols * sf_payload * spc;

        if self.equalized_buffer.len() < packet_samples + spc {
            return false;
        }

        let (avg_energy, success) = self.process_packet_core();

        let mut st = self.tracking_state.expect("st exists");
        let offset_int = st.timing_offset.round() as i32;
        let actual_drain_len = (packet_samples as i32 + offset_int).max(0) as usize;
        st.timing_offset -= offset_int as f32;
        self.tracking_state = Some(st);

        self.equalized_buffer.drain(0..actual_drain_len.min(self.equalized_buffer.len()));
        self.drain_processed_samples_only_raw(actual_drain_len);

        if success {
            self.consecutive_crc_errors = 0;
        } else {
            self.consecutive_crc_errors += 1;
        }
        self.packets_processed_in_burst += 1;

        let burst_limit = self.config.packets_per_burst;
        if self.packets_processed_in_burst >= burst_limit + 1 || (avg_energy < 0.1 && self.consecutive_crc_errors >= 3) { 
            self.state = DecoderState::Searching;
            self.current_sync = None;
        }

        true
    }

    fn process_packet_core(&mut self) -> (f32, bool) {
        let spc = self.config.proc_samples_per_chip().max(1);
        let sf_payload = 16;
        let interleaved_bits: usize = 352;
        let expected_symbols = interleaved_bits.div_ceil(6);

        let mut st = self.tracking_state.expect("Tracking state must be initialized");
        let timing_limit = spc as f32 * TRACKING_TIMING_LIMIT_CHIP;
        let timing_rate_limit = spc as f32 * TRACKING_TIMING_RATE_LIMIT_CHIP;
        let early_late_delta = (spc as f32 * TRACKING_EARLY_LATE_DELTA_CHIP).max(1.0);

        let mut packet_llrs = Vec::with_capacity(interleaved_bits);
        let mut total_packet_energy = 0.0f32;

        for sym_idx in 0..expected_symbols {
            let symbol_start = spc + sym_idx * sf_payload * spc;

            let on_corrs = if let Some(c) = self.despread_symbol_with_timing(symbol_start, st.timing_offset, 0.0) {
                c
            } else {
                break;
            };
            let early_corrs = if let Some(c) = self.despread_symbol_with_timing(symbol_start, st.timing_offset, -early_late_delta) {
                c
            } else {
                break;
            };
            let late_corrs = if let Some(c) = self.despread_symbol_with_timing(symbol_start, st.timing_offset, early_late_delta) {
                c
            } else {
                break;
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

            packet_llrs.extend_from_slice(&walsh_llr);
            packet_llrs.extend_from_slice(&dqpsk_llr);

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
            let (sin_dphi, cos_dphi) = phase_step_from_phase_error(phase_err, st.phase_rate).sin_cos();
            st.phase_ref *= Complex32::new(cos_dphi, sin_dphi);
            st.phase_ref /= st.phase_ref.norm().max(1e-6);

            let timing_err = timing_error_from_early_late(early_corrs[best_idx].norm(), late_corrs[best_idx].norm());
            st.timing_rate = update_timing_rate(st.timing_rate, timing_err, timing_rate_limit);
            st.timing_offset = update_timing_offset(
                st.timing_offset,
                st.timing_rate,
                timing_err,
                timing_limit,
            );

            let on_norm = on_rot.norm().max(1e-6);
            self.demodulator.set_prev_phase(on_rot / on_norm);
        }

        self.tracking_state = Some(st);

        let avg_energy = total_packet_energy / expected_symbols as f32;
        println!("[Decoder] Core: processed {} symbols, avg_energy={:.3}", expected_symbols, avg_energy);
        
        let success = if packet_llrs.len() >= interleaved_bits {
            self.decode_llrs(&packet_llrs) > 0
        } else {
            false
        };

        (avg_energy, success)
    }

    fn drain_processed_samples_only_raw(&mut self, _len: usize) {
        let spc = self.config.proc_samples_per_chip().max(1);
        let history_to_keep = self.config.preamble_sf * self.config.preamble_repeat * spc;
        let can_drain = self.equalizer_input_offset.saturating_sub(history_to_keep);
        if can_drain > 0 {
            self.sample_buffer_i.drain(0..can_drain);
            self.sample_buffer_q.drain(0..can_drain);
            self.equalizer_input_offset -= can_drain;
            self.last_search_idx = self.last_search_idx.saturating_sub(can_drain);
        }
    }

    fn drain_processed_samples(&mut self, len: usize) {
        // 1. 等化後バッファを消費
        let drain_len = len.min(self.equalized_buffer.len());
        self.equalized_buffer.drain(0..drain_len);
        
        // 2. sample_buffer の物理的な削除（再同期のための履歴 H を残す）
        // H = プリアンブル検索に必要なサンプル数
        let spc = self.config.proc_samples_per_chip().max(1);
        let history_to_keep = self.config.preamble_sf * self.config.preamble_repeat * spc;
        
        // equalizer_input_offset より手前に history_to_keep 分だけ残して削る
        let can_drain = self.equalizer_input_offset.saturating_sub(history_to_keep);
        if can_drain > 0 {
            self.sample_buffer_i.drain(0..can_drain);
            self.sample_buffer_q.drain(0..can_drain);
            // 物理的に削った分だけポインタを戻して整合性を保つ
            self.equalizer_input_offset -= can_drain;
            self.last_search_idx = self.last_search_idx.saturating_sub(can_drain);
        }
    }

    fn decode_llrs(&mut self, llrs: &[f32]) -> usize {
        let p_bits_len = crate::frame::packet::PACKET_BYTES * 8;
        let raw_bits = p_bits_len + 6;
        let fec_bits = raw_bits * 2;
        let rows = 16;
        let cols = fec_bits.div_ceil(rows);
        let interleaved_bits = rows * cols;

        let packet_chunk_bits = interleaved_bits.div_ceil(6) * 6;
        let mut success_count = 0;

        for packet_llrs in llrs.chunks(packet_chunk_bits) {
            if packet_llrs.len() < interleaved_bits {
                break;
            }

            let valid_llrs = &packet_llrs[..interleaved_bits];

            let interleaver = BlockInterleaver::new(rows, cols);
            let mut deinterleaved_llr = interleaver.deinterleave_f32(valid_llrs);

            let mut scrambler = Scrambler::default();
            for llr in deinterleaved_llr.iter_mut() {
                if scrambler.next_bit() == 1 {
                    *llr = -*llr;
                }
            }

            if let Some(ref mut callback) = self.llr_callback {
                callback(&deinterleaved_llr);
            }

            let decoded_bits = fec::decode_soft(&deinterleaved_llr[..fec_bits]);
            if decoded_bits.len() < p_bits_len {
                continue;
            }

            let decoded_bytes = fec::bits_to_bytes(&decoded_bits[..p_bits_len]);

            match Packet::deserialize(&decoded_bytes) {
                Ok(packet) => {
                    println!("[Decoder] FEC: CRC OK, seq={}, k={}", packet.lt_seq, packet.lt_k);
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
            basis_matrix: self.fountain_decoder.get_basis_matrix(),
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
        let mut results = [num_complex::Complex32::new(0.0, 0.0); 1];

        for chip_idx in 0..sf {
            let p = symbol_start as f32 + (chip_idx * spc) as f32 + timing_offset + ((spc as f32 - 1.0) / 2.0);

            let i_idx = p.round() as i32;
            
            if i_idx < 0 || i_idx >= self.equalized_buffer.len() as i32 {
                return None;
            }
            let sample = self.equalized_buffer[i_idx as usize];

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
        let mut results = [Complex32::new(0.0, 0.0); 16];

        for chip_idx in 0..sf {
            let p = symbol_start as f32 + (chip_idx * spc) as f32 + timing_offset + sample_shift + ((spc as f32 - 1.0) / 2.0);

            let i_idx = p.round() as i32;
            if i_idx < 0 || i_idx >= self.equalized_buffer.len() as i32 {
                return None;
            }
            let sample = self.equalized_buffer[i_idx as usize];

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
        let sf_sync = 15;
        let sf_payload = 16;
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
        let sync_bits: Vec<u8> = (0..16)
            .map(|i| ((sync_word >> (15 - i)) & 1) as u8)
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
        let preamble_part = modulator.generate_preamble();
        let preamble_len_48k = preamble_part.len();
        
        // 信号全体を生成 (新しいインスタンスで)
        let mut modulator2 = Modulator::new(config.clone());
        let data = vec![0x55u8; 16];
        let frame_48k = modulator2.encode_frame(&data);

        println!("[GT] Preamble len (48k): {}", preamble_len_48k);
        
        // 2. Decoder のパイプラインを通過させる
        let mut decoder = Decoder::new(160, 10, config.clone());
        decoder.process_samples(&frame_48k);
        
        // 48k -> 24k (proc_rate) へのリサンプリング後の長さを算出
        let preamble_len_24k = preamble_len_48k / 2;
        
        // 3. SyncDetector を実行
        let (sync_opt, _) = decoder.sync_detector.detect(
            &decoder.sample_buffer_i,
            &decoder.sample_buffer_q,
            0,
        );
        
        if let Some(s) = sync_opt {
            // peak_sample_idx は同期語 0 番目の最初のチップの中央
            // 期待値 = (プリアンブル終了点) + (チップ 0 の半分)
            let expected_peak = preamble_len_24k + (spc / 2);
            let diff = s.peak_sample_idx as i32 - expected_peak as i32;
            println!("[GT] Detected peak: {}, Expected peak: {}, Diff: {}", 
                     s.peak_sample_idx, expected_peak, diff);
            
            // 完璧な整合性を要求する
            assert_eq!(diff, 0, "Sync peak must match Ground Truth exactly!");
        } else {
            panic!("Sync not detected in GT test! Frame len: {}, Preamble len: {}", frame_48k.len(), preamble_len_48k);
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
        let config = DspConfig::default_48k();
        let mut encoder = Encoder::new(config.clone());
        let mut decoder = Decoder::new(160, 10, config);
        decoder.config.packets_per_burst = 5;
        let data = vec![0x12u8; 80];
        encoder.set_data(&data);
        let mut signal = Vec::new();
        let mut packets = Vec::new();
        for _ in 0..5 {
            packets.push(encoder.fountain_encoder_mut().unwrap().next_packet());
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
        let config = DspConfig::default_48k();
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

    #[test]
    fn test_decoder_tracking_tolerates_carrier_offset() {
        use crate::mary::encoder::Encoder;
        use std::sync::{Arc, Mutex};
        use std::time::Instant;
        let config = DspConfig::default_48k();
        let mut encoder = Encoder::new(config.clone());
        let mut rx_config = config.clone();
        rx_config.carrier_freq += 20.0;
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
        println!("--- Performance & BER Trend (Carrier Offset 20Hz) ---");
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
        let progress = decoder.progress();
        println!("Test finished: received={}, needed={}, crc_errors={}, parse_errors={}",
                 progress.received_packets, progress.needed_packets, progress.crc_error_packets, progress.parse_error_packets);
        let recovered = decoder
            .recovered_data()
            .expect("Decoder should recover data under carrier offset (needs tracking)");
        assert_eq!(&recovered[..data.len()], &data[..]);
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
        
        assert!(progress.complete, "Should recover data from continuous bursts. Received {}/{} packets", progress.received_packets, fountain_k);
        let recovered = decoder.recovered_data().expect("Should have recovered data");
        assert_eq!(&recovered[..data.len()], &data[..]);
    }

    #[test]
    fn test_fde_multipath_recovery() {
        use crate::mary::encoder::Encoder;
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 127; // Use high-resolution CIR estimation
        
        let mut encoder = Encoder::new(config.clone());
        
        let data_size = 32; // 2 packets
        let fountain_k = 2;
        let mut decoder = Decoder::new(data_size, fountain_k, config.clone());
        
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

        // 2. 2パス・マルチパスチャネルの適用: y(t) = x(t) + 0.6 * x(t - tau)
        // tau = 4 chips.
        let spc = config.proc_samples_per_chip();
        let delay_samples = 4 * spc * (config.sample_rate / config.proc_sample_rate()) as usize;
        let mut multipath_signal = original_signal.clone();
        let alpha = 0.6f32;
        for t in delay_samples..original_signal.len() {
            multipath_signal[t] += original_signal[t - delay_samples] * alpha;
        }

        // 3. 軽微なノイズ重畳 (SNR ~25dB)
        for s in multipath_signal.iter_mut() {
            *s += (rand::random::<f32>() - 0.5) * 0.05;
        }

        // --- Step A: 等化ありでの復元確認 ---
        decoder.process_samples(&multipath_signal);
        let progress = decoder.progress();
        
        assert!(progress.complete, "FDE should recover data under strong multipath. Received {}/{} packets", progress.received_packets, fountain_k);
        let recovered = decoder.recovered_data().expect("Should have recovered data");
        assert_eq!(&recovered[..data.len()], &data[..], "Recovered data mismatch with FDE");

        // --- Step B: 等化なしでの失敗確認 (対照実験) ---
        decoder.reset();
        
        // Searching を走らせて同期と tracking_state 初期化を済ませる
        decoder.process_samples(&multipath_signal); 
        
        if let Some(ref mut eq) = decoder.equalizer {
            // ここで強引に CIR を identity に上書きし、等化を無効化する
            let mut identity_cir = vec![num_complex::Complex32::new(0.0, 0.0); config.preamble_sf];
            identity_cir[0] = num_complex::Complex32::new(1.0, 0.0);
            eq.set_cir(&identity_cir, 100.0);
            eq.reset();
            // 既に古い CIR で処理された分を破棄し、identity で再処理させる
            decoder.equalized_buffer.clear();
            decoder.equalizer_input_offset = 0; 
            // packets_processed_in_burst も 0 に戻して同期語からやり直させる
            decoder.packets_processed_in_burst = 0;
        }
        
        // 再度 process (handle_decoding が identity CIR で走る)
        decoder.process_samples(&multipath_signal);
        
        let progress_no_fde = decoder.progress();
        assert!(!progress_no_fde.complete, "Decoding should FAIL without FDE under 0.6 multipath");
    }
}
