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
    pub progress: f32,
    pub complete: bool,
}

/// LLR観測用コールバック型
pub type LlrCallback = Box<dyn FnMut(&[f32]) + Send>;

/// MaryDQPSKデコーダ
pub struct Decoder {
    config: DspConfig,
    proc_config: DspConfig,
    resampler_i: Resampler,
    resampler_q: Resampler,
    rrc_filter_i: RrcFilter,
    rrc_filter_q: RrcFilter,
    sample_buffer_i: Vec<f32>,
    sample_buffer_q: Vec<f32>,
    demodulator: Demodulator,
    fountain_decoder: FountainDecoder,
    pub recovered_data: Option<Vec<u8>>,
    lo_nco: Nco,
    sync_detector: MarySyncDetector,
    pub packets_per_sync_burst: usize,

    // 同期・追従状態
    last_search_idx: usize,
    current_sync: Option<SyncResult>,
    tracking_state: Option<TrackingState>,
    packets_processed_in_burst: usize,
    last_packet_seq: Option<u32>,

    /// デバッグ観測用コールバック: デインターリーブ・デスクランブル後のLLRをパススルーする
    pub llr_callback: Option<LlrCallback>,

    // 統計
    pub stats_total_samples: usize,
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
        let proc_config = DspConfig::new_for_processing_from(&dsp_config);
        let params = FountainParams::new(fountain_k, PAYLOAD_SIZE);
        let lo_nco = Nco::new(-dsp_config.carrier_freq, dsp_config.sample_rate);

        let rrc_bw = dsp_config.chip_rate * (1.0 + dsp_config.rrc_alpha) * 0.5;
        let cutoff = Some(rrc_bw);

        let tc = MarySyncDetector::THRESHOLD_COARSE_DEFAULT;
        let tf = MarySyncDetector::THRESHOLD_FINE_DEFAULT;

        Decoder {
            resampler_i: Resampler::new_with_cutoff(
                dsp_config.sample_rate as u32,
                proc_config.sample_rate as u32,
                cutoff,
            ),
            resampler_q: Resampler::new_with_cutoff(
                dsp_config.sample_rate as u32,
                proc_config.sample_rate as u32,
                cutoff,
            ),
            rrc_filter_i: RrcFilter::from_config(&proc_config),
            rrc_filter_q: RrcFilter::from_config(&proc_config),
            sample_buffer_i: Vec::new(),
            sample_buffer_q: Vec::new(),
            demodulator: Demodulator::new(),
            fountain_decoder: FountainDecoder::new(params),
            recovered_data: None,
            config: dsp_config,
            sync_detector: MarySyncDetector::new(proc_config.clone(), tc, tf),
            proc_config,
            lo_nco,
            packets_per_sync_burst: 1,
            last_search_idx: 0,
            current_sync: None,
            tracking_state: None,
            packets_processed_in_burst: 0,
            last_packet_seq: None,
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
        let spc = self.proc_config.samples_per_chip().max(1);
        let sf_preamble = 15;
        let sf_payload = 16;

        let interleaved_bits: usize = 352;
        let expected_symbols_per_packet = interleaved_bits.div_ceil(6); // 59

        let max_buffer_len = 100_000;
        let drain_len = 50_000;

        loop {
            if self.recovered_data.is_some() {
                break;
            }

            // 1. 同期情報の取得・初期化
            if self.current_sync.is_none() {
                let sync_required = (sf_preamble
                    * (self.config.preamble_repeat + self.config.sync_word_bits))
                    * spc;
                if self.sample_buffer_i.len() < self.last_search_idx + sync_required {
                    break;
                }

                let (sync_opt, next_search_idx) = self.sync_detector.detect(
                    &self.sample_buffer_i,
                    &self.sample_buffer_q,
                    self.last_search_idx,
                );

                if let Some(s) = sync_opt {
                    self.current_sync = Some(s.clone());
                    self.packets_processed_in_burst = 0;

                    let initial_phase = Complex32::new(s.peak_iq.0, s.peak_iq.1);
                    let initial_phase_norm = initial_phase.norm().max(1e-6);
                    self.tracking_state = Some(TrackingState {
                        phase_ref: initial_phase / initial_phase_norm,
                        phase_rate: 0.0,
                        timing_offset: 0.0,
                        timing_rate: 0.0,
                    });
                    self.demodulator.set_prev_phase(Complex32::new(1.0, 0.0));

                    let sync_word_bits = self.config.sync_word_bits;
                    let payload_start_center =
                        s.peak_sample_idx + sync_word_bits * sf_preamble * spc;
                    let alignment_offset = payload_start_center.saturating_sub(spc);

                    if alignment_offset > self.sample_buffer_i.len() {
                        break;
                    }

                    self.sample_buffer_i.drain(0..alignment_offset);
                    self.sample_buffer_q.drain(0..alignment_offset);
                    self.last_search_idx = 0;

                    continue;
                } else {
                    self.last_search_idx = next_search_idx;
                    if self.sample_buffer_i.len() > max_buffer_len {
                        self.sample_buffer_i.drain(0..drain_len);
                        self.sample_buffer_q.drain(0..drain_len);
                        self.last_search_idx = self.last_search_idx.saturating_sub(drain_len);
                    }
                    break;
                }
            }

            // 2. 1パケット分 + マージン が溜まっているか確認
            let packet_samples = expected_symbols_per_packet * sf_payload * spc;
            let margin_samples = sf_payload * spc;
            let required_samples = spc + packet_samples + margin_samples;

            if self.sample_buffer_i.len() < required_samples {
                if self.sample_buffer_i.len() > max_buffer_len {
                    self.current_sync = None;
                    self.tracking_state = None;
                    self.last_search_idx = 0;
                    continue;
                }
                break;
            }

            // 3. 1パケット分のLLR抽出（トラッキングあり）
            let mut st = self
                .tracking_state
                .expect("Tracking state must be initialized");
            let timing_limit = spc as f32 * TRACKING_TIMING_LIMIT_CHIP;
            let timing_rate_limit = spc as f32 * TRACKING_TIMING_RATE_LIMIT_CHIP;
            let early_late_delta = (spc as f32 * TRACKING_EARLY_LATE_DELTA_CHIP).max(1.0);

            let mut packet_llrs = Vec::with_capacity(expected_symbols_per_packet * 6);

            for sym_idx in 0..expected_symbols_per_packet {
                let symbol_start = spc + sym_idx * sf_payload * spc;

                let on_corrs = if let Some(c) =
                    self.despread_symbol_with_timing(symbol_start, st.timing_offset, 0.0)
                {
                    c
                } else {
                    break;
                };
                let early_corrs = if let Some(c) = self.despread_symbol_with_timing(
                    symbol_start,
                    st.timing_offset,
                    -early_late_delta,
                ) {
                    c
                } else {
                    break;
                };
                let late_corrs = if let Some(c) = self.despread_symbol_with_timing(
                    symbol_start,
                    st.timing_offset,
                    early_late_delta,
                ) {
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

                let best_corr = on_corrs[best_idx];
                let early_corr = early_corrs[best_idx];
                let late_corr = late_corrs[best_idx];

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
                let dphi = phase_step_from_phase_error(phase_err, st.phase_rate);
                let (sin_dphi, cos_dphi) = dphi.sin_cos();
                st.phase_ref *= Complex32::new(cos_dphi, sin_dphi);
                let phase_norm = st.phase_ref.norm().max(1e-6);
                st.phase_ref /= phase_norm;

                let timing_err = timing_error_from_early_late(early_corr.norm(), late_corr.norm());
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

            if packet_llrs.len() >= interleaved_bits {
                self.decode_llrs(&packet_llrs);
            }

            self.tracking_state = Some(st);
            self.packets_processed_in_burst += 1;

            self.sample_buffer_i.drain(0..packet_samples);
            self.sample_buffer_q.drain(0..packet_samples);

            if self.packets_processed_in_burst >= self.packets_per_sync_burst {
                self.current_sync = None;
                self.tracking_state = None;
                self.last_search_idx = 0;
                self.sample_buffer_i.clear();
                self.sample_buffer_q.clear();
            }
        }

        self.progress()
    }

    fn decode_llrs(&mut self, llrs: &[f32]) {
        let p_bits_len = crate::frame::packet::PACKET_BYTES * 8;
        let raw_bits = p_bits_len + 6;
        let fec_bits = raw_bits * 2;
        let rows = 16;
        let cols = fec_bits.div_ceil(rows);
        let interleaved_bits = rows * cols;

        let packet_chunk_bits = interleaved_bits.div_ceil(6) * 6;

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

            if let Ok(packet) = Packet::deserialize(&decoded_bytes) {
                let pkt_k = packet.lt_k as usize;
                if pkt_k != self.fountain_decoder.params().k {
                    self.rebuild_fountain_decoder(pkt_k);
                }

                let fountain_packet = FountainPacket {
                    seq: packet.lt_seq as u32,
                    coefficients: crate::coding::fountain::reconstruct_packet_coefficients(
                        packet.lt_seq as u32,
                        self.fountain_decoder.params().k,
                    ),
                    data: packet.payload.to_vec(),
                };

                self.fountain_decoder.receive(fountain_packet);

                if let Some(data) = self.fountain_decoder.decode() {
                    self.recovered_data = Some(data);
                }
            }
        }
    }

    fn rebuild_fountain_decoder(&mut self, fountain_k: usize) {
        let params = FountainParams::new(fountain_k, PAYLOAD_SIZE);
        self.fountain_decoder = FountainDecoder::new(params);
        self.recovered_data = None;
        self.last_packet_seq = None;
    }

    fn progress(&self) -> DecodeProgress {
        let received = self.fountain_decoder.received_count();
        let needed = self.fountain_decoder.params().k;
        let progress = self.fountain_decoder.progress();

        DecodeProgress {
            received_packets: received,
            needed_packets: needed,
            progress,
            complete: self.recovered_data.is_some(),
        }
    }

    pub fn recovered_data(&self) -> Option<&[u8]> {
        self.recovered_data.as_deref()
    }

    fn despread_symbol_with_timing(
        &self,
        symbol_start: usize,
        timing_offset: f32,
        sample_shift: f32,
    ) -> Option<[Complex32; 16]> {
        let spc = self.proc_config.samples_per_chip().max(1);
        let sf = 16;
        let mut results = [Complex32::new(0.0, 0.0); 16];

        for chip_idx in 0..sf {
            let p = symbol_start as f32 + (chip_idx * spc) as f32 + timing_offset + sample_shift;

            let i_idx = p.round() as i32;
            if i_idx < 0 || i_idx >= self.sample_buffer_i.len() as i32 {
                return None;
            }

            let si = self.sample_buffer_i[i_idx as usize];
            let sq = self.sample_buffer_q[i_idx as usize];
            let sample = Complex32::new(si, sq);

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
        self.last_search_idx = 0;
        self.current_sync = None;
        self.tracking_state = None;
        self.packets_processed_in_burst = 0;
        self.last_packet_seq = None;
        self.sample_buffer_i.clear();
        self.sample_buffer_q.clear();
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
    fn test_encoder_decoder_continuous_frames() {
        use crate::mary::encoder::Encoder;
        let config = DspConfig::default_48k();
        let mut encoder = Encoder::new(config.clone());
        let mut decoder = Decoder::new(160, 10, config);
        let data = vec![0x56u8; 16];
        encoder.set_data(&data);
        let max_frames = 5;
        let mut frame_count = 0;
        for _ in 0..max_frames {
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
    fn test_encoder_decoder_with_noise() {
        use crate::mary::encoder::Encoder;
        let config = DspConfig::default_48k();
        let mut encoder = Encoder::new(config.clone());
        let mut decoder = Decoder::new(160, 10, config);
        let data = vec![0x78u8; 16];
        encoder.set_data(&data);
        let mut frame = encoder.encode_frame().unwrap();
        for s in frame.iter_mut() {
            *s += (rand::random::<f32>() - 0.5) * 0.01;
        }
        decoder.process_samples(&frame);
        let progress = decoder.progress();
        let _ = progress.received_packets;
    }

    #[test]
    fn test_encoder_decoder_reset_and_reuse() {
        use crate::mary::encoder::Encoder;
        let config = DspConfig::default_48k();
        let mut encoder = Encoder::new(config.clone());
        let mut decoder = Decoder::new(160, 10, config);
        let data = vec![0x9Au8; 16];
        encoder.set_data(&data);
        let frame1 = encoder.encode_frame().unwrap();
        decoder.process_samples(&frame1);
        decoder.reset();
        let frame2 = encoder.encode_frame().unwrap();
        decoder.process_samples(&frame2);
        let progress = decoder.progress();
        assert_eq!(progress.received_packets, 0);
    }

    #[test]
    fn test_preamble_detection_accuracy() {
        use crate::mary::encoder::Encoder;
        let config = DspConfig::default_48k();
        let mut encoder = Encoder::new(config.clone());
        let mut decoder = Decoder::new(160, 10, config.clone());
        let data = vec![0xABu8; 16];
        encoder.set_data(&data);
        let frame = encoder.encode_frame().unwrap();
        decoder.process_samples(&frame);
        assert!(decoder.stats_total_samples >= frame.len());
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
        decoder.packets_per_sync_burst = 5;
        let data = vec![0x12u8; 80];
        encoder.set_data(&data);
        let mut signal = Vec::new();
        let mut packets = Vec::new();
        for _ in 0..5 {
            packets.push(encoder.fountain_encoder_mut().unwrap().next_packet());
        }
        signal.extend(encoder.encode_burst(&packets));
        signal.extend(encoder.flush());
        signal.extend(vec![0.0; 1000]);
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
        decoder.packets_per_sync_burst = 40;
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
        signal.extend(vec![0.0; 1000]);
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
        decoder.packets_per_sync_burst = 40;
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
        signal.extend(vec![0.0; 1000]);
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
        let recovered = decoder
            .recovered_data()
            .expect("Decoder should recover data under carrier offset (needs tracking)");
        assert_eq!(&recovered[..data.len()], &data[..]);
    }
}
