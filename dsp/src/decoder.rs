//! 受信パイプライン (統合デコーダ)

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::common::nco::complex_mul_interleaved2_simd;
use crate::{
    coding::fec,
    coding::fountain::{FountainDecoder, FountainPacket, FountainParams, ReceiveOutcome},
    coding::interleaver::BlockInterleaver,
    common::nco::Nco,
    common::rrc_filter::DecimatingRrcFilter,
    frame::packet::{Packet, PacketParseError, PACKET_BYTES},
    params::{MODULATION, PACKETS_PER_SYNC_BURST, PAYLOAD_SIZE, SYNC_WORD_BITS},
    phy::sync::{SyncDetector, SyncResult},
    DifferentialModulation, DspConfig,
};
use num_complex::Complex32;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use std::arch::wasm32::{f32x4, v128, v128_store};
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

const TRACKING_TIMING_PROP_GAIN: f32 = 0.18;
const TRACKING_TIMING_RATE_GAIN: f32 = 0.01;
const TRACKING_PHASE_PROP_GAIN: f32 = 0.22;
const TRACKING_PHASE_FREQ_GAIN: f32 = 0.015;
const TRACKING_TIMING_LIMIT_CHIP: f32 = 2.0;
const TRACKING_TIMING_RATE_LIMIT_CHIP: f32 = 0.25;
const TRACKING_EARLY_LATE_DELTA_CHIP: f32 = 0.5;
const TRACKING_PHASE_RATE_LIMIT_RAD: f32 = 2.6;
const TRACKING_PHASE_STEP_CLAMP: f32 = 2.8;
const ITERATION_BUDGET_MIN: usize = 2;
const ITERATION_BUDGET_MAX: usize = 8;
const ITERATION_BUDGET_HEADROOM: usize = 1;
const LLR_CLIP_ABS: f32 = 6.0;
const LLR_NOISE_EMA_ALPHA: f32 = 0.04;
const LLR_NOISE_VAR_MIN: f32 = 0.02;
const LLR_NOISE_VAR_MAX: f32 = 2.0;
const LLR_PHASE_ERR_ERASE_RAD: f32 = 0.55;
const LLR_TIMING_ERR_ERASE: f32 = 0.45;

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
}

pub struct Decoder {
    config: DspConfig,
    proc_config: DspConfig,
    rrc_decim_i: DecimatingRrcFilter,
    rrc_decim_q: DecimatingRrcFilter,
    sample_buffer_i: Vec<f32>,
    sample_buffer_q: Vec<f32>,
    sync_detector: SyncDetector,
    interleaver: BlockInterleaver,
    fountain_decoder: FountainDecoder,
    recovered_data: Option<Vec<u8>>,
    pub agc_peak_fast: f32,
    pub agc_peak_slow: f32,
    decimation_factor: usize,
    packets_per_sync_burst: usize,
    lo_nco: Nco,

    // --- 同期状態 ---
    last_search_idx: usize,
    current_sync: Option<SyncResult>,
    last_packet_seq: Option<u32>,
    last_rank_up_seq: Option<u32>,
    dependent_packets: usize,
    duplicate_packets: usize,
    crc_error_packets: usize,
    parse_error_packets: usize,
    invalid_neighbor_packets: usize,

    // --- 統計用 ---
    pub stats_sync_calls: usize,
    pub stats_sync_time: Duration,
    pub stats_total_samples: usize,
}

#[derive(Clone, Copy, Debug)]
struct TrackingState {
    phase_ref: Complex32,
    prev_symbol: Complex32,
    phase_rate: f32,
    timing_offset: f32,
    timing_rate: f32,
    noise_var: f32,
}

#[derive(Debug)]
struct DecodedSoftBits {
    llrs: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
struct SymbolSoftDecision {
    decided: Complex32,
    llrs: [f32; 2],
    llr_count: usize,
}

impl Decoder {
    pub fn new(_data_size: usize, fountain_k: usize, dsp_config: DspConfig) -> Self {
        let decimation_factor = choose_decimation_factor(&dsp_config);
        let proc_config = build_proc_config(&dsp_config, decimation_factor);
        let params = FountainParams::new(fountain_k, PAYLOAD_SIZE);
        let raw_bits = PACKET_BYTES * 8 + 6;
        let fec_bits = raw_bits * 2;
        let il_rows = 16;
        let il_cols = fec_bits.div_ceil(16);
        let lo_nco = Nco::new(-dsp_config.carrier_freq, dsp_config.sample_rate);

        Decoder {
            rrc_decim_i: DecimatingRrcFilter::from_config(&dsp_config, decimation_factor),
            rrc_decim_q: DecimatingRrcFilter::from_config(&dsp_config, decimation_factor),
            sample_buffer_i: Vec::new(),
            sample_buffer_q: Vec::new(),
            sync_detector: SyncDetector::new(proc_config.clone()),
            interleaver: BlockInterleaver::new(il_rows, il_cols),
            fountain_decoder: FountainDecoder::new(params),
            recovered_data: None,
            config: dsp_config,
            proc_config,
            agc_peak_fast: 0.5,
            agc_peak_slow: 0.5,
            decimation_factor,
            packets_per_sync_burst: PACKETS_PER_SYNC_BURST,
            lo_nco,
            last_search_idx: 0,
            current_sync: None,
            last_packet_seq: None,
            last_rank_up_seq: None,
            dependent_packets: 0,
            duplicate_packets: 0,
            crc_error_packets: 0,
            parse_error_packets: 0,
            invalid_neighbor_packets: 0,
            stats_sync_calls: 0,
            stats_sync_time: Duration::ZERO,
            stats_total_samples: 0,
        }
    }

    pub fn process_samples(&mut self, samples: &[f32]) -> DecodeProgress {
        if self.recovered_data.is_some() {
            return self.progress();
        }
        self.stats_total_samples += samples.len();

        let spc = self.proc_config.samples_per_chip();
        let sf = self.config.spread_factor();
        let max_buffer_len = (100_000 / self.decimation_factor.max(1)).max(1);
        let drain_len = (50_000 / self.decimation_factor.max(1)).max(1);

        let mut i_mixed = Vec::with_capacity(samples.len());
        let mut q_mixed = Vec::with_capacity(samples.len());
        self.mix_real_to_iq(samples, &mut i_mixed, &mut q_mixed);

        let mut i_decimated = Vec::new();
        let mut q_decimated = Vec::new();
        self.rrc_decim_i.process_block(&i_mixed, &mut i_decimated);
        self.rrc_decim_q.process_block(&q_mixed, &mut q_decimated);
        self.sample_buffer_i.extend_from_slice(&i_decimated);
        self.sample_buffer_q.extend_from_slice(&q_decimated);

        let sync_bits_len = SYNC_WORD_BITS;
        let sync_symbol_len = sync_bits_len;
        let bits_per_symbol_payload = MODULATION.bits_per_symbol();
        let fec_bits_len = self.interleaver.rows() * self.interleaver.cols();
        let burst_data_bits_len = fec_bits_len * self.packets_per_sync_burst.max(1);
        let payload_symbols = burst_data_bits_len.div_ceil(bits_per_symbol_payload);
        let total_symbols = sync_symbol_len + payload_symbols;
        let symbol_len = sf * spc;
        let frame_samples = (total_symbols * symbol_len).max(1);
        let queued_frames = self.sample_buffer_i.len() / frame_samples;
        // バッファ量に応じて反復回数を決める。過不足を避けるために上下限を設ける。
        let mut iteration_budget = (queued_frames + ITERATION_BUDGET_HEADROOM)
            .clamp(ITERATION_BUDGET_MIN, ITERATION_BUDGET_MAX);

        loop {
            if iteration_budget == 0 {
                break;
            }
            iteration_budget -= 1;

            if self.recovered_data.is_some() {
                break;
            }

            // 1. 同期情報の取得
            let sync = if let Some(s) = self.current_sync.clone() {
                s
            } else {
                #[cfg(not(target_arch = "wasm32"))]
                let sync_start = Instant::now();
                let (sync_opt, next_search_idx) = self.sync_detector.detect(
                    &self.sample_buffer_i,
                    &self.sample_buffer_q,
                    self.last_search_idx,
                );
                self.stats_sync_calls += 1;
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.stats_sync_time += sync_start.elapsed();
                }

                if let Some(s) = sync_opt {
                    self.current_sync = Some(s.clone());
                    s
                } else {
                    self.last_search_idx = next_search_idx;
                    if self.sample_buffer_i.len() > max_buffer_len {
                        let drain = drain_len;
                        self.sample_buffer_i.drain(0..drain);
                        self.sample_buffer_q.drain(0..drain);
                        self.last_search_idx = self.last_search_idx.saturating_sub(drain);
                    }
                    break;
                }
            };

            let start = sync.peak_sample_idx;
            let data_end_sample = start + total_symbols * sf * spc;

            // データが溜まるのを待つ (start は既に SYNC_WORD の開始点付近)
            if self.sample_buffer_i.len() < data_end_sample {
                // タイムアウト監視
                if start + symbol_len < self.sample_buffer_i.len().saturating_sub(max_buffer_len) {
                    self.current_sync = None;
                    self.last_search_idx = 0;
                    continue;
                }
                break;
            }

            // --- Unified Sync Trust Integration ---
            // SyncDetector が既に 36シンボルで SYNC_WORD を検証済みのため、
            // 改めて Probing する必要はない。

            // start は SYNC_WORD 第1シンボルの中心を指している。
            // ペイロードの開始位置はそこから 32シンボル先。
            let payload_start = start + sync_symbol_len * sf * spc;

            // SyncDetector が提供した IQ値を基準位相として使用する。
            // これにより位相反転の曖昧さが完全に解消される。
            let initial_ref = Complex32::new(sync.peak_iq.0, sync.peak_iq.1);
            let initial_ref_norm = initial_ref.norm().max(1e-6);

            let best_tracking_after_sync = TrackingState {
                phase_ref: initial_ref / initial_ref_norm,
                prev_symbol: initial_ref / initial_ref_norm,
                phase_rate: 0.0,
                timing_offset: 0.0,
                timing_rate: 0.0,
                noise_var: 0.2,
            };

            let mut mseq = crate::common::msequence::MSequence::new(self.config.mseq_order);
            let pn: Vec<f32> = mseq.generate(sf).into_iter().map(|v| v as f32).collect();

            let p_bits_len = PACKET_BYTES * 8;
            let Some((decoded_packets, crc_errors, parse_errors)) = self
                .decode_payload_with_timing_retries(
                    payload_start,
                    burst_data_bits_len,
                    best_tracking_after_sync,
                    &pn,
                    fec_bits_len,
                    p_bits_len,
                )
            else {
                // デコードに失敗した場合も、同じ場所を繰り返さないよう進める。
                self.last_search_idx = (start + symbol_len).min(self.sample_buffer_i.len());
                self.current_sync = None;
                continue;
            };
            self.crc_error_packets += crc_errors;
            self.parse_error_packets += parse_errors;

            if decoded_packets.is_empty() {
                // 同期語一致後に payload が全滅したら境界ずれを疑う
                self.last_search_idx = (start + (symbol_len / 2).max(spc)).min(self.sample_buffer_i.len());
                self.current_sync = None;
                continue;
            }

            for packet in decoded_packets {
                // ... (FountainDecoder への供給ロジック) ...
                let pkt_k = packet.lt_k as usize;
                if pkt_k != self.fountain_decoder.params().k {
                    self.rebuild_fountain_decoder(pkt_k);
                }
                let seq = packet.lt_seq as u32;
                let coefficients = crate::coding::fountain::reconstruct_packet_coefficients(
                    seq,
                    self.fountain_decoder.params().k,
                );
                let outcome = self.fountain_decoder.receive_with_outcome(FountainPacket {
                    seq,
                    coefficients,
                    data: packet.payload.to_vec(),
                });
                match outcome {
                    ReceiveOutcome::AcceptedRankUp => {
                        self.last_packet_seq = Some(seq);
                        self.last_rank_up_seq = Some(seq);
                    }
                    ReceiveOutcome::AcceptedNoRankUp => {
                        self.last_packet_seq = Some(seq);
                        self.dependent_packets += 1;
                    }
                    ReceiveOutcome::DuplicateSeq => {
                        self.duplicate_packets += 1;
                    }
                    ReceiveOutcome::InvalidPacket => {
                        self.invalid_neighbor_packets += 1;
                    }
                }
                if let Some(data) = self.fountain_decoder.decode() {
                    self.recovered_data = Some(data);
                    break;
                }
            }
            // 受信窓をバースト分前進させる
            let actual_end = (payload_start + burst_data_bits_len / bits_per_symbol_payload * sf * spc).min(self.sample_buffer_i.len());
            self.sample_buffer_i.drain(0..actual_end);
            self.sample_buffer_q.drain(0..actual_end);
            self.last_search_idx = 0;
            self.current_sync = None;
        }

        self.progress()
    }

    fn despread_symbol_with_timing(
        &self,
        symbol_start: usize,
        pn: &[f32],
        timing_offset: f32,
        sample_shift: f32,
    ) -> Option<Complex32> {
        let spc = self.proc_config.samples_per_chip().max(1);
        let mut sum_i = 0.0f32;
        let mut sum_q = 0.0f32;
        for (chip_idx, &pn_val) in pn.iter().enumerate() {
            let p = symbol_start as f32
                + (chip_idx * spc + (spc / 2)) as f32
                + timing_offset
                + sample_shift;
            let si = sample_at_fractional(&self.sample_buffer_i, p)?;
            let sq = sample_at_fractional(&self.sample_buffer_q, p)?;
            sum_i += si * pn_val;
            sum_q += sq * pn_val;
        }
        let inv_sf = 1.0f32 / pn.len() as f32;
        Some(Complex32::new(sum_i * inv_sf, sum_q * inv_sf))
    }

    fn decode_bits_with_tracking(
        &self,
        start_sample: usize,
        num_bits: usize,
        initial_state: TrackingState,
        pn: &[f32],
    ) -> Option<DecodedSoftBits> {
        let sf = self.config.spread_factor();
        let spc = self.proc_config.samples_per_chip().max(1);
        let symbol_len = sf * spc;
        let bits_per_symbol = MODULATION.bits_per_symbol();
        let symbols_needed = num_bits.div_ceil(bits_per_symbol);

        let mut phase_ref = initial_state.phase_ref;
        let mut prev_symbol = initial_state.prev_symbol;
        let mut phase_rate = initial_state.phase_rate;
        let mut timing_offset = initial_state.timing_offset;
        let mut timing_rate = initial_state.timing_rate;
        let timing_limit = spc as f32 * TRACKING_TIMING_LIMIT_CHIP;
        let timing_rate_limit = spc as f32 * TRACKING_TIMING_RATE_LIMIT_CHIP;
        let early_late_delta = (spc as f32 * TRACKING_EARLY_LATE_DELTA_CHIP).max(1.0);
        let mut llrs = Vec::with_capacity(num_bits);
        let mut noise_var = initial_state
            .noise_var
            .clamp(LLR_NOISE_VAR_MIN, LLR_NOISE_VAR_MAX);

        for s_idx in 0..symbols_needed {
            let symbol_start = start_sample + s_idx * symbol_len;
            let on = self.despread_symbol_with_timing(symbol_start, pn, timing_offset, 0.0)?;
            let early = self.despread_symbol_with_timing(
                symbol_start,
                pn,
                timing_offset,
                -early_late_delta,
            )?;
            let late = self.despread_symbol_with_timing(
                symbol_start,
                pn,
                timing_offset,
                early_late_delta,
            )?;

            let on_rot = on * phase_ref.conj();
            let diff = on_rot * prev_symbol.conj();
            let soft = decode_diff_symbol_soft(diff);
            let decided = soft.decided;
            noise_var = update_noise_var_ema(noise_var, diff, decided);
            let phase_err = phase_error_from_diff(diff, decided);
            phase_rate = update_phase_rate(phase_rate, phase_err);
            let dphi = phase_step_from_phase_error(phase_err, phase_rate);
            let (sin_dphi, cos_dphi) = dphi.sin_cos();
            phase_ref *= Complex32::new(cos_dphi, sin_dphi);
            let phase_norm = phase_ref.norm().max(1e-6);
            phase_ref /= phase_norm;

            // Early/Late timing tracking (PI loop)
            let early_mag = early.norm();
            let late_mag = late.norm();
            let timing_err = timing_error_from_early_late(early_mag, late_mag);
            timing_rate = update_timing_rate(timing_rate, timing_err, timing_rate_limit);
            timing_offset =
                update_timing_offset(timing_offset, timing_rate, timing_err, timing_limit);

            // LLR品質向上:
            // 1) 差動誤差から推定した雑音分散で正規化
            // 2) 位相誤差/タイミング誤差で減衰
            // 3) クリップで過信を抑制
            let on_norm = on_rot.norm();
            let quality = llr_quality(phase_err, timing_err);
            for &raw_llr in soft.llrs.iter().take(soft.llr_count) {
                if llrs.len() >= num_bits {
                    break;
                }
                let llr = condition_llr(raw_llr, noise_var, quality);
                llrs.push(llr);
            }

            if on_norm > 1e-4 {
                prev_symbol = on_rot / on_norm;
            } else {
                prev_symbol = decided;
            }
        }

        llrs.truncate(num_bits);
        Some(DecodedSoftBits { llrs })
    }

    fn decode_payload_with_timing_retries(
        &self,
        payload_start: usize,
        burst_data_bits_len: usize,
        initial_state: TrackingState,
        pn: &[f32],
        fec_bits_len: usize,
        p_bits_len: usize,
    ) -> Option<(Vec<Packet>, usize, usize)> {
        let spc = self.proc_config.samples_per_chip().max(1) as f32;
        let timing_limit = spc * TRACKING_TIMING_LIMIT_CHIP;
        let timing_biases = [-0.75f32 * spc, 0.0, 0.75f32 * spc];

        let mut best: Option<(Vec<Packet>, usize, usize)> = None;
        for bias in timing_biases {
            let mut st = initial_state;
            st.timing_offset = (st.timing_offset + bias).clamp(-timing_limit, timing_limit);
            let Some(decoded_soft_bits) =
                self.decode_bits_with_tracking(payload_start, burst_data_bits_len, st, pn)
            else {
                continue;
            };
            let candidate =
                self.parse_payload_packets(&decoded_soft_bits.llrs, fec_bits_len, p_bits_len);
            let replace = match &best {
                None => true,
                Some((best_packets, best_crc, best_parse)) => {
                    candidate.0.len() > best_packets.len()
                        || (candidate.0.len() == best_packets.len()
                            && (candidate.1 < *best_crc
                                || (candidate.1 == *best_crc && candidate.2 < *best_parse)))
                }
            };
            if replace {
                best = Some(candidate);
            }
        }

        best
    }

    fn parse_payload_packets(
        &self,
        payload_llrs: &[f32],
        fec_bits_len: usize,
        p_bits_len: usize,
    ) -> (Vec<Packet>, usize, usize) {
        let mut decoded_packets = Vec::new();
        let mut crc_errors = 0usize;
        let mut parse_errors = 0usize;
        for packet_llrs in payload_llrs.chunks_exact(fec_bits_len) {
            let mut deinterleaved_llr = self.interleaver.deinterleave_f32(packet_llrs);

            let mut scrambler = crate::coding::scrambler::Scrambler::default();
            for llr in deinterleaved_llr.iter_mut() {
                if scrambler.next_bit() == 1 {
                    *llr = -*llr;
                }
            }

            let decoded_soft = fec::decode_soft(&deinterleaved_llr);
            match parse_packet_from_decoded_bits(&decoded_soft, p_bits_len) {
                Ok(packet) => decoded_packets.push(packet),
                Err(PacketParseError::CrcMismatch { .. }) => crc_errors += 1,
                Err(PacketParseError::InvalidLength { .. }) => parse_errors += 1,
            }
        }
        (decoded_packets, crc_errors, parse_errors)
    }

    fn progress(&self) -> DecodeProgress {
        let received = self.fountain_decoder.received_count();
        let needed = self.fountain_decoder.needed_count();
        let rank = self.fountain_decoder.rank();
        DecodeProgress {
            received_packets: received,
            needed_packets: needed,
            rank_packets: rank,
            stalled_packets: received.saturating_sub(rank),
            dependent_packets: self.dependent_packets,
            duplicate_packets: self.duplicate_packets,
            crc_error_packets: self.crc_error_packets,
            parse_error_packets: self.parse_error_packets,
            invalid_neighbor_packets: self.invalid_neighbor_packets,
            last_packet_seq: self.last_packet_seq.map(|v| v as i32).unwrap_or(-1),
            last_rank_up_seq: self.last_rank_up_seq.map(|v| v as i32).unwrap_or(-1),
            progress: self.fountain_decoder.progress(),
            complete: self.recovered_data.is_some(),
        }
    }

    fn rebuild_fountain_decoder(&mut self, fountain_k: usize) {
        let params = FountainParams::new(fountain_k, PAYLOAD_SIZE);
        self.fountain_decoder = FountainDecoder::new(params);
        self.recovered_data = None;
        self.last_packet_seq = None;
        self.last_rank_up_seq = None;
        self.dependent_packets = 0;
        self.duplicate_packets = 0;
        self.crc_error_packets = 0;
        self.parse_error_packets = 0;
        self.invalid_neighbor_packets = 0;
    }

    #[inline]
    fn agc_scale(&mut self, sample: f32) -> f32 {
        let sample_abs = sample.abs();

        // 高速EMA: 信号の急激な変化に対応
        let fast_alpha_rise = 0.1;
        let fast_alpha_fall = 0.001;
        if sample_abs > self.agc_peak_fast {
            self.agc_peak_fast =
                self.agc_peak_fast * (1.0 - fast_alpha_rise) + sample_abs * fast_alpha_rise;
        } else {
            self.agc_peak_fast =
                self.agc_peak_fast * (1.0 - fast_alpha_fall) + sample_abs * fast_alpha_fall;
        }

        // 低速EMA: 安定したレベル推定
        let slow_alpha_rise = 0.01;
        let slow_alpha_fall = 0.0005;
        if sample_abs > self.agc_peak_slow {
            self.agc_peak_slow =
                self.agc_peak_slow * (1.0 - slow_alpha_rise) + sample_abs * slow_alpha_rise;
        } else {
            self.agc_peak_slow =
                self.agc_peak_slow * (1.0 - slow_alpha_fall) + sample_abs * slow_alpha_fall;
        }

        // 雑音過敏を防ぐため、低速EMAの1.5倍以上高速EMAが大きい場合のみ高速EMAを採用
        let peak = if self.agc_peak_fast > self.agc_peak_slow * 1.5 {
            self.agc_peak_fast
        } else {
            self.agc_peak_slow
        };

        let gain = if peak > 1e-6 { 0.5 / peak } else { 1.0 };
        sample * gain * 2.0
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn mix_real_to_iq(&mut self, samples: &[f32], i_out: &mut Vec<f32>, q_out: &mut Vec<f32>) {
        i_out.clear();
        q_out.clear();
        i_out.reserve(samples.len());
        q_out.reserve(samples.len());

        let mut idx = 0usize;
        let mut interleaved = [0.0f32; 16];
        while idx + 8 <= samples.len() {
            let s0 = self.agc_scale(samples[idx]);
            let s1 = self.agc_scale(samples[idx + 1]);
            let s2 = self.agc_scale(samples[idx + 2]);
            let s3 = self.agc_scale(samples[idx + 3]);
            let s4 = self.agc_scale(samples[idx + 4]);
            let s5 = self.agc_scale(samples[idx + 5]);
            let s6 = self.agc_scale(samples[idx + 6]);
            let s7 = self.agc_scale(samples[idx + 7]);

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
                i_out.push(pair[0]);
                q_out.push(pair[1]);
            }
            idx += 8;
        }

        for &sample in &samples[idx..] {
            let s = self.agc_scale(sample);
            let lo = self.lo_nco.step();
            i_out.push(s * lo.re);
            q_out.push(s * lo.im);
        }
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    fn mix_real_to_iq(&mut self, samples: &[f32], i_out: &mut Vec<f32>, q_out: &mut Vec<f32>) {
        i_out.clear();
        q_out.clear();
        i_out.reserve(samples.len());
        q_out.reserve(samples.len());
        for &sample in samples {
            let s = self.agc_scale(sample);
            let lo = self.lo_nco.step();
            i_out.push(s * lo.re);
            q_out.push(s * lo.im);
        }
    }

    pub fn reset(&mut self) {
        self.interleaver.reset();
        self.rrc_decim_i.reset();
        self.rrc_decim_q.reset();
        self.sample_buffer_i.clear();
        self.sample_buffer_q.clear();
        self.lo_nco.reset();
        self.recovered_data = None;
        self.last_search_idx = 0;
        self.current_sync = None;
        self.last_packet_seq = None;
        self.last_rank_up_seq = None;
        self.dependent_packets = 0;
        self.duplicate_packets = 0;
        self.crc_error_packets = 0;
        self.parse_error_packets = 0;
        self.invalid_neighbor_packets = 0;
        self.rebuild_fountain_decoder(self.fountain_decoder.params().k);
    }

    pub fn recovered_data(&self) -> Option<&[u8]> {
        self.recovered_data.as_deref()
    }
}

#[inline]
fn timing_error_from_early_late(early_mag: f32, late_mag: f32) -> f32 {
    (late_mag - early_mag) / (late_mag + early_mag + 1e-6)
}

#[inline]
fn sample_at_fractional(buf: &[f32], pos: f32) -> Option<f32> {
    if pos < 0.0 {
        return None;
    }
    let i0 = pos.floor() as usize;
    let frac = (pos - i0 as f32).clamp(0.0, 1.0);

    if i0 >= buf.len() {
        return None;
    }
    if i0 + 1 >= buf.len() {
        if frac <= 1e-6 {
            return Some(buf[i0]);
        }
        return None;
    }

    let a = buf[i0];
    let b = buf[i0 + 1];
    Some(a + (b - a) * frac)
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
fn decode_diff_symbol_soft(diff: Complex32) -> SymbolSoftDecision {
    match MODULATION {
        DifferentialModulation::Dbpsk => {
            let decided = if diff.re >= 0.0 {
                Complex32::new(1.0, 0.0)
            } else {
                Complex32::new(-1.0, 0.0)
            };
            SymbolSoftDecision {
                decided,
                llrs: [4.0 * diff.re, 0.0],
                llr_count: 1,
            }
        }
        DifferentialModulation::Dqpsk => {
            let (symbol, _pair, pair_llr) = dqpsk_hard_bits_and_llr(diff);
            SymbolSoftDecision {
                decided: symbol,
                llrs: pair_llr,
                llr_count: 2,
            }
        }
    }
}

#[inline]
fn dqpsk_hard_bits_and_llr(diff: Complex32) -> (Complex32, [u8; 2], [f32; 2]) {
    // マッピング:
    // +1 -> 00, +j -> 01, -1 -> 11, -j -> 10
    let llr0 = 2.0 * (diff.re + diff.im);
    let llr1 = 2.0 * (diff.re - diff.im);

    let b0 = if llr0 >= 0.0 { 0u8 } else { 1u8 };
    let b1 = if llr1 >= 0.0 { 0u8 } else { 1u8 };

    let symbol = match (b0, b1) {
        (0, 0) => Complex32::new(1.0, 0.0),
        (0, 1) => Complex32::new(0.0, 1.0),
        (1, 0) => Complex32::new(0.0, -1.0),
        (1, 1) => Complex32::new(-1.0, 0.0),
        _ => unreachable!(),
    };

    (symbol, [b0, b1], [llr0, llr1])
}

#[inline]
fn condition_llr(raw_llr: f32, noise_var: f32, quality: f32) -> f32 {
    let nv = noise_var.clamp(LLR_NOISE_VAR_MIN, LLR_NOISE_VAR_MAX);
    (raw_llr * quality / nv).clamp(-LLR_CLIP_ABS, LLR_CLIP_ABS)
}

#[inline]
fn llr_quality(phase_err: f32, timing_err: f32) -> f32 {
    if phase_err.abs() > LLR_PHASE_ERR_ERASE_RAD || timing_err.abs() > LLR_TIMING_ERR_ERASE {
        return 0.0;
    }
    let phase_q = (1.0 - phase_err.abs() / 0.9).clamp(0.0, 1.0);
    let timing_q = (1.0 - timing_err.abs()).clamp(0.0, 1.0);
    phase_q * timing_q
}

#[inline]
fn estimate_noise_var_from_diff(diff: Complex32, decided_symbol: Complex32) -> f32 {
    let amp = diff.norm().max(1e-6);
    let diff_n = diff / amp;
    // 振幅のフェージング分散を含めると過少評価されるため、位相ズレをベースにする
    0.5 * (diff_n - decided_symbol).norm_sqr()
}

#[inline]
fn update_noise_var_ema(prev: f32, diff: Complex32, decided_symbol: Complex32) -> f32 {
    let inst = estimate_noise_var_from_diff(diff, decided_symbol)
        .clamp(LLR_NOISE_VAR_MIN, LLR_NOISE_VAR_MAX);
    ((1.0 - LLR_NOISE_EMA_ALPHA) * prev + LLR_NOISE_EMA_ALPHA * inst)
        .clamp(LLR_NOISE_VAR_MIN, LLR_NOISE_VAR_MAX)
}

#[inline]
fn parse_packet_from_decoded_bits(
    decoded_bits: &[u8],
    p_bits_len: usize,
) -> Result<Packet, PacketParseError> {
    if decoded_bits.len() < p_bits_len {
        return Err(PacketParseError::InvalidLength {
            actual: decoded_bits.len() / 8,
        });
    }
    let d_bytes = fec::bits_to_bytes(&decoded_bits[..p_bits_len]);
    Packet::deserialize(&d_bytes)
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

fn choose_decimation_factor(config: &DspConfig) -> usize {
    let spc = config.samples_per_chip();
    if spc >= 6 {
        3
    } else if spc >= 4 {
        2
    } else {
        1
    }
}

fn build_proc_config(config: &DspConfig, decimation_factor: usize) -> DspConfig {
    let mut proc_config = config.clone();
    proc_config.sample_rate = config.sample_rate / decimation_factor as f32;
    proc_config
}

impl Drop for Decoder {
    fn drop(&mut self) {
        let enable_stats = std::env::var("MISTCAST_DECODER_STATS")
            .map(|v| v == "1")
            .unwrap_or(false);
        if !enable_stats {
            return;
        }
        println!("\n--- Decoder Statistics ---");
        println!("  Total samples processed: {}", self.stats_total_samples);
        println!("  Total detect() calls: {}", self.stats_sync_calls);
        println!("  Total time in detect(): {:?}", self.stats_sync_time);
        if self.stats_sync_calls > 0 {
            println!(
                "  Avg time per detect(): {:?}",
                self.stats_sync_time / self.stats_sync_calls as u32
            );
        }
        println!("--------------------------\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coding::fountain::{FountainEncoder, FountainParams};
    use crate::params::FIXED_K;
    use crate::{
        encoder::{Encoder, EncoderConfig},
        DspConfig,
    };

    #[test]
    fn test_choose_decimation_factor() {
        assert_eq!(choose_decimation_factor(&DspConfig::default_48k()), 2);
        assert_eq!(choose_decimation_factor(&DspConfig::default_44k()), 1);
        assert_eq!(choose_decimation_factor(&DspConfig::new(16000.0)), 1);
    }

    #[test]
    fn test_build_proc_config() {
        let dsp_config = DspConfig::default_48k();
        let proc = build_proc_config(&dsp_config, 2);
        assert_eq!(proc.sample_rate, 24_000.0);
        assert_eq!(proc.carrier_freq, dsp_config.carrier_freq);
        assert_eq!(proc.chip_rate, dsp_config.chip_rate);
    }

    #[test]
    fn test_decoder_silence_input_does_not_complete() {
        let dsp_config = DspConfig::default_48k();
        let mut decoder = Decoder::new(32, FIXED_K, dsp_config);

        let silence = vec![0.0f32; 4096];
        let mut progress = decoder.process_samples(&silence);
        for _ in 0..4 {
            progress = decoder.process_samples(&silence);
        }

        assert!(!progress.complete);
        assert_eq!(progress.received_packets, 0);
        assert!(decoder.recovered_data().is_none());
    }

    #[test]
    fn test_agc_fast_attack_response() {
        let dsp_config = DspConfig::default_48k();
        let mut decoder = Decoder::new(32, FIXED_K, dsp_config);

        // 初期状態: 低レベルで十分に安定化
        let low_level = vec![0.01f32; 100];
        for _ in 0..100 {
            decoder.process_samples(&low_level);
        }
        let initial_fast_peak = decoder.agc_peak_fast;
        let _initial_slow_peak = decoder.agc_peak_slow;

        // 急激なレベル上昇（10倍）
        let high_level = vec![0.1f32; 100];
        decoder.process_samples(&high_level);

        // 高速EMAが追従していることを確認
        let fast_peak = decoder.agc_peak_fast;
        assert!(
            fast_peak > initial_fast_peak * 1.05,
            "Fast EMA should track level increase: initial={}, fast={}",
            initial_fast_peak,
            fast_peak
        );
    }

    #[test]
    fn test_agc_slow_decay_stability() {
        let dsp_config = DspConfig::default_48k();
        let mut decoder = Decoder::new(32, FIXED_K, dsp_config);

        // 高レベルで安定化
        let high_level = vec![0.1f32; 1000];
        for _ in 0..20 {
            decoder.process_samples(&high_level);
        }
        let stabilized_fast_peak = decoder.agc_peak_fast;
        let stabilized_slow_peak = decoder.agc_peak_slow;

        // 急激なレベル下降（1/10）
        let low_level = vec![0.01f32; 100];
        decoder.process_samples(&low_level);

        // 低速EMAは緩やかに下降する（すぐには下がりすぎない）
        let decayed_fast_peak = decoder.agc_peak_fast;
        let decayed_slow_peak = decoder.agc_peak_slow;
        assert!(
            decayed_slow_peak > stabilized_slow_peak * 0.5,
            "Slow EMA should decay gradually: stabilized={}, decayed={}",
            stabilized_slow_peak,
            decayed_slow_peak
        );
        assert!(
            decayed_fast_peak < stabilized_fast_peak * 0.95,
            "Fast EMA should decay faster: stabilized={}, decayed={}",
            stabilized_fast_peak,
            decayed_fast_peak
        );
        assert!(
            decayed_slow_peak > stabilized_slow_peak * 0.5,
            "Slow EMA should decay gradually: stabilized={}, decayed={}",
            stabilized_slow_peak,
            decayed_slow_peak
        );
    }

    #[test]
    fn test_decoder_reset_after_silence() {
        let dsp_config = DspConfig::default_48k();
        let mut decoder = Decoder::new(16, FIXED_K, dsp_config);

        let silence = vec![0.0f32; 2048];
        let _ = decoder.process_samples(&silence);
        decoder.reset();
        let progress = decoder.process_samples(&[]);

        assert!(!progress.complete);
        assert_eq!(progress.received_packets, 0);
        assert!(decoder.recovered_data().is_none());
    }

    fn build_test_signal(data: &[u8], k: usize, frames: usize, gap_samples: usize) -> Vec<f32> {
        let mut enc_cfg = EncoderConfig::new(DspConfig::default_48k());
        enc_cfg.fountain_k = k;
        let burst_count = enc_cfg.packets_per_sync_burst.max(1);
        let mut encoder = Encoder::new(enc_cfg);
        let params = FountainParams::new(k, PAYLOAD_SIZE);
        let mut fountain_encoder = FountainEncoder::new(data, params);

        let mut signal = Vec::new();
        for _ in 0..frames {
            let mut packets = Vec::with_capacity(burst_count);
            for _ in 0..burst_count {
                packets.push(fountain_encoder.next_packet());
            }
            let frame = encoder.encode_burst(&packets);
            signal.extend_from_slice(&frame);
            if gap_samples > 0 {
                signal.extend(encoder.modulate_silence(gap_samples));
            }
        }
        signal.extend(encoder.flush());
        // 信号の末尾に十分なマージンを追加し、デコーダのバッファチェックで落ちないようにする
        signal.extend(vec![0.0f32; 1024]);
        signal
    }

    fn decode_signal(data: &[u8], k: usize, config: DspConfig, signal: &[f32]) -> Option<Vec<u8>> {
        let mut decoder = Decoder::new(data.len(), k, config);
        for chunk in signal.chunks(2048) {
            let progress = decoder.process_samples(chunk);
            if progress.complete {
                return decoder.recovered_data().map(|v| v.to_vec());
            }
        }
        decoder.recovered_data().map(|v| v.to_vec())
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
    fn test_decoder_tracking_tolerates_clock_drift_ppm() {
        let data = b"tracking timing drift payload";
        let k = data.len().div_ceil(crate::params::PAYLOAD_SIZE);
        let signal = build_test_signal(data, k, 6, 64);
        // 実オーディオI/Fで起きるオーダー(100ppm前後)のクロックずれを模擬。
        let drifted = apply_clock_drift_ppm(&signal, 120.0);
        let recovered = decode_signal(data, k, DspConfig::default_48k(), &drifted)
            .expect("decoder should recover under realistic clock drift");
        assert_eq!(&recovered[..data.len()], data);
    }

    #[test]
    fn test_decoder_tracking_tolerates_carrier_offset() {
        let data = b"tracking carrier offset payload";
        let k = data.len().div_ceil(crate::params::PAYLOAD_SIZE);
        let signal = build_test_signal(data, k, 3, 64);

        let mut rx_cfg = DspConfig::default_48k();
        // 受信LOずれを模擬: 音響リンクで現実的な小さなCFO。
        rx_cfg.carrier_freq += 10.0;

        let recovered = decode_signal(data, k, rx_cfg, &signal)
            .expect("decoder should recover under mild carrier offset");
        assert_eq!(&recovered[..data.len()], data);
    }

    #[test]
    fn test_sync_word_tracking_tolerates_offset_and_cfo() {
        let data = b"sync word header";
        let k = data.len().div_ceil(crate::params::PAYLOAD_SIZE);
        let mut signal = vec![0.0f32; 137];
        signal.extend(build_test_signal(data, k, 3, 64));

        let mut rx_cfg = DspConfig::default_48k();
        rx_cfg.carrier_freq += 10.0;

        let recovered = decode_signal(data, k, rx_cfg, &signal)
            .expect("decoder should recover with offset start + mild CFO");
        assert_eq!(&recovered[..data.len()], data);
    }

    #[test]
    fn test_tracking_timing_error_sign() {
        let pos = timing_error_from_early_late(0.6, 1.2);
        let neg = timing_error_from_early_late(1.2, 0.6);
        let zero = timing_error_from_early_late(1.0, 1.0);
        assert!(pos > 0.0);
        assert!(neg < 0.0);
        assert!(zero.abs() < 1e-6);
    }

    #[test]
    fn test_tracking_phase_error_sign_and_clamp() {
        let p_err = phase_error_from_diff(Complex32::new(0.8, 0.8), Complex32::new(1.0, 0.0));
        let n_err = phase_error_from_diff(Complex32::new(0.8, -0.8), Complex32::new(1.0, 0.0));
        let mut rate = 0.0f32;
        rate = update_phase_rate(rate, p_err);
        let p_step = phase_step_from_phase_error(p_err, rate);
        rate = update_phase_rate(rate, n_err);
        let n_step = phase_step_from_phase_error(n_err, rate);

        let huge_err = phase_error_from_diff(Complex32::new(-1.0, 1e-6), Complex32::new(1.0, 0.0));
        let huge_rate = update_phase_rate(TRACKING_PHASE_RATE_LIMIT_RAD, huge_err);
        let c = phase_step_from_phase_error(huge_err, huge_rate);

        assert!(p_err > 0.0);
        assert!(n_err < 0.0);
        assert!(p_step > 0.0);
        assert!(n_step < 0.0);
        assert!(c.abs() <= TRACKING_PHASE_STEP_CLAMP + 1e-6);
    }

    #[test]
    fn test_sample_at_fractional_linear_interp() {
        let buf = [0.0f32, 10.0, 20.0];
        assert_eq!(sample_at_fractional(&buf, -0.1), None);
        assert_eq!(sample_at_fractional(&buf, 0.0), Some(0.0));
        assert_eq!(sample_at_fractional(&buf, 2.0), Some(20.0));
        assert_eq!(sample_at_fractional(&buf, 2.1), None);
        assert_eq!(sample_at_fractional(&buf, 0.5), Some(5.0));
        assert_eq!(sample_at_fractional(&buf, 1.25), Some(12.5));
    }

    #[test]
    fn test_tracking_timing_pi_loop_direction() {
        let timing_limit = 4.0f32;
        let timing_rate_limit = 1.0f32;

        let mut timing_rate = 0.0f32;
        let mut timing_offset = 0.0f32;
        for _ in 0..8 {
            timing_rate = update_timing_rate(timing_rate, 0.5, timing_rate_limit);
            timing_offset = update_timing_offset(timing_offset, timing_rate, 0.5, timing_limit);
        }
        assert!(timing_rate > 0.0);
        assert!(timing_offset > 0.0);

        for _ in 0..8 {
            timing_rate = update_timing_rate(timing_rate, -0.5, timing_rate_limit);
            timing_offset = update_timing_offset(timing_offset, timing_rate, -0.5, timing_limit);
        }
        assert!(timing_offset < 0.5);
    }

    #[test]
    fn test_dqpsk_hard_bits_and_llr_at_ideal_points() {
        let (_s0, b0, l0) = dqpsk_hard_bits_and_llr(Complex32::new(1.0, 0.0));
        assert_eq!(b0, [0, 0]);
        assert!(l0[0] > 0.0 && l0[1] > 0.0);

        let (_s1, b1, l1) = dqpsk_hard_bits_and_llr(Complex32::new(0.0, 1.0));
        assert_eq!(b1, [0, 1]);
        assert!(l1[0] > 0.0 && l1[1] < 0.0);

        let (_s2, b2, l2) = dqpsk_hard_bits_and_llr(Complex32::new(-1.0, 0.0));
        assert_eq!(b2, [1, 1]);
        assert!(l2[0] < 0.0 && l2[1] < 0.0);

        let (_s3, b3, l3) = dqpsk_hard_bits_and_llr(Complex32::new(0.0, -1.0));
        assert_eq!(b3, [1, 0]);
        assert!(l3[0] < 0.0 && l3[1] > 0.0);
    }

    #[test]
    fn test_decode_diff_symbol_soft_outputs_expected_llr_count() {
        let s0 = decode_diff_symbol_soft(Complex32::new(0.0, 1.0));
        assert_eq!(s0.decided, Complex32::new(0.0, 1.0));
        assert_eq!(s0.llr_count, MODULATION.bits_per_symbol());
        assert!(s0.llrs[0].is_finite());

        let s1 = decode_diff_symbol_soft(Complex32::new(-1.0, 0.0));
        assert_eq!(s1.decided, Complex32::new(-1.0, 0.0));
        assert_eq!(s1.llr_count, MODULATION.bits_per_symbol());
        assert!(s1.llrs[0].is_finite());
    }

    #[test]
    fn test_condition_llr_clips_and_preserves_sign() {
        let p = condition_llr(10.0, LLR_NOISE_VAR_MIN, 1.0);
        let n = condition_llr(-10.0, LLR_NOISE_VAR_MIN, 1.0);
        assert_eq!(p, LLR_CLIP_ABS);
        assert_eq!(n, -LLR_CLIP_ABS);

        let m = condition_llr(0.8, 1.5, 0.5);
        assert!(m > 0.0);
        assert!(m < LLR_CLIP_ABS);
    }

    #[test]
    fn test_condition_llr_quality_reduces_magnitude() {
        let hi = condition_llr(1.0, 1.0, 1.0);
        let lo = condition_llr(1.0, 1.0, 0.2);
        assert!(lo.abs() < hi.abs());
    }

    #[test]
    fn test_condition_llr_noise_var_reduces_magnitude() {
        let low_noise = condition_llr(1.0, 0.1, 1.0);
        let high_noise = condition_llr(1.0, 1.0, 1.0);
        assert!(high_noise.abs() < low_noise.abs());
    }

    #[test]
    fn test_llr_quality_erases_on_large_phase_or_timing_error() {
        let q_ok = llr_quality(0.05, 0.05);
        let q_phase_erase = llr_quality(LLR_PHASE_ERR_ERASE_RAD + 0.01, 0.0);
        let q_timing_erase = llr_quality(0.0, LLR_TIMING_ERR_ERASE + 0.01);
        assert!(q_ok > 0.0);
        assert_eq!(q_phase_erase, 0.0);
        assert_eq!(q_timing_erase, 0.0);
    }

    #[test]
    fn test_decode_diff_symbol_soft_scales_with_amplitude() {
        let a = decode_diff_symbol_soft(Complex32::new(0.5, 0.5));
        let b = decode_diff_symbol_soft(Complex32::new(2.0, 2.0));
        assert_eq!(a.decided, b.decided);
        assert_eq!(a.llr_count, b.llr_count);
        for i in 0..a.llr_count {
            assert!((a.llrs[i] * 4.0 - b.llrs[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_estimate_noise_var_from_diff_is_amplitude_invariant() {
        let s = Complex32::new(1.0, 0.0);
        let e0 = estimate_noise_var_from_diff(Complex32::new(0.9, 0.1), s);
        let e1 = estimate_noise_var_from_diff(Complex32::new(1.8, 0.2), s);
        assert!((e0 - e1).abs() < 1e-4);
    }

    #[test]
    fn test_reproduce_vitest_regression() {
        let data = vec![0xAAu8; 160]; // k=10
        let config = DspConfig::default_48k();
        let mut enc_cfg = EncoderConfig::new(config.clone());
        enc_cfg.fountain_k = 10;
        let mut encoder = Encoder::new(enc_cfg);
        let mut decoder = Decoder::new(data.len(), 10, config);

        // ウォームアップ
        decoder.process_samples(&vec![0.0f32; 4096]);

        let mut seen_ranks = Vec::new();
        let mut complete = false;

        let params = crate::coding::fountain::FountainParams::new(10, crate::params::PAYLOAD_SIZE);
        let mut fountain_encoder = crate::coding::fountain::FountainEncoder::new(&data, params);

        for i in 0..40 {
            let mut packets = Vec::new();
            for _ in 0..2 {
                packets.push(fountain_encoder.next_packet());
            }
            let frame = encoder.encode_burst(&packets);

            if (5..=9).contains(&i) {
                continue;
            }

            let mut signal = frame;
            signal.extend(encoder.modulate_silence(4800)); // 物理的に正しい隙間

            let progress = decoder.process_samples(&signal);
            println!(
                "Iteration {}: rank={}, sync={:?}, buf_len={}, last_seq={:?}",
                i,
                progress.rank_packets,
                decoder.current_sync.as_ref().map(|s| s.peak_sample_idx),
                decoder.sample_buffer_i.len(),
                progress.last_packet_seq
            );
            seen_ranks.push(progress.rank_packets);

            if progress.complete {
                complete = true;
                break;
            }
        }

        println!("Seen ranks: {:?}", seen_ranks);
        assert!(complete, "Should complete eventually");
        assert!(
            seen_ranks.len() <= 6,
            "Regression detected: took {} iterations, expected <= 6",
            seen_ranks.len()
        );
    }
}
