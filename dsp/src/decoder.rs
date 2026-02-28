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
    params::{MODULATION, PACKETS_PER_SYNC_BURST, PAYLOAD_SIZE, SYNC_WORD, SYNC_WORD_BITS},
    phy::demodulator::Demodulator,
    phy::sync::{SyncDetector, SyncResult},
    DifferentialModulation, DspConfig,
};
use num_complex::Complex32;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use std::arch::wasm32::{f32x4, v128, v128_store};
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

const TRACKING_TIMING_LOOP_GAIN: f32 = 0.18;
const TRACKING_PHASE_LOOP_GAIN: f32 = 0.04;
const TRACKING_TIMING_LIMIT_CHIP: f32 = 0.5;
const TRACKING_EARLY_LATE_DELTA_CHIP: f32 = 0.5;
const TRACKING_PHASE_STEP_CLAMP: f32 = 0.15;

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
    demodulator: Demodulator,
    interleaver: BlockInterleaver,
    fountain_decoder: FountainDecoder,
    recovered_data: Option<Vec<u8>>,
    agc_peak: f32,
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
            demodulator: Demodulator::new_with_mode(
                proc_config.clone(),
                DifferentialModulation::Dbpsk,
            ),
            interleaver: BlockInterleaver::new(il_rows, il_cols),
            fountain_decoder: FountainDecoder::new(params),
            recovered_data: None,
            config: dsp_config,
            proc_config,
            agc_peak: 0.5,
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
        // 1回の process_samples で探索しすぎると detect() が支配的になるため、
        // 反復数を最小限に抑えてストリーミング側へ制御を返す。
        let mut iteration_budget = 2usize;

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

            // データが溜まるのを待つ
            if self.sample_buffer_i.len() < data_end_sample + spc {
                // タイムアウト監視: 同期位置がバッファの遥か後方を指している場合や、
                // 既にバッファが十分長いのに同期位置が古すぎる場合は、偽同期とみなす
                if start + symbol_len < self.sample_buffer_i.len().saturating_sub(max_buffer_len) {
                    self.current_sync = None;
                    self.last_search_idx = 0;
                    continue;
                }
                break;
            }

            // --- Robust Header Search (Probing) ---
            let mut found_header = false;
            let mut best_sym_shift = 0;
            let mut best_t_shift = 0;
            let mut best_invert = false;

            'probe: for sym_shift in -2i32..=2i32 {
                let base_start = start as i32 + sym_shift * symbol_len as i32;
                for t_shift in -((spc as i32) * 2)..=((spc as i32) * 2) {
                    let p_start_i32 = base_start + t_shift;
                    if p_start_i32 < 0 {
                        continue;
                    }
                    let p_start = p_start_i32 as usize;
                    let ref_sym_start = p_start.saturating_sub(sf * spc);
                    let (ref_i, ref_q) = self.despread_at(ref_sym_start);

                    for invert in [false, true] {
                        self.demodulator.reset();
                        if invert {
                            self.demodulator.set_reference_phase(-ref_i, -ref_q);
                        } else {
                            self.demodulator.set_reference_phase(ref_i, ref_q);
                        }

                        let mut sync_chips_i = Vec::with_capacity(sync_symbol_len * sf);
                        let mut sync_chips_q = Vec::with_capacity(sync_symbol_len * sf);
                        for c_idx in 0..(sync_symbol_len * sf) {
                            let cp = p_start + c_idx * spc + (spc / 2);
                            if cp >= self.sample_buffer_i.len() {
                                sync_chips_i.clear();
                                sync_chips_q.clear();
                                break;
                            }
                            sync_chips_i.push(self.sample_buffer_i[cp]);
                            sync_chips_q.push(self.sample_buffer_q[cp]);
                        }
                        if sync_chips_i.len() != sync_symbol_len * sf {
                            continue;
                        }

                        let mut bits = self
                            .demodulator
                            .demodulate_chips(&sync_chips_i, &sync_chips_q);
                        if bits.len() < sync_bits_len {
                            continue;
                        }
                        bits.truncate(sync_bits_len);

                        let mut val = 0u32;
                        for b in bits {
                            val = (val << 1) | (b as u32);
                        }

                        if val == SYNC_WORD {
                            found_header = true;
                            best_sym_shift = sym_shift;
                            best_t_shift = t_shift;
                            best_invert = invert;
                            break 'probe;
                        }
                    }
                }
            }

            if found_header {
                let p_start =
                    (start as i32 + best_sym_shift * symbol_len as i32 + best_t_shift) as usize;
                let last_sync_sym_start = p_start + (sync_symbol_len - 1) * symbol_len;
                let (sync_last_i, sync_last_q) = self.despread_at(last_sync_sym_start);
                let initial_ref = if best_invert {
                    Complex32::new(-sync_last_i, -sync_last_q)
                } else {
                    Complex32::new(sync_last_i, sync_last_q)
                };
                let payload_start = p_start + sync_symbol_len * symbol_len;
                let Some(payload_bits) =
                    self.decode_bits_with_tracking(payload_start, burst_data_bits_len, initial_ref)
                else {
                    let skip = (start + symbol_len).min(self.sample_buffer_i.len());
                    self.sample_buffer_i.drain(0..skip);
                    self.sample_buffer_q.drain(0..skip);
                    self.last_search_idx = 0;
                    self.current_sync = None;
                    continue;
                };
                let p_bits_len = PACKET_BYTES * 8;
                let mut _valid_packet_count = 0usize;
                for packet_bits in payload_bits.chunks_exact(fec_bits_len) {
                    let deinterleaved = self.interleaver.deinterleave(packet_bits);
                    let decoded_bits = fec::decode(&deinterleaved);
                    if decoded_bits.len() < p_bits_len {
                        self.parse_error_packets += 1;
                        continue;
                    }
                    let d_bytes = fec::bits_to_bytes(&decoded_bits[..p_bits_len]);
                    match Packet::deserialize(&d_bytes) {
                        Ok(packet) => {
                            _valid_packet_count += 1;
                            let pkt_k = packet.lt_k as usize;
                            if pkt_k != self.fountain_decoder.params().k {
                                self.rebuild_fountain_decoder(pkt_k);
                            }
                            let seq = packet.lt_seq as u32;
                            let coefficients =
                                crate::coding::fountain::reconstruct_packet_coefficients(
                                    seq,
                                    self.fountain_decoder.params().k,
                                );
                            let outcome =
                                self.fountain_decoder.receive_with_outcome(FountainPacket {
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
                        Err(PacketParseError::CrcMismatch { .. }) => {
                            self.crc_error_packets += 1;
                        }
                        Err(PacketParseError::InvalidLength { .. }) => {
                            self.parse_error_packets += 1;
                        }
                    }
                }
                // lock判定は preamble+sync のみで行う。
                // CRC失敗パケットは捨てるだけにして、受信窓は常にバースト分前進させる。
                let actual_end =
                    (p_start + total_symbols * sf * spc).min(self.sample_buffer_i.len());
                self.sample_buffer_i.drain(0..actual_end);
                self.sample_buffer_q.drain(0..actual_end);
                self.last_search_idx = 0;
                self.current_sync = None;
                continue;
            }

            // デコード失敗またはヘッダ未検出: 偽同期として捨てて進める
            let skip = (start + symbol_len).min(self.sample_buffer_i.len());
            self.sample_buffer_i.drain(0..skip);
            self.sample_buffer_q.drain(0..skip);
            self.last_search_idx = 0;
            self.current_sync = None;
            continue;
        }

        self.progress()
    }

    fn despread_at(&self, start_sample: usize) -> (f32, f32) {
        let sf = self.config.spread_factor();
        let spc = self.proc_config.samples_per_chip();
        let mut mseq = crate::common::msequence::MSequence::new(self.config.mseq_order);
        let pn = mseq.generate(sf);

        let mut sum_i = 0.0;
        let mut sum_q = 0.0;
        for (j, &chip) in pn.iter().enumerate() {
            let p = start_sample + j * spc + (spc / 2);
            if let (Some(&si), Some(&sq)) =
                (self.sample_buffer_i.get(p), self.sample_buffer_q.get(p))
            {
                sum_i += si * chip as f32;
                sum_q += sq * chip as f32;
            }
        }
        (sum_i / sf as f32, sum_q / sf as f32)
    }

    fn despread_symbol_with_timing(
        &self,
        symbol_start: usize,
        pn: &[f32],
        timing_offset: f32,
        sample_shift: f32,
    ) -> Option<Complex32> {
        let spc = self.proc_config.samples_per_chip().max(1);
        let shift = (timing_offset + sample_shift).round() as isize;
        let mut sum_i = 0.0f32;
        let mut sum_q = 0.0f32;
        for (chip_idx, &pn_val) in pn.iter().enumerate() {
            let p = symbol_start as isize + (chip_idx * spc + (spc / 2)) as isize + shift;
            if p < 0 {
                return None;
            }
            let p = p as usize;
            let si = *self.sample_buffer_i.get(p)?;
            let sq = *self.sample_buffer_q.get(p)?;
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
        initial_phase_ref: Complex32,
    ) -> Option<Vec<u8>> {
        let sf = self.config.spread_factor();
        let spc = self.proc_config.samples_per_chip().max(1);
        let symbol_len = sf * spc;
        let bits_per_symbol = MODULATION.bits_per_symbol();
        let symbols_needed = num_bits.div_ceil(bits_per_symbol);
        let mut mseq = crate::common::msequence::MSequence::new(self.config.mseq_order);
        let pn: Vec<f32> = mseq.generate(sf).into_iter().map(|v| v as f32).collect();

        let mut phase_ref = if initial_phase_ref.norm_sqr() > 1e-8 {
            initial_phase_ref / initial_phase_ref.norm()
        } else {
            Complex32::new(1.0, 0.0)
        };
        let mut prev_symbol = phase_ref;
        let mut timing_offset = 0.0f32;
        let timing_limit = spc as f32 * TRACKING_TIMING_LIMIT_CHIP;
        let early_late_delta = (spc as f32 * TRACKING_EARLY_LATE_DELTA_CHIP).max(1.0);
        let mut bits = Vec::with_capacity(num_bits);

        for s_idx in 0..symbols_needed {
            let symbol_start = start_sample + s_idx * symbol_len;
            let on = self.despread_symbol_with_timing(symbol_start, &pn, timing_offset, 0.0)?;
            let early = self.despread_symbol_with_timing(
                symbol_start,
                &pn,
                timing_offset,
                -early_late_delta,
            )?;
            let late = self.despread_symbol_with_timing(
                symbol_start,
                &pn,
                timing_offset,
                early_late_delta,
            )?;

            let on_rot = on * phase_ref.conj();
            let diff = on_rot * prev_symbol.conj();
            let decided = decode_diff_symbol_and_push_bits(diff, &mut bits, num_bits);
            let dphi = phase_step_from_diff(diff, decided);
            let (sin_dphi, cos_dphi) = dphi.sin_cos();
            phase_ref *= Complex32::new(cos_dphi, sin_dphi);
            let phase_norm = phase_ref.norm().max(1e-6);
            phase_ref /= phase_norm;

            // Early/Late timing tracking
            let early_mag = early.norm();
            let late_mag = late.norm();
            let timing_err = timing_error_from_early_late(early_mag, late_mag);
            timing_offset = (timing_offset + TRACKING_TIMING_LOOP_GAIN * timing_err)
                .clamp(-timing_limit, timing_limit);

            let on_norm = on_rot.norm();
            if on_norm > 1e-4 {
                prev_symbol = on_rot / on_norm;
            } else {
                prev_symbol = decided;
            }
        }

        bits.truncate(num_bits);
        Some(bits)
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
        self.agc_peak = self.agc_peak * 0.999 + sample.abs() * 0.001;
        let gain = if self.agc_peak > 1e-6 {
            0.5 / self.agc_peak
        } else {
            1.0
        };
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
        self.demodulator.reset();
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
fn decode_diff_symbol_and_push_bits(
    diff: Complex32,
    bits: &mut Vec<u8>,
    target_bits: usize,
) -> Complex32 {
    match MODULATION {
        DifferentialModulation::Dbpsk => {
            if bits.len() < target_bits {
                bits.push(if diff.re >= 0.0 { 0 } else { 1 });
            }
            if diff.re >= 0.0 {
                Complex32::new(1.0, 0.0)
            } else {
                Complex32::new(-1.0, 0.0)
            }
        }
        DifferentialModulation::Dqpsk => {
            let (symbol, pair) = if diff.re.abs() >= diff.im.abs() {
                if diff.re >= 0.0 {
                    (Complex32::new(1.0, 0.0), [0u8, 0u8])
                } else {
                    (Complex32::new(-1.0, 0.0), [1u8, 1u8])
                }
            } else if diff.im >= 0.0 {
                (Complex32::new(0.0, 1.0), [0u8, 1u8])
            } else {
                (Complex32::new(0.0, -1.0), [1u8, 0u8])
            };
            for &b in &pair {
                if bits.len() < target_bits {
                    bits.push(b);
                }
            }
            symbol
        }
    }
}

#[inline]
fn phase_step_from_diff(diff: Complex32, decided_symbol: Complex32) -> f32 {
    let diff_data_removed = diff * decided_symbol.conj();
    let phase_err = diff_data_removed.im / (diff_data_removed.re.abs() + 1e-6);
    (TRACKING_PHASE_LOOP_GAIN * phase_err)
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
        let mut encoder = Encoder::new(enc_cfg);
        let mut stream = encoder.encode_stream(data);

        let mut signal = Vec::new();
        for _ in 0..frames {
            if let Some(frame) = stream.next() {
                signal.extend_from_slice(&frame);
                signal.extend(std::iter::repeat_n(0.0f32, gap_samples));
            }
        }
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
        let period = (1_000_000.0 / ppm.abs()).round() as usize;
        if period < 2 {
            return input.to_vec();
        }

        if ppm > 0.0 {
            // 正のppm: 受信側クロックが遅い想定 -> 波形がわずかに伸びる（サンプル重複）
            let mut out = Vec::with_capacity(input.len() + input.len() / period + 8);
            for (i, &s) in input.iter().enumerate() {
                out.push(s);
                if (i + 1) % period == 0 {
                    out.push(s);
                }
            }
            out
        } else {
            // 負のppm: 受信側クロックが速い想定 -> 波形がわずかに縮む（サンプル間引き）
            let mut out = Vec::with_capacity(input.len().saturating_sub(input.len() / period));
            for (i, &s) in input.iter().enumerate() {
                if (i + 1) % period == 0 {
                    continue;
                }
                out.push(s);
            }
            out
        }
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
        rx_cfg.carrier_freq += 5.0;

        let recovered = decode_signal(data, k, rx_cfg, &signal)
            .expect("decoder should recover under mild carrier offset");
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
    fn test_tracking_phase_step_sign_and_clamp() {
        let p = phase_step_from_diff(Complex32::new(0.8, 0.8), Complex32::new(1.0, 0.0));
        let n = phase_step_from_diff(Complex32::new(0.8, -0.8), Complex32::new(1.0, 0.0));
        let c = phase_step_from_diff(Complex32::new(1e-6, 10.0), Complex32::new(1.0, 0.0));
        assert!(p > 0.0);
        assert!(n < 0.0);
        assert!(c.abs() <= TRACKING_PHASE_STEP_CLAMP + 1e-6);
    }
}
