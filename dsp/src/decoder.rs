//! 受信パイプライン (統合デコーダ)

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::common::nco::complex_mul_interleaved2_simd;
use crate::{
    coding::fec,
    coding::fountain::{EncodedPacket, LtDecoder, LtParams},
    coding::interleaver::BlockInterleaver,
    common::nco::Nco,
    common::rrc_filter::DecimatingRrcFilter,
    frame::packet::{Packet, PACKET_BYTES},
    params::PAYLOAD_SIZE,
    phy::demodulator::Demodulator,
    phy::sync::{SyncDetector, SyncResult},
    DspConfig,
};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use std::arch::wasm32::{f32x4, v128, v128_store};
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct DecodeProgress {
    pub received_packets: usize,
    pub needed_packets: usize,
    pub rank_packets: usize,
    pub stalled_packets: usize,
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
    lt_decoder: LtDecoder,
    recovered_data: Option<Vec<u8>>,
    agc_peak: f32,
    decimation_factor: usize,
    lo_nco: Nco,

    // --- 同期状態 ---
    last_search_idx: usize,
    current_sync: Option<SyncResult>,
    last_packet_seq: Option<u32>,
    last_rank_up_seq: Option<u32>,

    // --- 統計用 ---
    pub stats_sync_calls: usize,
    pub stats_sync_time: Duration,
    pub stats_total_samples: usize,
}

impl Decoder {
    pub fn new(_data_size: usize, lt_k: usize, dsp_config: DspConfig) -> Self {
        let decimation_factor = choose_decimation_factor(&dsp_config);
        let proc_config = build_proc_config(&dsp_config, decimation_factor);
        let params = LtParams::new(lt_k, PAYLOAD_SIZE);
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
            demodulator: Demodulator::new(proc_config.clone()),
            interleaver: BlockInterleaver::new(il_rows, il_cols),
            lt_decoder: LtDecoder::new(params),
            recovered_data: None,
            config: dsp_config,
            proc_config,
            agc_peak: 0.5,
            decimation_factor,
            lo_nco,
            last_search_idx: 0,
            current_sync: None,
            last_packet_seq: None,
            last_rank_up_seq: None,
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

        let sync_bits_len = 32;
        let fec_bits_len = self.interleaver.rows() * self.interleaver.cols();
        let total_bits = sync_bits_len + fec_bits_len;
        let symbol_len = sf * spc;

        loop {
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
            let data_end_sample = start + total_bits * sf * spc;

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

                        let mut sync_chips_i = Vec::with_capacity(sync_bits_len * sf);
                        let mut sync_chips_q = Vec::with_capacity(sync_bits_len * sf);
                        for c_idx in 0..(sync_bits_len * sf) {
                            let cp = p_start + c_idx * spc + (spc / 2);
                            if cp >= self.sample_buffer_i.len() {
                                sync_chips_i.clear();
                                sync_chips_q.clear();
                                break;
                            }
                            sync_chips_i.push(self.sample_buffer_i[cp]);
                            sync_chips_q.push(self.sample_buffer_q[cp]);
                        }
                        if sync_chips_i.len() != sync_bits_len * sf {
                            continue;
                        }

                        let llrs = self
                            .demodulator
                            .demodulate_chips_soft(&sync_chips_i, &sync_chips_q);
                        let bits: Vec<u8> = llrs
                            .iter()
                            .map(|&l| if l > 0.0 { 0u8 } else { 1u8 })
                            .collect();

                        let mut val = 0u32;
                        for b in &bits[0..32] {
                            val = (val << 1) | (*b as u32);
                        }

                        if val == crate::params::SYNC_WORD {
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
                let ref_sym_start = p_start.saturating_sub(sf * spc);
                let (ref_i, ref_q) = self.despread_at(ref_sym_start);

                self.demodulator.reset();
                if best_invert {
                    self.demodulator.set_reference_phase(-ref_i, -ref_q);
                } else {
                    self.demodulator.set_reference_phase(ref_i, ref_q);
                }

                let mut chips_i = Vec::with_capacity(total_bits * sf);
                let mut chips_q = Vec::with_capacity(total_bits * sf);
                for c_idx in 0..(total_bits * sf) {
                    let cp = p_start + c_idx * spc + (spc / 2);
                    if cp >= self.sample_buffer_i.len() {
                        chips_i.clear();
                        chips_q.clear();
                        break;
                    }
                    chips_i.push(self.sample_buffer_i[cp]);
                    chips_q.push(self.sample_buffer_q[cp]);
                }
                if chips_i.len() != total_bits * sf {
                    let skip = (start + symbol_len).min(self.sample_buffer_i.len());
                    self.sample_buffer_i.drain(0..skip);
                    self.sample_buffer_q.drain(0..skip);
                    self.last_search_idx = 0;
                    self.current_sync = None;
                    continue;
                }

                let llrs = self.demodulator.demodulate_chips_soft(&chips_i, &chips_q);
                let bits: Vec<u8> = llrs
                    .iter()
                    .map(|&l| if l > 0.0 { 0u8 } else { 1u8 })
                    .collect();

                let deinterleaved = self
                    .interleaver
                    .deinterleave(&bits[sync_bits_len..total_bits]);
                let decoded_bits = fec::decode(&deinterleaved);
                let p_bits_len = PACKET_BYTES * 8;

                if decoded_bits.len() >= p_bits_len {
                    let d_bytes = fec::bits_to_bytes(&decoded_bits[..p_bits_len]);
                    if let Some(packet) = Packet::deserialize(&d_bytes) {
                        let seq = packet.lt_seq as u32;
                        let (degree, neighbors) =
                            crate::coding::fountain::reconstruct_packet_metadata(
                                seq,
                                self.lt_decoder.params().k,
                                self.lt_decoder.params().c,
                                self.lt_decoder.params().delta,
                            );
                        let before_received = self.lt_decoder.received_count();
                        let before_rank = self.lt_decoder.rank();
                        self.lt_decoder.receive(EncodedPacket {
                            seq,
                            degree,
                            neighbors,
                            data: packet.payload.to_vec(),
                        });
                        let after_received = self.lt_decoder.received_count();
                        let after_rank = self.lt_decoder.rank();
                        if after_received > before_received {
                            self.last_packet_seq = Some(seq);
                            if after_rank > before_rank {
                                self.last_rank_up_seq = Some(seq);
                            }
                        }

                        if let Some(data) = self.lt_decoder.decode() {
                            self.recovered_data = Some(data);
                        }
                        let actual_end = p_start + total_bits * sf * spc;
                        self.sample_buffer_i.drain(0..actual_end);
                        self.sample_buffer_q.drain(0..actual_end);
                        self.last_search_idx = 0;
                        self.current_sync = None;
                        continue;
                    }
                }
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

    fn progress(&self) -> DecodeProgress {
        let received = self.lt_decoder.received_count();
        let needed = self.lt_decoder.needed_count();
        let rank = self.lt_decoder.rank();
        DecodeProgress {
            received_packets: received,
            needed_packets: needed,
            rank_packets: rank,
            stalled_packets: received.saturating_sub(rank),
            last_packet_seq: self.last_packet_seq.map(|v| v as i32).unwrap_or(-1),
            last_rank_up_seq: self.last_rank_up_seq.map(|v| v as i32).unwrap_or(-1),
            progress: self.lt_decoder.progress(),
            complete: self.recovered_data.is_some(),
        }
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
        let params = self.lt_decoder.params().clone();
        self.lt_decoder = LtDecoder::new(params);
    }

    pub fn recovered_data(&self) -> Option<&[u8]> {
        self.recovered_data.as_deref()
    }
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

    #[test]
    fn test_choose_decimation_factor() {
        assert_eq!(choose_decimation_factor(&DspConfig::default_48k()), 3);
        assert_eq!(choose_decimation_factor(&DspConfig::default_44k()), 2);
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
}
