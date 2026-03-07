//! パケット復調モジュール
//!
//! 等化済みシンボル列から1パケット分の LLR を生成し、
//! デインターリーブ・デスクランブル・soft-list 復号までを担当する。

use super::decoder_stats::DecoderStats;
use super::tracking::{
    self, TrackingState, PHASE_ERR_ABS_THRESH_0P5_RAD, PHASE_ERR_ABS_THRESH_1P0_RAD,
    TRACKING_PHASE_ERR_GATE_DQPSK_CONF_HIGH, TRACKING_PHASE_ERR_GATE_RAD,
    TRACKING_PHASE_FREQ_GAIN_OFF, TRACKING_PHASE_OFF_ERR_CLAMP, TRACKING_PHASE_PROP_GAIN_OFF,
    TRACKING_PHASE_RATE_HOLD_DECAY, TRACKING_PHASE_STEP_CLAMP,
};
use crate::coding::fec;
use crate::coding::interleaver::BlockInterleaver;
use crate::coding::scrambler::Scrambler;
use crate::frame::packet::Packet;
use crate::mary::demodulator::Demodulator;
use crate::mary::interleaver_config;
use crate::mary::params::PAYLOAD_SPREAD_FACTOR;
use num_complex::Complex32;

/// LLR観測用コールバック型
pub type LlrCallback = Box<dyn FnMut(&[f32]) + Send>;

#[derive(Clone, Copy, Debug)]
pub(crate) enum PacketDecodeError {
    Crc,
    Parse,
}

pub(crate) struct PacketDecodeOptions {
    pub spc: usize,
    pub early_late_delta: f32,
    pub viterbi_list_size: usize,
    pub llr_erasure_second_pass_enabled: bool,
    pub llr_erasure_quantile: f32,
    pub llr_erasure_list_size: usize,
}

pub(crate) struct PacketDecodeBuffers {
    pub packet_llrs_buffer: Vec<f32>,
    pub deinterleave_buffer: Vec<f32>,
    pub erasure_llr_buffer: Vec<f32>,
}

impl PacketDecodeBuffers {
    pub fn new() -> Self {
        let cap = interleaver_config::interleaved_bits();
        Self {
            packet_llrs_buffer: Vec::with_capacity(cap),
            deinterleave_buffer: Vec::with_capacity(cap),
            erasure_llr_buffer: Vec::with_capacity(cap),
        }
    }

    pub fn clear(&mut self) {
        self.packet_llrs_buffer.clear();
        self.deinterleave_buffer.clear();
        self.erasure_llr_buffer.clear();
    }
}

pub(crate) struct PacketProcessResult {
    pub processed: bool,
    pub packet: Option<Packet>,
}

pub(crate) struct PacketDecodeRuntime<'a> {
    pub demodulator: &'a Demodulator,
    pub prev_phase: &'a mut Complex32,
    pub tracking_state: &'a mut TrackingState,
    pub stats: &'a mut DecoderStats,
    pub buffers: &'a mut PacketDecodeBuffers,
    pub llr_callback: &'a mut Option<LlrCallback>,
}

struct DecodeLayout {
    rows: usize,
    cols: usize,
    interleaved_bits: usize,
    fec_bits: usize,
    payload_bits_len: usize,
}

struct DecodeCandidateContext<'a> {
    stats: &'a mut DecoderStats,
    buffers: &'a mut PacketDecodeBuffers,
    llr_callback: &'a mut Option<LlrCallback>,
    options: &'a PacketDecodeOptions,
}

pub(crate) fn process_packet_core<D>(
    runtime: PacketDecodeRuntime<'_>,
    options: &PacketDecodeOptions,
    mut despread_symbol: D,
) -> PacketProcessResult
where
    D: FnMut(usize, f32, f32) -> Option<[Complex32; 16]>,
{
    let PacketDecodeRuntime {
        demodulator,
        prev_phase,
        tracking_state,
        stats,
        buffers,
        llr_callback,
    } = runtime;
    let interleaved_bits = interleaver_config::interleaved_bits();
    let expected_symbols = interleaver_config::mary_symbols();
    let timing_limit = options.spc as f32 * tracking::TRACKING_TIMING_LIMIT_CHIP;
    let timing_rate_limit = options.spc as f32 * tracking::TRACKING_TIMING_RATE_LIMIT_CHIP;

    buffers.packet_llrs_buffer.clear();
    let mut total_packet_energy = 0.0f32;

    for sym_idx in 0..expected_symbols {
        let symbol_start = options.spc + sym_idx * PAYLOAD_SPREAD_FACTOR * options.spc;

        let on_corrs =
            if let Some(c) = despread_symbol(symbol_start, tracking_state.timing_offset, 0.0) {
                c
            } else {
                return PacketProcessResult {
                    processed: false,
                    packet: None,
                };
            };
        let early_corrs = if let Some(c) = despread_symbol(
            symbol_start,
            tracking_state.timing_offset,
            -options.early_late_delta,
        ) {
            c
        } else {
            return PacketProcessResult {
                processed: false,
                packet: None,
            };
        };
        let late_corrs = if let Some(c) = despread_symbol(
            symbol_start,
            tracking_state.timing_offset,
            options.early_late_delta,
        ) {
            c
        } else {
            return PacketProcessResult {
                processed: false,
                packet: None,
            };
        };

        let mut max_energy = 0.0f32;
        let mut second_energy = 0.0f32;
        let mut best_idx = 0usize;
        for (idx, corr) in on_corrs.iter().enumerate() {
            let energy = corr.norm_sqr();
            if energy > max_energy {
                second_energy = max_energy;
                max_energy = energy;
                best_idx = idx;
            } else if energy > second_energy {
                second_energy = energy;
            }
        }
        total_packet_energy += max_energy;

        let best_corr = on_corrs[best_idx];
        let on_rot = best_corr * tracking_state.phase_ref.conj();
        let diff = on_rot * prev_phase.conj();

        let energies: [f32; 16] =
            on_corrs.map(|c| (c * tracking_state.phase_ref.conj()).norm_sqr());
        let walsh_llr = demodulator.walsh_llr(&energies, max_energy);
        let dqpsk_norm = on_rot.norm().max(1e-6);
        let dqpsk_llr = demodulator.dqpsk_llr(diff, dqpsk_norm);

        buffers.packet_llrs_buffer.extend_from_slice(&walsh_llr);
        buffers.packet_llrs_buffer.extend_from_slice(&dqpsk_llr);

        let walsh_conf = ((max_energy - second_energy).max(0.0)) / max_energy.max(1e-6);
        let energy_sum = energies.iter().sum::<f32>();
        let noise_floor = ((energy_sum - max_energy).max(0.0)) / 15.0;
        let snr_proxy = max_energy / (noise_floor + 1e-6);
        let dqpsk_conf = dqpsk_llr[0].abs() + dqpsk_llr[1].abs();
        tracking_state.phase_gate_enabled = tracking::next_phase_gate_enabled(
            tracking_state.phase_gate_enabled,
            dqpsk_conf,
            walsh_conf,
            snr_proxy,
        );
        if tracking_state.phase_gate_enabled {
            stats.phase_gate_on_symbols += 1;
        } else {
            stats.phase_gate_off_symbols += 1;
        }

        let decided = decide_dqpsk_symbol_from_llr(dqpsk_llr);
        let phase_err = tracking::phase_error_from_diff(diff, decided);
        let phase_err_abs = phase_err.abs();
        stats.phase_err_abs_sum_rad += phase_err_abs as f64;
        stats.phase_err_abs_count += 1;
        if phase_err_abs >= PHASE_ERR_ABS_THRESH_0P5_RAD {
            stats.phase_err_abs_ge_0p5_symbols += 1;
        }
        if phase_err_abs >= PHASE_ERR_ABS_THRESH_1P0_RAD {
            stats.phase_err_abs_ge_1p0_symbols += 1;
        }
        let innovation_rejected = tracking_state.phase_gate_enabled
            && phase_err_abs > TRACKING_PHASE_ERR_GATE_RAD
            && dqpsk_conf < TRACKING_PHASE_ERR_GATE_DQPSK_CONF_HIGH;
        if innovation_rejected {
            stats.phase_innovation_reject_symbols += 1;
        }
        let phase_step = if tracking_state.phase_gate_enabled {
            tracking_state.phase_rate =
                tracking::update_phase_rate(tracking_state.phase_rate, phase_err);
            tracking::phase_step_from_phase_error(phase_err, tracking_state.phase_rate)
        } else {
            let damped_err =
                phase_err.clamp(-TRACKING_PHASE_OFF_ERR_CLAMP, TRACKING_PHASE_OFF_ERR_CLAMP);
            tracking_state.phase_rate = (tracking_state.phase_rate
                * TRACKING_PHASE_RATE_HOLD_DECAY
                + TRACKING_PHASE_FREQ_GAIN_OFF * damped_err)
                .clamp(
                    -tracking::TRACKING_PHASE_RATE_LIMIT_RAD,
                    tracking::TRACKING_PHASE_RATE_LIMIT_RAD,
                );
            (tracking_state.phase_rate + TRACKING_PHASE_PROP_GAIN_OFF * damped_err)
                .clamp(-TRACKING_PHASE_STEP_CLAMP, TRACKING_PHASE_STEP_CLAMP)
        };
        let (sin_dphi, cos_dphi) = phase_step.sin_cos();
        tracking_state.phase_ref *= Complex32::new(cos_dphi, sin_dphi);
        tracking_state.phase_ref /= tracking_state.phase_ref.norm().max(1e-6);

        let timing_err = tracking::timing_error_from_early_late(
            early_corrs[best_idx].norm(),
            late_corrs[best_idx].norm(),
        );
        tracking_state.timing_rate =
            tracking::update_timing_rate(tracking_state.timing_rate, timing_err, timing_rate_limit);
        tracking_state.timing_offset = tracking::update_timing_offset(
            tracking_state.timing_offset,
            tracking_state.timing_rate,
            timing_err,
            timing_limit,
        );

        let on_norm = on_rot.norm().max(1e-6);
        *prev_phase = on_rot / on_norm;
    }

    let _avg_energy = total_packet_energy / expected_symbols as f32;
    let mut llr_buf = std::mem::take(&mut buffers.packet_llrs_buffer);
    let packet_llrs_len = llr_buf.len().min(interleaved_bits);
    let packet = if packet_llrs_len >= interleaved_bits {
        decode_llrs(
            &llr_buf[..packet_llrs_len],
            stats,
            buffers,
            llr_callback,
            options,
        )
    } else {
        None
    };
    llr_buf.clear();
    buffers.packet_llrs_buffer = llr_buf;

    PacketProcessResult {
        processed: true,
        packet,
    }
}

fn decode_llrs(
    llrs: &[f32],
    stats: &mut DecoderStats,
    buffers: &mut PacketDecodeBuffers,
    llr_callback: &mut Option<LlrCallback>,
    options: &PacketDecodeOptions,
) -> Option<Packet> {
    let p_bits_len = crate::frame::packet::PACKET_BYTES * 8;
    let fec_bits = interleaver_config::fec_bits();
    let rows = interleaver_config::INTERLEAVER_ROWS;
    let cols = interleaver_config::INTERLEAVER_COLS;
    let interleaved_bits = interleaver_config::interleaved_bits();
    let packet_chunk_bits = interleaver_config::mary_aligned_bits();
    let layout = DecodeLayout {
        rows,
        cols,
        interleaved_bits,
        fec_bits,
        payload_bits_len: p_bits_len,
    };
    let mut context = DecodeCandidateContext {
        stats,
        buffers,
        llr_callback,
        options,
    };
    for packet_llrs in llrs.chunks(packet_chunk_bits) {
        if packet_llrs.len() < interleaved_bits {
            break;
        }

        let valid_llrs = &packet_llrs[..interleaved_bits];
        match decode_single_llr_candidate(valid_llrs, &layout, &mut context) {
            Ok(packet) => {
                return Some(packet);
            }
            Err(PacketDecodeError::Crc) => {
                context.stats.crc_error_packets += 1;
            }
            Err(PacketDecodeError::Parse) => {
                context.stats.parse_error_packets += 1;
            }
        }
    }
    None
}

fn decode_single_llr_candidate(
    llrs: &[f32],
    layout: &DecodeLayout,
    context: &mut DecodeCandidateContext<'_>,
) -> Result<Packet, PacketDecodeError> {
    let interleaver = BlockInterleaver::new(layout.rows, layout.cols);
    context
        .buffers
        .deinterleave_buffer
        .resize(layout.interleaved_bits, 0.0);
    interleaver.deinterleave_f32_in_place(
        llrs,
        &mut context.buffers.deinterleave_buffer[..layout.interleaved_bits],
    );

    let mut scrambler = Scrambler::default();
    for llr in context.buffers.deinterleave_buffer[..layout.interleaved_bits].iter_mut() {
        if scrambler.next_bit() == 1 {
            *llr = -*llr;
        }
    }

    if let Some(callback) = context.llr_callback.as_mut() {
        callback(&context.buffers.deinterleave_buffer[..layout.interleaved_bits]);
    }

    let first_candidates = fec::decode_soft_list(
        &context.buffers.deinterleave_buffer[..layout.fec_bits],
        context.options.viterbi_list_size,
    );
    let first_attempt = try_decode_soft_list_candidates(first_candidates, layout.payload_bits_len);
    if let Ok(packet) = first_attempt {
        return Ok(packet);
    }

    let mut saw_crc = matches!(first_attempt, Err(PacketDecodeError::Crc));

    if context.options.llr_erasure_second_pass_enabled && saw_crc {
        context.stats.llr_second_pass_attempts += 1;
        context
            .buffers
            .erasure_llr_buffer
            .resize(layout.interleaved_bits, 0.0);
        context.buffers.erasure_llr_buffer[..layout.interleaved_bits]
            .copy_from_slice(&context.buffers.deinterleave_buffer[..layout.interleaved_bits]);
        apply_llr_erasure_quantile(
            &mut context.buffers.erasure_llr_buffer[..layout.fec_bits],
            context.options.llr_erasure_quantile,
        );
        let second_candidates = fec::decode_soft_list(
            &context.buffers.erasure_llr_buffer[..layout.fec_bits],
            context.options.llr_erasure_list_size,
        );
        match try_decode_soft_list_candidates(second_candidates, layout.payload_bits_len) {
            Ok(packet) => {
                context.stats.llr_second_pass_rescued += 1;
                return Ok(packet);
            }
            Err(PacketDecodeError::Crc) => saw_crc = true,
            Err(PacketDecodeError::Parse) => {}
        }
    }

    if saw_crc {
        Err(PacketDecodeError::Crc)
    } else {
        Err(PacketDecodeError::Parse)
    }
}

fn try_decode_soft_list_candidates(
    decoded_candidates: Vec<Vec<u8>>,
    p_bits_len: usize,
) -> Result<Packet, PacketDecodeError> {
    let mut saw_crc = false;
    for decoded_bits in decoded_candidates {
        if decoded_bits.len() < p_bits_len {
            continue;
        }
        match decode_packet(&decoded_bits[..p_bits_len]) {
            Ok(packet) => return Ok(packet),
            Err(PacketDecodeError::Crc) => saw_crc = true,
            Err(PacketDecodeError::Parse) => {}
        }
    }
    if saw_crc {
        Err(PacketDecodeError::Crc)
    } else {
        Err(PacketDecodeError::Parse)
    }
}

fn decode_packet(packet_bits: &[u8]) -> Result<Packet, PacketDecodeError> {
    let decoded_bytes = fec::bits_to_bytes(packet_bits);
    match Packet::deserialize(&decoded_bytes) {
        Ok(packet) => Ok(packet),
        Err(crate::frame::packet::PacketParseError::CrcMismatch { .. }) => {
            Err(PacketDecodeError::Crc)
        }
        Err(_) => Err(PacketDecodeError::Parse),
    }
}

#[inline]
pub(crate) fn apply_llr_erasure_quantile(llrs: &mut [f32], quantile: f32) {
    if llrs.is_empty() {
        return;
    }
    let q = quantile.clamp(0.0, 1.0);
    if q <= 0.0 {
        return;
    }
    let erase_count = ((llrs.len() as f32) * q).round() as usize;
    if erase_count == 0 {
        return;
    }

    let mut abs_vals = llrs.iter().map(|v| v.abs()).collect::<Vec<_>>();
    abs_vals.sort_by(|a, b| a.total_cmp(b));
    let threshold_idx = erase_count.saturating_sub(1).min(abs_vals.len() - 1);
    let threshold = abs_vals[threshold_idx];

    for llr in llrs.iter_mut() {
        if llr.abs() <= threshold {
            *llr = 0.0;
        }
    }
}

#[inline]
pub(crate) fn decide_dqpsk_symbol_from_llr(dqpsk_llr: [f32; 2]) -> Complex32 {
    if dqpsk_llr[0] >= 0.0 && dqpsk_llr[1] >= 0.0 {
        Complex32::new(1.0, 0.0)
    } else if dqpsk_llr[0] >= 0.0 && dqpsk_llr[1] < 0.0 {
        Complex32::new(0.0, 1.0)
    } else if dqpsk_llr[0] < 0.0 && dqpsk_llr[1] < 0.0 {
        Complex32::new(-1.0, 0.0)
    } else {
        Complex32::new(0.0, -1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coding::fec;
    use crate::frame::packet::{Packet, PACKET_BYTES};
    use crate::mary::demodulator::Demodulator;

    #[test]
    fn test_apply_llr_erasure_quantile_zeroes_small_abs_values() {
        let mut llrs = vec![0.1, -0.2, 0.5, -1.0, 2.0];
        apply_llr_erasure_quantile(&mut llrs, 0.4);
        assert_eq!(llrs, vec![0.0, 0.0, 0.5, -1.0, 2.0]);
    }

    #[test]
    fn test_decide_dqpsk_symbol_from_llr_maps_all_quadrants() {
        assert_eq!(
            decide_dqpsk_symbol_from_llr([1.0, 1.0]),
            Complex32::new(1.0, 0.0)
        );
        assert_eq!(
            decide_dqpsk_symbol_from_llr([1.0, -1.0]),
            Complex32::new(0.0, 1.0)
        );
        assert_eq!(
            decide_dqpsk_symbol_from_llr([-1.0, -1.0]),
            Complex32::new(-1.0, 0.0)
        );
        assert_eq!(
            decide_dqpsk_symbol_from_llr([-1.0, 1.0]),
            Complex32::new(0.0, -1.0)
        );
    }

    #[test]
    fn test_process_packet_core_returns_unprocessed_when_despread_fails() {
        let demodulator = Demodulator::new();
        let mut prev_phase = Complex32::new(1.0, 0.0);
        let mut tracking_state = TrackingState::new();
        let mut stats = DecoderStats::new();
        let mut buffers = PacketDecodeBuffers::new();
        let mut llr_callback: Option<LlrCallback> = None;
        let options = PacketDecodeOptions {
            spc: 2,
            early_late_delta: 1.0,
            viterbi_list_size: 1,
            llr_erasure_second_pass_enabled: false,
            llr_erasure_quantile: 0.2,
            llr_erasure_list_size: 1,
        };

        let result = process_packet_core(
            PacketDecodeRuntime {
                demodulator: &demodulator,
                prev_phase: &mut prev_phase,
                tracking_state: &mut tracking_state,
                stats: &mut stats,
                buffers: &mut buffers,
                llr_callback: &mut llr_callback,
            },
            &options,
            |_symbol_start, _timing_offset, _sample_shift| None,
        );

        assert!(!result.processed);
        assert!(result.packet.is_none());
        assert!(buffers.packet_llrs_buffer.is_empty());
    }

    #[test]
    fn test_try_decode_soft_list_candidates_returns_packet_for_valid_bits() {
        let packet = Packet::new(7, 3, &[0x5a; crate::params::PAYLOAD_SIZE]);
        let bits = fec::bytes_to_bits(&packet.serialize());

        let decoded = try_decode_soft_list_candidates(vec![bits], PACKET_BYTES * 8).unwrap();

        assert_eq!(decoded, packet);
    }

    #[test]
    fn test_try_decode_soft_list_candidates_distinguishes_crc_and_parse_errors() {
        let packet = Packet::new(1, 2, &[0x11; crate::params::PAYLOAD_SIZE]);
        let mut crc_bits = fec::bytes_to_bits(&packet.serialize());
        crc_bits[0] ^= 1;

        let crc_err = try_decode_soft_list_candidates(vec![crc_bits], PACKET_BYTES * 8);
        assert!(matches!(crc_err, Err(PacketDecodeError::Crc)));

        let parse_err =
            try_decode_soft_list_candidates(vec![vec![0; PACKET_BYTES * 8 - 1]], PACKET_BYTES * 8);
        assert!(matches!(parse_err, Err(PacketDecodeError::Parse)));
    }
}
