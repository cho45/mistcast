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
    pub fec_candidates_buffer: Vec<Vec<u8>>,
    pub fec_workspace: fec::FecDecodeWorkspace,
}

impl PacketDecodeBuffers {
    pub fn new() -> Self {
        let cap = interleaver_config::interleaved_bits();
        Self {
            packet_llrs_buffer: Vec::with_capacity(cap),
            deinterleave_buffer: Vec::with_capacity(cap),
            erasure_llr_buffer: Vec::with_capacity(cap),
            fec_candidates_buffer: Vec::new(),
            fec_workspace: fec::FecDecodeWorkspace::new(),
        }
    }

    pub fn clear(&mut self) {
        self.packet_llrs_buffer.clear();
        self.deinterleave_buffer.clear();
        self.erasure_llr_buffer.clear();
        self.fec_candidates_buffer.clear();
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

    fec::decode_soft_list_into(
        &context.buffers.deinterleave_buffer[..layout.fec_bits],
        context.options.viterbi_list_size,
        &mut context.buffers.fec_candidates_buffer,
        &mut context.buffers.fec_workspace,
    );
    let first_attempt = try_decode_soft_list_candidates(
        &context.buffers.fec_candidates_buffer,
        layout.payload_bits_len,
    );
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
        fec::decode_soft_list_into(
            &context.buffers.erasure_llr_buffer[..layout.fec_bits],
            context.options.llr_erasure_list_size,
            &mut context.buffers.fec_candidates_buffer,
            &mut context.buffers.fec_workspace,
        );
        match try_decode_soft_list_candidates(
            &context.buffers.fec_candidates_buffer,
            layout.payload_bits_len,
        ) {
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
    decoded_candidates: &[Vec<u8>],
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
    use crate::coding::interleaver::BlockInterleaver;
    use crate::coding::scrambler::Scrambler;
    use crate::frame::packet::{Packet, PACKET_BYTES};
    use crate::mary::demodulator::Demodulator;
    use crate::mary::interleaver_config;
    use std::sync::{Arc, Mutex};

    fn default_options() -> PacketDecodeOptions {
        PacketDecodeOptions {
            spc: 2,
            early_late_delta: 1.0,
            viterbi_list_size: 1,
            llr_erasure_second_pass_enabled: true,
            llr_erasure_quantile: 0.2,
            llr_erasure_list_size: 1,
        }
    }

    fn decode_layout() -> DecodeLayout {
        DecodeLayout {
            rows: interleaver_config::INTERLEAVER_ROWS,
            cols: interleaver_config::INTERLEAVER_COLS,
            interleaved_bits: interleaver_config::interleaved_bits(),
            fec_bits: interleaver_config::fec_bits(),
            payload_bits_len: PACKET_BYTES * 8,
        }
    }

    fn encode_packet_to_interleaved_llrs(packet: &Packet, magnitude: f32) -> Vec<f32> {
        encode_packet_to_interleaved_bits(packet)
            .into_iter()
            .map(|bit| if bit == 0 { magnitude } else { -magnitude })
            .collect()
    }

    fn encode_packet_to_descrambled_llrs(packet: &Packet, magnitude: f32) -> Vec<f32> {
        fec::encode(&fec::bytes_to_bits(&packet.serialize()))
            .into_iter()
            .map(|bit| if bit == 0 { magnitude } else { -magnitude })
            .collect()
    }

    fn encode_packet_to_interleaved_bits(packet: &Packet) -> Vec<u8> {
        let bits = fec::bytes_to_bits(&packet.serialize());
        let mut fec_bits = fec::encode(&bits);
        let mut scrambler = Scrambler::default();
        scrambler.process_bits(&mut fec_bits);
        let interleaver = BlockInterleaver::new(
            interleaver_config::INTERLEAVER_ROWS,
            interleaver_config::INTERLEAVER_COLS,
        );
        interleaver.interleave(&fec_bits)
    }

    fn dqpsk_symbol_from_bits(bits: &[u8]) -> Complex32 {
        match (bits[0], bits[1]) {
            (0, 0) => Complex32::new(1.0, 0.0),
            (0, 1) => Complex32::new(0.0, 1.0),
            (1, 1) => Complex32::new(-1.0, 0.0),
            (1, 0) => Complex32::new(0.0, -1.0),
            _ => unreachable!(),
        }
    }

    fn build_symbol_correlation_sequence(packet: &Packet, amplitude: f32) -> Vec<[Complex32; 16]> {
        let bits = encode_packet_to_interleaved_bits(packet);
        let mut prev = Complex32::new(1.0, 0.0);
        bits.chunks_exact(6)
            .map(|chunk| {
                let walsh_idx = chunk[..4]
                    .iter()
                    .fold(0usize, |acc, &bit| (acc << 1) | bit as usize);
                let diff = dqpsk_symbol_from_bits(&chunk[4..6]);
                let current = prev * diff;
                prev = current;
                let mut corrs = [Complex32::new(0.0, 0.0); 16];
                corrs[walsh_idx] = current * amplitude;
                corrs
            })
            .collect()
    }

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
        let options = default_options();

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

        let candidates = vec![bits];
        let decoded = try_decode_soft_list_candidates(&candidates, PACKET_BYTES * 8).unwrap();

        assert_eq!(decoded, packet);
    }

    #[test]
    fn test_try_decode_soft_list_candidates_distinguishes_crc_and_parse_errors() {
        let packet = Packet::new(1, 2, &[0x11; crate::params::PAYLOAD_SIZE]);
        let mut crc_bits = fec::bytes_to_bits(&packet.serialize());
        crc_bits[0] ^= 1;

        let crc_candidates = vec![crc_bits];
        let crc_err = try_decode_soft_list_candidates(&crc_candidates, PACKET_BYTES * 8);
        assert!(matches!(crc_err, Err(PacketDecodeError::Crc)));

        let parse_candidates = vec![vec![0; PACKET_BYTES * 8 - 1]];
        let parse_err = try_decode_soft_list_candidates(&parse_candidates, PACKET_BYTES * 8);
        assert!(matches!(parse_err, Err(PacketDecodeError::Parse)));
    }

    #[test]
    fn test_decode_packet_distinguishes_crc_and_parse_errors() {
        let packet = Packet::new(9, 4, &[0x33; crate::params::PAYLOAD_SIZE]);
        let valid_bits = fec::bytes_to_bits(&packet.serialize());
        assert_eq!(decode_packet(&valid_bits).unwrap(), packet);

        let mut crc_bits = valid_bits.clone();
        crc_bits[5] ^= 1;
        assert!(matches!(
            decode_packet(&crc_bits),
            Err(PacketDecodeError::Crc)
        ));

        let mut parse_bits = valid_bits.clone();
        parse_bits.extend_from_slice(&[0; 8]);
        assert!(matches!(
            decode_packet(&parse_bits),
            Err(PacketDecodeError::Parse)
        ));
    }

    #[test]
    fn test_decode_single_llr_candidate_recovers_packet_and_invokes_callback() {
        let packet = Packet::new(5, 8, &[0x6c; crate::params::PAYLOAD_SIZE]);
        let magnitude = 8.0;
        let llrs = encode_packet_to_interleaved_llrs(&packet, magnitude);
        let expected_callback_llrs = encode_packet_to_descrambled_llrs(&packet, magnitude);
        let layout = decode_layout();
        let mut stats = DecoderStats::new();
        let mut buffers = PacketDecodeBuffers::new();
        let captured = Arc::new(Mutex::new(Vec::<f32>::new()));
        let captured_for_cb = Arc::clone(&captured);
        let mut llr_callback: Option<LlrCallback> = Some(Box::new(move |values| {
            let mut slot = captured_for_cb.lock().unwrap();
            slot.clear();
            slot.extend_from_slice(values);
        }));
        let options = default_options();
        let mut context = DecodeCandidateContext {
            stats: &mut stats,
            buffers: &mut buffers,
            llr_callback: &mut llr_callback,
            options: &options,
        };

        let decoded = decode_single_llr_candidate(&llrs, &layout, &mut context).unwrap();

        assert_eq!(decoded, packet);
        let observed = captured.lock().unwrap().clone();
        assert_eq!(observed.len(), interleaver_config::fec_bits());
        assert_eq!(observed, expected_callback_llrs);
        assert_eq!(stats.llr_second_pass_attempts, 0);
        assert_eq!(stats.llr_second_pass_rescued, 0);
    }

    #[test]
    fn test_decode_single_llr_candidate_reports_crc_for_corrupted_llrs_without_second_pass() {
        let packet = Packet::new(4, 9, &[0x3c; crate::params::PAYLOAD_SIZE]);
        let llrs = encode_packet_to_interleaved_llrs(&packet, 8.0)
            .into_iter()
            .map(|llr| -llr)
            .collect::<Vec<_>>();
        let layout = decode_layout();
        let mut options = default_options();
        options.llr_erasure_second_pass_enabled = false;
        let mut stats = DecoderStats::new();
        let mut buffers = PacketDecodeBuffers::new();
        let mut llr_callback: Option<LlrCallback> = None;
        let mut context = DecodeCandidateContext {
            stats: &mut stats,
            buffers: &mut buffers,
            llr_callback: &mut llr_callback,
            options: &options,
        };
        let result = decode_single_llr_candidate(&llrs, &layout, &mut context);

        assert!(matches!(result, Err(PacketDecodeError::Crc)));
        assert_eq!(stats.llr_second_pass_attempts, 0);
        assert_eq!(stats.llr_second_pass_rescued, 0);
    }

    #[test]
    fn test_decode_llrs_skips_first_invalid_chunk_and_decodes_later_valid_chunk() {
        let packet = Packet::new(3, 7, &[0x44; crate::params::PAYLOAD_SIZE]);
        let valid_llrs = encode_packet_to_interleaved_llrs(&packet, 8.0);
        let mut combined = valid_llrs.iter().map(|llr| -*llr).collect::<Vec<_>>();
        combined.extend_from_slice(&valid_llrs);
        let mut stats = DecoderStats::new();
        let mut buffers = PacketDecodeBuffers::new();
        let mut llr_callback: Option<LlrCallback> = None;
        let options = default_options();

        let decoded = decode_llrs(
            &combined,
            &mut stats,
            &mut buffers,
            &mut llr_callback,
            &options,
        )
        .unwrap();

        assert_eq!(decoded, packet);
        assert_eq!(stats.crc_error_packets, 1);
        assert_eq!(stats.parse_error_packets, 0);
    }

    #[test]
    fn test_process_packet_core_recovers_packet_and_updates_packet_stats() {
        let packet = Packet::new(13, 6, &[0x55; crate::params::PAYLOAD_SIZE]);
        let symbol_corrs = build_symbol_correlation_sequence(&packet, 10.0);
        let demodulator = Demodulator::new();
        let mut prev_phase = Complex32::new(1.0, 0.0);
        let mut tracking_state = TrackingState::new();
        let mut stats = DecoderStats::new();
        let mut buffers = PacketDecodeBuffers::new();
        let mut llr_callback: Option<LlrCallback> = None;
        let options = default_options();

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
            |symbol_start, _timing_offset, _sample_shift| {
                let sym_idx = (symbol_start - options.spc) / (PAYLOAD_SPREAD_FACTOR * options.spc);
                symbol_corrs.get(sym_idx).copied()
            },
        );

        assert!(result.processed);
        assert_eq!(result.packet, Some(packet));
        assert_eq!(
            stats.phase_gate_on_symbols + stats.phase_gate_off_symbols,
            interleaver_config::mary_symbols()
        );
        assert_eq!(
            stats.phase_err_abs_count,
            interleaver_config::mary_symbols()
        );
    }

    #[test]
    fn test_decode_llrs_stops_on_short_trailing_chunk() {
        let packet = Packet::new(11, 1, &[0x22; crate::params::PAYLOAD_SIZE]);
        let mut llrs = encode_packet_to_interleaved_llrs(&packet, 8.0);
        llrs.truncate(interleaver_config::interleaved_bits() - 1);
        let mut stats = DecoderStats::new();
        let mut buffers = PacketDecodeBuffers::new();
        let mut llr_callback: Option<LlrCallback> = None;
        let options = default_options();

        let decoded = decode_llrs(&llrs, &mut stats, &mut buffers, &mut llr_callback, &options);

        assert!(decoded.is_none());
        assert_eq!(stats.crc_error_packets, 0);
        assert_eq!(stats.parse_error_packets, 0);
    }
}
