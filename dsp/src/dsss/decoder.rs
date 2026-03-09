//! 受信パイプライン (統合デコーダ)

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::common::nco::complex_mul_interleaved2_simd;
use crate::{
    coding::fec,
    coding::fountain::{FountainDecoder, FountainParams, ReceiveOutcome},
    coding::interleaver::BlockInterleaver,
    common::nco::Nco,
    common::resample::Resampler,
    common::rrc_filter::RrcFilter,
    dsss::sync::{SyncDetector, SyncResult},
    frame::packet::{Packet, PacketParseError, PACKET_BYTES},
    params::{MODULATION, PAYLOAD_SIZE},
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
    pub synced_frames: usize,
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

pub type LlrCallback = Box<dyn Fn(&[f32]) + Send + Sync>;

pub struct Decoder {
    pub config: DspConfig,
    resampler_i: Resampler,
    resampler_q: Resampler,
    rrc_filter_i: RrcFilter,
    rrc_filter_q: RrcFilter,
    sample_buffer_i: Vec<f32>,
    sample_buffer_q: Vec<f32>,
    sample_buffer_start: usize,
    mix_buffer_i: Vec<f32>,
    mix_buffer_q: Vec<f32>,
    resample_buffer_i: Vec<f32>,
    resample_buffer_q: Vec<f32>,
    pn_chips: Vec<f32>,
    pn_cache_mseq_order: usize,
    pn_cache_sf: usize,
    sync_detector: SyncDetector,
    interleaver: BlockInterleaver,
    deinterleave_buffer: Vec<f32>,
    decoded_bits_buffer: Vec<u8>,
    decoded_bytes_buffer: Vec<u8>,
    fec_workspace: fec::FecDecodeWorkspace,
    fountain_decoder: FountainDecoder,
    recovered_data: Option<Vec<u8>>,
    pub agc_peak_fast: f32,
    pub agc_peak_slow: f32,
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
    pub stats_synced_frames: usize,
    tracking_llr_buffer: Vec<f32>,
    best_attempt_llr_buffer: Vec<f32>,
    tracking_packets_buffer: Vec<Packet>,
    best_attempt_packets_buffer: Vec<Packet>,
    pub llr_callback: Option<LlrCallback>,
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
struct PayloadDecodeAttempt {
    packet_count: usize,
    crc_errors: usize,
    parse_errors: usize,
}

#[derive(Clone, Copy, Debug)]
struct SymbolSoftDecision {
    decided: Complex32,
    llrs: [f32; 2],
    llr_count: usize,
}

impl Decoder {
    #[inline]
    fn active_sample_buffer_i(&self) -> &[f32] {
        &self.sample_buffer_i[self.sample_buffer_start..]
    }

    #[inline]
    fn active_sample_buffer_q(&self) -> &[f32] {
        &self.sample_buffer_q[self.sample_buffer_start..]
    }

    #[inline]
    fn active_sample_len(&self) -> usize {
        self.sample_buffer_i
            .len()
            .saturating_sub(self.sample_buffer_start)
    }

    #[inline]
    fn consume_sample_front(&mut self, n: usize) {
        let consume = n.min(self.active_sample_len());
        self.sample_buffer_start =
            (self.sample_buffer_start + consume).min(self.sample_buffer_i.len());
    }

    #[inline]
    fn maybe_compact_sample_buffers(&mut self, max_active_len: usize) {
        if self.sample_buffer_start == 0 {
            return;
        }
        let len = self.sample_buffer_i.len();
        let should_compact = self.sample_buffer_start >= (max_active_len / 2).max(1)
            || self.sample_buffer_start * 2 >= len
            || len > max_active_len.saturating_mul(2);

        if should_compact {
            self.sample_buffer_i.drain(0..self.sample_buffer_start);
            self.sample_buffer_q.drain(0..self.sample_buffer_start);
            self.sample_buffer_start = 0;
        }
    }

    fn build_pn_chips(mseq_order: usize, sf: usize) -> Vec<f32> {
        let mut mseq = crate::common::msequence::MSequence::new(mseq_order);
        let mut chips = Vec::with_capacity(sf);
        mseq.generate_into(sf, &mut chips);
        chips.into_iter().map(|v| v as f32).collect()
    }

    fn refresh_pn_cache_if_needed(&mut self) {
        let sf = self.config.spread_factor();
        let order = self.config.mseq_order;
        if self.pn_cache_mseq_order != order || self.pn_cache_sf != sf {
            self.pn_chips = Self::build_pn_chips(order, sf);
            self.pn_cache_mseq_order = order;
            self.pn_cache_sf = sf;
        }
    }

    pub fn new(_data_size: usize, fountain_k: usize, dsp_config: DspConfig) -> Self {
        let proc_sample_rate = dsp_config.proc_sample_rate();
        let params = FountainParams::new(fountain_k, PAYLOAD_SIZE);
        let raw_bits = PACKET_BYTES * 8 + 6;
        let fec_bits = raw_bits * 2;
        let il_rows = 16;
        let il_cols = fec_bits.div_ceil(16);
        let packets_per_burst = dsp_config.packets_per_burst.max(1);
        let lo_nco = Nco::new(-dsp_config.carrier_freq, dsp_config.sample_rate);

        // リサンプラのカットオフ設定:
        // RRCのロールオフを含めた帯域 Rc*(1+alpha)/2 を保護しつつ、
        // エイリアシングを防ぐために proc_sample_rate / 2 以下に設定する。
        let rrc_bw = dsp_config.chip_rate * (1.0 + dsp_config.rrc_alpha) * 0.5;
        let cutoff = Some(rrc_bw);

        // 拡散率に応じて同期しきい値をスケールさせる。
        // 処理利得 (Processing Gain) によりノイズフロアが 1/sf に比例して下がるため、
        // しきい値もそれに合わせて引き下げることで感度を維持する。
        let sf = dsp_config.spread_factor();
        let pn_chips = Self::build_pn_chips(dsp_config.mseq_order, sf);
        let scale = 15.0 / sf as f32;
        let tc = SyncDetector::THRESHOLD_COARSE_DEFAULT * scale;
        let tf = SyncDetector::THRESHOLD_FINE_DEFAULT * scale;
        let mut fec_workspace = fec::FecDecodeWorkspace::new();
        fec_workspace.preallocate_for_llr_len(fec_bits, 1);

        Decoder {
            resampler_i: Resampler::new_with_cutoff(
                dsp_config.sample_rate as u32,
                proc_sample_rate as u32,
                cutoff,
                Some(dsp_config.rx_resampler_taps),
            ),
            resampler_q: Resampler::new_with_cutoff(
                dsp_config.sample_rate as u32,
                proc_sample_rate as u32,
                cutoff,
                Some(dsp_config.rx_resampler_taps),
            ),
            rrc_filter_i: RrcFilter::from_config(&dsp_config),
            rrc_filter_q: RrcFilter::from_config(&dsp_config),
            sample_buffer_i: Vec::new(),
            sample_buffer_q: Vec::new(),
            sample_buffer_start: 0,
            mix_buffer_i: Vec::with_capacity(4096),
            mix_buffer_q: Vec::with_capacity(4096),
            resample_buffer_i: Vec::with_capacity(6144),
            resample_buffer_q: Vec::with_capacity(6144),
            pn_chips,
            pn_cache_mseq_order: dsp_config.mseq_order,
            pn_cache_sf: sf,
            sync_detector: SyncDetector::new(dsp_config.clone(), tc, tf),
            interleaver: BlockInterleaver::new(il_rows, il_cols),
            deinterleave_buffer: Vec::with_capacity(fec_bits),
            decoded_bits_buffer: Vec::with_capacity(raw_bits),
            decoded_bytes_buffer: Vec::with_capacity(PACKET_BYTES),
            fec_workspace,
            fountain_decoder: FountainDecoder::new(params),
            recovered_data: None,
            config: dsp_config,
            agc_peak_fast: 0.5,
            agc_peak_slow: 0.5,
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
            stats_synced_frames: 0,
            tracking_llr_buffer: Vec::with_capacity(fec_bits),
            best_attempt_llr_buffer: Vec::with_capacity(fec_bits),
            tracking_packets_buffer: Vec::with_capacity(packets_per_burst),
            best_attempt_packets_buffer: Vec::with_capacity(packets_per_burst),
            llr_callback: None,
        }
    }

    pub fn process_samples(&mut self, samples: &[f32]) -> DecodeProgress {
        if self.recovered_data.is_some() {
            return self.progress();
        }
        self.stats_total_samples += samples.len();

        let spc = self.config.proc_samples_per_chip();
        let sf = self.config.spread_factor();
        // 処理レート基準でバッファ制限を計算
        let max_buffer_len = 100_000;
        let drain_len = 50_000;
        self.refresh_pn_cache_if_needed();
        let pn = std::mem::take(&mut self.pn_chips);

        // 1. 高レート混合 (at fs_in)
        let mut i_mixed = std::mem::take(&mut self.mix_buffer_i);
        let mut q_mixed = std::mem::take(&mut self.mix_buffer_q);
        self.mix_real_to_iq(samples, &mut i_mixed, &mut q_mixed);

        // 2. ベースバンド・リサンプリング (fs_in -> fs_proc)
        let mut i_resampled = std::mem::take(&mut self.resample_buffer_i);
        let mut q_resampled = std::mem::take(&mut self.resample_buffer_q);
        i_resampled.clear();
        q_resampled.clear();
        self.resampler_i.process(&i_mixed, &mut i_resampled);
        self.resampler_q.process(&q_mixed, &mut q_resampled);

        // 3. マッチドフィルタリング (at fs_proc)
        self.rrc_filter_i.process_block_in_place(&mut i_resampled);
        self.rrc_filter_q.process_block_in_place(&mut q_resampled);

        // 4. 処理用バッファへ追加
        self.sample_buffer_i.extend_from_slice(&i_resampled);
        self.sample_buffer_q.extend_from_slice(&q_resampled);

        let sync_bits_len = self.config.sync_word_bits;
        let sync_symbol_len = sync_bits_len;
        let bits_per_symbol_payload = MODULATION.bits_per_symbol();
        let fec_bits_len = self.interleaver.rows() * self.interleaver.cols();
        let burst_data_bits_len = fec_bits_len * self.config.packets_per_burst.max(1);
        let payload_symbols = burst_data_bits_len.div_ceil(bits_per_symbol_payload);
        let total_symbols = sync_symbol_len + payload_symbols;
        let symbol_len = sf * spc;
        let frame_samples = (total_symbols * symbol_len).max(1);
        let queued_frames = self.active_sample_len() / frame_samples;
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
                    self.active_sample_buffer_i(),
                    self.active_sample_buffer_q(),
                    self.last_search_idx,
                );
                self.stats_sync_calls += 1;
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.stats_sync_time += sync_start.elapsed();
                }

                if let Some(s) = sync_opt {
                    self.stats_synced_frames += 1;
                    self.last_search_idx = next_search_idx + 1;
                    self.current_sync = Some(s.clone());
                    s
                } else {
                    self.last_search_idx = next_search_idx;
                    if self.active_sample_len() > max_buffer_len {
                        let drain = drain_len;
                        self.consume_sample_front(drain);
                        self.last_search_idx = self.last_search_idx.saturating_sub(drain);
                        self.maybe_compact_sample_buffers(max_buffer_len);
                    }
                    break;
                }
            };

            let start = sync.peak_sample_idx;
            let data_end_sample = start + total_symbols * sf * spc;
            let decode_guard_samples = spc * 2 + 1;
            let decode_ready_end = data_end_sample + decode_guard_samples;
            let next_search_after_candidate = data_end_sample.min(self.active_sample_len());

            // データが溜まるのを待つ (start は既に SYNC_WORD の開始点付近)
            if self.active_sample_len() < decode_ready_end {
                // タイムアウト監視
                if start + symbol_len < self.active_sample_len().saturating_sub(max_buffer_len) {
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
                prev_symbol: Complex32::new(1.0, 0.0),
                phase_rate: 0.0,
                timing_offset: 0.0,
                timing_rate: 0.0,
                noise_var: 0.2,
            };

            let p_bits_len = PACKET_BYTES * 8;
            let Some(attempt) = self.decode_payload_with_timing_retries(
                payload_start,
                burst_data_bits_len,
                best_tracking_after_sync,
                &pn,
                fec_bits_len,
                p_bits_len,
            ) else {
                // 同じ候補フレームの payload 上を再探索するとゾンビ同期が連鎖する。
                // 失敗した候補はフレーム末尾までまとめてスキップする。
                self.last_search_idx = next_search_after_candidate;
                self.current_sync = None;
                continue;
            };
            let crc_errors = attempt.crc_errors;
            let parse_errors = attempt.parse_errors;
            self.crc_error_packets += crc_errors;
            self.parse_error_packets += parse_errors;

            if attempt.packet_count == 0 || crc_errors > 0 {
                // payload が全滅した候補は境界ずれとして破棄する。
                if self.fountain_decoder.received_count() == 0 && self.crc_error_packets < 20 {
                    // 起動直後だけ同期長ぶん先へ寄せて再探索し、誤同期固定化を崩す。
                    let startup_search_step =
                        (self.config.preamble_repeat + self.config.sync_word_bits) * symbol_len;
                    self.last_search_idx =
                        start.saturating_add(startup_search_step.max(symbol_len));
                } else {
                    // payload 内の擬似ピークを拾わないよう、候補フレーム全体を飛ばす。
                    self.last_search_idx = next_search_after_candidate;
                }
                self.current_sync = None;
                continue;
            }

            let mut decoded_packets = Vec::new();
            std::mem::swap(&mut decoded_packets, &mut self.best_attempt_packets_buffer);
            for packet in decoded_packets.drain(..) {
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
                let outcome = self.fountain_decoder.receive_payload_array_with_outcome(
                    seq,
                    coefficients,
                    packet.payload,
                );
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
                }
            }
            decoded_packets.clear();
            std::mem::swap(&mut decoded_packets, &mut self.best_attempt_packets_buffer);
            // 受信窓をバースト分前進させる
            let actual_end = (payload_start
                + burst_data_bits_len / bits_per_symbol_payload * sf * spc)
                .min(self.active_sample_len());
            self.consume_sample_front(actual_end);
            self.last_search_idx = 0;
            self.current_sync = None;
            self.maybe_compact_sample_buffers(max_buffer_len);
        }

        i_mixed.clear();
        q_mixed.clear();
        i_resampled.clear();
        q_resampled.clear();
        self.mix_buffer_i = i_mixed;
        self.mix_buffer_q = q_mixed;
        self.resample_buffer_i = i_resampled;
        self.resample_buffer_q = q_resampled;
        self.pn_chips = pn;

        self.progress()
    }

    fn despread_symbol_with_timing(
        &self,
        symbol_start: usize,
        pn: &[f32],
        timing_offset: f32,
        sample_shift: f32,
    ) -> Option<Complex32> {
        let spc = self.config.proc_samples_per_chip().max(1);
        let active_len = self.active_sample_len() as i32;
        let mut sum_i = 0.0f32;
        let mut sum_q = 0.0f32;
        for (chip_idx, &pn_val) in pn.iter().enumerate() {
            let p = symbol_start as f32
                + (chip_idx * spc + (spc / 2)) as f32
                + timing_offset
                + sample_shift;

            let i0 = p.floor() as i32;
            let frac = p - i0 as f32;
            let i1 = i0 + 1;
            if i0 < 0 || i1 >= active_len {
                return None;
            }
            let base = self.sample_buffer_start;
            let i0u = base + i0 as usize;
            let i1u = base + i1 as usize;
            let w0 = 1.0 - frac;
            let w1 = frac;
            let si = self.sample_buffer_i[i0u] * w0 + self.sample_buffer_i[i1u] * w1;
            let sq = self.sample_buffer_q[i0u] * w0 + self.sample_buffer_q[i1u] * w1;
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
        llrs: &mut Vec<f32>,
    ) -> Option<()> {
        let sf = self.config.spread_factor();
        let spc = self.config.proc_samples_per_chip().max(1);
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
        llrs.clear();
        if llrs.capacity() < num_bits {
            llrs.reserve(num_bits - llrs.capacity());
        }
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
        Some(())
    }

    #[allow(clippy::too_many_arguments)]
    fn try_decode_payload_once(
        &mut self,
        payload_start: usize,
        burst_data_bits_len: usize,
        timing_bias: f32,
        mut initial_state: TrackingState,
        pn: &[f32],
        fec_bits_len: usize,
        p_bits_len: usize,
        llrs: &mut Vec<f32>,
        decoded_packets: &mut Vec<Packet>,
    ) -> Option<PayloadDecodeAttempt> {
        let spc = self.config.proc_samples_per_chip().max(1) as f32;
        let timing_limit = spc * TRACKING_TIMING_LIMIT_CHIP;
        initial_state.timing_offset =
            (initial_state.timing_offset + timing_bias).clamp(-timing_limit, timing_limit);
        self.decode_bits_with_tracking(
            payload_start,
            burst_data_bits_len,
            initial_state,
            pn,
            llrs,
        )?;
        let (crc_errors, parse_errors) =
            self.parse_payload_packets_into(llrs, fec_bits_len, p_bits_len, decoded_packets);
        Some(PayloadDecodeAttempt {
            packet_count: decoded_packets.len(),
            crc_errors,
            parse_errors,
        })
    }

    fn is_better_payload_attempt(
        candidate: &PayloadDecodeAttempt,
        best: &PayloadDecodeAttempt,
    ) -> bool {
        candidate.packet_count > best.packet_count
            || (candidate.packet_count == best.packet_count
                && (candidate.crc_errors < best.crc_errors
                    || (candidate.crc_errors == best.crc_errors
                        && candidate.parse_errors < best.parse_errors)))
    }

    fn decode_payload_with_timing_retries(
        &mut self,
        payload_start: usize,
        burst_data_bits_len: usize,
        initial_state: TrackingState,
        pn: &[f32],
        fec_bits_len: usize,
        p_bits_len: usize,
    ) -> Option<PayloadDecodeAttempt> {
        let spc = self.config.proc_samples_per_chip().max(1) as f32;
        let timing_biases = [-0.75f32 * spc, 0.0, 0.75f32 * spc];

        let mut attempt_llrs = Vec::new();
        let mut best_llrs = Vec::new();
        let mut attempt_packets = Vec::new();
        let mut best_packets = Vec::new();
        std::mem::swap(&mut attempt_llrs, &mut self.tracking_llr_buffer);
        std::mem::swap(&mut best_llrs, &mut self.best_attempt_llr_buffer);
        std::mem::swap(&mut attempt_packets, &mut self.tracking_packets_buffer);
        std::mem::swap(&mut best_packets, &mut self.best_attempt_packets_buffer);
        attempt_packets.clear();
        best_packets.clear();

        let mut best: Option<PayloadDecodeAttempt> = None;
        for timing_bias in timing_biases {
            if let Some(attempt) = self.try_decode_payload_once(
                payload_start,
                burst_data_bits_len,
                timing_bias,
                initial_state,
                pn,
                fec_bits_len,
                p_bits_len,
                &mut attempt_llrs,
                &mut attempt_packets,
            ) {
                let replace = best
                    .as_ref()
                    .is_none_or(|current| Self::is_better_payload_attempt(&attempt, current));
                if replace {
                    best_llrs.clear();
                    best_llrs.extend_from_slice(&attempt_llrs);
                    std::mem::swap(&mut best_packets, &mut attempt_packets);
                    best = Some(attempt);
                }
            }
        }
        if best.is_some() {
            self.emit_packet_llr_callbacks(&best_llrs, fec_bits_len);
        } else {
            best_packets.clear();
        }

        attempt_llrs.clear();
        best_llrs.clear();
        attempt_packets.clear();
        std::mem::swap(&mut attempt_llrs, &mut self.tracking_llr_buffer);
        std::mem::swap(&mut best_llrs, &mut self.best_attempt_llr_buffer);
        std::mem::swap(&mut attempt_packets, &mut self.tracking_packets_buffer);
        std::mem::swap(&mut best_packets, &mut self.best_attempt_packets_buffer);

        best
    }

    fn parse_payload_packets_into(
        &mut self,
        payload_llrs: &[f32],
        fec_bits_len: usize,
        p_bits_len: usize,
        decoded_packets: &mut Vec<Packet>,
    ) -> (usize, usize) {
        decoded_packets.clear();
        let packet_capacity = if fec_bits_len > 0 {
            payload_llrs.len() / fec_bits_len
        } else {
            0
        };
        if decoded_packets.capacity() < packet_capacity {
            decoded_packets.reserve(packet_capacity - decoded_packets.capacity());
        }
        let mut crc_errors = 0usize;
        let mut parse_errors = 0usize;
        self.deinterleave_buffer.resize(fec_bits_len, 0.0);
        for packet_llrs in payload_llrs.chunks_exact(fec_bits_len) {
            self.interleaver
                .deinterleave_f32_in_place(packet_llrs, &mut self.deinterleave_buffer);

            let mut scrambler = crate::coding::scrambler::Scrambler::default();
            for llr in self.deinterleave_buffer.iter_mut() {
                if scrambler.next_bit() == 1 {
                    *llr = -*llr;
                }
            }

            fec::decode_soft_into(
                &self.deinterleave_buffer,
                &mut self.decoded_bits_buffer,
                &mut self.fec_workspace,
            );
            match parse_packet_from_decoded_bits(
                &self.decoded_bits_buffer,
                p_bits_len,
                &mut self.decoded_bytes_buffer,
            ) {
                Ok(packet) => decoded_packets.push(packet),
                Err(PacketParseError::CrcMismatch { .. }) => crc_errors += 1,
                Err(PacketParseError::InvalidLength { .. }) => parse_errors += 1,
            }
        }
        (crc_errors, parse_errors)
    }

    fn emit_packet_llr_callbacks(&mut self, payload_llrs: &[f32], fec_bits_len: usize) {
        let Some(callback) = self.llr_callback.as_mut() else {
            return;
        };
        self.deinterleave_buffer.resize(fec_bits_len, 0.0);
        for packet_llrs in payload_llrs.chunks_exact(fec_bits_len) {
            self.interleaver
                .deinterleave_f32_in_place(packet_llrs, &mut self.deinterleave_buffer);
            let mut scrambler = crate::coding::scrambler::Scrambler::default();
            for llr in self.deinterleave_buffer.iter_mut() {
                if scrambler.next_bit() == 1 {
                    *llr = -*llr;
                }
            }
            callback(&self.deinterleave_buffer);
        }
    }

    fn progress(&self) -> DecodeProgress {
        let received = self.fountain_decoder.received_count();
        let needed = self.fountain_decoder.needed_count();
        let rank = self.fountain_decoder.rank();
        DecodeProgress {
            synced_frames: self.stats_synced_frames,
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
            basis_matrix: self.fountain_decoder.get_basis_matrix(),
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

    pub fn reset_fountain_decoder(&mut self) {
        self.rebuild_fountain_decoder(self.fountain_decoder.params().k);
        self.stats_synced_frames = 0;
    }

    pub fn reset(&mut self) {
        self.interleaver.reset();
        self.resampler_i.reconfigure(
            self.config.sample_rate as u32,
            self.config.proc_sample_rate() as u32,
            Some(self.config.chip_rate * (1.0 + self.config.rrc_alpha) * 0.5),
            Some(self.config.rx_resampler_taps),
        );
        self.resampler_q.reconfigure(
            self.config.sample_rate as u32,
            self.config.proc_sample_rate() as u32,
            Some(self.config.chip_rate * (1.0 + self.config.rrc_alpha) * 0.5),
            Some(self.config.rx_resampler_taps),
        );
        self.rrc_filter_i.reset();
        self.rrc_filter_q.reset();
        self.sample_buffer_i.clear();
        self.sample_buffer_q.clear();
        self.sample_buffer_start = 0;
        self.mix_buffer_i.clear();
        self.mix_buffer_q.clear();
        self.resample_buffer_i.clear();
        self.resample_buffer_q.clear();
        self.refresh_pn_cache_if_needed();
        self.lo_nco.reset();
        self.agc_peak_fast = 0.5;
        self.agc_peak_slow = 0.5;
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
        self.stats_sync_calls = 0;
        self.stats_sync_time = Duration::ZERO;
        self.stats_total_samples = 0;
        self.stats_synced_frames = 0;
        self.tracking_llr_buffer.clear();
        self.best_attempt_llr_buffer.clear();
        self.tracking_packets_buffer.clear();
        self.best_attempt_packets_buffer.clear();
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
    decoded_bytes: &mut Vec<u8>,
) -> Result<Packet, PacketParseError> {
    if decoded_bits.len() < p_bits_len {
        return Err(PacketParseError::InvalidLength {
            actual: decoded_bits.len() / 8,
        });
    }
    fec::bits_to_bytes_into(&decoded_bits[..p_bits_len], decoded_bytes);
    Packet::deserialize(decoded_bytes)
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
    use crate::coding::scrambler::Scrambler;
    use crate::params::FIXED_K;
    use crate::{
        dsss::encoder::{Encoder, EncoderConfig},
        DspConfig,
    };
    use std::sync::{Arc, Mutex};
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn scalar_mix_reference(
        decoder: &mut Decoder,
        samples: &[f32],
        i_out: &mut Vec<f32>,
        q_out: &mut Vec<f32>,
    ) {
        i_out.clear();
        q_out.clear();
        i_out.reserve(samples.len());
        q_out.reserve(samples.len());
        for &sample in samples {
            let s = decoder.agc_scale(sample);
            let lo = decoder.lo_nco.step();
            i_out.push(s * lo.re);
            q_out.push(s * lo.im);
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn assert_vec_close(actual: &[f32], expected: &[f32], eps: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= eps,
                "idx={} actual={} expected={} eps={}",
                idx,
                a,
                e,
                eps
            );
        }
    }

    fn encode_packet_bits_to_interleaved_llrs(bits: &[u8], magnitude: f32) -> Vec<f32> {
        let mut fec_bits = fec::encode(bits);
        let mut scrambler = Scrambler::default();
        scrambler.process_bits(&mut fec_bits);
        let interleaver = BlockInterleaver::new(16, fec_bits.len().div_ceil(16));
        interleaver
            .interleave(&fec_bits)
            .into_iter()
            .map(|bit| if bit == 0 { magnitude } else { -magnitude })
            .collect()
    }

    #[test]
    fn test_decoder_silence_input_does_not_complete() {
        let dsp_config = crate::dsss::params::dsp_config_48k();
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
    fn test_unsynced_long_run_keeps_physical_buffer_bounded() {
        let dsp_config = crate::dsss::params::dsp_config_48k();
        let mut decoder = Decoder::new(32, FIXED_K, dsp_config);
        let chunk = vec![0.0f32; 2048];

        let mut peak_physical_len = 0usize;
        let mut peak_active_len = 0usize;
        for _ in 0..600 {
            let progress = decoder.process_samples(&chunk);
            assert!(
                !progress.complete,
                "silence should remain unsynced during long run"
            );

            peak_physical_len = peak_physical_len.max(decoder.sample_buffer_i.len());
            peak_active_len = peak_active_len.max(decoder.active_sample_len());
            assert_eq!(decoder.sample_buffer_i.len(), decoder.sample_buffer_q.len());
            assert!(decoder.sample_buffer_start <= decoder.sample_buffer_i.len());
        }

        // active は max_buffer_len(100_000) を大きく超えて増え続けないこと
        assert!(
            peak_active_len <= 102_048,
            "active buffer grew too large: peak_active_len={}",
            peak_active_len
        );

        // physical も compaction により上限内に収まること
        assert!(
            peak_physical_len <= 120_000,
            "physical buffer grew too large: peak_physical_len={}",
            peak_physical_len
        );
    }

    #[test]
    fn test_agc_fast_attack_response() {
        let dsp_config = crate::dsss::params::dsp_config_48k();
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
        let dsp_config = crate::dsss::params::dsp_config_48k();
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
        let dsp_config = crate::dsss::params::dsp_config_48k();
        let mut decoder = Decoder::new(16, FIXED_K, dsp_config);

        let silence = vec![0.0f32; 2048];
        let _ = decoder.process_samples(&silence);
        decoder.reset();
        let progress = decoder.process_samples(&[]);

        assert!(!progress.complete);
        assert_eq!(progress.received_packets, 0);
        assert!(decoder.recovered_data().is_none());
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[wasm_bindgen_test]
    fn test_mix_real_to_iq_simd_matches_scalar_reference() {
        let config = crate::dsss::params::dsp_config_48k();
        let mut simd_decoder = Decoder::new(32, FIXED_K, config.clone());
        let mut scalar_decoder = Decoder::new(32, FIXED_K, config);

        let samples = vec![
            0.03, -0.11, 0.17, -0.23, 0.29, -0.31, 0.37, -0.41, 0.47, -0.53, 0.59, -0.61, 0.67,
        ];

        let mut simd_i = Vec::new();
        let mut simd_q = Vec::new();
        simd_decoder.mix_real_to_iq(&samples, &mut simd_i, &mut simd_q);

        let mut expected_i = Vec::new();
        let mut expected_q = Vec::new();
        scalar_mix_reference(
            &mut scalar_decoder,
            &samples,
            &mut expected_i,
            &mut expected_q,
        );

        assert_vec_close(&simd_i, &expected_i, 1e-6);
        assert_vec_close(&simd_q, &expected_q, 1e-6);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[wasm_bindgen_test]
    fn test_mix_real_to_iq_simd_matches_scalar_across_calls() {
        let config = crate::dsss::params::dsp_config_48k();
        let mut simd_decoder = Decoder::new(32, FIXED_K, config.clone());
        let mut scalar_decoder = Decoder::new(32, FIXED_K, config);

        let first = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let second = vec![0.12, 0.24, -0.36, 0.48, -0.6, 0.72, -0.84, 0.96, -0.08];

        let mut simd_i = Vec::new();
        let mut simd_q = Vec::new();
        let mut expected_i = Vec::new();
        let mut expected_q = Vec::new();

        simd_decoder.mix_real_to_iq(&first, &mut simd_i, &mut simd_q);
        scalar_mix_reference(
            &mut scalar_decoder,
            &first,
            &mut expected_i,
            &mut expected_q,
        );
        assert_vec_close(&simd_i, &expected_i, 1e-6);
        assert_vec_close(&simd_q, &expected_q, 1e-6);

        simd_decoder.mix_real_to_iq(&second, &mut simd_i, &mut simd_q);
        scalar_mix_reference(
            &mut scalar_decoder,
            &second,
            &mut expected_i,
            &mut expected_q,
        );
        assert_vec_close(&simd_i, &expected_i, 1e-6);
        assert_vec_close(&simd_q, &expected_q, 1e-6);
    }

    fn build_test_signal(data: &[u8], k: usize, frames: usize, gap_samples: usize) -> Vec<f32> {
        let mut enc_cfg = EncoderConfig::new(crate::dsss::params::dsp_config_48k());
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

    fn run_decode_window_after_midstream_reset(
        signal: &[f32],
        reset_offset: usize,
        window_samples_after_reset: usize,
        chunk_size: usize,
    ) -> (usize, usize, bool, usize) {
        let data_len = 160usize;
        let k = data_len.div_ceil(crate::params::PAYLOAD_SIZE);
        let mut decoder = Decoder::new(data_len, k, crate::dsss::params::dsp_config_48k());
        decoder.config.packets_per_burst = 1;

        for chunk in signal[..reset_offset.min(signal.len())].chunks(chunk_size) {
            let _ = decoder.process_samples(chunk);
        }
        decoder.reset();

        let end = (reset_offset + window_samples_after_reset).min(signal.len());
        let mut max_received = 0usize;
        let mut max_crc = 0usize;
        let mut max_synced = 0usize;
        let mut complete = false;
        for chunk in signal[reset_offset..end].chunks(chunk_size) {
            let p = decoder.process_samples(chunk);
            max_received = max_received.max(p.received_packets);
            max_crc = max_crc.max(p.crc_error_packets);
            max_synced = max_synced.max(p.synced_frames);
            if p.complete {
                complete = true;
                break;
            }
        }
        (max_received, max_crc, complete, max_synced)
    }

    fn build_runtime_like_continuous_signal(data: &[u8], k: usize, frames: usize) -> Vec<f32> {
        let mut enc_cfg = EncoderConfig::new(crate::dsss::params::dsp_config_48k());
        enc_cfg.fountain_k = k;
        enc_cfg.packets_per_sync_burst = 1;
        let mut encoder = Encoder::new(enc_cfg);
        let params = FountainParams::new(k, PAYLOAD_SIZE);
        let mut fountain_encoder = FountainEncoder::new(data, params);

        let mut signal = Vec::new();
        for _ in 0..frames {
            let packet = fountain_encoder.next_packet();
            signal.extend(encoder.encode_burst(&[packet]));
        }
        // 実運用では送信継続中だが、テスト終端でデコード窓を確保するため末尾に余白を足す
        signal.extend(vec![0.0f32; 8192]);
        signal
    }

    fn append_processed_samples(decoder: &mut Decoder, samples: &[f32]) {
        let mut i_mixed = Vec::with_capacity(samples.len());
        let mut q_mixed = Vec::with_capacity(samples.len());
        decoder.mix_real_to_iq(samples, &mut i_mixed, &mut q_mixed);

        let mut i_resampled = Vec::new();
        let mut q_resampled = Vec::new();
        decoder.resampler_i.process(&i_mixed, &mut i_resampled);
        decoder.resampler_q.process(&q_mixed, &mut q_resampled);

        let i_filtered = decoder.rrc_filter_i.process_block(&i_resampled);
        let q_filtered = decoder.rrc_filter_q.process_block(&q_resampled);
        decoder.sample_buffer_i.extend_from_slice(&i_filtered);
        decoder.sample_buffer_q.extend_from_slice(&q_filtered);
    }

    use crate::common::channel::apply_clock_drift_ppm;

    #[test]
    fn test_decoder_tracking_tolerates_clock_drift_ppm() {
        let data = b"tracking timing drift payload";
        let k = data.len().div_ceil(crate::params::PAYLOAD_SIZE);
        let signal = build_test_signal(data, k, 6, 64);
        // 実オーディオI/Fで起きるオーダー(100ppm前後)のクロックずれを模擬。
        let drifted = apply_clock_drift_ppm(&signal, 120.0);
        let recovered = decode_signal(data, k, crate::dsss::params::dsp_config_48k(), &drifted)
            .expect("decoder should recover under realistic clock drift");
        assert_eq!(&recovered[..data.len()], data);
    }

    #[test]
    fn test_decoder_tracking_tolerates_carrier_offset() {
        let data = b"tracking carrier offset payload";
        let k = data.len().div_ceil(crate::params::PAYLOAD_SIZE);
        let signal = build_test_signal(data, k, 3, 64);

        let mut rx_cfg = crate::dsss::params::dsp_config_48k();
        // 受信LOずれを模擬: 音響リンクで現実的な小さなCFO。
        rx_cfg.carrier_freq += 10.0;

        let recovered = decode_signal(data, k, rx_cfg, &signal)
            .expect("decoder should recover under mild carrier offset");
        assert_eq!(&recovered[..data.len()], data);
    }

    #[test]
    fn test_decoder_direct_first_frame_payload_decode_has_no_crc_errors() {
        let data = vec![0xAAu8; 160];
        let k = data.len().div_ceil(crate::params::PAYLOAD_SIZE);
        let signal = build_test_signal(&data, k, 1, 64);
        let mut decoder = Decoder::new(data.len(), k, crate::dsss::params::dsp_config_48k());
        decoder.config.packets_per_burst = 1;
        append_processed_samples(&mut decoder, &signal);

        let spc = decoder.config.proc_samples_per_chip();
        let sf = decoder.config.spread_factor();
        let sync_bits_len = decoder.config.sync_word_bits;
        let fec_bits_len = decoder.interleaver.rows() * decoder.interleaver.cols();
        let burst_data_bits_len = fec_bits_len * decoder.config.packets_per_burst.max(1);
        let p_bits_len = PACKET_BYTES * 8;

        let (sync_opt, _) =
            decoder
                .sync_detector
                .detect(&decoder.sample_buffer_i, &decoder.sample_buffer_q, 0);
        let sync = sync_opt.expect("sync should be detected on ideal first frame");
        let payload_start = sync.peak_sample_idx + sync_bits_len * sf * spc;

        let initial_ref = Complex32::new(sync.peak_iq.0, sync.peak_iq.1);
        let initial_ref_norm = initial_ref.norm().max(1e-6);
        let tracking = TrackingState {
            phase_ref: initial_ref / initial_ref_norm,
            prev_symbol: initial_ref / initial_ref_norm,
            phase_rate: 0.0,
            timing_offset: 0.0,
            timing_rate: 0.0,
            noise_var: 0.2,
        };

        let mut mseq = crate::common::msequence::MSequence::new(decoder.config.mseq_order);
        let pn: Vec<f32> = mseq.generate(sf).into_iter().map(|v| v as f32).collect();

        let attempt = decoder
            .decode_payload_with_timing_retries(
                payload_start,
                burst_data_bits_len,
                tracking,
                &pn,
                fec_bits_len,
                p_bits_len,
            )
            .expect("payload decode attempt should be produced");

        assert_eq!(
            attempt.crc_errors, 0,
            "ideal first frame should have no crc errors"
        );
        assert_eq!(
            attempt.parse_errors, 0,
            "ideal first frame should have no parse errors"
        );
        assert_eq!(
            attempt.packet_count, 1,
            "ideal first frame should decode one packet"
        );
    }

    #[test]
    fn test_decoder_direct_isolated_frames_have_no_crc_errors() {
        let data = vec![0xAAu8; 160];
        let k = data.len().div_ceil(crate::params::PAYLOAD_SIZE);
        let config = crate::dsss::params::dsp_config_48k();
        let mut enc_cfg = EncoderConfig::new(config.clone());
        enc_cfg.fountain_k = k;
        enc_cfg.packets_per_sync_burst = 1;
        let params = FountainParams::new(k, PAYLOAD_SIZE);
        let mut fountain_encoder = FountainEncoder::new(&data, params);

        for frame_idx in 0..20 {
            let packet = fountain_encoder.next_packet();
            let mut encoder = Encoder::new(enc_cfg.clone());
            let mut signal = encoder.encode_burst(&[packet]);
            signal.extend(encoder.modulate_silence(64));
            signal.extend(encoder.flush());
            signal.extend(vec![0.0f32; 1024]);

            let mut decoder = Decoder::new(data.len(), k, config.clone());
            decoder.config.packets_per_burst = 1;
            append_processed_samples(&mut decoder, &signal);

            let spc = decoder.config.proc_samples_per_chip();
            let sf = decoder.config.spread_factor();
            let sync_bits_len = decoder.config.sync_word_bits;
            let fec_bits_len = decoder.interleaver.rows() * decoder.interleaver.cols();
            let burst_data_bits_len = fec_bits_len * decoder.config.packets_per_burst.max(1);
            let p_bits_len = PACKET_BYTES * 8;

            let (sync_opt, _) =
                decoder
                    .sync_detector
                    .detect(&decoder.sample_buffer_i, &decoder.sample_buffer_q, 0);
            let sync = sync_opt.expect("sync should be detected on isolated ideal frame");
            let payload_start = sync.peak_sample_idx + sync_bits_len * sf * spc;

            let initial_ref = Complex32::new(sync.peak_iq.0, sync.peak_iq.1);
            let initial_ref_norm = initial_ref.norm().max(1e-6);
            let mut mseq = crate::common::msequence::MSequence::new(decoder.config.mseq_order);
            let pn: Vec<f32> = mseq.generate(sf).into_iter().map(|v| v as f32).collect();

            let tracking = TrackingState {
                phase_ref: initial_ref / initial_ref_norm,
                prev_symbol: initial_ref / initial_ref_norm,
                phase_rate: 0.0,
                timing_offset: 0.0,
                timing_rate: 0.0,
                noise_var: 0.2,
            };
            let attempt = decoder
                .decode_payload_with_timing_retries(
                    payload_start,
                    burst_data_bits_len,
                    tracking,
                    &pn,
                    fec_bits_len,
                    p_bits_len,
                )
                .unwrap_or_else(|| {
                    panic!("frame {} should produce payload decode attempt", frame_idx)
                });

            assert_eq!(
                attempt.crc_errors, 0,
                "isolated ideal frame {} should have no crc errors",
                frame_idx
            );
            assert_eq!(
                attempt.parse_errors, 0,
                "isolated ideal frame {} should have no parse errors",
                frame_idx
            );
            assert_eq!(
                attempt.packet_count, 1,
                "isolated ideal frame {} should decode one packet",
                frame_idx
            );
        }
    }

    #[test]
    fn test_parse_payload_packets_counts_crc_errors() {
        let dsp_config = crate::dsss::params::dsp_config_48k();
        let mut decoder = Decoder::new(32, FIXED_K, dsp_config);
        let fec_bits_len = decoder.interleaver.rows() * decoder.interleaver.cols();
        let p_bits_len = PACKET_BYTES * 8;

        let valid_packet = Packet::new(7, 3, &[0x5a; crate::params::PAYLOAD_SIZE]);
        let valid_bits = fec::bytes_to_bits(&valid_packet.serialize());

        let mut crc_mismatch_bits = valid_bits.clone();
        crc_mismatch_bits[0] ^= 1;

        let mut payload_llrs = encode_packet_bits_to_interleaved_llrs(&valid_bits, 8.0);
        payload_llrs.extend(encode_packet_bits_to_interleaved_llrs(
            &crc_mismatch_bits,
            8.0,
        ));

        let mut packets = Vec::new();
        let (crc_errors, parse_errors) = decoder.parse_payload_packets_into(
            &payload_llrs,
            fec_bits_len,
            p_bits_len,
            &mut packets,
        );

        assert_eq!(packets, vec![valid_packet]);
        assert_eq!(crc_errors, 1);
        assert_eq!(parse_errors, 0);
    }

    #[test]
    fn test_decoder_llr_callback_emits_one_codeword_per_packet_in_burst() {
        let data = vec![0x55u8; 64];
        let k = data.len().div_ceil(crate::params::PAYLOAD_SIZE);
        let packets_per_burst = 3;

        let mut config = crate::dsss::params::dsp_config_48k();
        config.packets_per_burst = packets_per_burst;

        let mut enc_cfg = EncoderConfig::new(config.clone());
        enc_cfg.fountain_k = k;
        enc_cfg.packets_per_sync_burst = packets_per_burst;
        let mut encoder = Encoder::new(enc_cfg);

        let params = FountainParams::new(k, PAYLOAD_SIZE);
        let mut fountain_encoder = FountainEncoder::new(&data, params);
        let mut packets = Vec::with_capacity(packets_per_burst);
        for _ in 0..packets_per_burst {
            packets.push(fountain_encoder.next_packet());
        }

        let mut signal = encoder.encode_burst(&packets);
        signal.extend(encoder.flush());
        signal.extend(vec![0.0f32; 1024]);

        let callback_lengths = Arc::new(Mutex::new(Vec::new()));
        let callback_lengths_capture = Arc::clone(&callback_lengths);
        let mut decoder = Decoder::new(data.len(), k, config);
        decoder.config.packets_per_burst = packets_per_burst;
        decoder.llr_callback = Some(Box::new(move |llrs: &[f32]| {
            callback_lengths_capture.lock().unwrap().push(llrs.len());
        }));

        for chunk in signal.chunks(2048) {
            decoder.process_samples(chunk);
        }
        let final_progress = decoder.process_samples(&[]);

        assert_eq!(final_progress.synced_frames, 1);
        assert_eq!(final_progress.received_packets, packets_per_burst);
        assert_eq!(final_progress.crc_error_packets, 0);

        let observed = callback_lengths.lock().unwrap().clone();
        let fec_bits_len = decoder.interleaver.rows() * decoder.interleaver.cols();
        assert_eq!(observed.len(), packets_per_burst);
        assert!(
            observed.iter().all(|&len| len == fec_bits_len),
            "callback must receive one full FEC codeword per packet: {:?}",
            observed
        );
    }

    #[test]
    fn test_sync_word_tracking_tolerates_offset_and_cfo() {
        let data = b"sync word header";
        let k = data.len().div_ceil(crate::params::PAYLOAD_SIZE);
        let mut signal = vec![0.0f32; 137];
        signal.extend(build_test_signal(data, k, 3, 64));

        let mut rx_cfg = crate::dsss::params::dsp_config_48k();
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
        let mut config = crate::dsss::params::dsp_config_48k();
        // このテストは以前のデフォルト値での挙動を前提としているため、明示的に指定する
        config.chip_rate = 12000.0;
        config.sync_word_bits = 32;
        config.preamble_repeat = 4;

        let mut enc_cfg = EncoderConfig::new(config.clone());
        enc_cfg.fountain_k = 10;
        enc_cfg.packets_per_sync_burst = 2;

        let mut encoder = Encoder::new(enc_cfg);
        let mut decoder = Decoder::new(data.len(), 10, config);
        decoder.config.packets_per_burst = 2;

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

    #[test]
    fn test_decoder_does_not_overcount_syncs_continuous_frames() {
        // dsss_e2e_eval と同じ定義で synced_frame_ratio を評価する。
        let data = vec![0xAAu8; 160];
        let k = data.len().div_ceil(crate::params::PAYLOAD_SIZE);
        let num_frames = 20;
        let signal = build_test_signal(&data, k, num_frames, 64);

        let mut decoder = Decoder::new(data.len(), k, crate::dsss::params::dsp_config_48k());
        decoder.config.packets_per_burst = 1;

        let mut total_synced_frames = 0usize;
        let mut total_accepted_packets = 0usize;
        let mut total_crc_error_packets = 0usize;
        for chunk in signal.chunks(2048) {
            let progress = decoder.process_samples(chunk);
            if progress.complete {
                total_synced_frames += progress.synced_frames;
                total_accepted_packets += progress.received_packets;
                total_crc_error_packets += progress.crc_error_packets;
                decoder.reset_fountain_decoder();
            }
        }
        let final_progress = decoder.process_samples(&[]);
        total_synced_frames += final_progress.synced_frames;
        total_accepted_packets += final_progress.received_packets;
        total_crc_error_packets += final_progress.crc_error_packets;

        let synced_frame_ratio = total_synced_frames as f32 / num_frames as f32;
        assert_eq!(total_accepted_packets, num_frames);
        assert_eq!(total_crc_error_packets, 0);
        assert!(
            (synced_frame_ratio - 1.0).abs() < 1e-6,
            "synced_frame_ratio mismatch: {}",
            synced_frame_ratio
        );
    }

    #[test]
    fn test_decoder_ideal_continuous_frames_do_not_accumulate_crc_errors() {
        let data = vec![0xAAu8; 160];
        let k = data.len().div_ceil(crate::params::PAYLOAD_SIZE);
        let num_frames = 20;
        let signal = build_test_signal(&data, k, num_frames, 64);

        let mut decoder = Decoder::new(data.len(), k, crate::dsss::params::dsp_config_48k());
        decoder.config.packets_per_burst = 1;

        let mut total_accepted_packets = 0usize;
        let mut total_crc_error_packets = 0usize;
        let mut total_parse_error_packets = 0usize;
        for chunk in signal.chunks(2048) {
            let progress = decoder.process_samples(chunk);
            if progress.complete {
                total_accepted_packets += progress.received_packets;
                total_crc_error_packets += progress.crc_error_packets;
                total_parse_error_packets += progress.parse_error_packets;
                decoder.reset_fountain_decoder();
            }
        }
        let final_progress = decoder.process_samples(&[]);
        total_accepted_packets += final_progress.received_packets;
        total_crc_error_packets += final_progress.crc_error_packets;
        total_parse_error_packets += final_progress.parse_error_packets;

        assert_eq!(
            total_accepted_packets, num_frames,
            "ideal continuous frames should accept every packet"
        );
        assert_eq!(
            total_crc_error_packets, 0,
            "ideal continuous frames should not accumulate crc errors"
        );
        assert_eq!(
            total_parse_error_packets, 0,
            "ideal continuous frames should not accumulate parse errors"
        );
    }

    #[test]
    fn test_decoder_counts_all_packets_in_completed_burst() {
        let data = vec![0xAAu8; 64];
        let k = data.len().div_ceil(crate::params::PAYLOAD_SIZE);
        let packets_per_burst = 6;

        let mut config = crate::dsss::params::dsp_config_48k();
        config.packets_per_burst = packets_per_burst;

        let mut enc_cfg = EncoderConfig::new(config.clone());
        enc_cfg.fountain_k = k;
        enc_cfg.packets_per_sync_burst = packets_per_burst;
        let mut encoder = Encoder::new(enc_cfg);
        let mut decoder = Decoder::new(data.len(), k, config);
        decoder.config.packets_per_burst = packets_per_burst;

        let params = FountainParams::new(k, PAYLOAD_SIZE);
        let mut fountain_encoder = FountainEncoder::new(&data, params);
        let mut packets = Vec::with_capacity(packets_per_burst);
        for _ in 0..packets_per_burst {
            packets.push(fountain_encoder.next_packet());
        }

        let mut signal = encoder.encode_burst(&packets);
        signal.extend(encoder.flush());
        signal.extend(vec![0.0f32; 1024]);

        for chunk in signal.chunks(2048) {
            decoder.process_samples(chunk);
        }
        let final_progress = decoder.process_samples(&[]);

        assert!(final_progress.complete);
        assert_eq!(final_progress.synced_frames, 1);
        assert_eq!(final_progress.crc_error_packets, 0);
        assert_eq!(final_progress.parse_error_packets, 0);
        assert_eq!(
            final_progress.received_packets, packets_per_burst,
            "completed burst should count every packet in the frame"
        );
    }

    #[test]
    #[ignore = "manual reproduction: mid-stream sweep is expensive"]
    fn test_repro_reset_midstream_can_fall_into_crc_only_window() {
        // 仮説2の再現 (loopback/ideal channel):
        // 連続送信中に reset した後、受理0のままCRCのみ増える窓があるか探索する。
        let data = vec![0xA5u8; 160];
        let k = data.len().div_ceil(crate::params::PAYLOAD_SIZE);
        let signal = build_runtime_like_continuous_signal(&data, k, 120);

        let mut found_crc_only_offset = None;
        let window_after_reset = 96_000;
        let search_end = signal.len().saturating_sub(window_after_reset);
        for offset in (4096..search_end).step_by(4096).take(80) {
            let (max_received, max_crc, complete, _max_synced) =
                run_decode_window_after_midstream_reset(&signal, offset, window_after_reset, 4096);
            if !complete && max_received == 0 && max_crc >= 4 {
                found_crc_only_offset = Some((offset, max_crc));
                break;
            }
        }

        assert!(
            found_crc_only_offset.is_some(),
            "mid-stream reset reproduction was not observed in current sweep"
        );
    }

    #[test]
    fn test_reset_midstream_at_chunk_boundary_should_recover_loopback() {
        // RED:
        // loopback(理想チャネル)では、chunk境界(4096)で reset しても復帰できるべき。
        let data = vec![0xA5u8; 160];
        let k = data.len().div_ceil(crate::params::PAYLOAD_SIZE);
        let signal = build_runtime_like_continuous_signal(&data, k, 120);

        let (max_received, max_crc, complete, max_synced) =
            run_decode_window_after_midstream_reset(&signal, 4096, 96_000, 4096);

        assert!(
            max_received > 0,
            "decoder should recover packets even when reset occurs at chunk boundary: received={}, crc={}, complete={}, synced={}",
            max_received,
            max_crc,
            complete,
            max_synced
        );
        assert!(
            max_crc < max_received.saturating_mul(4).saturating_add(8),
            "decoder should avoid crc-dominant lock after reset: received={}, crc={}, complete={}, synced={}",
            max_received,
            max_crc,
            complete,
            max_synced
        );
    }

    #[test]
    fn test_reset_restores_agc_state_to_default() {
        // RED:
        // reset() 後は AGC peak が new 時の初期値へ戻るべき。
        let mut decoder = Decoder::new(32, FIXED_K, crate::dsss::params::dsp_config_48k());

        // AGC状態を偏らせる
        let hot_signal = vec![4.0f32; 16_384];
        let _ = decoder.process_samples(&hot_signal);
        let before_fast = decoder.agc_peak_fast;
        let before_slow = decoder.agc_peak_slow;
        assert!(before_fast > 1.0);
        assert!(before_slow > 1.0);

        decoder.reset();

        // 期待仕様: reset後に初期値へ復帰する
        assert!(
            (decoder.agc_peak_fast - 0.5).abs() < 1e-6,
            "agc_peak_fast must be reset to default: before={}, after={}",
            before_fast,
            decoder.agc_peak_fast
        );
        assert!(
            (decoder.agc_peak_slow - 0.5).abs() < 1e-6,
            "agc_peak_slow must be reset to default: before={}, after={}",
            before_slow,
            decoder.agc_peak_slow
        );
    }
}
