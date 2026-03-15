//! パケット復調モジュール
//!
//! 等化済みシンボル列から1パケット分の LLR を生成し、
//! デインターリーブ・デスクランブル・soft-list 復号までを担当する。

use super::decoder_stats::DecoderStats;
use super::tracking::{
    self, TrackingState, PHASE_ERR_ABS_THRESH_0P5_RAD, PHASE_ERR_ABS_THRESH_1P0_RAD,
    TRACKING_PHASE_DQPSK_CONF_ON_MIN, TRACKING_PHASE_ERR_GATE_DQPSK_CONF_HIGH,
    TRACKING_PHASE_ERR_GATE_RAD, TRACKING_PHASE_FREQ_GAIN_OFF, TRACKING_PHASE_OFF_ERR_CLAMP,
    TRACKING_PHASE_PROP_GAIN_OFF, TRACKING_PHASE_RATE_HOLD_DECAY, TRACKING_PHASE_STEP_CLAMP,
};
use crate::coding::fec;
use crate::coding::interleaver::BlockInterleaver;
use crate::coding::scrambler::Scrambler;
use crate::frame::packet::{Packet, PACKET_BYTES};
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
    pub llr_abs_scratch: Vec<f32>,
    pub fec_candidate_bits_buffer: Vec<u8>,
    pub decoded_bytes_buffer: Vec<u8>,
    pub fec_workspace: fec::FecDecodeWorkspace,
}

impl PacketDecodeBuffers {
    pub fn new() -> Self {
        let cap = interleaver_config::interleaved_bits();
        let mut fec_workspace = fec::FecDecodeWorkspace::new();
        fec_workspace.preallocate_for_llr_len(cap, 1);
        Self {
            packet_llrs_buffer: Vec::with_capacity(cap),
            deinterleave_buffer: Vec::with_capacity(cap),
            erasure_llr_buffer: Vec::with_capacity(cap),
            llr_abs_scratch: Vec::with_capacity(cap),
            fec_candidate_bits_buffer: Vec::with_capacity(PACKET_BYTES * 8),
            decoded_bytes_buffer: Vec::with_capacity(PACKET_BYTES),
            fec_workspace,
        }
    }

    pub fn clear(&mut self) {
        self.packet_llrs_buffer.clear();
        self.deinterleave_buffer.clear();
        self.erasure_llr_buffer.clear();
        self.llr_abs_scratch.clear();
        self.fec_candidate_bits_buffer.clear();
        self.decoded_bytes_buffer.clear();
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

#[derive(Clone, Copy)]
struct SymbolObservation {
    on_corrs: [Complex32; 16],
    phase_ref: Complex32,
    prev_phase: Complex32,
    max_energy: f32,
    second_energy: f32,
    on_rot_best: Complex32,
    diff_best: Complex32,
    prev_phase_kappa: f32,
    phase_err: f32,
    phase_fit_weight: f32,
}

// DQPSK LLR は (Walsh仮説 h, 位相仮説 s) の同時尤度を
// 全Walsh仮説で周辺化して計算する。
// walsh_conf は位相追跡のゲート制御にも使うため継続して算出する。
const DQPSK_WALSH_HYPOTHESES: usize = 16;
// walsh_conf = (E1-E2)/E1 の観測閾値（統計/テストの判定用）。
// 範囲: [0, 1]。大きくすると「曖昧」と判定される領域が広がる。
#[cfg(test)]
const DQPSK_WALSH_TOPK_ENABLE_CONF_THRESH: f32 = 0.30;
// DQPSK尤度計算で使う雑音分散の下限（数値安定用）。
// 0に近づけるほどLLRが過大化しやすく、上げすぎると過小信頼になりやすい。
const DQPSK_NOISE_VAR_FLOOR: f32 = 1e-6;
// 振幅正規化後の位相観測分散の下限。
// 小さすぎると高SNRでLLRが暴走しやすく、大きすぎると全域で過小信頼になりやすい。
// 妥当目安: 1e-4..1e-2
const DQPSK_PHASE_OBS_VAR_FLOOR: f32 = 1e-3;
// 位相尤度のモデル誤差分散（正規化領域）。
// 観測分散に加算して sigma_eff^2 = sigma_obs^2 + sigma_model^2 とし、
// モデルずれ・残留位相誤差でのLLR過信を抑える。
// 小さくすると鋭いLLRになり、大きくすると保守化される。
// 妥当目安: 0.5..2.0
const DQPSK_PHASE_MODEL_VAR: f32 = 1.0;
// prev_phase の位相不確かさ分散推定（EWMA）に使う係数。
// 妥当範囲: 0.0..1.0（1.0は更新停止に近いので非推奨）。
// 大きいほど長期平均寄りで安定だが追従は遅く、小さいほど追従は速いが雑音に敏感。
const DQPSK_PREV_PHASE_VAR_EWMA_ALPHA: f32 = 0.95;
// 位相分散の上限 [rad]。外れ値で分散推定が暴走しないためのクリップ。
// 妥当目安: 0.5..PI。小さすぎると常時保守化、大きすぎると外れ値耐性が下がる。
const DQPSK_PREV_PHASE_SIGMA_MAX_RAD: f32 = 1.2;
// Walsh事後平均で位相追跡に使う複素振幅の最小ノルム。
// 小さすぎると雑音起因のほぼゼロベクトルで位相更新しやすく、
// 大きすぎるとフォールバックが増えて best 仮説依存が強くなる。
// 妥当目安: 1e-6..1e-3
const DQPSK_TRACKING_POSTERIOR_NORM_MIN: f32 = 1e-5;
// 期待シンボルの振幅がこの値未満なら、方向が不安定とみなして hard 判定へフォールバックする。
const DQPSK_PHASE_SOFT_SYMBOL_NORM_MIN: f32 = 1e-3;
// パケット内の残留位相傾き（rad/symbol）を非因果推定で補償する際の上限。
// これを超える推定は外れ値起因とみなし、補償を無効化する。
const DQPSK_NONCAUSAL_PHASE_SLOPE_MAX_RAD_PER_SYMBOL: f32 = 0.35;
// 非因果位相フィットに使う最小有効シンボル数。
const DQPSK_NONCAUSAL_PHASE_MIN_VALID_SYMBOLS: usize = 8;
// 非因果位相フィットに使う重みの下限。
const DQPSK_NONCAUSAL_PHASE_MIN_WEIGHT: f32 = 0.2;

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

#[inline]
fn log_add_exp(a: f32, b: f32) -> f32 {
    if !a.is_finite() {
        return b;
    }
    if !b.is_finite() {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

#[inline]
fn dqpsk_noise_var_from_energy(energy_sum: f32, energy_h: f32) -> f32 {
    ((energy_sum - energy_h).max(0.0) / 15.0).max(DQPSK_NOISE_VAR_FLOOR)
}

#[inline]
fn dqpsk_phase_log_metrics(
    diff: Complex32,
    amp: f32,
    noise_var: f32,
    prev_phase_kappa: f32,
) -> [f32; 4] {
    // 位相不確かさを含む観測モデル（周辺化）:
    //   y = diff / amp_ref ≈ s * exp(jδ) + n, n ~ CN(0, sigma2), δ ~ von Mises(kappa, mean=0)
    //   p(y|s) ∝ ∫ exp(kappa*cosδ + (2/sigma2)*Re{y*conj(s)*exp(-jδ)}) dδ
    //          ∝ I0(|kappa + (2/sigma2) * conj(y) * s|)
    // よって log-metric は log I0 の形になる。
    let amp_ref = amp.max(1e-6);
    let y = diff / amp_ref;
    let sigma2_obs = (noise_var / (amp_ref * amp_ref)).max(DQPSK_PHASE_OBS_VAR_FLOOR);
    let sigma2_eff = sigma2_obs + DQPSK_PHASE_MODEL_VAR;
    let eta = 2.0 / sigma2_eff;
    let prior = Complex32::new(prev_phase_kappa.clamp(0.0, 1.0), 0.0);
    let z0 = prior + y.conj() * Complex32::new(1.0, 0.0) * eta; // s0: 00
    let z1 = prior + y.conj() * Complex32::new(0.0, 1.0) * eta; // s1: 01
    let z2 = prior + y.conj() * Complex32::new(-1.0, 0.0) * eta; // s2: 11
    let z3 = prior + y.conj() * Complex32::new(0.0, -1.0) * eta; // s3: 10
    [
        log_i0_approx(z0.norm()),
        log_i0_approx(z1.norm()),
        log_i0_approx(z2.norm()),
        log_i0_approx(z3.norm()),
    ]
}

#[inline]
fn dqpsk_prev_phase_kappa_from_sigma2(sigma2: f32) -> f32 {
    let sigma2_clamped = sigma2.clamp(0.0, DQPSK_PREV_PHASE_SIGMA_MAX_RAD.powi(2));
    (-0.5 * sigma2_clamped).exp().clamp(0.0, 1.0)
}

#[inline]
fn log_i0_approx(x: f32) -> f32 {
    // 変形ベッセル関数 I0 の近似係数は Cephes math library (i0f) 由来。
    // 参考: S. L. Moshier, Cephes Math Library（Numerical Recipes掲載式と同系）。
    // 区分点 3.75 も同近似の既知パラメータで、小x/大xで安定な多項式を使い分ける。
    // ここでは I0(x) を直接計算せず log-domain に変換してオーバーフローを避ける。
    let ax = x.abs();
    if ax < 3.75 {
        let y = (ax / 3.75) * (ax / 3.75);
        let i0 = 1.0
            + y * (3.515_623
                + y * (3.089_942
                    + y * (1.206_749 + y * (0.265_973 + y * (0.036_076_8 + y * 0.004_581_3)))));
        i0.ln()
    } else {
        let y = 3.75 / ax;
        let poly = 0.398_942_3
            + y * (0.013_285_92
                + y * (0.002_253_19
                    + y * (-0.001_575_65
                        + y * (0.009_162_81
                            + y * (-0.020_577_06
                                + y * (0.026_355_37 + y * (-0.016_476_33 + y * 0.003_923_77)))))));
        ax + poly.ln() - 0.5 * ax.ln()
    }
}

#[inline]
fn walsh_posterior_weighted_on_rot(
    on_corrs: &[Complex32; 16],
    phase_ref: Complex32,
    energies: &[f32; 16],
    max_energy: f32,
    on_rot_best: Complex32,
) -> Complex32 {
    let energy_sum = energies.iter().sum::<f32>();
    let walsh_temp = dqpsk_noise_var_from_energy(energy_sum, max_energy);

    let mut max_log = f32::NEG_INFINITY;
    let mut log_weights = [0.0f32; 16];
    for idx in 0..16 {
        let lp = (energies[idx] - max_energy) / walsh_temp;
        log_weights[idx] = lp;
        if lp > max_log {
            max_log = lp;
        }
    }

    let mut weighted = Complex32::new(0.0, 0.0);
    let mut wsum = 0.0f32;
    for idx in 0..16 {
        let w = (log_weights[idx] - max_log).exp();
        wsum += w;
        weighted += on_corrs[idx] * phase_ref.conj() * w;
    }
    if wsum <= 0.0 {
        return on_rot_best;
    }
    let posterior = weighted / wsum;
    if posterior.norm() >= DQPSK_TRACKING_POSTERIOR_NORM_MIN {
        posterior
    } else {
        on_rot_best
    }
}

fn dqpsk_llr_from_walsh_hypotheses(
    on_corrs: &[Complex32; 16],
    phase_ref: Complex32,
    prev_phase: Complex32,
    prev_phase_kappa: f32,
    max_energy: f32,
    second_energy: f32,
    on_rot_best: Complex32,
    diff_best: Complex32,
    residual_rot: Complex32,
) -> ([f32; 2], f32, usize, [f32; 16]) {
    let energies: [f32; 16] = on_corrs.map(|c| c.norm_sqr());
    let walsh_conf = ((max_energy - second_energy).max(0.0)) / max_energy.max(1e-6);
    let energy_sum = energies.iter().sum::<f32>();
    let walsh_noise_floor_e = dqpsk_noise_var_from_energy(energy_sum, max_energy);

    // p(s|y) ∝ Σ_h p(y|h) p(y|s,h) を log-domain で周辺化する。
    let mut sym_logs = [f32::NEG_INFINITY; 4];
    for idx in 0..DQPSK_WALSH_HYPOTHESES {
        let energy_h = energies[idx];
        let on_rot_h = on_corrs[idx] * phase_ref.conj();
        let diff_h = on_rot_h * prev_phase.conj() * residual_rot;
        let noise_var_h = dqpsk_noise_var_from_energy(energy_sum, energy_h);
        let phase_logs =
            dqpsk_phase_log_metrics(diff_h, on_rot_h.norm(), noise_var_h, prev_phase_kappa);
        // 相対ログ重みを使って数値桁落ちを避ける（定数項はLLR差分で相殺される）。
        let log_p_h = (energy_h - max_energy) / walsh_noise_floor_e;
        for s in 0..4 {
            sym_logs[s] = log_add_exp(sym_logs[s], log_p_h + phase_logs[s]);
        }
    }

    let dqpsk_llr = if sym_logs.iter().all(|v| v.is_finite()) {
        let bit0_0 = log_add_exp(sym_logs[0], sym_logs[1]);
        let bit0_1 = log_add_exp(sym_logs[2], sym_logs[3]);
        let bit1_0 = log_add_exp(sym_logs[0], sym_logs[3]);
        let bit1_1 = log_add_exp(sym_logs[1], sym_logs[2]);
        [bit0_0 - bit0_1, bit1_0 - bit1_1]
    } else {
        let noise_var_best = dqpsk_noise_var_from_energy(energy_sum, max_energy);
        let phase_logs = dqpsk_phase_log_metrics(
            diff_best * residual_rot,
            on_rot_best.norm(),
            noise_var_best,
            prev_phase_kappa,
        );
        [
            log_add_exp(phase_logs[0], phase_logs[1]) - log_add_exp(phase_logs[2], phase_logs[3]),
            log_add_exp(phase_logs[0], phase_logs[3]) - log_add_exp(phase_logs[1], phase_logs[2]),
        ]
    };

    (dqpsk_llr, walsh_conf, DQPSK_WALSH_HYPOTHESES, energies)
}

#[inline]
fn wrap_to_pi(mut x: f32) -> f32 {
    while x > std::f32::consts::PI {
        x -= 2.0 * std::f32::consts::PI;
    }
    while x < -std::f32::consts::PI {
        x += 2.0 * std::f32::consts::PI;
    }
    x
}

fn fit_noncausal_phase_line(observations: &[SymbolObservation]) -> Option<(f32, f32)> {
    let mut xs = Vec::with_capacity(observations.len());
    let mut ys = Vec::with_capacity(observations.len());
    let mut ws = Vec::with_capacity(observations.len());

    let mut last_raw = 0.0f32;
    let mut last_unwrapped = 0.0f32;
    let mut has_prev = false;
    for (i, obs) in observations.iter().enumerate() {
        let w = obs.phase_fit_weight;
        if !w.is_finite() || w < DQPSK_NONCAUSAL_PHASE_MIN_WEIGHT || !obs.phase_err.is_finite() {
            continue;
        }
        let raw = wrap_to_pi(obs.phase_err);
        let unwrapped = if has_prev {
            last_unwrapped + wrap_to_pi(raw - last_raw)
        } else {
            raw
        };
        has_prev = true;
        last_raw = raw;
        last_unwrapped = unwrapped;
        xs.push(i as f32);
        ys.push(unwrapped);
        ws.push(w);
    }

    if xs.len() < DQPSK_NONCAUSAL_PHASE_MIN_VALID_SYMBOLS {
        return None;
    }

    let mut sw = 0.0f32;
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    let mut sxx = 0.0f32;
    let mut sxy = 0.0f32;
    for i in 0..xs.len() {
        let w = ws[i];
        let x = xs[i];
        let y = ys[i];
        sw += w;
        sx += w * x;
        sy += w * y;
        sxx += w * x * x;
        sxy += w * x * y;
    }
    if sw <= 0.0 {
        return None;
    }

    let denom = sw * sxx - sx * sx;
    if denom.abs() < 1e-6 {
        return None;
    }

    let slope = (sw * sxy - sx * sy) / denom;
    if slope.abs() > DQPSK_NONCAUSAL_PHASE_SLOPE_MAX_RAD_PER_SYMBOL {
        return None;
    }
    let intercept = (sy - slope * sx) / sw;
    Some((intercept, slope))
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
    let mut prev_phase_sigma2 = 0.0f32;
    let mut observations = Vec::with_capacity(expected_symbols);

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

        let phase_ref_sym = tracking_state.phase_ref;
        let prev_phase_sym = *prev_phase;
        let best_corr = on_corrs[best_idx];
        let on_rot = best_corr * phase_ref_sym.conj();
        let diff_best = on_rot * prev_phase_sym.conj();
        let prev_phase_kappa = dqpsk_prev_phase_kappa_from_sigma2(prev_phase_sigma2);

        let (dqpsk_llr, walsh_conf, _topk_used, energies) = dqpsk_llr_from_walsh_hypotheses(
            &on_corrs,
            phase_ref_sym,
            prev_phase_sym,
            prev_phase_kappa,
            max_energy,
            second_energy,
            on_rot,
            diff_best,
            Complex32::new(1.0, 0.0),
        );
        let on_rot_tracking = walsh_posterior_weighted_on_rot(
            &on_corrs,
            phase_ref_sym,
            &energies,
            max_energy,
            on_rot,
        );
        let diff_tracking = on_rot_tracking * prev_phase_sym.conj();

        let energy_sum = energies.iter().sum::<f32>();
        let noise_floor = ((energy_sum - max_energy).max(0.0)) / 15.0;
        let snr_proxy = max_energy / (noise_floor + 1e-6);
        let dqpsk_conf = dqpsk_llr[0].abs() + dqpsk_llr[1].abs();
        // 位相追跡の信頼度は DQPSK 単独では過大になりやすいため、
        // Walsh識別の確からしさを掛け合わせて低SNR時の追従暴走を抑える。
        let dqpsk_conf_tracking = dqpsk_conf * walsh_conf.clamp(0.0, 1.0);
        tracking_state.phase_gate_enabled = tracking::next_phase_gate_enabled(
            tracking_state.phase_gate_enabled,
            dqpsk_conf_tracking,
            walsh_conf,
            snr_proxy,
        );
        if tracking_state.phase_gate_enabled {
            stats.phase_gate_on_symbols += 1;
        } else {
            stats.phase_gate_off_symbols += 1;
        }

        let decided = decide_dqpsk_symbol_for_phase_update(dqpsk_llr);
        let phase_err = tracking::phase_error_from_diff(diff_tracking, decided);
        let phase_err_abs = phase_err.abs();
        stats.phase_err_abs_sum_rad += phase_err_abs as f64;
        stats.phase_err_abs_count += 1;
        if phase_err_abs >= PHASE_ERR_ABS_THRESH_0P5_RAD {
            stats.phase_err_abs_ge_0p5_symbols += 1;
        }
        if phase_err_abs >= PHASE_ERR_ABS_THRESH_1P0_RAD {
            stats.phase_err_abs_ge_1p0_symbols += 1;
        }
        observations.push(SymbolObservation {
            on_corrs,
            phase_ref: phase_ref_sym,
            prev_phase: prev_phase_sym,
            max_energy,
            second_energy,
            on_rot_best: on_rot,
            diff_best,
            prev_phase_kappa,
            phase_err,
            phase_fit_weight: dqpsk_conf_tracking.clamp(0.0, 8.0),
        });
        let phase_var_sample =
            (phase_err_abs * phase_err_abs).min(DQPSK_PREV_PHASE_SIGMA_MAX_RAD.powi(2));
        prev_phase_sigma2 = DQPSK_PREV_PHASE_VAR_EWMA_ALPHA * prev_phase_sigma2
            + (1.0 - DQPSK_PREV_PHASE_VAR_EWMA_ALPHA) * phase_var_sample;
        let innovation_rejected = tracking_state.phase_gate_enabled
            && phase_err_abs > TRACKING_PHASE_ERR_GATE_RAD
            && dqpsk_conf_tracking < TRACKING_PHASE_ERR_GATE_DQPSK_CONF_HIGH;
        let phase_rate_update_enabled = tracking_state.phase_gate_enabled
            && tracking::phase_rate_update_enabled(dqpsk_conf_tracking, walsh_conf, snr_proxy);
        if innovation_rejected {
            stats.phase_innovation_reject_symbols += 1;
        }
        let phase_step = if tracking_state.phase_gate_enabled {
            // DQPSK信頼度が低いときは位相更新の駆動誤差を抑え、
            // 誤ったイノベーションでループが振れるのを防ぐ。
            let dqpsk_phase_weight =
                (dqpsk_conf_tracking / TRACKING_PHASE_DQPSK_CONF_ON_MIN).clamp(0.0, 1.0);
            let phase_err_for_update = if innovation_rejected || !phase_rate_update_enabled {
                0.0
            } else {
                phase_err * dqpsk_phase_weight
            };
            tracking_state.phase_rate = update_phase_rate_when_gate_on(
                tracking_state.phase_rate,
                phase_err_for_update,
                phase_rate_update_enabled,
            );
            tracking::phase_step_from_phase_error(phase_err_for_update, tracking_state.phase_rate)
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

        let on_norm = on_rot_tracking.norm().max(1e-6);
        *prev_phase = on_rot_tracking / on_norm;
    }

    let phase_line = fit_noncausal_phase_line(&observations);
    for (sym_idx, obs) in observations.iter().enumerate() {
        let residual_rot = if let Some((intercept, slope)) = phase_line {
            let phase = -(intercept + slope * sym_idx as f32);
            let (sin_p, cos_p) = phase.sin_cos();
            Complex32::new(cos_p, sin_p)
        } else {
            Complex32::new(1.0, 0.0)
        };
        let (dqpsk_llr, _walsh_conf, _topk_used, energies) = dqpsk_llr_from_walsh_hypotheses(
            &obs.on_corrs,
            obs.phase_ref,
            obs.prev_phase,
            obs.prev_phase_kappa,
            obs.max_energy,
            obs.second_energy,
            obs.on_rot_best,
            obs.diff_best,
            residual_rot,
        );
        let walsh_llr = demodulator.walsh_llr(&energies, obs.max_energy);
        buffers.packet_llrs_buffer.extend_from_slice(&walsh_llr);
        buffers.packet_llrs_buffer.extend_from_slice(&dqpsk_llr);
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

#[inline]
fn update_phase_rate_when_gate_on(
    phase_rate: f32,
    phase_err_for_update: f32,
    phase_rate_update_enabled: bool,
) -> f32 {
    if phase_rate_update_enabled {
        tracking::update_phase_rate(phase_rate, phase_err_for_update)
    } else {
        // ゲートON中でも低信頼シンボルでは phase_rate を更新せず減衰保持する。
        (phase_rate * TRACKING_PHASE_RATE_HOLD_DECAY).clamp(
            -tracking::TRACKING_PHASE_RATE_LIMIT_RAD,
            tracking::TRACKING_PHASE_RATE_LIMIT_RAD,
        )
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
    context.stats.viterbi_packet_decode_attempts += 1;
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

    let first_attempt = try_decode_soft_list_llrs(
        &context.buffers.deinterleave_buffer[..layout.fec_bits],
        context.options.viterbi_list_size,
        layout.payload_bits_len,
        &mut context.buffers.fec_candidate_bits_buffer,
        &mut context.buffers.decoded_bytes_buffer,
        &mut context.buffers.fec_workspace,
        context.stats,
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
        apply_llr_erasure_quantile_with_scratch(
            &mut context.buffers.erasure_llr_buffer[..layout.fec_bits],
            context.options.llr_erasure_quantile,
            &mut context.buffers.llr_abs_scratch,
        );
        match try_decode_soft_list_llrs(
            &context.buffers.erasure_llr_buffer[..layout.fec_bits],
            context.options.llr_erasure_list_size,
            layout.payload_bits_len,
            &mut context.buffers.fec_candidate_bits_buffer,
            &mut context.buffers.decoded_bytes_buffer,
            &mut context.buffers.fec_workspace,
            context.stats,
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

fn try_decode_soft_list_llrs(
    llrs: &[f32],
    list_size: usize,
    p_bits_len: usize,
    candidate_bits_scratch: &mut Vec<u8>,
    decoded_bytes: &mut Vec<u8>,
    fec_workspace: &mut fec::FecDecodeWorkspace,
    stats: &mut DecoderStats,
) -> Result<Packet, PacketDecodeError> {
    let mut saw_crc = false;
    let mut saw_parse = false;
    let packet = fec_workspace.decode_soft_list_try(
        llrs,
        list_size,
        candidate_bits_scratch,
        |decoded_bits, _rank, _score| {
            if decoded_bits.len() < p_bits_len {
                saw_parse = true;
                return None;
            }
            stats.viterbi_crc_candidate_checks += 1;
            match decode_packet(&decoded_bits[..p_bits_len], decoded_bytes) {
                Ok(packet) => Some(packet),
                Err(PacketDecodeError::Crc) => {
                    saw_crc = true;
                    None
                }
                Err(PacketDecodeError::Parse) => {
                    saw_parse = true;
                    None
                }
            }
        },
    );

    if let Some(packet) = packet {
        Ok(packet)
    } else if saw_crc {
        Err(PacketDecodeError::Crc)
    } else if saw_parse {
        Err(PacketDecodeError::Parse)
    } else {
        Err(PacketDecodeError::Parse)
    }
}

fn decode_packet(
    packet_bits: &[u8],
    decoded_bytes: &mut Vec<u8>,
) -> Result<Packet, PacketDecodeError> {
    fec::bits_to_bytes_into(packet_bits, decoded_bytes);
    match Packet::deserialize(decoded_bytes) {
        Ok(packet) => Ok(packet),
        Err(crate::frame::packet::PacketParseError::CrcMismatch { .. }) => {
            Err(PacketDecodeError::Crc)
        }
        Err(_) => Err(PacketDecodeError::Parse),
    }
}

#[cfg(test)]
#[inline]
pub(crate) fn apply_llr_erasure_quantile(llrs: &mut [f32], quantile: f32) {
    let mut scratch = Vec::new();
    apply_llr_erasure_quantile_with_scratch(llrs, quantile, &mut scratch);
}

#[inline]
pub(crate) fn apply_llr_erasure_quantile_with_scratch(
    llrs: &mut [f32],
    quantile: f32,
    abs_vals: &mut Vec<f32>,
) {
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

    abs_vals.clear();
    abs_vals.reserve(llrs.len());
    abs_vals.extend(llrs.iter().map(|v| v.abs()));
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

#[inline]
fn prob_bit0_from_llr(llr: f32) -> f32 {
    if llr >= 0.0 {
        1.0 / (1.0 + (-llr).exp())
    } else {
        let e = llr.exp();
        e / (1.0 + e)
    }
}

/// 位相更新専用の DQPSK シンボル推定。
/// LLR から bit 事後確率を作り、4位相の期待値方向を使って位相誤差を計算する。
/// 曖昧すぎて期待値振幅が極小のときは hard 判定にフォールバックする。
#[inline]
pub(crate) fn decide_dqpsk_symbol_for_phase_update(dqpsk_llr: [f32; 2]) -> Complex32 {
    let p_b0_0 = prob_bit0_from_llr(dqpsk_llr[0]);
    let p_b1_0 = prob_bit0_from_llr(dqpsk_llr[1]);
    let p_b0_1 = 1.0 - p_b0_0;
    let p_b1_1 = 1.0 - p_b1_0;

    // Gray マッピング:
    // s0:(0,0)->+1, s1:(0,1)->+j, s2:(1,1)->-1, s3:(1,0)->-j
    let w0 = p_b0_0 * p_b1_0;
    let w1 = p_b0_0 * p_b1_1;
    let w2 = p_b0_1 * p_b1_1;
    let w3 = p_b0_1 * p_b1_0;
    let soft = Complex32::new(w0 - w2, w1 - w3);
    let norm = soft.norm();
    if norm > DQPSK_PHASE_SOFT_SYMBOL_NORM_MIN {
        soft / norm
    } else {
        decide_dqpsk_symbol_from_llr(dqpsk_llr)
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

    fn assert_close(actual: f32, expected: f32, tol: f32, label: &str) {
        assert!(
            (actual - expected).abs() <= tol,
            "{} mismatch: actual={}, expected={}, tol={}",
            label,
            actual,
            expected,
            tol
        );
    }

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
    fn test_dqpsk_llr_uses_single_hypothesis_when_walsh_conf_is_high() {
        let mut on_corrs = [Complex32::new(0.0, 0.0); 16];
        on_corrs[0] = Complex32::new(10.0, 0.0); // energy=100
        on_corrs[1] = Complex32::new(1.0, 0.0); // energy=1

        let phase_ref = Complex32::new(1.0, 0.0);
        let prev_phase = Complex32::new(1.0, 0.0);
        let on_rot_best = on_corrs[0];
        let diff_best = on_corrs[0];

        let (dqpsk_llr, walsh_conf, topk_used, _energies) = dqpsk_llr_from_walsh_hypotheses(
            &on_corrs,
            phase_ref,
            prev_phase,
            1.0,
            100.0,
            1.0,
            on_rot_best,
            diff_best,
            Complex32::new(1.0, 0.0),
        );
        let energies: [f32; 16] = on_corrs.map(|c| c.norm_sqr());
        let energy_sum = energies.iter().sum::<f32>();
        let noise_var_best = dqpsk_noise_var_from_energy(energy_sum, energies[0]);
        let phase_logs =
            dqpsk_phase_log_metrics(diff_best, on_rot_best.norm(), noise_var_best, 1.0);
        let expected = [
            log_add_exp(phase_logs[0], phase_logs[1]) - log_add_exp(phase_logs[2], phase_logs[3]),
            log_add_exp(phase_logs[0], phase_logs[3]) - log_add_exp(phase_logs[1], phase_logs[2]),
        ];

        assert!(walsh_conf > DQPSK_WALSH_TOPK_ENABLE_CONF_THRESH);
        assert_eq!(topk_used, DQPSK_WALSH_HYPOTHESES);
        // Walsh尤度が十分に鋭い場合は joint 周辺化でも単一仮説に近づく。
        assert_close(dqpsk_llr[0], expected[0], 2e-2, "llr0");
        assert_close(dqpsk_llr[1], expected[1], 2e-2, "llr1");
    }

    #[test]
    fn test_dqpsk_llr_uses_joint_marginalization_when_walsh_conf_is_low() {
        let mut on_corrs = [Complex32::new(0.0, 0.0); 16];
        on_corrs[0] = Complex32::new(10.0, 0.0); // energy=100, phase0
        on_corrs[1] = Complex32::new(0.0, 8.944272); // energy≈80, phase1

        let phase_ref = Complex32::new(1.0, 0.0);
        let prev_phase = Complex32::new(1.0, 0.0);
        let on_rot_best = on_corrs[0];
        let diff_best = on_corrs[0];

        let (dqpsk_llr, walsh_conf, topk_used, _energies) = dqpsk_llr_from_walsh_hypotheses(
            &on_corrs,
            phase_ref,
            prev_phase,
            1.0,
            100.0,
            80.0,
            on_rot_best,
            diff_best,
            Complex32::new(1.0, 0.0),
        );

        let energies: [f32; 16] = on_corrs.map(|c| c.norm_sqr());
        let energy_sum = energies.iter().sum::<f32>();
        let walsh_noise_floor_e = dqpsk_noise_var_from_energy(energy_sum, 100.0);
        let mut sym_logs = [f32::NEG_INFINITY; 4];
        for idx in 0..DQPSK_WALSH_HYPOTHESES {
            let energy_h = energies[idx];
            let on_rot_h = on_corrs[idx] * phase_ref.conj();
            let diff_h = on_rot_h * prev_phase.conj();
            let noise_var_h = dqpsk_noise_var_from_energy(energy_sum, energy_h);
            let m = dqpsk_phase_log_metrics(diff_h, on_rot_h.norm(), noise_var_h, 1.0);
            let log_p_h = (energy_h - 100.0) / walsh_noise_floor_e;
            for s in 0..4 {
                sym_logs[s] = log_add_exp(sym_logs[s], log_p_h + m[s]);
            }
        }
        let expected = [
            log_add_exp(sym_logs[0], sym_logs[1]) - log_add_exp(sym_logs[2], sym_logs[3]),
            log_add_exp(sym_logs[0], sym_logs[3]) - log_add_exp(sym_logs[1], sym_logs[2]),
        ];

        assert!(walsh_conf < DQPSK_WALSH_TOPK_ENABLE_CONF_THRESH);
        assert_eq!(topk_used, DQPSK_WALSH_HYPOTHESES);
        assert_close(dqpsk_llr[0], expected[0], 1e-4, "llr0");
        assert_close(dqpsk_llr[1], expected[1], 1e-4, "llr1");
        assert!(dqpsk_llr[0].is_finite() && dqpsk_llr[1].is_finite());
    }

    #[test]
    fn test_walsh_posterior_weighted_on_rot_tracks_dominant_hypothesis() {
        let mut on_corrs = [Complex32::new(0.0, 0.0); 16];
        on_corrs[0] = Complex32::new(10.0, 0.0); // dominant
        on_corrs[1] = Complex32::new(1.0, 3.0);
        let energies = on_corrs.map(|c| c.norm_sqr());
        let on_rot_best = on_corrs[0];

        let mixed = walsh_posterior_weighted_on_rot(
            &on_corrs,
            Complex32::new(1.0, 0.0),
            &energies,
            energies[0],
            on_rot_best,
        );

        assert!(mixed.re > 8.0);
        assert!(mixed.im.abs() < 2.0);
    }

    #[test]
    fn test_walsh_posterior_weighted_on_rot_falls_back_when_posterior_cancels() {
        let mut on_corrs = [Complex32::new(0.0, 0.0); 16];
        on_corrs[0] = Complex32::new(10.0, 0.0);
        on_corrs[1] = Complex32::new(-10.0, 0.0);
        let energies = on_corrs.map(|c| c.norm_sqr());
        let on_rot_best = on_corrs[0];

        let mixed = walsh_posterior_weighted_on_rot(
            &on_corrs,
            Complex32::new(1.0, 0.0),
            &energies,
            energies[0],
            on_rot_best,
        );

        assert_close(mixed.re, on_rot_best.re, 1e-6, "mixed_re");
        assert_close(mixed.im, on_rot_best.im, 1e-6, "mixed_im");
    }

    #[test]
    fn test_log_i0_approx_is_finite_across_range() {
        for &x in &[0.0_f32, 1e-3, 0.1, 1.0, 5.0, 10.0, 50.0, 500.0] {
            let v = log_i0_approx(x);
            assert!(v.is_finite(), "x={x}, log_i0={v}");
        }
    }

    #[test]
    fn test_log_i0_approx_monotonic_non_decreasing() {
        let xs = [0.0_f32, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
        let mut prev = log_i0_approx(xs[0]);
        for &x in &xs[1..] {
            let cur = log_i0_approx(x);
            assert!(cur >= prev - 1e-6, "x={x}, prev={prev}, cur={cur}");
            prev = cur;
        }
    }

    fn log_i0_reference_series(x: f32) -> f32 {
        // I0(x) = Σ_{k=0..∞} ((x^2/4)^k / (k!)^2)
        // 精度テスト用の参照実装（f64）。大きいxは使わず収束が十分速い範囲で比較する。
        let x64 = x as f64;
        let t = 0.25 * x64 * x64;
        let mut sum = 1.0f64;
        let mut term = 1.0f64;
        for k in 1..=200 {
            let kk = k as f64;
            term *= t / (kk * kk);
            sum += term;
            if term.abs() <= 1e-15 * sum.abs() {
                break;
            }
        }
        (sum.ln()) as f32
    }

    #[test]
    fn test_log_i0_approx_matches_reference_series() {
        // Cephes区分点(3.75)近傍を含めて近似誤差を監視する。
        let xs = [0.0_f32, 0.1, 1.0, 2.5, 3.75, 4.0, 6.0, 10.0];
        for &x in &xs {
            let approx = log_i0_approx(x);
            let reference = log_i0_reference_series(x);
            let err = (approx - reference).abs();
            assert!(
                err < 2e-4,
                "x={x}, approx={approx}, ref={reference}, err={err}"
            );
        }
    }

    fn make_phase_obs(phase_err: f32, weight: f32) -> SymbolObservation {
        SymbolObservation {
            on_corrs: [Complex32::new(0.0, 0.0); 16],
            phase_ref: Complex32::new(1.0, 0.0),
            prev_phase: Complex32::new(1.0, 0.0),
            max_energy: 1.0,
            second_energy: 0.0,
            on_rot_best: Complex32::new(1.0, 0.0),
            diff_best: Complex32::new(1.0, 0.0),
            prev_phase_kappa: 1.0,
            phase_err,
            phase_fit_weight: weight,
        }
    }

    #[test]
    fn test_fit_noncausal_phase_line_recovers_linear_phase() {
        let intercept = 0.12f32;
        let slope = 0.04f32;
        let observations: Vec<SymbolObservation> = (0..20)
            .map(|i| make_phase_obs(intercept + slope * i as f32, 1.0))
            .collect();
        let (a, b) = fit_noncausal_phase_line(&observations).expect("fit should succeed");
        assert_close(a, intercept, 1e-4, "intercept");
        assert_close(b, slope, 1e-4, "slope");
    }

    #[test]
    fn test_fit_noncausal_phase_line_unwraps_wrapped_sequence() {
        let intercept = 2.9f32;
        let slope = 0.08f32;
        let observations: Vec<SymbolObservation> = (0..24)
            .map(|i| make_phase_obs(wrap_to_pi(intercept + slope * i as f32), 1.0))
            .collect();
        let (_a, b) = fit_noncausal_phase_line(&observations).expect("fit should succeed");
        assert_close(b, slope, 2e-3, "slope_wrapped");
    }

    #[test]
    fn test_fit_noncausal_phase_line_rejects_too_steep_slope() {
        let observations: Vec<SymbolObservation> = (0..24)
            .map(|i| make_phase_obs(0.8 * i as f32, 1.0))
            .collect();
        assert!(fit_noncausal_phase_line(&observations).is_none());
    }

    #[test]
    fn test_fit_noncausal_phase_line_ignores_non_finite_phase_errors() {
        let mut observations: Vec<SymbolObservation> = (0..24)
            .map(|i| make_phase_obs(0.1 + 0.03 * i as f32, 1.0))
            .collect();
        observations[5].phase_err = f32::NAN;
        observations[9].phase_err = f32::INFINITY;
        let (_a, b) = fit_noncausal_phase_line(&observations).expect("fit should survive NaN/Inf");
        assert_close(b, 0.03, 3e-3, "slope_non_finite_filtered");
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
    fn test_decide_dqpsk_symbol_for_phase_update_falls_back_when_llr_is_ambiguous() {
        let hard = decide_dqpsk_symbol_from_llr([0.0, 0.0]);
        let soft = decide_dqpsk_symbol_for_phase_update([0.0, 0.0]);
        assert_eq!(soft, hard);
    }

    #[test]
    fn test_decide_dqpsk_symbol_for_phase_update_uses_soft_direction_when_llr_is_asymmetric() {
        let soft = decide_dqpsk_symbol_for_phase_update([2.0, 0.5]);
        assert!(soft.re > 0.0);
        assert!(soft.im > 0.0);
        assert_close(soft.norm(), 1.0, 1e-5, "soft_symbol_norm");
    }

    #[test]
    fn test_soft_phase_update_symbol_reduces_phase_error_vs_hard_decision() {
        let llr = [2.0, 0.5];
        let hard = decide_dqpsk_symbol_from_llr(llr);
        let soft = decide_dqpsk_symbol_for_phase_update(llr);
        let diff = Complex32::new(20.0f32.to_radians().cos(), 20.0f32.to_radians().sin());

        let err_hard = tracking::phase_error_from_diff(diff, hard).abs();
        let err_soft = tracking::phase_error_from_diff(diff, soft).abs();
        assert!(
            err_soft < err_hard,
            "soft phase update should reduce phase error: hard={} soft={}",
            err_hard,
            err_soft
        );
    }

    #[test]
    fn test_update_phase_rate_when_gate_on_updates_with_innovation_when_enabled() {
        let phase_rate = 0.7;
        let phase_err_for_update = 0.2;
        let updated = update_phase_rate_when_gate_on(phase_rate, phase_err_for_update, true);
        let expected = tracking::update_phase_rate(phase_rate, phase_err_for_update);
        assert_close(updated, expected, 1e-7, "phase_rate_update_enabled");
    }

    #[test]
    fn test_update_phase_rate_when_gate_on_holds_and_decays_when_disabled() {
        let phase_rate = 0.7;
        let updated_small_err = update_phase_rate_when_gate_on(phase_rate, 0.1, false);
        let updated_large_err = update_phase_rate_when_gate_on(phase_rate, 10.0, false);
        let expected = (phase_rate * TRACKING_PHASE_RATE_HOLD_DECAY).clamp(
            -tracking::TRACKING_PHASE_RATE_LIMIT_RAD,
            tracking::TRACKING_PHASE_RATE_LIMIT_RAD,
        );
        assert_close(
            updated_small_err,
            expected,
            1e-7,
            "phase_rate_hold_decay_small",
        );
        assert_close(
            updated_large_err,
            expected,
            1e-7,
            "phase_rate_hold_decay_large",
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
    fn test_try_decode_soft_list_llrs_returns_packet_for_valid_bits() {
        let packet = Packet::new(7, 3, &[0x5a; crate::params::PAYLOAD_SIZE]);
        let llrs = encode_packet_to_descrambled_llrs(&packet, 8.0);
        let mut candidate_bits = Vec::new();
        let mut decoded_bytes = Vec::new();
        let mut workspace = fec::FecDecodeWorkspace::new();
        let mut stats = DecoderStats::new();
        let decoded = try_decode_soft_list_llrs(
            &llrs,
            1,
            PACKET_BYTES * 8,
            &mut candidate_bits,
            &mut decoded_bytes,
            &mut workspace,
            &mut stats,
        )
        .unwrap();

        assert_eq!(decoded, packet);
        assert_eq!(stats.viterbi_crc_candidate_checks, 1);
    }

    #[test]
    fn test_try_decode_soft_list_llrs_distinguishes_crc_and_parse_errors() {
        let packet = Packet::new(1, 2, &[0x11; crate::params::PAYLOAD_SIZE]);
        let mut crc_bytes = packet.serialize();
        crc_bytes[0] ^= 1;
        let crc_llrs: Vec<f32> = fec::encode(&fec::bytes_to_bits(&crc_bytes))
            .into_iter()
            .map(|bit| if bit == 0 { 8.0 } else { -8.0 })
            .collect();
        let mut candidate_bits = Vec::new();
        let mut decoded_bytes = Vec::new();
        let mut workspace = fec::FecDecodeWorkspace::new();
        let mut stats = DecoderStats::new();
        let crc_err = try_decode_soft_list_llrs(
            &crc_llrs,
            1,
            PACKET_BYTES * 8,
            &mut candidate_bits,
            &mut decoded_bytes,
            &mut workspace,
            &mut stats,
        );
        assert!(matches!(crc_err, Err(PacketDecodeError::Crc)));

        let llrs = encode_packet_to_descrambled_llrs(&packet, 8.0);
        let parse_err = try_decode_soft_list_llrs(
            &llrs,
            1,
            0,
            &mut candidate_bits,
            &mut decoded_bytes,
            &mut workspace,
            &mut stats,
        );
        assert!(matches!(parse_err, Err(PacketDecodeError::Parse)));
        assert_eq!(stats.viterbi_crc_candidate_checks, 2);
    }

    #[test]
    fn test_decode_packet_distinguishes_crc_and_parse_errors() {
        let packet = Packet::new(9, 4, &[0x33; crate::params::PAYLOAD_SIZE]);
        let valid_bits = fec::bytes_to_bits(&packet.serialize());
        let mut decoded_bytes = Vec::new();
        assert_eq!(
            decode_packet(&valid_bits, &mut decoded_bytes).unwrap(),
            packet
        );

        let mut crc_bits = valid_bits.clone();
        crc_bits[5] ^= 1;
        assert!(matches!(
            decode_packet(&crc_bits, &mut decoded_bytes),
            Err(PacketDecodeError::Crc)
        ));

        let mut parse_bits = valid_bits.clone();
        parse_bits.extend_from_slice(&[0; 8]);
        assert!(matches!(
            decode_packet(&parse_bits, &mut decoded_bytes),
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
        assert_eq!(stats.viterbi_packet_decode_attempts, 1);
        assert_eq!(stats.viterbi_crc_candidate_checks, 1);
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
        assert_eq!(stats.viterbi_packet_decode_attempts, 1);
        assert_eq!(stats.viterbi_crc_candidate_checks, 1);
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
