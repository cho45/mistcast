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
use crate::mary::params::{
    payload_data_index_for_symbol_slot, payload_total_symbols, PAYLOAD_PILOT_DQPSK_BITS,
    PAYLOAD_PILOT_WALSH_INDEX, PAYLOAD_SPREAD_FACTOR,
};
use num_complex::Complex32;
use std::sync::OnceLock;

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
    pub hmm_slot_logz: Vec<f32>,
    pub hmm_theta_log_prior: Vec<f32>,
    pub hmm_pred: Vec<f32>,
    pub hmm_alpha: Vec<f32>,
    pub hmm_beta: Vec<f32>,
    pub hmm_symbol_llrs: Vec<[f32; 6]>,
    phase_slot_observations: Vec<PhaseSlotObservation>,
}

impl PacketDecodeBuffers {
    pub fn new() -> Self {
        let cap = interleaver_config::interleaved_bits();
        let slots = payload_total_symbols(interleaver_config::mary_symbols());
        let hmm_cap = slots * DQPSK_SEMI_COHERENT_PHASE_STATES.max(1);
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
            hmm_slot_logz: Vec::with_capacity(hmm_cap),
            hmm_theta_log_prior: Vec::with_capacity(hmm_cap),
            hmm_pred: Vec::with_capacity(hmm_cap),
            hmm_alpha: Vec::with_capacity(hmm_cap),
            hmm_beta: Vec::with_capacity(hmm_cap),
            hmm_symbol_llrs: Vec::with_capacity(slots),
            phase_slot_observations: Vec::with_capacity(slots),
        }
    }

    pub fn clear(&mut self) {
        self.packet_llrs_buffer.clear();
        self.deinterleave_buffer.clear();
        self.erasure_llr_buffer.clear();
        self.llr_abs_scratch.clear();
        self.fec_candidate_bits_buffer.clear();
        self.decoded_bytes_buffer.clear();
        self.hmm_slot_logz.clear();
        self.hmm_theta_log_prior.clear();
        self.hmm_pred.clear();
        self.hmm_alpha.clear();
        self.hmm_beta.clear();
        self.hmm_symbol_llrs.clear();
        self.phase_slot_observations.clear();
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
struct PhaseSlotObservation {
    on_corrs: [Complex32; 16],
    energies: [f32; 16],
    energy_sum: f32,
    phase_ref_conj: Complex32,
    max_energy: f32,
    prev_phase_kappa: f32,
    snr_proxy: f32,
    avg_noise_var: f32,
    is_data: bool,
}

// DQPSK LLR は (Walsh仮説 h, 位相仮説 s) の同時尤度を
// 全Walsh仮説で周辺化して計算する。
// walsh_conf は位相追跡のゲート制御にも使うため継続して算出する。
const DQPSK_WALSH_HYPOTHESES: usize = 16;
const DQPSK_PHASE_SYMBOL_HYPOTHESES: usize = 4;
const DQPSK_JOINT_HYPOTHESES: usize = DQPSK_WALSH_HYPOTHESES * DQPSK_PHASE_SYMBOL_HYPOTHESES;
// HMM観測での Walsh 周辺化の上位仮説数（SNR適応）。
const DQPSK_HMM_WALSH_TOPK_HIGH_SNR: usize = 4;
const DQPSK_HMM_WALSH_TOPK_MID_SNR: usize = 8;
const DQPSK_HMM_WALSH_TOPK_LOW_SNR: usize = 16;
const DQPSK_HMM_WALSH_TOPK_MID_SNR_PROXY: f32 = 7.0;
const DQPSK_HMM_WALSH_TOPK_LOW_SNR_PROXY: f32 = 4.0;
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
// 既知パイロットで phase_rate を直接更新するときの低ゲイン設定。
// データ判定由来の更新より小さいゲインで、長尺時の累積ズレだけを穏やかに補正する。
const PILOT_PHASE_PROP_GAIN: f32 = 0.12;
const PILOT_PHASE_RATE_GAIN: f32 = 0.015;
const PILOT_PHASE_ERR_CLAMP: f32 = 0.60;
const PILOT_PHASE_WALSH_CONF_MIN: f32 = 0.20;
const PILOT_PHASE_SNR_PROXY_MIN: f32 = 2.0;
// 準コヒーレント復調の位相状態数（[-pi, pi) の離散グリッド）。
const DQPSK_SEMI_COHERENT_PHASE_STATES: usize = 32;
// 位相状態遷移のランダムウォーク分散 [rad^2]（小さいほど連続性を強く仮定）。
const DQPSK_SEMI_COHERENT_PHASE_TRANS_VAR: f32 = 0.30 * 0.30;
// 遷移分散に対する SNR/位相不確かさの加重係数。
const DQPSK_SEMI_COHERENT_TRANS_KAPPA_SNR_WEIGHT: f32 = 0.20;
const DQPSK_SEMI_COHERENT_TRANS_KAPPA_PHASE_WEIGHT: f32 = 0.20;
// 観測分散に対する SNR/位相不確かさの加重係数。
const DQPSK_PHASE_MODEL_VAR_SNR_WEIGHT: f32 = 0.20;
const DQPSK_PHASE_MODEL_VAR_PHASE_WEIGHT: f32 = 0.20;
// Walsh仮説重みの温度化係数（低SNR/高位相不確かさでソフト化）。
const DQPSK_WALSH_TEMP_SNR_WEIGHT: f32 = 0.50;
const DQPSK_WALSH_TEMP_PHASE_WEIGHT: f32 = 0.30;
// Walsh LLR と DQPSK(HMM) LLR はモデル由来が異なるため、
// 現段階では相対ゲインで合成する。
const DQPSK_OUTPUT_LLR_REL_GAIN: f32 = 0.25;
struct PhaseHmmCache {
    state_angles: Vec<f32>,
    state_residual_rots: Vec<Complex32>,
    transition_cos: Vec<f32>,
}

fn phase_hmm_cache() -> &'static PhaseHmmCache {
    static CACHE: OnceLock<PhaseHmmCache> = OnceLock::new();
    CACHE.get_or_init(|| {
        let m_len = DQPSK_SEMI_COHERENT_PHASE_STATES.max(1);
        let mut state_angles = Vec::with_capacity(m_len);
        let mut state_residual_rots = Vec::with_capacity(m_len);
        for i in 0..m_len {
            let theta =
                -std::f32::consts::PI + (2.0 * std::f32::consts::PI * i as f32) / m_len as f32;
            state_angles.push(theta);
            let (sin_p, cos_p) = (-theta).sin_cos();
            state_residual_rots.push(Complex32::new(cos_p, sin_p));
        }

        let mut transition_cos = vec![0.0f32; m_len * m_len];
        for prev in 0..m_len {
            for curr in 0..m_len {
                let d = wrap_to_pi(state_angles[curr] - state_angles[prev]);
                transition_cos[prev * m_len + curr] = d.cos();
            }
        }
        PhaseHmmCache {
            state_angles,
            state_residual_rots,
            transition_cos,
        }
    })
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
    max_energy: f32,
    avg_noise_var: f32,
    prev_phase_kappa: f32,
) -> [f32; 4] {
    // スケーリングに依存しないよう、diff を正規化して unit circle 上で扱う。
    let y = diff / diff.norm().max(1e-6);
    // 観測分散 sigma2_obs は、処理利得 SF を考慮したシンボルSNRの逆数に比例する。
    // max_energy = (A*SF)^2, avg_noise_var = sigma2*SF なので、
    // noise_var / max_energy = sigma2 / (A^2 * SF) = 1 / (SNR * SF)
    let sigma2_obs = (2.0 * avg_noise_var / max_energy.max(1e-6)).max(DQPSK_PHASE_OBS_VAR_FLOOR);
    let snr_proxy = max_energy / avg_noise_var.max(DQPSK_NOISE_VAR_FLOOR);
    let sigma2_eff = sigma2_obs + phase_model_var_from_kappa_snr(prev_phase_kappa, snr_proxy);
    let eta = 2.0 / sigma2_eff;
    let prior = Complex32::new(prev_phase_kappa.clamp(0.0, 1.0), 0.0);
    let z0 = prior + y.conj() * Complex32::new(1.0, 0.0) * eta;
    let z1 = prior + y.conj() * Complex32::new(0.0, 1.0) * eta;
    let z2 = prior + y.conj() * Complex32::new(-1.0, 0.0) * eta;
    let z3 = prior + y.conj() * Complex32::new(0.0, -1.0) * eta;
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
fn phase_sigma2_from_kappa(kappa: f32) -> f32 {
    let k = kappa.clamp(1e-4, 1.0);
    (-2.0 * k.ln()).clamp(0.0, DQPSK_PREV_PHASE_SIGMA_MAX_RAD.powi(2))
}

#[inline]
fn snr_penalty_from_proxy(snr_proxy: f32) -> f32 {
    1.0 / (snr_proxy.max(0.0) + 1.0)
}

#[inline]
fn transition_kappa_from_observations(prev: &PhaseSlotObservation, curr: &PhaseSlotObservation) -> f32 {
    let sigma2_prev = phase_sigma2_from_kappa(prev.prev_phase_kappa);
    let sigma2_curr = phase_sigma2_from_kappa(curr.prev_phase_kappa);
    let snr_pair = 0.5 * (prev.snr_proxy + curr.snr_proxy);
    let sigma2_pair = 0.5 * (sigma2_prev + sigma2_curr);
    let sigma2_trans = DQPSK_SEMI_COHERENT_PHASE_TRANS_VAR
        * (1.0 + DQPSK_SEMI_COHERENT_TRANS_KAPPA_PHASE_WEIGHT * sigma2_pair)
        * (1.0 + DQPSK_SEMI_COHERENT_TRANS_KAPPA_SNR_WEIGHT * snr_penalty_from_proxy(snr_pair));
    (1.0 / sigma2_trans.max(1e-4)).clamp(0.2, 16.0)
}

#[inline]
#[cfg(test)]
fn observation_model_var_from_observation(obs: &PhaseSlotObservation) -> f32 {
    phase_model_var_from_kappa_snr(obs.prev_phase_kappa, obs.snr_proxy)
}

#[inline]
fn phase_model_var_from_kappa_snr(prev_phase_kappa: f32, snr_proxy: f32) -> f32 {
    let sigma2_prior = phase_sigma2_from_kappa(prev_phase_kappa);
    let snr_penalty = snr_penalty_from_proxy(snr_proxy);
    DQPSK_PHASE_MODEL_VAR
        * (1.0 + DQPSK_PHASE_MODEL_VAR_PHASE_WEIGHT * sigma2_prior)
        * (1.0 + DQPSK_PHASE_MODEL_VAR_SNR_WEIGHT * snr_penalty)
}

#[inline]
fn walsh_temperature_scale(prev_phase_kappa: f32, snr_proxy: f32) -> f32 {
    let sigma2_prior = phase_sigma2_from_kappa(prev_phase_kappa);
    let snr_penalty = snr_penalty_from_proxy(snr_proxy);
    1.0 + DQPSK_WALSH_TEMP_PHASE_WEIGHT * sigma2_prior + DQPSK_WALSH_TEMP_SNR_WEIGHT * snr_penalty
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

#[allow(clippy::too_many_arguments)]
fn dqpsk_llr_from_walsh_hypotheses(
    on_corrs: &[Complex32; 16],
    phase_ref: Complex32,
    prev_phase: Complex32,
    prev_phase_kappa: f32,
    max_energy: f32,
    second_energy: f32,
    snr_proxy: f32,
    avg_noise_var: f32,
    _on_rot_best: Complex32,
    diff_best: Complex32,
    residual_rot: Complex32,
) -> ([f32; 2], f32, usize, [f32; 16]) {
    let energies: [f32; 16] = on_corrs.map(|c| c.norm_sqr());
    let walsh_conf = ((max_energy - second_energy).max(0.0)) / max_energy.max(1e-6);
    let _energy_sum = energies.iter().sum::<f32>();
    let walsh_temp_scale = walsh_temperature_scale(prev_phase_kappa, snr_proxy);

    let mut sym_logs = [f32::NEG_INFINITY; 4];
    for idx in 0..DQPSK_WALSH_HYPOTHESES {
        let energy_h = energies[idx];
        let on_rot_h = on_corrs[idx] * phase_ref.conj();
        let diff_h = on_rot_h * prev_phase.conj() * residual_rot;
        let phase_logs =
            dqpsk_phase_log_metrics(diff_h, max_energy, avg_noise_var, prev_phase_kappa);
        let log_p_h = (energy_h - max_energy) / (avg_noise_var * walsh_temp_scale);
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
        let phase_logs = dqpsk_phase_log_metrics(
            diff_best * residual_rot,
            max_energy,
            avg_noise_var,
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

#[inline]
fn normalize_log_row(row: &mut [f32]) {
    let mut max_v = f32::NEG_INFINITY;
    for &v in row.iter() {
        if v > max_v {
            max_v = v;
        }
    }
    if max_v.is_finite() {
        for v in row.iter_mut() {
            *v -= max_v;
        }
    }
}

#[inline]
fn dqpsk_symbol_index_from_bits_pair(b0: u8, b1: u8) -> usize {
    match (b0, b1) {
        (0, 0) => 0,
        (0, 1) => 1,
        (1, 1) => 2,
        (1, 0) => 3,
        _ => 0,
    }
}

#[inline]
fn joint_hypothesis_index(walsh_idx: usize, dqpsk_sym_idx: usize) -> usize {
    walsh_idx * DQPSK_PHASE_SYMBOL_HYPOTHESES + dqpsk_sym_idx
}

#[inline]
fn log_sum_exp_slice(values: &[f32]) -> f32 {
    let mut v = f32::NEG_INFINITY;
    for &x in values {
        v = log_add_exp(v, x);
    }
    v
}

fn joint_symbol_logs_for_phase_state(
    obs: &PhaseSlotObservation,
    residual_rot: Complex32,
) -> [f32; DQPSK_JOINT_HYPOTHESES] {
    let snr_instant = obs.max_energy / (((obs.energy_sum - obs.max_energy).max(0.0) / 15.0) + 1e-6);
    let walsh_temp_scale = walsh_temperature_scale(obs.prev_phase_kappa, snr_instant);
    let walsh_noise_floor_e = ((obs.energy_sum - obs.max_energy).max(0.0) / 15.0).max(DQPSK_NOISE_VAR_FLOOR);

    let mut joint = [f32::NEG_INFINITY; DQPSK_JOINT_HYPOTHESES];
    let mut topk_threshold = f32::NEG_INFINITY;
    if obs.is_data {
        let k = if obs.snr_proxy < DQPSK_HMM_WALSH_TOPK_LOW_SNR_PROXY {
            DQPSK_HMM_WALSH_TOPK_LOW_SNR
        } else if obs.snr_proxy < DQPSK_HMM_WALSH_TOPK_MID_SNR_PROXY {
            DQPSK_HMM_WALSH_TOPK_MID_SNR
        } else {
            DQPSK_HMM_WALSH_TOPK_HIGH_SNR
        };
        if k < DQPSK_WALSH_HYPOTHESES {
            let mut sorted = obs.energies;
            sorted.sort_by(|a: &f32, b: &f32| b.total_cmp(a));
            topk_threshold = sorted[k - 1];
        }
    }
    for walsh_idx in 0..DQPSK_WALSH_HYPOTHESES {
        if !obs.is_data && walsh_idx != PAYLOAD_PILOT_WALSH_INDEX {
            continue;
        }
        if obs.is_data && obs.energies[walsh_idx] < topk_threshold {
            continue;
        }
        let energy_h = obs.energies[walsh_idx];
        let on_rot_h = obs.on_corrs[walsh_idx] * obs.phase_ref_conj;
        let diff_h = on_rot_h * residual_rot;
        let phase_logs =
            dqpsk_phase_log_metrics(diff_h, obs.max_energy, obs.avg_noise_var, obs.prev_phase_kappa);
        let log_p_h = if obs.is_data {
            (energy_h - obs.max_energy) / (walsh_noise_floor_e * walsh_temp_scale)
        } else {
            0.0
        };
        for (dqpsk_sym_idx, &phase_log) in phase_logs.iter().enumerate() {
            joint[joint_hypothesis_index(walsh_idx, dqpsk_sym_idx)] = log_p_h + phase_log;
        }
    }
    joint
}

fn extract_differential_llrs_from_joint_post(
    joint_post: &[f32; DQPSK_JOINT_HYPOTHESES],
    prev_a: &[f32; 4],
    out_a: &mut [f32; 4],
) -> [f32; 6] {
    let mut out = [0.0f32; 6];

    for walsh_idx in 0..DQPSK_WALSH_HYPOTHESES {
        for dqpsk_sym_idx in 0..DQPSK_PHASE_SYMBOL_HYPOTHESES {
            let v = joint_post[joint_hypothesis_index(walsh_idx, dqpsk_sym_idx)];
            out_a[dqpsk_sym_idx] = log_add_exp(out_a[dqpsk_sym_idx], v);
        }
    }
    normalize_log_row(out_a);

    // Walsh bits: out[0]=bit3(MSB), out[1]=bit2, out[2]=bit1, out[3]=bit0(LSB)
    for (i, item) in out.iter_mut().take(4).enumerate() {
        let bit = 3 - i;
        let mut bit0 = f32::NEG_INFINITY;
        let mut bit1 = f32::NEG_INFINITY;
        for walsh_idx in 0..DQPSK_WALSH_HYPOTHESES {
            let mut p_w = f32::NEG_INFINITY;
            for dqpsk_sym_idx in 0..DQPSK_PHASE_SYMBOL_HYPOTHESES {
                p_w = log_add_exp(p_w, joint_post[joint_hypothesis_index(walsh_idx, dqpsk_sym_idx)]);
            }
            if ((walsh_idx >> bit) & 1) == 0 {
                bit0 = log_add_exp(bit0, p_w);
            } else {
                bit1 = log_add_exp(bit1, p_w);
            }
        }
        *item = bit0 - bit1;
    }

    let mut diff_logs = [f32::NEG_INFINITY; 4];
    for (a, &a_val) in out_a.iter().enumerate().take(DQPSK_PHASE_SYMBOL_HYPOTHESES) {
        for (b, &b_val) in prev_a.iter().enumerate().take(DQPSK_PHASE_SYMBOL_HYPOTHESES) {
            let d = (a + 4 - b) % 4;
            diff_logs[d] = log_add_exp(diff_logs[d], a_val + b_val);
        }
    }

    let b0_0 = log_add_exp(diff_logs[0], diff_logs[1]);
    let b0_1 = log_add_exp(diff_logs[2], diff_logs[3]);
    let b1_0 = log_add_exp(diff_logs[0], diff_logs[3]);
    let b1_1 = log_add_exp(diff_logs[1], diff_logs[2]);
    out[4] = b0_0 - b0_1;
    out[5] = b1_0 - b1_1;

    out
}

#[cfg(test)]
fn dqpsk_symbol_logs_for_phase_state(
    obs: &PhaseSlotObservation,
    residual_rot: Complex32,
) -> [f32; 4] {
    let walsh_noise_floor_e = dqpsk_noise_var_from_energy(obs.energy_sum, obs.max_energy);
    let walsh_temp_scale = walsh_temperature_scale(obs.prev_phase_kappa, obs.snr_proxy);
    let mut sym_logs = [f32::NEG_INFINITY; 4];
    let mut topk_threshold = f32::NEG_INFINITY;
    if obs.is_data {
        let k = if obs.snr_proxy < DQPSK_HMM_WALSH_TOPK_LOW_SNR_PROXY {
            DQPSK_HMM_WALSH_TOPK_LOW_SNR
        } else if obs.snr_proxy < DQPSK_HMM_WALSH_TOPK_MID_SNR_PROXY {
            DQPSK_HMM_WALSH_TOPK_MID_SNR
        } else {
            DQPSK_HMM_WALSH_TOPK_HIGH_SNR
        };
        if k < DQPSK_WALSH_HYPOTHESES {
            let mut sorted = obs.energies;
            sorted.sort_by(|a, b| b.total_cmp(a));
            topk_threshold = sorted[k - 1];
        }
    }
    for idx in 0..DQPSK_WALSH_HYPOTHESES {
        if !obs.is_data && idx != PAYLOAD_PILOT_WALSH_INDEX {
            continue;
        }
        if obs.is_data && obs.energies[idx] < topk_threshold {
            continue;
        }
        let energy_h = obs.energies[idx];
        let on_rot_h = obs.on_corrs[idx] * obs.phase_ref_conj;
        let diff_h = on_rot_h * residual_rot;
        let noise_var_h = dqpsk_noise_var_from_energy(obs.energy_sum, energy_h);
        let phase_logs =
            dqpsk_phase_log_metrics(diff_h, on_rot_h.norm(), noise_var_h, obs.prev_phase_kappa);
        let log_p_h = if obs.is_data {
            (energy_h - obs.max_energy) / (walsh_noise_floor_e * walsh_temp_scale)
        } else {
            0.0
        };
        for s in 0..4 {
            sym_logs[s] = log_add_exp(sym_logs[s], log_p_h + phase_logs[s]);
        }
    }
    sym_logs
}

fn symbol_llrs_with_phase_hmm(
    slot_obs: &[PhaseSlotObservation],
    out_llrs: &mut Vec<[f32; 6]>,
    buffers: &mut PacketDecodeBuffers,
) {
    out_llrs.clear();
    if slot_obs.is_empty() {
        return;
    }

    let cache = phase_hmm_cache();
    let t_len = slot_obs.len();
    let m_len = cache.state_angles.len();
    let flat_len = t_len * m_len;
    let pilot_sym_idx =
        dqpsk_symbol_index_from_bits_pair(PAYLOAD_PILOT_DQPSK_BITS.0, PAYLOAD_PILOT_DQPSK_BITS.1);

    buffers.hmm_slot_logz.resize(flat_len, f32::NEG_INFINITY);
    buffers.hmm_theta_log_prior.resize(flat_len, 0.0);
    buffers.hmm_pred.resize(flat_len, f32::NEG_INFINITY);
    buffers.hmm_alpha.resize(flat_len, f32::NEG_INFINITY);
    buffers.hmm_beta.resize(flat_len, 0.0);
    // resize は既存要素を初期化しないため、前パケットの残値混入を防ぐために毎回明示ゼロ化する。
    buffers.hmm_beta.fill(0.0);

    for (t, obs) in slot_obs.iter().enumerate() {
        let kappa_prior = obs.prev_phase_kappa.clamp(0.0, 1.0);
        for m in 0..m_len {
            let joint = joint_symbol_logs_for_phase_state(obs, cache.state_residual_rots[m]);
            let flat = t * m_len + m;
            buffers.hmm_theta_log_prior[flat] = kappa_prior * cache.state_angles[m].cos();
            buffers.hmm_slot_logz[flat] = if obs.is_data {
                log_sum_exp_slice(&joint) + buffers.hmm_theta_log_prior[flat]
            } else {
                joint[joint_hypothesis_index(PAYLOAD_PILOT_WALSH_INDEX, pilot_sym_idx)]
                    + buffers.hmm_theta_log_prior[flat]
            };
        }
    }

    let log_pi = -(m_len as f32).ln();
    for m in 0..m_len {
        buffers.hmm_pred[m] = log_pi;
        buffers.hmm_alpha[m] = log_pi + buffers.hmm_slot_logz[m];
    }
    normalize_log_row(&mut buffers.hmm_alpha[..m_len]);

    for t in 1..t_len {
        let kappa_trans = transition_kappa_from_observations(&slot_obs[t - 1], &slot_obs[t]);
        for curr in 0..m_len {
            let mut v = f32::NEG_INFINITY;
            for prev in 0..m_len {
                v = log_add_exp(
                    v,
                    buffers.hmm_alpha[(t - 1) * m_len + prev]
                        + kappa_trans * cache.transition_cos[prev * m_len + curr],
                );
            }
            buffers.hmm_pred[t * m_len + curr] = v;
            buffers.hmm_alpha[t * m_len + curr] = v + buffers.hmm_slot_logz[t * m_len + curr];
        }
        normalize_log_row(&mut buffers.hmm_alpha[t * m_len..(t + 1) * m_len]);
    }

    if t_len >= 2 {
        for t in (0..(t_len - 1)).rev() {
            let kappa_trans = transition_kappa_from_observations(&slot_obs[t], &slot_obs[t + 1]);
            for prev in 0..m_len {
                let mut v = f32::NEG_INFINITY;
                for curr in 0..m_len {
                    v = log_add_exp(
                        v,
                        kappa_trans * cache.transition_cos[prev * m_len + curr]
                            + buffers.hmm_slot_logz[(t + 1) * m_len + curr]
                            + buffers.hmm_beta[(t + 1) * m_len + curr],
                    );
                }
                buffers.hmm_beta[t * m_len + prev] = v;
            }
            normalize_log_row(&mut buffers.hmm_beta[t * m_len..(t + 1) * m_len]);
        }
    }

    out_llrs.reserve(slot_obs.len());
    let mut prev_a = [f32::NEG_INFINITY; 4];
    // 初期位相はPLLによって同期語の最後のシンボル(絶対位相0)にロックされているため
    prev_a[0] = 0.0;

    for (t, obs) in slot_obs.iter().enumerate() {
        let mut joint_post = [f32::NEG_INFINITY; DQPSK_JOINT_HYPOTHESES];
        for m in 0..m_len {
            let base = buffers.hmm_pred[t * m_len + m]
                + buffers.hmm_beta[t * m_len + m]
                + buffers.hmm_theta_log_prior[t * m_len + m];
            let joint = joint_symbol_logs_for_phase_state(obs, cache.state_residual_rots[m]);
            for idx in 0..DQPSK_JOINT_HYPOTHESES {
                joint_post[idx] = log_add_exp(joint_post[idx], base + joint[idx]);
            }
        }

        let mut out_a = [f32::NEG_INFINITY; 4];
        let llrs = extract_differential_llrs_from_joint_post(&joint_post, &prev_a, &mut out_a);
        if obs.is_data {
            out_llrs.push(llrs);
        }
        prev_a = out_a;
    }
}

#[cfg(test)]
fn dqpsk_llrs_with_phase_hmm(
    slot_obs: &[PhaseSlotObservation],
    out_llrs: &mut Vec<[f32; 2]>,
    buffers: &mut PacketDecodeBuffers,
) {
    let mut sym_llrs = Vec::<[f32; 6]>::new();
    symbol_llrs_with_phase_hmm(slot_obs, &mut sym_llrs, buffers);
    out_llrs.clear();
    out_llrs.reserve(sym_llrs.len());
    for llr in sym_llrs {
        out_llrs.push([llr[4], llr[5]]);
    }
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
    buffers.phase_slot_observations.clear();
    let mut avg_h_norm = prev_phase.norm().max(1.0);
    // 初期ノイズは SNR=10dB 程度と仮定して初期化するが、即座に適応させる。
    let mut avg_noise_var = (avg_h_norm * avg_h_norm * 0.1).max(DQPSK_NOISE_VAR_FLOOR);
    let mut total_packet_energy = 0.0f32;
    let mut prev_phase_sigma2 = 0.0f32;
    let total_symbol_slots = payload_total_symbols(expected_symbols);
    buffers.phase_slot_observations.reserve(total_symbol_slots);

    for slot_idx in 0..total_symbol_slots {
        let is_data_slot = payload_data_index_for_symbol_slot(slot_idx).is_some();
        let symbol_start = options.spc + slot_idx * PAYLOAD_SPREAD_FACTOR * options.spc;

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

        let energies_on = on_corrs.map(|corr| corr.norm_sqr());
        let best_idx = if is_data_slot {
            let mut best = 0usize;
            let mut best_energy = energies_on[0];
            for (idx, &energy) in energies_on.iter().enumerate().skip(1) {
                if energy > best_energy {
                    best = idx;
                    best_energy = energy;
                }
            }
            best
        } else {
            // パイロットは送信側で Walsh[0] 固定なので、既知系列を基準に追従更新する。
            PAYLOAD_PILOT_WALSH_INDEX
        };
        let max_energy = energies_on[best_idx];
        let mut second_energy = 0.0f32;
        for (idx, &energy) in energies_on.iter().enumerate() {
            if idx != best_idx && energy > second_energy {
                second_energy = energy;
            }
        }
        if is_data_slot {
            total_packet_energy += max_energy;
        }

        let prev_phase_kappa = dqpsk_prev_phase_kappa_from_sigma2(prev_phase_sigma2);
        let energy_sum = energies_on.iter().sum::<f32>();
        let _snr_instant = max_energy / (((energy_sum - max_energy).max(0.0) / 15.0) + 1e-6);

        let (
            walsh_conf,
            on_rot_tracking,
            phase_err,
            dqpsk_conf_tracking,
        ) = if is_data_slot {
                let on_rot = on_corrs[best_idx] * tracking_state.phase_ref.conj();
                let diff_best = on_rot * prev_phase.conj();
                let snr_proxy_val = (avg_h_norm * avg_h_norm) / avg_noise_var.max(DQPSK_NOISE_VAR_FLOOR);
                let (dqpsk_llr, walsh_conf, _topk_used, energies) = dqpsk_llr_from_walsh_hypotheses(
                    &on_corrs,
                    tracking_state.phase_ref,
                    *prev_phase,
                    prev_phase_kappa,
                    max_energy,
                    second_energy,
                    snr_proxy_val,
                    avg_noise_var,
                    on_rot,
                    diff_best,
                    Complex32::new(1.0, 0.0),
                );
                let on_rot_tracking = walsh_posterior_weighted_on_rot(
                    &on_corrs,
                    tracking_state.phase_ref,
                    &energies,
                    max_energy,
                    on_rot,
                );
                let diff_tracking = on_rot_tracking * prev_phase.conj();
                let dqpsk_conf = dqpsk_llr[0].abs() + dqpsk_llr[1].abs();
                let dqpsk_conf_tracking = dqpsk_conf * walsh_conf.clamp(0.0, 1.0);
                let decided = dqpsk_symbol_from_bits_pair(
                    if dqpsk_llr[0] > 0.0 { 0 } else { 1 },
                    if dqpsk_llr[1] > 0.0 { 0 } else { 1 },
                );
                let phase_err = tracking::phase_error_from_diff(diff_tracking, decided);
                (
                    walsh_conf,
                    on_rot_tracking,
                    phase_err,
                    dqpsk_conf_tracking,
                )
            } else {
                let pilot_on_rot = on_corrs[PAYLOAD_PILOT_WALSH_INDEX] * tracking_state.phase_ref.conj();
                let pilot_diff = pilot_on_rot * prev_phase.conj();
                let pilot_decided = dqpsk_symbol_from_bits_pair(
                    PAYLOAD_PILOT_DQPSK_BITS.0,
                    PAYLOAD_PILOT_DQPSK_BITS.1,
                );
                let phase_err = tracking::phase_error_from_diff(pilot_diff, pilot_decided);
                let pilot_energy = energies_on[PAYLOAD_PILOT_WALSH_INDEX];
                let walsh_conf = ((pilot_energy - second_energy).max(0.0)) / pilot_energy.max(1e-6);
                let dqpsk_conf_tracking = TRACKING_PHASE_DQPSK_CONF_ON_MIN;
                (
                    walsh_conf,
                    pilot_on_rot,
                    phase_err,
                    dqpsk_conf_tracking,
                )
            };

        let should_update_stats = if !is_data_slot {
            true
        } else {
            walsh_conf >= 0.75
        };
        if should_update_stats {
            let h_instant = max_energy.sqrt();
            avg_h_norm = 0.8 * avg_h_norm + 0.2 * h_instant;
            let noise_instant = ((energy_sum - max_energy).max(0.0)) / 15.0;
            avg_noise_var = 0.8 * avg_noise_var + 0.2 * noise_instant;
        }

        let snr_proxy = (avg_h_norm * avg_h_norm) / avg_noise_var.max(DQPSK_NOISE_VAR_FLOOR);
        tracking_state.phase_gate_enabled = tracking::next_phase_gate_enabled(
            tracking_state.phase_gate_enabled,
            dqpsk_conf_tracking,
            walsh_conf,
            snr_proxy,
        );
        if is_data_slot {
            if tracking_state.phase_gate_enabled {
                stats.phase_gate_on_symbols += 1;
            } else {
                stats.phase_gate_off_symbols += 1;
            }
        }

        let phase_err_abs = f32::abs(phase_err);
        buffers.phase_slot_observations.push(PhaseSlotObservation {
            on_corrs,
            energies: energies_on,
            energy_sum,
            phase_ref_conj: tracking_state.phase_ref.conj(),
            max_energy,
            prev_phase_kappa,
            snr_proxy,
            avg_noise_var,
            is_data: is_data_slot,
        });
        if is_data_slot {
            stats.phase_err_abs_sum_rad += phase_err_abs as f64;
            stats.phase_err_abs_count += 1;
            if phase_err_abs >= PHASE_ERR_ABS_THRESH_0P5_RAD {
                stats.phase_err_abs_ge_0p5_symbols += 1;
            }
            if phase_err_abs >= PHASE_ERR_ABS_THRESH_1P0_RAD {
                stats.phase_err_abs_ge_1p0_symbols += 1;
            }
        }
        let phase_var_sample =
            (phase_err_abs * phase_err_abs).min(DQPSK_PREV_PHASE_SIGMA_MAX_RAD.powi(2));
        prev_phase_sigma2 = DQPSK_PREV_PHASE_VAR_EWMA_ALPHA * prev_phase_sigma2
            + (1.0 - DQPSK_PREV_PHASE_VAR_EWMA_ALPHA) * phase_var_sample;
        let phase_step = if is_data_slot {
            let innovation_rejected = tracking_state.phase_gate_enabled
                && phase_err_abs > TRACKING_PHASE_ERR_GATE_RAD
                && dqpsk_conf_tracking < TRACKING_PHASE_ERR_GATE_DQPSK_CONF_HIGH;
            let phase_rate_update_enabled = tracking_state.phase_gate_enabled
                && tracking::phase_rate_update_enabled(dqpsk_conf_tracking, walsh_conf, snr_proxy);
            if innovation_rejected {
                stats.phase_innovation_reject_symbols += 1;
            }
            if tracking_state.phase_gate_enabled {
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
                tracking::phase_step_from_phase_error(
                    phase_err_for_update,
                    tracking_state.phase_rate,
                )
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
            }
        } else {
            // 既知パイロットは低ゲインで phase_rate を直接補正する。
            // データ復調の判定誤差を介さず、長尺時の残留位相傾きを前向きに抑える。
            let pilot_update_enabled =
                walsh_conf >= PILOT_PHASE_WALSH_CONF_MIN && snr_proxy >= PILOT_PHASE_SNR_PROXY_MIN;
            let pilot_err = if pilot_update_enabled {
                phase_err.clamp(-PILOT_PHASE_ERR_CLAMP, PILOT_PHASE_ERR_CLAMP)
            } else {
                0.0
            };
            if pilot_update_enabled {
                tracking_state.phase_rate =
                    (tracking_state.phase_rate + PILOT_PHASE_RATE_GAIN * pilot_err).clamp(
                        -tracking::TRACKING_PHASE_RATE_LIMIT_RAD,
                        tracking::TRACKING_PHASE_RATE_LIMIT_RAD,
                    );
            } else {
                tracking_state.phase_rate =
                    (tracking_state.phase_rate * TRACKING_PHASE_RATE_HOLD_DECAY).clamp(
                        -tracking::TRACKING_PHASE_RATE_LIMIT_RAD,
                        tracking::TRACKING_PHASE_RATE_LIMIT_RAD,
                    );
            }
            (tracking_state.phase_rate + PILOT_PHASE_PROP_GAIN * pilot_err)
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

        let on_norm = Complex32::norm(on_rot_tracking).max(1e-6);
        *prev_phase = on_rot_tracking / on_norm;
    }

    // 準コヒーレント復調:
    // 位相を離散状態 θ_n の隠れマルコフ過程として forward/backward で周辺化し、
    // pilot(既知シンボル)を尤度拘束として取り込む。
    let mut phase_slot_observations = std::mem::take(&mut buffers.phase_slot_observations);
    let mut symbol_llrs_tmp = std::mem::take(&mut buffers.hmm_symbol_llrs);
    symbol_llrs_with_phase_hmm(&phase_slot_observations, &mut symbol_llrs_tmp, buffers);
    let mut sym_idx = 0usize;
    for obs in phase_slot_observations.iter() {
        if !obs.is_data {
            continue;
        }
        let walsh_llr = demodulator.walsh_llr(&obs.energies, obs.max_energy);
        buffers.packet_llrs_buffer.extend_from_slice(&walsh_llr);
        
        let llr6 = symbol_llrs_tmp[sym_idx];
        buffers.packet_llrs_buffer.push(llr6[4] * DQPSK_OUTPUT_LLR_REL_GAIN);
        buffers.packet_llrs_buffer.push(llr6[5] * DQPSK_OUTPUT_LLR_REL_GAIN);
        sym_idx += 1;
    }
    debug_assert_eq!(sym_idx, symbol_llrs_tmp.len());
    symbol_llrs_tmp.clear();
    buffers.hmm_symbol_llrs = symbol_llrs_tmp;
    phase_slot_observations.clear();
    buffers.phase_slot_observations = phase_slot_observations;

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
        // NOTE: DQPSK 2bit限定 erasure も評価したが、
        // 2026-03-15のAWGN sweep (sigma=0.8..1.2) では改善せず、
        // sigma=1.2で crc_pass/goodput がわずかに悪化したため不採用とした。
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
    } else {
        Err(PacketDecodeError::Parse)
    }
}

#[inline]
fn dqpsk_symbol_from_bits_pair(b0: u8, b1: u8) -> Complex32 {
    match (b0, b1) {
        (0, 0) => Complex32::new(1.0, 0.0),
        (0, 1) => Complex32::new(0.0, 1.0),
        (1, 1) => Complex32::new(-1.0, 0.0),
        (1, 0) => Complex32::new(0.0, -1.0),
        _ => Complex32::new(1.0, 0.0),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coding::fec;
    use crate::coding::interleaver::BlockInterleaver;
    use crate::coding::scrambler::Scrambler;
    use crate::frame::packet::{Packet, PACKET_BYTES};
    use crate::mary::demodulator::Demodulator;
    use crate::mary::interleaver_config;
    use crate::mary::params;
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
        let data_symbols = bits.len().div_ceil(6);
        let mut out = Vec::with_capacity(params::payload_total_symbols(data_symbols));
        for (sym_idx, chunk) in bits.chunks_exact(6).enumerate() {
            let data_walsh_idx = chunk[..4]
                .iter()
                .fold(0usize, |acc, &bit| (acc << 1) | bit as usize);
            let data_diff = dqpsk_symbol_from_bits(&chunk[4..6]);
            let data_current = prev * data_diff;
            prev = data_current;
            let mut data_corrs = [Complex32::new(0.0, 0.0); 16];
            data_corrs[data_walsh_idx] = data_current * amplitude;
            out.push(data_corrs);

            if params::PAYLOAD_PILOT_INTERVAL_SYMBOLS > 0
                && (sym_idx + 1) % params::PAYLOAD_PILOT_INTERVAL_SYMBOLS == 0
                && (sym_idx + 1) < data_symbols
            {
                let pilot_diff = dqpsk_symbol_from_bits(&[
                    params::PAYLOAD_PILOT_DQPSK_BITS.0,
                    params::PAYLOAD_PILOT_DQPSK_BITS.1,
                ]);
                let pilot_current = prev * pilot_diff;
                prev = pilot_current;
                let mut pilot_corrs = [Complex32::new(0.0, 0.0); 16];
                pilot_corrs[params::PAYLOAD_PILOT_WALSH_INDEX] = pilot_current * amplitude;
                out.push(pilot_corrs);
            }
        }
        out
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
            1000.0, // snr_proxy (10.0^2 / 0.1)
            0.1,  // avg_noise_var
            on_rot_best,
            diff_best,
            Complex32::new(1.0, 0.0),
        );
        let energies: [f32; 16] = on_corrs.map(|c| c.norm_sqr());
        let energy_sum = energies.iter().sum::<f32>();
        let _noise_var_best = dqpsk_noise_var_from_energy(energy_sum, energies[0]);
        let phase_logs =
            dqpsk_phase_log_metrics(diff_best, 100.0, 1.6, 1.0);
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
            1000.0, // snr_proxy (10.0^2 / 0.1)
            0.1,  // avg_noise_var
            on_rot_best,
            diff_best,
            Complex32::new(1.0, 0.0),
        );

        let energies: [f32; 16] = on_corrs.map(|c| c.norm_sqr());
        let avg_h_norm = 10.0;
        let avg_noise_var = 0.1;
        let snr_proxy = (avg_h_norm * avg_h_norm) / avg_noise_var;
        let walsh_temp_scale = walsh_temperature_scale(1.0, snr_proxy);
        let mut sym_logs = [f32::NEG_INFINITY; 4];
        for idx in 0..DQPSK_WALSH_HYPOTHESES {
            let energy_h = energies[idx];
            let on_rot_h = on_corrs[idx] * phase_ref.conj();
            let diff_h = on_rot_h * prev_phase.conj();
            let m = dqpsk_phase_log_metrics(diff_h, energy_h, avg_noise_var, 1.0);
            let log_p_h = (energy_h - 100.0) / (avg_noise_var * walsh_temp_scale);
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
    fn test_dqpsk_llrs_with_phase_hmm_single_slot_is_finite() {
        let mut on_corrs = [Complex32::new(0.0, 0.0); 16];
        on_corrs[0] = Complex32::new(10.0, 0.0);
        on_corrs[1] = Complex32::new(1.0, 0.2);
        let energies = on_corrs.map(|c| c.norm_sqr());
        let energy_sum = energies.iter().sum::<f32>();
        let slots = [PhaseSlotObservation {
            on_corrs,
            energies,
            energy_sum,
            phase_ref_conj: Complex32::new(1.0, 0.0).conj(),
            max_energy: on_corrs[0].norm_sqr(),
            prev_phase_kappa: 1.0,
            snr_proxy: 20.0,
            avg_noise_var: 0.1, is_data: true,
        }];
        let mut buffers = PacketDecodeBuffers::new();
        let mut llrs = Vec::new();
        dqpsk_llrs_with_phase_hmm(&slots, &mut llrs, &mut buffers);
        assert_eq!(llrs.len(), 1);
        assert!(llrs[0][0].is_finite());
        assert!(llrs[0][1].is_finite());
    }

    #[test]
    fn test_dqpsk_llrs_with_phase_hmm_uses_pilot_constraint() {
        let omega = 0.45f32;
        let (sin_o, cos_o) = omega.sin_cos();
        let rot = Complex32::new(cos_o, sin_o);
        let amp = 10.0f32;

        let mut pilot_corrs = [Complex32::new(0.0, 0.0); 16];
        pilot_corrs[PAYLOAD_PILOT_WALSH_INDEX] = rot * amp;
        let pilot_energies = pilot_corrs.map(|c| c.norm_sqr());
        let pilot_energy_sum = pilot_energies.iter().sum::<f32>();

        let mut data_corrs = [Complex32::new(0.0, 0.0); 16];
        data_corrs[0] = rot * amp;
        let data_energies = data_corrs.map(|c| c.norm_sqr());
        let data_energy_sum = data_energies.iter().sum::<f32>();

        let slots = [
            PhaseSlotObservation {
                on_corrs: pilot_corrs,
                energies: pilot_energies,
                energy_sum: pilot_energy_sum,
                phase_ref_conj: Complex32::new(1.0, 0.0).conj(),
                max_energy: pilot_corrs[PAYLOAD_PILOT_WALSH_INDEX].norm_sqr(),
                prev_phase_kappa: 1.0,
                snr_proxy: 20.0,
                avg_noise_var: 0.1, is_data: false,
            },
            PhaseSlotObservation {
                on_corrs: data_corrs,
                energies: data_energies,
                energy_sum: data_energy_sum,
                phase_ref_conj: Complex32::new(1.0, 0.0).conj(),
                max_energy: data_corrs[0].norm_sqr(),
                prev_phase_kappa: 1.0,
                snr_proxy: 20.0,
                avg_noise_var: 0.1, is_data: true,
            },
        ];
        let mut buffers = PacketDecodeBuffers::new();
        let mut llrs = Vec::new();
        dqpsk_llrs_with_phase_hmm(&slots, &mut llrs, &mut buffers);
        assert_eq!(llrs.len(), 1);
        // data symbol=00 なので両bitとも 0 側に寄る。
        assert!(llrs[0][0] > 0.0);
        assert!(llrs[0][1] > 0.0);
    }

    #[test]
    fn test_symbol_llrs_with_phase_hmm_single_slot_signs() {
        let mut on_corrs = [Complex32::new(0.0, 0.0); 16];
        on_corrs[0] = Complex32::new(10.0, 0.0); // Walsh[0], DQPSK=00
        let energies = on_corrs.map(|c| c.norm_sqr());
        let energy_sum = energies.iter().sum::<f32>();
        let slots = [PhaseSlotObservation {
            on_corrs,
            energies,
            energy_sum,
            phase_ref_conj: Complex32::new(1.0, 0.0),
            max_energy: energies[0],
            prev_phase_kappa: 1.0,
            snr_proxy: 20.0,
            avg_noise_var: 0.1, is_data: true,
        }];
        let mut buffers = PacketDecodeBuffers::new();
        let mut llrs = Vec::new();
        symbol_llrs_with_phase_hmm(&slots, &mut llrs, &mut buffers);
        assert_eq!(llrs.len(), 1);
        for (bit, &val) in llrs[0].iter().enumerate().take(6) {
            assert!(val > 0.0, "bit{} llr={}", bit, val);
        }
    }

    #[test]
    fn test_symbol_llrs_with_phase_hmm_walsh_bit_order() {
        for walsh_idx in 0..16 {
            let mut on_corrs = [Complex32::new(0.0, 0.0); 16];
            on_corrs[walsh_idx] = Complex32::new(10.0, 0.0); // DQPSK=00
            let energies = on_corrs.map(|c| c.norm_sqr());
            let energy_sum = energies.iter().sum::<f32>();
            let slots = [PhaseSlotObservation {
                on_corrs,
                energies,
                energy_sum,
                phase_ref_conj: Complex32::new(1.0, 0.0),
                max_energy: energies[walsh_idx],
                prev_phase_kappa: 1.0,
                snr_proxy: 20.0,
                avg_noise_var: 0.1, is_data: true,
            }];
            let mut buffers = PacketDecodeBuffers::new();
            let mut llrs = Vec::new();
            symbol_llrs_with_phase_hmm(&slots, &mut llrs, &mut buffers);
            assert_eq!(llrs.len(), 1);
            for (out_idx, &val) in llrs[0].iter().enumerate().take(4) {
                let bit = 3 - out_idx; // out[0]=bit3(MSB) ... out[3]=bit0
                let expected_zero = ((walsh_idx >> bit) & 1) == 0;
                assert_eq!(
                    val > 0.0,
                    expected_zero,
                    "walsh_idx={} out_idx={} llr={}",
                    walsh_idx,
                    out_idx,
                    val
                );
            }
            assert!(llrs[0][4] > 0.0, "dqpsk bit0 llr={}", llrs[0][4]);
            assert!(llrs[0][5] > 0.0, "dqpsk bit1 llr={}", llrs[0][5]);
        }
    }

    #[test]
    fn test_symbol_llrs_with_phase_hmm_dqpsk_bit_mapping() {
        let cases = [
            (Complex32::new(10.0, 0.0), true, true),   // 00
            (Complex32::new(0.0, 10.0), true, false),  // 01
            (Complex32::new(-10.0, 0.0), false, false), // 11
            (Complex32::new(0.0, -10.0), false, true), // 10
        ];
        for (sym, bit0_zero, bit1_zero) in cases {
            let mut on_corrs = [Complex32::new(0.0, 0.0); 16];
            on_corrs[0] = sym;
            let energies = on_corrs.map(|c| c.norm_sqr());
            let energy_sum = energies.iter().sum::<f32>();
            let slots = [PhaseSlotObservation {
                on_corrs,
                energies,
                energy_sum,
                phase_ref_conj: Complex32::new(1.0, 0.0),
                max_energy: energies[0],
                prev_phase_kappa: 1.0,
                snr_proxy: 20.0,
                avg_noise_var: 0.1, is_data: true,
            }];
            let mut buffers = PacketDecodeBuffers::new();
            let mut llrs = Vec::new();
            symbol_llrs_with_phase_hmm(&slots, &mut llrs, &mut buffers);
            assert_eq!(llrs.len(), 1);
            assert_eq!(llrs[0][4] > 0.0, bit0_zero, "dqpsk bit0 llr={}", llrs[0][4]);
            assert_eq!(llrs[0][5] > 0.0, bit1_zero, "dqpsk bit1 llr={}", llrs[0][5]);
        }
    }

    #[test]
    fn test_hmm_symbol_logs_match_single_hypothesis_phase_model() {
        let mut on_corrs = [Complex32::new(0.0, 0.0); 16];
        on_corrs[PAYLOAD_PILOT_WALSH_INDEX] = Complex32::new(3.0, 4.0);
        let energies = on_corrs.map(|c| c.norm_sqr());
        let energy_sum = energies.iter().sum::<f32>();
        let obs = PhaseSlotObservation {
            on_corrs,
            energies,
            energy_sum,
            phase_ref_conj: Complex32::new(1.0, 0.0),
            max_energy: energies[PAYLOAD_PILOT_WALSH_INDEX],
            prev_phase_kappa: 0.8,
            snr_proxy: 20.0,
            avg_noise_var: 0.1, is_data: false,
        };
        let actual = dqpsk_symbol_logs_for_phase_state(&obs, Complex32::new(1.0, 0.0));
        let noise_var = dqpsk_noise_var_from_energy(
            obs.energy_sum,
            obs.energies[PAYLOAD_PILOT_WALSH_INDEX],
        );
        let expected = dqpsk_phase_log_metrics(
            on_corrs[PAYLOAD_PILOT_WALSH_INDEX],
            obs.energies[PAYLOAD_PILOT_WALSH_INDEX],
            noise_var,
            obs.prev_phase_kappa,
        );
        for i in 0..4 {
            assert_close(actual[i], expected[i], 1e-5, "sym_log");
        }
    }

    #[test]
    fn test_transition_kappa_respects_snr_and_phase_uncertainty() {
        let base = PhaseSlotObservation {
            on_corrs: [Complex32::new(0.0, 0.0); 16],
            energies: [0.0; 16],
            energy_sum: 0.0,
            phase_ref_conj: Complex32::new(1.0, 0.0),
            max_energy: 1.0,
            prev_phase_kappa: 1.0,
            snr_proxy: 20.0,
            avg_noise_var: 0.1, is_data: true,
        };
        let mut noisy = base;
        noisy.prev_phase_kappa = 0.2;
        noisy.snr_proxy = 1.0;
        let kappa_good = transition_kappa_from_observations(&base, &base);
        let kappa_noisy = transition_kappa_from_observations(&noisy, &noisy);
        assert!(kappa_good > kappa_noisy);
    }

    #[test]
    fn test_observation_model_var_respects_snr_and_phase_uncertainty() {
        let base = PhaseSlotObservation {
            on_corrs: [Complex32::new(0.0, 0.0); 16],
            energies: [0.0; 16],
            energy_sum: 0.0,
            phase_ref_conj: Complex32::new(1.0, 0.0),
            max_energy: 1.0,
            prev_phase_kappa: 1.0,
            snr_proxy: 20.0,
            avg_noise_var: 0.1, is_data: true,
        };
        let mut noisy = base;
        noisy.prev_phase_kappa = 0.2;
        noisy.snr_proxy = 1.0;
        let sigma2_good = observation_model_var_from_observation(&base);
        let sigma2_noisy = observation_model_var_from_observation(&noisy);
        assert!(sigma2_noisy > sigma2_good);
    }

    #[test]
    fn test_walsh_temperature_scale_respects_snr_and_phase_uncertainty() {
        let scale_good = walsh_temperature_scale(1.0, 20.0);
        let scale_noisy = walsh_temperature_scale(0.2, 1.0);
        assert!(scale_noisy > scale_good);
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

    #[test]
    fn test_apply_llr_erasure_quantile_zeroes_small_abs_values() {
        let mut llrs = vec![0.1, -0.2, 0.5, -1.0, 2.0];
        apply_llr_erasure_quantile(&mut llrs, 0.4);
        assert_eq!(llrs, vec![0.0, 0.0, 0.5, -1.0, 2.0]);
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
        let captured_llrs = Arc::new(Mutex::new(Vec::<f32>::new()));
        let captured_llrs_cloned = Arc::clone(&captured_llrs);
        let mut llr_callback: Option<LlrCallback> = Some(Box::new(move |values| {
            let mut guard = captured_llrs_cloned.lock().unwrap();
            guard.clear();
            guard.extend_from_slice(values);
        }));
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
                let slot_idx = (symbol_start - options.spc) / (PAYLOAD_SPREAD_FACTOR * options.spc);
                symbol_corrs.get(slot_idx).copied()
            },
        );

        assert!(result.processed);
        if result.packet.is_none() {
            let observed = captured_llrs.lock().unwrap().clone();
            let expected = encode_packet_to_descrambled_llrs(&packet, 8.0);
            let compare_len = observed.len().min(expected.len());
            let mut sign_mismatch = 0usize;
            let mut non_finite = 0usize;
            let mut max_abs = 0.0f32;
            for i in 0..compare_len {
                if (observed[i] >= 0.0) != (expected[i] >= 0.0) {
                    sign_mismatch += 1;
                }
                if !observed[i].is_finite() {
                    non_finite += 1;
                }
                max_abs = max_abs.max(observed[i].abs());
            }
            panic!(
                "decode failed: captured_len={} expected_len={} sign_mismatch={} non_finite={} max_abs={}",
                observed.len(),
                expected.len(),
                sign_mismatch,
                non_finite,
                max_abs
            );
        }
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
