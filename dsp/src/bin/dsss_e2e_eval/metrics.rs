use crate::utils::{PostFecStats, PreFecStats};

/// 評価シミュレーションの集計メトリクス（兼 実行状態）
#[derive(Default, Clone, Debug)]
pub struct Metrics {
    pub total_sim_sec: f32,
    pub total_frame_attempts: usize,
    pub packets_per_frame: usize,
    pub total_synced_frames: usize,
    pub total_accepted_packets: usize,
    pub total_crc_error_packets: usize,
    pub total_successes: usize,
    pub total_bit_errors: usize,
    pub total_bits_compared: usize,
    pub dropped_frames: usize,
    pub total_tx_signal_energy: f64,
    pub total_tx_signal_samples: usize,
    pub total_process_time_ns: u64,
    pub completion_secs: Vec<f32>,
    pub total_raw_bit_errors: usize,
    pub total_raw_bits_compared: usize,
    pub total_raw_walsh_bit_errors: usize,
    pub total_raw_walsh_bits_compared: usize,
    pub total_raw_dqpsk_bit_errors: usize,
    pub total_raw_dqpsk_bits_compared: usize,
    pub total_raw_dqpsk_walsh_weak_bit_errors: usize,
    pub total_raw_dqpsk_walsh_weak_bits_compared: usize,
    pub total_raw_dqpsk_walsh_strong_bit_errors: usize,
    pub total_raw_dqpsk_walsh_strong_bits_compared: usize,
    pub total_raw_error_runs: usize,
    pub total_raw_error_run_bits: usize,
    pub max_raw_error_run_len: usize,
    pub total_codewords: usize,
    pub total_codeword_error_sum: usize,
    pub max_codeword_error: usize,
    pub codeword_error_weights: Vec<usize>,
    pub total_post_bit_errors: usize,
    pub total_post_bits_compared: usize,
    pub total_post_error_runs: usize,
    pub total_post_error_run_bits: usize,
    pub max_post_error_run_len: usize,
    pub total_post_codewords: usize,
    pub total_post_codeword_error_sum: usize,
    pub max_post_codeword_error: usize,
    pub post_codeword_error_weights: Vec<usize>,
    pub total_post_decode_attempts: usize,
    pub total_post_decode_matched: usize,
    pub total_viterbi_packet_decode_attempts: usize,
    pub total_viterbi_crc_candidate_checks: usize,
    pub total_false_accepted_packets: usize,
    pub sum_last_est_snr_db: f64,
    pub count_last_est_snr_db: usize,
    pub total_phase_gate_on_symbols: usize,
    pub total_phase_gate_off_symbols: usize,
    pub total_phase_innovation_reject_symbols: usize,
    pub total_phase_err_abs_sum_rad: f64,
    pub total_phase_err_abs_count: usize,
    pub total_phase_err_abs_ge_0p5_symbols: usize,
    pub total_phase_err_abs_ge_1p0_symbols: usize,
    pub total_llr_second_pass_attempts: usize,
    pub total_llr_second_pass_rescued: usize,
}

pub struct MetricContext<'a> {
    pub scenario: &'a str,
    pub phy: &'a str,
    pub mary_fde_mode: &'a str,
    pub payload_bits: usize,
    pub sigma: f32,
    pub multipath_name: &'a str,
    pub sample_rate: f32,
    pub chip_rate: f32,
    pub bit_rate: f32,
}

#[derive(Debug, Clone)]
pub enum MetricValue {
    Float(f32),
    Int(usize),
    Text(String),
    Null,
}

impl From<f32> for MetricValue {
    fn from(v: f32) -> Self {
        Self::Float(v)
    }
}

impl From<usize> for MetricValue {
    fn from(v: usize) -> Self {
        Self::Int(v)
    }
}

impl From<String> for MetricValue {
    fn from(v: String) -> Self {
        Self::Text(v)
    }
}

impl From<&str> for MetricValue {
    fn from(v: &str) -> Self {
        Self::Text(v.to_string())
    }
}

impl<T: Into<MetricValue>> From<Option<T>> for MetricValue {
    fn from(v: Option<T>) -> Self {
        match v {
            Some(x) => x.into(),
            None => Self::Null,
        }
    }
}

pub struct MetricDef {
    pub id: &'static str,
    pub description: &'static str,
    pub extractor: fn(&MetricContext, &Metrics) -> MetricValue,
}

#[macro_export]
macro_rules! define_metrics {
    ($($id:ident, $desc:expr, $extractor:expr $(, $default:ident)? ;)*) => {
        pub const METRICS_DEFS: &[$crate::metrics::MetricDef] = &[
            $(
                $crate::metrics::MetricDef {
                    id: stringify!($id),
                    description: $desc,
                    extractor: $extractor,
                },
            )*
        ];

        pub const ALL_COLUMNS: &[&str] = &[
            $( stringify!($id) ),*
        ];

        pub const DEFAULT_COLUMNS: &[&str] = &[
            $(
                $(
                    $crate::define_metrics!(@id $id $default),
                )?
            )*
        ];

        pub fn get_default_columns() -> Vec<&'static str> {
            DEFAULT_COLUMNS.to_vec()
        }
    };

    (@id $id:ident default) => {
        stringify!($id)
    };
}

impl Metrics {
    pub fn new(packets_per_frame: usize) -> Self {
        Self {
            packets_per_frame,
            ..Self::default()
        }
    }

    pub fn add_frame_event(
        &mut self,
        synced: usize,
        accepted_packets: usize,
        crc_error_packets: usize,
    ) {
        self.total_synced_frames += synced;
        self.total_accepted_packets += accepted_packets;
        self.total_crc_error_packets += crc_error_packets;
    }

    pub fn add_recovery_event(
        &mut self,
        completion_sec: f32,
        bit_errors: usize,
        bits_compared: usize,
    ) {
        self.completion_secs.push(completion_sec);
        self.total_successes += 1;
        self.total_bit_errors += bit_errors;
        self.total_bits_compared += bits_compared;
    }

    pub fn set_mary_raw_ber(&mut self, stats: PreFecStats) {
        self.total_raw_bit_errors = stats.bit_errors;
        self.total_raw_bits_compared = stats.bits_compared;
        self.total_raw_walsh_bit_errors = stats.walsh_bit_errors;
        self.total_raw_walsh_bits_compared = stats.walsh_bits_compared;
        self.total_raw_dqpsk_bit_errors = stats.dqpsk_bit_errors;
        self.total_raw_dqpsk_bits_compared = stats.dqpsk_bits_compared;
        self.total_raw_dqpsk_walsh_weak_bit_errors = stats.dqpsk_walsh_weak_bit_errors;
        self.total_raw_dqpsk_walsh_weak_bits_compared = stats.dqpsk_walsh_weak_bits_compared;
        self.total_raw_dqpsk_walsh_strong_bit_errors = stats.dqpsk_walsh_strong_bit_errors;
        self.total_raw_dqpsk_walsh_strong_bits_compared = stats.dqpsk_walsh_strong_bits_compared;
        self.total_raw_error_runs = stats.error_runs;
        self.total_raw_error_run_bits = stats.error_run_bits;
        self.max_raw_error_run_len = stats.error_run_max;
        self.total_codewords = stats.codeword_count;
        self.total_codeword_error_sum = stats.codeword_error_sum;
        self.max_codeword_error = stats.codeword_error_max;
        self.codeword_error_weights = stats.codeword_error_weights;
    }

    pub fn set_mary_post_ber(&mut self, stats: PostFecStats) {
        self.total_post_bit_errors = stats.bit_errors;
        self.total_post_bits_compared = stats.bits_compared;
        self.total_post_error_runs = stats.error_runs;
        self.total_post_error_run_bits = stats.error_run_bits;
        self.max_post_error_run_len = stats.error_run_max;
        self.total_post_codewords = stats.codeword_count;
        self.total_post_codeword_error_sum = stats.codeword_error_sum;
        self.max_post_codeword_error = stats.codeword_error_max;
        self.post_codeword_error_weights = stats.codeword_error_weights;
    }

    pub fn add_mary_phase(&mut self, stats: PhaseStats) {
        if stats.last_est_snr_db.is_finite() {
            self.sum_last_est_snr_db += stats.last_est_snr_db as f64;
            self.count_last_est_snr_db += 1;
        }
        self.total_phase_gate_on_symbols += stats.phase_gate_on_symbols;
        self.total_phase_gate_off_symbols += stats.phase_gate_off_symbols;
        self.total_phase_innovation_reject_symbols += stats.phase_innovation_reject_symbols;
        self.total_phase_err_abs_sum_rad += stats.phase_err_abs_sum_rad;
        self.total_phase_err_abs_count += stats.phase_err_abs_count;
        self.total_phase_err_abs_ge_0p5_symbols += stats.phase_err_abs_ge_0p5_symbols;
        self.total_phase_err_abs_ge_1p0_symbols += stats.phase_err_abs_ge_1p0_symbols;
    }

    pub fn add_mary_llr(&mut self, attempts: usize, rescued: usize) {
        self.total_llr_second_pass_attempts += attempts;
        self.total_llr_second_pass_rescued += rescued;
    }

    pub fn add_mary_decode_stats(&mut self, attempts: usize, matched: usize) {
        self.total_post_decode_attempts += attempts;
        self.total_post_decode_matched += matched;
    }

    pub fn add_mary_viterbi_stats(&mut self, packet_attempts: usize, crc_candidate_checks: usize) {
        self.total_viterbi_packet_decode_attempts += packet_attempts;
        self.total_viterbi_crc_candidate_checks += crc_candidate_checks;
    }

    pub fn add_false_accepted_packets(&mut self, false_accepted_packets: usize) {
        self.total_false_accepted_packets += false_accepted_packets;
    }

    pub fn packet_accept_ratio(&self) -> f32 {
        let total_packets_sent = self.total_frame_attempts * self.packets_per_frame;
        ratio(self.total_accepted_packets, total_packets_sent)
    }

    pub fn synced_frame_ratio(&self) -> f32 {
        ratio(self.total_synced_frames, self.total_frame_attempts)
    }

    pub fn crc_pass_ratio(&self) -> f32 {
        ratio(
            self.total_accepted_packets,
            self.total_accepted_packets + self.total_crc_error_packets,
        )
    }

    pub fn llr_second_pass_trigger_ratio(&self) -> f32 {
        ratio(
            self.total_llr_second_pass_attempts,
            self.total_accepted_packets + self.total_crc_error_packets,
        )
    }

    pub fn llr_second_pass_rescue_ratio(&self) -> f32 {
        ratio(
            self.total_llr_second_pass_rescued,
            self.total_llr_second_pass_attempts,
        )
    }

    pub fn viterbi_crc_candidates_mean(&self) -> f32 {
        ratio(
            self.total_viterbi_crc_candidate_checks,
            self.total_viterbi_packet_decode_attempts,
        )
    }

    pub fn false_accept_ratio_per_accepted_packet(&self) -> f32 {
        ratio(
            self.total_false_accepted_packets,
            self.total_accepted_packets,
        )
    }

    pub fn ber(&self) -> f32 {
        if self.total_bits_compared == 0 {
            0.0
        } else {
            self.total_bit_errors as f32 / self.total_bits_compared as f32
        }
    }

    pub fn mean_completion_sec(&self) -> Option<f32> {
        if self.completion_secs.is_empty() {
            None
        } else {
            Some(self.completion_secs.iter().sum::<f32>() / self.completion_secs.len() as f32)
        }
    }

    pub fn p95_completion_sec(&self) -> Option<f32> {
        quantile(&self.completion_secs, 0.95)
    }

    pub fn goodput_effective_bps(&self) -> f32 {
        if self.total_sim_sec <= 0.0 {
            0.0
        } else {
            let packet_payload_bits = (dsp::params::PAYLOAD_SIZE * 8) as f32;
            packet_payload_bits * self.total_accepted_packets as f32 / self.total_sim_sec
        }
    }

    pub fn goodput_success_mean_bps(&self, payload_bits: usize) -> Option<f32> {
        if self.completion_secs.is_empty() {
            return None;
        }
        let completion_sum_sec = self.completion_secs.iter().sum::<f32>();
        if completion_sum_sec <= 0.0 {
            return None;
        }
        Some((payload_bits * self.total_successes) as f32 / completion_sum_sec)
    }

    pub fn tx_signal_power(&self) -> Option<f32> {
        if self.total_tx_signal_samples == 0 {
            None
        } else {
            Some((self.total_tx_signal_energy / self.total_tx_signal_samples as f64) as f32)
        }
    }

    pub fn avg_process_time_per_sample_ns(&self) -> f32 {
        if self.total_tx_signal_samples == 0 {
            0.0
        } else {
            self.total_process_time_ns as f32 / self.total_tx_signal_samples as f32
        }
    }

    pub fn raw_ber(&self) -> f32 {
        if self.total_raw_bits_compared == 0 {
            f32::NAN
        } else {
            self.total_raw_bit_errors as f32 / self.total_raw_bits_compared as f32
        }
    }

    pub fn raw_ber_walsh(&self) -> f32 {
        if self.total_raw_walsh_bits_compared == 0 {
            f32::NAN
        } else {
            self.total_raw_walsh_bit_errors as f32 / self.total_raw_walsh_bits_compared as f32
        }
    }

    pub fn raw_ber_dqpsk(&self) -> f32 {
        if self.total_raw_dqpsk_bits_compared == 0 {
            f32::NAN
        } else {
            self.total_raw_dqpsk_bit_errors as f32 / self.total_raw_dqpsk_bits_compared as f32
        }
    }

    pub fn raw_ber_dqpsk_over_walsh(&self) -> f32 {
        let walsh = self.raw_ber_walsh();
        let dqpsk = self.raw_ber_dqpsk();
        if !walsh.is_finite() || !dqpsk.is_finite() || walsh <= 0.0 {
            f32::NAN
        } else {
            dqpsk / walsh
        }
    }

    pub fn raw_ber_dqpsk_walsh_weak(&self) -> f32 {
        if self.total_raw_dqpsk_walsh_weak_bits_compared == 0 {
            f32::NAN
        } else {
            self.total_raw_dqpsk_walsh_weak_bit_errors as f32
                / self.total_raw_dqpsk_walsh_weak_bits_compared as f32
        }
    }

    pub fn raw_ber_dqpsk_walsh_strong(&self) -> f32 {
        if self.total_raw_dqpsk_walsh_strong_bits_compared == 0 {
            f32::NAN
        } else {
            self.total_raw_dqpsk_walsh_strong_bit_errors as f32
                / self.total_raw_dqpsk_walsh_strong_bits_compared as f32
        }
    }

    pub fn dqpsk_walsh_weak_bit_ratio(&self) -> f32 {
        let weak = self.total_raw_dqpsk_walsh_weak_bits_compared;
        let strong = self.total_raw_dqpsk_walsh_strong_bits_compared;
        ratio(weak, weak + strong)
    }

    pub fn raw_err_run_mean(&self) -> Option<f32> {
        if self.total_raw_error_runs == 0 {
            None
        } else {
            Some(self.total_raw_error_run_bits as f32 / self.total_raw_error_runs as f32)
        }
    }

    pub fn raw_err_run_max(&self) -> Option<usize> {
        if self.max_raw_error_run_len == 0 {
            None
        } else {
            Some(self.max_raw_error_run_len)
        }
    }

    pub fn err_w_cw_mean(&self) -> Option<f32> {
        if self.total_codewords == 0 {
            None
        } else {
            Some(self.total_codeword_error_sum as f32 / self.total_codewords as f32)
        }
    }

    pub fn err_w_cw_p50(&self) -> Option<f32> {
        quantile_usize(&self.codeword_error_weights, 0.5)
    }

    pub fn err_w_cw_p90(&self) -> Option<f32> {
        quantile_usize(&self.codeword_error_weights, 0.9)
    }

    pub fn err_w_cw_p99(&self) -> Option<f32> {
        quantile_usize(&self.codeword_error_weights, 0.99)
    }

    pub fn err_w_cw_max(&self) -> Option<usize> {
        if self.max_codeword_error == 0 && self.total_codewords == 0 {
            None
        } else {
            Some(self.max_codeword_error)
        }
    }

    pub fn err_w_cw_hist(&self) -> Option<String> {
        error_weight_hist(&self.codeword_error_weights)
    }

    pub fn post_ber(&self) -> f32 {
        if self.total_post_bits_compared == 0 {
            f32::NAN
        } else {
            self.total_post_bit_errors as f32 / self.total_post_bits_compared as f32
        }
    }

    pub fn post_decode_match_ratio(&self) -> f32 {
        ratio(
            self.total_post_decode_matched,
            self.total_post_decode_attempts,
        )
    }

    pub fn post_err_run_mean(&self) -> Option<f32> {
        if self.total_post_error_runs == 0 {
            None
        } else {
            Some(self.total_post_error_run_bits as f32 / self.total_post_error_runs as f32)
        }
    }

    pub fn post_err_run_max(&self) -> Option<usize> {
        if self.max_post_error_run_len == 0 {
            None
        } else {
            Some(self.max_post_error_run_len)
        }
    }

    pub fn post_err_w_cw_mean(&self) -> Option<f32> {
        if self.total_post_codewords == 0 {
            None
        } else {
            Some(self.total_post_codeword_error_sum as f32 / self.total_post_codewords as f32)
        }
    }

    pub fn post_err_w_cw_p50(&self) -> Option<f32> {
        quantile_usize(&self.post_codeword_error_weights, 0.5)
    }

    pub fn post_err_w_cw_p90(&self) -> Option<f32> {
        quantile_usize(&self.post_codeword_error_weights, 0.9)
    }

    pub fn post_err_w_cw_p99(&self) -> Option<f32> {
        quantile_usize(&self.post_codeword_error_weights, 0.99)
    }

    pub fn post_err_w_cw_max(&self) -> Option<usize> {
        if self.max_post_codeword_error == 0 && self.total_post_codewords == 0 {
            None
        } else {
            Some(self.max_post_codeword_error)
        }
    }

    pub fn post_err_w_cw_hist(&self) -> Option<String> {
        error_weight_hist(&self.post_codeword_error_weights)
    }

    pub fn avg_last_est_snr_db(&self) -> Option<f32> {
        if self.count_last_est_snr_db == 0 {
            None
        } else {
            Some((self.sum_last_est_snr_db / self.count_last_est_snr_db as f64) as f32)
        }
    }

    /// 受信器内部の推定 SNR から Eb/N0 を換算する。
    ///
    /// `snr_bandwidth_hz` には avg_last_est_snr_db を定義した帯域 B を渡す。
    /// （Mary の現実装では、内部推定SNRはチップ領域由来のため B≈chip_rate を使う）
    pub fn avg_last_est_ebn0_db(&self, snr_bandwidth_hz: f32, bit_rate: f32) -> Option<f32> {
        let est_snr_db = self.avg_last_est_snr_db()?;
        let rb = bit_rate.max(1e-6);
        let beq = snr_bandwidth_hz.max(1e-6);
        Some(dsp::common::channel::snr_db_to_ebn0_db(est_snr_db, beq, rb))
    }

    /// 受信器内部の推定 SNR から C/N0 を換算する。
    ///
    /// `snr_bandwidth_hz` は avg_last_est_snr_db の帯域 B と一致させる必要がある。
    pub fn avg_last_est_cn0_db(&self, snr_bandwidth_hz: f32) -> Option<f32> {
        let est_snr_db = self.avg_last_est_snr_db()?;
        let beq = snr_bandwidth_hz.max(1e-6);
        Some(dsp::common::channel::snr_db_to_cn0_db(est_snr_db, beq))
    }

    pub fn awgn_snr_db(&self, sigma: f32) -> Option<f32> {
        if sigma <= 0.0 {
            return None;
        }
        let p_sig = self.tx_signal_power()?;
        if p_sig <= 0.0 {
            return None;
        }
        let p_noise = sigma * sigma;
        Some(10.0 * (p_sig / p_noise).log10())
    }

    pub fn phase_gate_on_ratio(&self) -> f32 {
        ratio(
            self.total_phase_gate_on_symbols,
            self.total_phase_gate_on_symbols + self.total_phase_gate_off_symbols,
        )
    }

    pub fn phase_innovation_reject_ratio(&self) -> f32 {
        ratio(
            self.total_phase_innovation_reject_symbols,
            self.total_phase_gate_on_symbols + self.total_phase_gate_off_symbols,
        )
    }

    pub fn phase_err_abs_mean_rad(&self) -> Option<f32> {
        if self.total_phase_err_abs_count == 0 {
            None
        } else {
            Some((self.total_phase_err_abs_sum_rad / self.total_phase_err_abs_count as f64) as f32)
        }
    }

    pub fn phase_err_abs_ge_0p5_ratio(&self) -> f32 {
        ratio(
            self.total_phase_err_abs_ge_0p5_symbols,
            self.total_phase_err_abs_count,
        )
    }

    pub fn phase_err_abs_ge_1p0_ratio(&self) -> f32 {
        ratio(
            self.total_phase_err_abs_ge_1p0_symbols,
            self.total_phase_err_abs_count,
        )
    }
}

pub fn ratio(num: usize, den: usize) -> f32 {
    if den == 0 {
        0.0
    } else {
        num as f32 / den as f32
    }
}

pub fn quantile(values: &[f32], q: f32) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(f32::total_cmp);
    let q = q.clamp(0.0, 1.0);
    let idx = (((v.len() - 1) as f32) * q).round() as usize;
    v.get(idx).copied()
}

pub fn quantile_usize(values: &[usize], q: f32) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_unstable();
    let q = q.clamp(0.0, 1.0);
    let idx = (((v.len() - 1) as f32) * q).round() as usize;
    v.get(idx).map(|&x| x as f32)
}

pub fn error_weight_hist(weights: &[usize]) -> Option<String> {
    if weights.is_empty() {
        return None;
    }
    let mut b0 = 0usize;
    let mut b1 = 0usize;
    let mut b2 = 0usize;
    let mut b3_4 = 0usize;
    let mut b5_8 = 0usize;
    let mut b9_16 = 0usize;
    let mut b17_32 = 0usize;
    let mut b33p = 0usize;
    for &w in weights {
        match w {
            0 => b0 += 1,
            1 => b1 += 1,
            2 => b2 += 1,
            3..=4 => b3_4 += 1,
            5..=8 => b5_8 += 1,
            9..=16 => b9_16 += 1,
            17..=32 => b17_32 += 1,
            _ => b33p += 1,
        }
    }
    let n = weights.len() as f32;
    Some(format!(
        "0:{:.3}|1:{:.3}|2:{:.3}|3-4:{:.3}|5-8:{:.3}|9-16:{:.3}|17-32:{:.3}|33+:{:.3}",
        b0 as f32 / n,
        b1 as f32 / n,
        b2 as f32 / n,
        b3_4 as f32 / n,
        b5_8 as f32 / n,
        b9_16 as f32 / n,
        b17_32 as f32 / n,
        b33p as f32 / n,
    ))
}

pub struct PhaseStats {
    pub last_est_snr_db: f32,
    pub phase_gate_on_symbols: usize,
    pub phase_gate_off_symbols: usize,
    pub phase_innovation_reject_symbols: usize,
    pub phase_err_abs_sum_rad: f64,
    pub phase_err_abs_count: usize,
    pub phase_err_abs_ge_0p5_symbols: usize,
    pub phase_err_abs_ge_1p0_symbols: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_consistency() {
        let mut m = Metrics::new(3); // packets_per_frame = 3
        m.total_sim_sec = 10.0;
        m.total_frame_attempts = 100; // total_packets_sent = 300

        // 1. Packet stats
        m.total_synced_frames = 80;
        m.total_accepted_packets = 200;
        m.total_crc_error_packets = 40;

        assert_eq!(m.packet_accept_ratio(), 200.0 / 300.0); // 200 / (100 * 3)
        assert_eq!(m.synced_frame_ratio(), 80.0 / 100.0);
        assert_eq!(m.crc_pass_ratio(), 200.0 / 240.0); // 200 / (200 + 40)

        // 2. Bit errors
        m.total_bit_errors = 50;
        m.total_bits_compared = 10000;
        assert_eq!(m.ber(), 50.0 / 10000.0);

        // 3. Recovery and Goodput
        let payload_bits = 512;
        m.add_recovery_event(2.0, 0, 0); // success 1
        m.add_recovery_event(4.0, 0, 0); // success 2
        m.add_recovery_event(6.0, 0, 0); // success 3

        assert_eq!(m.total_successes, 3);
        assert_eq!(m.mean_completion_sec().unwrap(), (2.0 + 4.0 + 6.0) / 3.0);
        // goodput_effective_bps = (PAYLOAD_SIZE*8 * accepted_packets) / total_sim_sec
        let expected_effective =
            (dsp::params::PAYLOAD_SIZE * 8 * m.total_accepted_packets) as f32 / 10.0;
        assert_eq!(m.goodput_effective_bps(), expected_effective);
        // goodput_success_mean_bps = (payload_bits * total_successes) / sum(completion_secs)
        let expected_mean_bps = (512.0 * 3.0) / (2.0 + 4.0 + 6.0);
        assert!(
            (m.goodput_success_mean_bps(payload_bits).unwrap() - expected_mean_bps).abs() < 1e-4
        );

        // 4. Power and SNR
        m.total_tx_signal_energy = 1000.0;
        m.total_tx_signal_samples = 2000;
        // power = 1000 / 2000 = 0.5
        assert_eq!(m.tx_signal_power().unwrap(), 0.5);
        // AWGN SNR for sigma=1.0: 10 * log10(0.5 / 1.0) = -3.0103
        assert!((m.awgn_snr_db(1.0).unwrap() - (-3.0103)).abs() < 1e-4);

        // 5. Raw BER
        m.total_raw_bit_errors = 100;
        m.total_raw_bits_compared = 1000;
        m.total_raw_walsh_bit_errors = 60;
        m.total_raw_walsh_bits_compared = 700;
        m.total_raw_dqpsk_bit_errors = 40;
        m.total_raw_dqpsk_bits_compared = 300;
        m.total_raw_dqpsk_walsh_weak_bit_errors = 24;
        m.total_raw_dqpsk_walsh_weak_bits_compared = 120;
        m.total_raw_dqpsk_walsh_strong_bit_errors = 16;
        m.total_raw_dqpsk_walsh_strong_bits_compared = 180;
        assert_eq!(m.raw_ber(), 100.0 / 1000.0);
        assert_eq!(m.raw_ber_walsh(), 60.0 / 700.0);
        assert_eq!(m.raw_ber_dqpsk(), 40.0 / 300.0);
        assert_eq!(m.raw_ber_dqpsk_walsh_weak(), 24.0 / 120.0);
        assert_eq!(m.raw_ber_dqpsk_walsh_strong(), 16.0 / 180.0);
        assert_eq!(m.dqpsk_walsh_weak_bit_ratio(), 120.0 / 300.0);
        assert_eq!(
            m.raw_ber_dqpsk_over_walsh(),
            (40.0 / 300.0) / (60.0 / 700.0)
        );

        // 6. LLR Rescue
        m.total_llr_second_pass_attempts = 20;
        m.total_llr_second_pass_rescued = 5;
        assert_eq!(m.llr_second_pass_trigger_ratio(), 20.0 / 240.0);
        assert_eq!(m.llr_second_pass_rescue_ratio(), 5.0 / 20.0);

        // 6.5 Viterbi CRC candidate checks
        m.total_viterbi_packet_decode_attempts = 8;
        m.total_viterbi_crc_candidate_checks = 20;
        assert_eq!(m.viterbi_crc_candidates_mean(), 20.0 / 8.0);
        m.total_false_accepted_packets = 4;
        assert_eq!(m.false_accept_ratio_per_accepted_packet(), 4.0 / 200.0);

        // 7. Phase
        m.total_phase_gate_on_symbols = 700;
        m.total_phase_gate_off_symbols = 300;
        m.total_phase_err_abs_sum_rad = 50.0;
        m.total_phase_err_abs_count = 1000;
        m.total_phase_err_abs_ge_0p5_symbols = 100;

        assert_eq!(m.phase_gate_on_ratio(), 0.7);
        assert_eq!(m.phase_err_abs_mean_rad().unwrap(), 50.0 / 1000.0);
        assert_eq!(m.phase_err_abs_ge_0p5_ratio(), 100.0 / 1000.0);
    }

    #[test]
    fn test_quantiles() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile(&values, 0.0).unwrap(), 1.0);
        assert_eq!(quantile(&values, 0.5).unwrap(), 3.0);
        assert_eq!(quantile(&values, 1.0).unwrap(), 5.0);

        let usizes = vec![0, 10, 20, 30, 40];
        assert_eq!(quantile_usize(&usizes, 0.9).unwrap(), 40.0);
        assert_eq!(quantile_usize(&usizes, 0.1).unwrap(), 0.0);
    }

    #[test]
    fn test_viterbi_crc_candidates_mean_handles_zero_attempts() {
        let m = Metrics::new(1);
        assert_eq!(m.viterbi_crc_candidates_mean(), 0.0);
    }
}
