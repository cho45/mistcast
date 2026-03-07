use crate::utils::{PreFecStats, PostFecStats};

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

#[derive(Default, Clone, Debug)]
pub struct TrialResult {
    pub success: bool,
    pub completion_sec: Option<f32>,
    pub elapsed_sec: f32,
    pub attempts: usize,
    /// 同期成立したフレーム数
    pub synced_frames: usize,
    /// CRC通過フレーム数（Accepted）
    pub accepted_frames: usize,
    /// CRCエラーフレーム数
    pub crc_error_frames: usize,
    pub first_attempt_success: bool,
    pub bit_errors: usize,
    pub bits_compared: usize,
    pub dropped_attempts: usize,
    pub tx_signal_energy_sum: f64,
    pub tx_signal_samples: usize,
    pub process_time_ns: u64,
    /// FEC符号化ビットのハード判定BER（Fountain符号の外側の生BER）
    pub raw_bit_errors: usize,
    pub raw_bits_compared: usize,
    pub raw_error_runs: usize,
    pub raw_error_run_bits: usize,
    pub raw_error_run_max: usize,
    pub codeword_count: usize,
    pub codeword_error_sum: usize,
    pub codeword_error_max: usize,
    pub codeword_error_weights: Vec<usize>,
    pub post_bit_errors: usize,
    pub post_bits_compared: usize,
    pub post_error_runs: usize,
    pub post_error_run_bits: usize,
    pub post_error_run_max: usize,
    pub post_codeword_count: usize,
    pub post_codeword_error_sum: usize,
    pub post_codeword_error_max: usize,
    pub post_codeword_error_weights: Vec<usize>,
    pub post_decode_attempts: usize,
    pub post_decode_matched: usize,
    pub last_est_snr_db: f32,
    pub phase_gate_on_symbols: usize,
    pub phase_gate_off_symbols: usize,
    pub phase_innovation_reject_symbols: usize,
    pub phase_err_abs_sum_rad: f64,
    pub phase_err_abs_count: usize,
    pub phase_err_abs_ge_0p5_symbols: usize,
    pub phase_err_abs_ge_1p0_symbols: usize,
    pub llr_second_pass_attempts: usize,
    pub llr_second_pass_rescued: usize,
}

#[derive(Default, Clone, Debug)]
pub struct Metrics {
    pub trials: usize,
    pub successes: usize,
    pub first_attempt_successes: usize,
    pub total_attempts: usize,
    pub total_synced_frames: usize,
    pub total_accepted_frames: usize,
    pub total_crc_error_frames: usize,
    pub total_bit_errors: usize,
    pub total_bits_compared: usize,
    pub total_elapsed_sec: f32,
    pub dropped_attempts: usize,
    pub total_tx_signal_energy: f64,
    pub total_tx_signal_samples: usize,
    pub total_process_time_ns: u64,
    pub completion_secs: Vec<f32>,
    pub total_raw_bit_errors: usize,
    pub total_raw_bits_compared: usize,
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

pub struct TrialState {
    pub elapsed_sec: f32,
    pub attempts: usize,
    pub dropped_attempts: usize,
    pub tx_signal_energy_sum: f64,
    pub tx_signal_samples: usize,
    pub total_process_ns: u64,
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

pub struct TrialResultBuilder {
    pub success: bool,
    pub completion_sec: Option<f32>,
    pub elapsed_sec: f32,
    pub attempts: usize,
    pub synced_frames: usize,
    pub accepted_frames: usize,
    pub crc_error_frames: usize,
    pub first_attempt_success: bool,
    pub bit_errors: usize,
    pub bits_compared: usize,
    pub dropped_attempts: usize,
    pub tx_signal_energy_sum: f64,
    pub tx_signal_samples: usize,
    pub process_time_ns: u64,
    pub raw_bit_errors: usize,
    pub raw_bits_compared: usize,
    pub raw_error_runs: usize,
    pub raw_error_run_bits: usize,
    pub raw_error_run_max: usize,
    pub codeword_count: usize,
    pub codeword_error_sum: usize,
    pub codeword_error_max: usize,
    pub codeword_error_weights: Vec<usize>,
    pub post_bit_errors: usize,
    pub post_bits_compared: usize,
    pub post_error_runs: usize,
    pub post_error_run_bits: usize,
    pub post_error_run_max: usize,
    pub post_codeword_count: usize,
    pub post_codeword_error_sum: usize,
    pub post_codeword_error_max: usize,
    pub post_codeword_error_weights: Vec<usize>,
    pub post_decode_attempts: usize,
    pub post_decode_matched: usize,
    pub last_est_snr_db: f32,
    pub phase_gate_on_symbols: usize,
    pub phase_gate_off_symbols: usize,
    pub phase_innovation_reject_symbols: usize,
    pub phase_err_abs_sum_rad: f64,
    pub phase_err_abs_count: usize,
    pub phase_err_abs_ge_0p5_symbols: usize,
    pub phase_err_abs_ge_1p0_symbols: usize,
    pub llr_second_pass_attempts: usize,
    pub llr_second_pass_rescued: usize,
}

impl TrialResultBuilder {
    pub fn new(state: &TrialState) -> Self {
        Self {
            success: false,
            completion_sec: None,
            elapsed_sec: state.elapsed_sec,
            attempts: state.attempts,
            synced_frames: 0,
            accepted_frames: 0,
            crc_error_frames: 0,
            first_attempt_success: false,
            bit_errors: 0,
            bits_compared: 0,
            dropped_attempts: state.dropped_attempts,
            tx_signal_energy_sum: state.tx_signal_energy_sum,
            tx_signal_samples: state.tx_signal_samples,
            process_time_ns: state.total_process_ns,
            raw_bit_errors: 0,
            raw_bits_compared: 0,
            raw_error_runs: 0,
            raw_error_run_bits: 0,
            raw_error_run_max: 0,
            codeword_count: 0,
            codeword_error_sum: 0,
            codeword_error_max: 0,
            codeword_error_weights: Vec::new(),
            post_bit_errors: 0,
            post_bits_compared: 0,
            post_error_runs: 0,
            post_error_run_bits: 0,
            post_error_run_max: 0,
            post_codeword_count: 0,
            post_codeword_error_sum: 0,
            post_codeword_error_max: 0,
            post_codeword_error_weights: Vec::new(),
            post_decode_attempts: 0,
            post_decode_matched: 0,
            last_est_snr_db: f32::NAN,
            phase_gate_on_symbols: 0,
            phase_gate_off_symbols: 0,
            phase_innovation_reject_symbols: 0,
            phase_err_abs_sum_rad: 0.0,
            phase_err_abs_count: 0,
            phase_err_abs_ge_0p5_symbols: 0,
            phase_err_abs_ge_1p0_symbols: 0,
            llr_second_pass_attempts: 0,
            llr_second_pass_rescued: 0,
        }
    }

    pub fn success(mut self, success: bool, bit_errors: usize, bits_compared: usize) -> Self {
        self.success = success;
        self.completion_sec = if success {
            Some(self.elapsed_sec)
        } else {
            None
        };
        self.first_attempt_success = success && self.attempts == 1;
        self.bit_errors = bit_errors;
        self.bits_compared = bits_compared;
        self
    }

    pub fn frame_stats<SF, AF, EF>(
        mut self,
        synced_frames: SF,
        accepted_frames: AF,
        crc_error_frames: EF,
    ) -> Self
    where
        SF: Into<usize>,
        AF: Into<usize>,
        EF: Into<usize>,
    {
        self.synced_frames = synced_frames.into();
        self.accepted_frames = accepted_frames.into();
        self.crc_error_frames = crc_error_frames.into();
        self
    }

    pub fn mary_raw_ber(mut self, stats: PreFecStats) -> Self {
        self.raw_bit_errors = stats.bit_errors;
        self.raw_bits_compared = stats.bits_compared;
        self.raw_error_runs = stats.error_runs;
        self.raw_error_run_bits = stats.error_run_bits;
        self.raw_error_run_max = stats.error_run_max;
        self.codeword_count = stats.codeword_count;
        self.codeword_error_sum = stats.codeword_error_sum;
        self.codeword_error_max = stats.codeword_error_max;
        self.codeword_error_weights = stats.codeword_error_weights;
        self
    }

    pub fn mary_post_ber(mut self, stats: PostFecStats) -> Self {
        self.post_bit_errors = stats.bit_errors;
        self.post_bits_compared = stats.bits_compared;
        self.post_error_runs = stats.error_runs;
        self.post_error_run_bits = stats.error_run_bits;
        self.post_error_run_max = stats.error_run_max;
        self.post_codeword_count = stats.codeword_count;
        self.post_codeword_error_sum = stats.codeword_error_sum;
        self.post_codeword_error_max = stats.codeword_error_max;
        self.post_codeword_error_weights = stats.codeword_error_weights;
        self
    }

    pub fn mary_phase(mut self, stats: PhaseStats) -> Self {
        self.last_est_snr_db = stats.last_est_snr_db;
        self.phase_gate_on_symbols = stats.phase_gate_on_symbols;
        self.phase_gate_off_symbols = stats.phase_gate_off_symbols;
        self.phase_innovation_reject_symbols = stats.phase_innovation_reject_symbols;
        self.phase_err_abs_sum_rad = stats.phase_err_abs_sum_rad;
        self.phase_err_abs_count = stats.phase_err_abs_count;
        self.phase_err_abs_ge_0p5_symbols = stats.phase_err_abs_ge_0p5_symbols;
        self.phase_err_abs_ge_1p0_symbols = stats.phase_err_abs_ge_1p0_symbols;
        self
    }

    pub fn mary_llr(mut self, attempts: usize, rescued: usize) -> Self {
        self.llr_second_pass_attempts = attempts;
        self.llr_second_pass_rescued = rescued;
        self
    }

    pub fn mary_decode_stats(
        mut self,
        post_decode_attempts: usize,
        post_decode_matched: usize,
    ) -> Self {
        self.post_decode_attempts = post_decode_attempts;
        self.post_decode_matched = post_decode_matched;
        self
    }

    pub fn build(self) -> TrialResult {
        TrialResult {
            success: self.success,
            completion_sec: self.completion_sec,
            elapsed_sec: self.elapsed_sec,
            attempts: self.attempts,
            synced_frames: self.synced_frames,
            accepted_frames: self.accepted_frames,
            crc_error_frames: self.crc_error_frames,
            first_attempt_success: self.first_attempt_success,
            bit_errors: self.bit_errors,
            bits_compared: self.bits_compared,
            dropped_attempts: self.dropped_attempts,
            tx_signal_energy_sum: self.tx_signal_energy_sum,
            tx_signal_samples: self.tx_signal_samples,
            process_time_ns: self.process_time_ns,
            raw_bit_errors: self.raw_bit_errors,
            raw_bits_compared: self.raw_bits_compared,
            raw_error_runs: self.raw_error_runs,
            raw_error_run_bits: self.raw_error_run_bits,
            raw_error_run_max: self.raw_error_run_max,
            codeword_count: self.codeword_count,
            codeword_error_sum: self.codeword_error_sum,
            codeword_error_max: self.codeword_error_max,
            codeword_error_weights: self.codeword_error_weights,
            post_bit_errors: self.post_bit_errors,
            post_bits_compared: self.post_bits_compared,
            post_error_runs: self.post_error_runs,
            post_error_run_bits: self.post_error_run_bits,
            post_error_run_max: self.post_error_run_max,
            post_codeword_count: self.post_codeword_count,
            post_codeword_error_sum: self.post_codeword_error_sum,
            post_codeword_error_max: self.post_codeword_error_max,
            post_codeword_error_weights: self.post_codeword_error_weights,
            post_decode_attempts: self.post_decode_attempts,
            post_decode_matched: self.post_decode_matched,
            last_est_snr_db: self.last_est_snr_db,
            phase_gate_on_symbols: self.phase_gate_on_symbols,
            phase_gate_off_symbols: self.phase_gate_off_symbols,
            phase_innovation_reject_symbols: self.phase_innovation_reject_symbols,
            phase_err_abs_sum_rad: self.phase_err_abs_sum_rad,
            phase_err_abs_count: self.phase_err_abs_count,
            phase_err_abs_ge_0p5_symbols: self.phase_err_abs_ge_0p5_symbols,
            phase_err_abs_ge_1p0_symbols: self.phase_err_abs_ge_1p0_symbols,
            llr_second_pass_attempts: self.llr_second_pass_attempts,
            llr_second_pass_rescued: self.llr_second_pass_rescued,
        }
    }
}

impl Metrics {
    pub fn push(&mut self, t: TrialResult) {
        self.trials += 1;
        self.total_elapsed_sec += t.elapsed_sec;
        self.total_attempts += t.attempts;
        self.total_synced_frames += t.synced_frames;
        self.total_accepted_frames += t.accepted_frames;
        self.total_crc_error_frames += t.crc_error_frames;
        self.total_bit_errors += t.bit_errors;
        self.total_bits_compared += t.bits_compared;
        self.dropped_attempts += t.dropped_attempts;
        self.total_raw_bit_errors += t.raw_bit_errors;
        self.total_raw_bits_compared += t.raw_bits_compared;
        self.total_raw_error_runs += t.raw_error_runs;
        self.total_raw_error_run_bits += t.raw_error_run_bits;
        self.max_raw_error_run_len = self.max_raw_error_run_len.max(t.raw_error_run_max);
        self.total_codewords += t.codeword_count;
        self.total_codeword_error_sum += t.codeword_error_sum;
        self.max_codeword_error = self.max_codeword_error.max(t.codeword_error_max);
        self.codeword_error_weights.extend(t.codeword_error_weights);
        self.total_post_bit_errors += t.post_bit_errors;
        self.total_post_bits_compared += t.post_bits_compared;
        self.total_post_error_runs += t.post_error_runs;
        self.total_post_error_run_bits += t.post_error_run_bits;
        self.max_post_error_run_len = self.max_post_error_run_len.max(t.post_error_run_max);
        self.total_post_codewords += t.post_codeword_count;
        self.total_post_codeword_error_sum += t.post_codeword_error_sum;
        self.max_post_codeword_error = self.max_post_codeword_error.max(t.post_codeword_error_max);
        self.post_codeword_error_weights
            .extend(t.post_codeword_error_weights);
        self.total_post_decode_attempts += t.post_decode_attempts;
        self.total_post_decode_matched += t.post_decode_matched;
        self.total_tx_signal_energy += t.tx_signal_energy_sum;
        self.total_tx_signal_samples += t.tx_signal_samples;
        self.total_process_time_ns += t.process_time_ns;
        if t.last_est_snr_db.is_finite() {
            self.sum_last_est_snr_db += t.last_est_snr_db as f64;
            self.count_last_est_snr_db += 1;
        }
        self.total_phase_gate_on_symbols += t.phase_gate_on_symbols;
        self.total_phase_gate_off_symbols += t.phase_gate_off_symbols;
        self.total_phase_innovation_reject_symbols += t.phase_innovation_reject_symbols;
        self.total_phase_err_abs_sum_rad += t.phase_err_abs_sum_rad;
        self.total_phase_err_abs_count += t.phase_err_abs_count;
        self.total_phase_err_abs_ge_0p5_symbols += t.phase_err_abs_ge_0p5_symbols;
        self.total_phase_err_abs_ge_1p0_symbols += t.phase_err_abs_ge_1p0_symbols;
        self.total_llr_second_pass_attempts += t.llr_second_pass_attempts;
        self.total_llr_second_pass_rescued += t.llr_second_pass_rescued;

        if t.first_attempt_success {
            self.first_attempt_successes += 1;
        }
        if t.success {
            self.successes += 1;
            if let Some(c) = t.completion_sec {
                self.completion_secs.push(c);
            }
        }
    }

    pub fn p_complete(&self) -> f32 {
        ratio(self.successes, self.trials)
    }

    pub fn synced_frame_ratio(&self) -> f32 {
        ratio(self.total_synced_frames, self.total_attempts)
    }

    pub fn crc_pass_ratio(&self) -> f32 {
        ratio(
            self.total_accepted_frames,
            self.total_accepted_frames + self.total_crc_error_frames,
        )
    }

    pub fn llr_second_pass_trigger_ratio(&self) -> f32 {
        ratio(
            self.total_llr_second_pass_attempts,
            self.total_accepted_frames + self.total_crc_error_frames,
        )
    }

    pub fn llr_second_pass_rescue_ratio(&self) -> f32 {
        ratio(
            self.total_llr_second_pass_rescued,
            self.total_llr_second_pass_attempts,
        )
    }

    pub fn ber(&self) -> f32 {
        if self.total_bits_compared == 0 {
            0.0
        } else {
            self.total_bit_errors as f32 / self.total_bits_compared as f32
        }
    }

    pub fn p95_completion_sec(&self) -> Option<f32> {
        quantile(&self.completion_secs, 0.95)
    }

    pub fn mean_completion_sec(&self) -> Option<f32> {
        if self.completion_secs.is_empty() {
            None
        } else {
            Some(self.completion_secs.iter().sum::<f32>() / self.completion_secs.len() as f32)
        }
    }

    pub fn goodput_effective_bps(&self, payload_bits: usize) -> f32 {
        if self.total_elapsed_sec <= 0.0 {
            0.0
        } else {
            (payload_bits * self.successes) as f32 / self.total_elapsed_sec
        }
    }

    pub fn goodput_success_mean_bps(&self, payload_bits: usize) -> Option<f32> {
        if self.completion_secs.is_empty() {
            return None;
        }
        let sum = self
            .completion_secs
            .iter()
            .map(|&t| payload_bits as f32 / t.max(1e-6))
            .sum::<f32>();
        Some(sum / self.completion_secs.len() as f32)
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
