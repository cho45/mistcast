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

/// 評価シミュレーションの集計メトリクス（兼 実行状態）
#[derive(Default, Clone, Debug)]
pub struct Metrics {
    /// シミュレーション上の信号の経過時間 [sec]
    pub total_sim_sec: f32,
    /// 送信フレーム数
    pub total_frame_attempts: usize,
    /// 1フレームに含まれるパケット数
    pub packets_per_frame: usize,
    /// 受信側で同期が成立した総フレーム数
    pub total_synced_frames: usize,
    /// 正常に復元された総パケット数 (Fountain単位)
    pub total_accepted_packets: usize,
    /// CRCエラーとなった総パケット数
    pub total_crc_error_packets: usize,
    /// 総復元成功回数 (データ全体単位)
    pub total_successes: usize,
    /// 全復元データにおけるビットエラー総数
    pub total_bit_errors: usize,
    /// 比較された全ビット数
    pub total_bits_compared: usize,
    /// 消失（ドロップ）したフレーム数
    pub dropped_frames: usize,
    /// 送信信号の総エネルギー
    pub total_tx_signal_energy: f64,
    /// 送信信号の総サンプル数
    pub total_tx_signal_samples: usize,
    /// 信号処理に要した合計CPU時間 [ns]
    pub total_process_time_ns: u64,
    /// 各データ復元成功までの到達時間のリスト [sec]
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

impl Metrics {
    pub fn new(packets_per_frame: usize) -> Self {
        Self {
            packets_per_frame,
            ..Self::default()
        }
    }

    pub fn add_packet_stats<SF, AP, EP>(
        &mut self,
        synced_frames: SF,
        accepted_packets: AP,
        crc_error_packets: EP,
    ) where
        SF: Into<usize>,
        AP: Into<usize>,
        EP: Into<usize>,
    {
        self.total_synced_frames += synced_frames.into();
        self.total_accepted_packets += accepted_packets.into();
        self.total_crc_error_packets += crc_error_packets.into();
    }

    pub fn add_recovery_event(&mut self, completion_sec: f32, bit_errors: usize, bits_compared: usize) {
        self.completion_secs.push(completion_sec);
        self.total_successes += 1;
        self.total_bit_errors += bit_errors;
        self.total_bits_compared += bits_compared;
    }

    pub fn set_mary_raw_ber(&mut self, stats: PreFecStats) {
        self.total_raw_bit_errors += stats.bit_errors;
        self.total_raw_bits_compared += stats.bits_compared;
        self.total_raw_error_runs += stats.error_runs;
        self.total_raw_error_run_bits += stats.error_run_bits;
        self.max_raw_error_run_len = self.max_raw_error_run_len.max(stats.error_run_max);
        self.total_codewords += stats.codeword_count;
        self.total_codeword_error_sum += stats.codeword_error_sum;
        self.max_codeword_error = self.max_codeword_error.max(stats.codeword_error_max);
        self.codeword_error_weights.extend(stats.codeword_error_weights);
    }

    pub fn set_mary_post_ber(&mut self, stats: PostFecStats) {
        self.total_post_bit_errors += stats.bit_errors;
        self.total_post_bits_compared += stats.bits_compared;
        self.total_post_error_runs += stats.error_runs;
        self.total_post_error_run_bits += stats.error_run_bits;
        self.max_post_error_run_len = self.max_post_error_run_len.max(stats.error_run_max);
        self.total_post_codewords += stats.codeword_count;
        self.total_post_codeword_error_sum += stats.codeword_error_sum;
        self.max_post_codeword_error = self.max_post_codeword_error.max(stats.codeword_error_max);
        self.post_codeword_error_weights.extend(stats.codeword_error_weights);
    }

    pub fn set_mary_phase(&mut self, stats: PhaseStats) {
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

    pub fn set_mary_llr(&mut self, attempts: usize, rescued: usize) {
        self.total_llr_second_pass_attempts += attempts;
        self.total_llr_second_pass_rescued += rescued;
    }

    pub fn set_mary_decode_stats(&mut self, attempts: usize, matched: usize) {
        self.total_post_decode_attempts += attempts;
        self.total_post_decode_matched += matched;
    }

    /// パケット到達率 (PDR: Packet Delivery Ratio)
    pub fn p_complete(&self) -> f32 {
        let total_packets_sent = self.total_frame_attempts * self.packets_per_frame;
        ratio(self.total_accepted_packets, total_packets_sent)
    }

    /// フレーム同期成功率
    pub fn synced_frame_ratio(&self) -> f32 {
        ratio(self.total_synced_frames, self.total_frame_attempts)
    }

    /// CRC通過率
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

    /// ビットエラーレート (BER)
    pub fn ber(&self) -> f32 {
        if self.total_bits_compared == 0 {
            0.0
        } else {
            self.total_bit_errors as f32 / self.total_bits_compared as f32
        }
    }

    /// 平均データ復元成功時間 (MTTD)
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

    /// 実効スループット [bps]
    pub fn goodput_effective_bps(&self, payload_bits: usize) -> f32 {
        if self.total_sim_sec <= 0.0 {
            0.0
        } else {
            (payload_bits * self.total_successes) as f32 / self.total_sim_sec
        }
    }

    /// 成功パケットごとのスループット平均 [bps]
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
