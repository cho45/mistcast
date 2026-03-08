use crate::utils::{PostFecStats, PreFecStats};

macro_rules! define_metrics {
    ($($id:ident, $desc:expr, $extractor:expr $(, $default:ident)? ;)*) => {
        pub const METRICS_DEFS: &[MetricDef] = &[
            $(
                MetricDef {
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
                    define_metrics!(@id $id $default),
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


define_metrics! {
    scenario,                      "評価シナリオ名",               |ctx, _| ctx.scenario.into(), default;
    phy,                           "方式 (dsss, mary)",            |ctx, _| ctx.phy.into(), default;
    mary_fde_mode,                 "Mary FDEモード",               |ctx, _| ctx.mary_fde_mode.into(), default;
    total_sim_sec,                 "シミュレーション経過時間 [sec]",|_, m| m.total_sim_sec.into(), default;
    awgn_snr_db,                   "理論上のAWGN SNR [dB]",        |ctx, m| m.awgn_snr_db(ctx.sigma).into(), default;
    p_complete,                    "パケット到達率 (PDR)",         |_, m| m.p_complete().into(), default;
    ber,                           "ビットエラーレート (BER)",     |_, m| m.ber().into(), default;
    raw_ber,                       "FEC適用前の生BER",             |_, m| m.raw_ber().into(), default;
    goodput_effective_bps,         "実効スループット [bps]",       |ctx, m| m.goodput_effective_bps(ctx.payload_bits).into(), default;
    goodput_success_mean_bps,      "平均パケットスループット [bps]",|ctx, m| m.goodput_success_mean_bps(ctx.payload_bits).into(), default;
    p95_complete_s,                "95%パケット到達時間 [sec]",    |_, m| m.p95_completion_sec().into(), default;
    mean_complete_s,               "平均パケット到達時間 [sec]",   |_, m| m.mean_completion_sec().into(), default;
    avg_proc_ns_sample,            "平均処理時間 [ns/sample]",     |_, m| m.avg_process_time_per_sample_ns().into(), default;
    synced_frame_ratio,            "フレーム同期成功率",           |_, m| m.synced_frame_ratio().into(), default;
    crc_pass_ratio,                "CRC通過率",                    |_, m| m.crc_pass_ratio().into(), default;
    llr_second_pass_trigger_ratio, "LLR消去2ndパス起動率",         |_, m| m.llr_second_pass_trigger_ratio().into(), default;
    llr_second_pass_rescue_ratio,  "LLR消去2ndパス救済率",         |_, m| m.llr_second_pass_rescue_ratio().into(), default;
    phase_gate_on_ratio,           "位相ゲート有効率",             |_, m| m.phase_gate_on_ratio().into(), default;
    phase_innovation_reject_ratio, "位相変化棄却率",               |_, m| m.phase_innovation_reject_ratio().into(), default;
    phase_err_abs_mean_rad,        "平均絶対位相誤差 [rad]",       |_, m| m.phase_err_abs_mean_rad().into(), default;
    phase_err_abs_ge_0p5_ratio,    "0.5rad以上誤差率",             |_, m| m.phase_err_abs_ge_0p5_ratio().into(), default;
    phase_err_abs_ge_1p0_ratio,    "1.0rad以上誤差率",             |_, m| m.phase_err_abs_ge_1p0_ratio().into(), default;
    avg_last_est_snr_db,           "平均推定SNR [dB]",             |_, m| m.avg_last_est_snr_db().into(), default;
    multipath,                     "マルチパスプロファイル",       |ctx, _| ctx.multipath_name.into(), default;
    raw_err_run_mean,              "生エラーラン平均長",           |_, m| m.raw_err_run_mean().into();
    raw_err_run_max,               "生エラーラン最大長",           |_, m| m.raw_err_run_max().into();
    err_w_cw_mean,                 "CWあたり平均エラービット",     |_, m| m.err_w_cw_mean().into();
    err_w_cw_p50,                  "CWエラービット中央値",         |_, m| m.err_w_cw_p50().into();
    err_w_cw_p90,                  "CWエラービット90%点",          |_, m| m.err_w_cw_p90().into();
    err_w_cw_p99,                  "CWエラービット99%点",          |_, m| m.err_w_cw_p99().into();
    err_w_cw_max,                  "CWあたり最大エラービット",     |_, m| m.err_w_cw_max().into();
    err_w_cw_hist,                 "CWエラー重みヒストグラム",     |_, m| m.err_w_cw_hist().into();
    post_ber,                      "Viterbi後BER",                 |_, m| m.post_ber().into();
    post_decode_match_ratio,       "データ一致率",                 |_, m| m.post_decode_match_ratio().into();
    post_err_run_mean,             "Viterbi後エラーラン平均長",    |_, m| m.post_err_run_mean().into();
    post_err_run_max,              "Viterbi後エラーラン最大長",    |_, m| m.post_err_run_max().into();
    post_err_w_cw_mean,            "Viterbi後CWあたり平均エラー",  |_, m| m.post_err_w_cw_mean().into();
    post_err_w_cw_p50,             "Viterbi後CWエラー中央値",      |_, m| m.post_err_w_cw_p50().into();
    post_err_w_cw_p90,             "Viterbi後CWエラー90%点",       |_, m| m.post_err_w_cw_p90().into();
    post_err_w_cw_p99,             "Viterbi後CWエラー99%点",       |_, m| m.post_err_w_cw_p99().into();
    post_err_w_cw_max,             "Viterbi後CWあたり最大エラー",  |_, m| m.post_err_w_cw_max().into();
    post_err_w_cw_hist,            "Viterbi後CWエラーヒストグラム",|_, m| m.post_err_w_cw_hist().into();
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
    /// FEC前（生）のビットエラー総数
    pub total_raw_bit_errors: usize,
    /// FEC前（生）の比較ビット総数
    pub total_raw_bits_compared: usize,
    /// FEC前（生）のエラーラン総数
    pub total_raw_error_runs: usize,
    /// FEC前（生）のエラーランに含まれるビット総数
    pub total_raw_error_run_bits: usize,
    /// FEC前（生）の最大エラーラン長
    pub max_raw_error_run_len: usize,
    /// FEC前（生）の総コードワード数
    pub total_codewords: usize,
    /// FEC前（生）のコードワードあたりのエラービット総数
    pub total_codeword_error_sum: usize,
    /// FEC前（生）のコードワードあたりの最大エラービット数
    pub max_codeword_error: usize,
    /// FEC前（生）のコードワードエラー重みのリスト
    pub codeword_error_weights: Vec<usize>,
    /// FEC適用後のビットエラー総数
    pub total_post_bit_errors: usize,
    /// FEC適用後の比較ビット総数
    pub total_post_bits_compared: usize,
    /// FEC適用後のエラーラン総数
    pub total_post_error_runs: usize,
    /// FEC適用後のエラーランに含まれるビット総数
    pub total_post_error_run_bits: usize,
    /// FEC適用後の最大エラーラン長
    pub max_post_error_run_len: usize,
    /// FEC適用後の総コードワード数
    pub total_post_codewords: usize,
    /// FEC適用後のコードワードあたりのエラービット総数
    pub total_post_codeword_error_sum: usize,
    /// FEC適用後のコードワードあたりの最大エラービット数
    pub max_post_codeword_error: usize,
    /// FEC適用後のコードワードエラー重みのリスト
    pub post_codeword_error_weights: Vec<usize>,
    /// Post-FECデコード試行総数
    pub total_post_decode_attempts: usize,
    /// Post-FECデコード結果が送信データと一致した総数
    pub total_post_decode_matched: usize,
    /// 推定SNR의合計 (平均計算用)
    pub sum_last_est_snr_db: f64,
    /// 推定SNRのサンプル数
    pub count_last_est_snr_db: usize,
    /// 位相ゲートがONだったシンボル総数
    pub total_phase_gate_on_symbols: usize,
    /// 位相ゲートがOFFだったシンボル総数
    pub total_phase_gate_off_symbols: usize,
    /// 位相変化が棄却されたシンボル総数
    pub total_phase_innovation_reject_symbols: usize,
    /// 絶対位相誤差の合計 [rad]
    pub total_phase_err_abs_sum_rad: f64,
    /// 位相誤差が計算されたシンボル総数
    pub total_phase_err_abs_count: usize,
    /// 0.5 rad 以上の位相誤差が発生したシンボル総数
    pub total_phase_err_abs_ge_0p5_symbols: usize,
    /// 1.0 rad 以上の位相誤差が発生したシンボル総数
    pub total_phase_err_abs_ge_1p0_symbols: usize,
    /// LLR消去第2パスがトリガーされた回数
    pub total_llr_second_pass_attempts: usize,
    /// LLR消去第2パスによって救済されたパケット総数
    pub total_llr_second_pass_rescued: usize,
}

impl Metrics {
    pub fn new(packets_per_frame: usize) -> Self {
        Self {
            packets_per_frame,
            ..Self::default()
        }
    }

    pub fn add_frame_event(&mut self, synced: usize, accepted_packets: usize, crc_error_packets: usize) {
        self.total_synced_frames += synced;
        self.total_accepted_packets += accepted_packets;
        self.total_crc_error_packets += crc_error_packets;
    }

    /// データ復元成功イベントを記録する
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

    /// Mary生ビットエラー統計を反映
    pub fn set_mary_raw_ber(&mut self, stats: PreFecStats) {
        self.total_raw_bit_errors = stats.bit_errors;
        self.total_raw_bits_compared = stats.bits_compared;
        self.total_raw_error_runs = stats.error_runs;
        self.total_raw_error_run_bits = stats.error_run_bits;
        self.max_raw_error_run_len = stats.error_run_max;
        self.total_codewords = stats.codeword_count;
        self.total_codeword_error_sum = stats.codeword_error_sum;
        self.max_codeword_error = stats.codeword_error_max;
        self.codeword_error_weights = stats.codeword_error_weights;
    }

    /// Mary FEC適用後統計を反映
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

    /// Mary 位相統計を反映（加算）
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

    /// LLR救済統計を加算
    pub fn add_mary_llr(&mut self, attempts: usize, rescued: usize) {
        self.total_llr_second_pass_attempts += attempts;
        self.total_llr_second_pass_rescued += rescued;
    }

    /// Viterbiデコード一致統計を加算
    pub fn add_mary_decode_stats(&mut self, attempts: usize, matched: usize) {
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

    /// LLR消去第2パスのトリガー率
    pub fn llr_second_pass_trigger_ratio(&self) -> f32 {
        ratio(
            self.total_llr_second_pass_attempts,
            self.total_accepted_packets + self.total_crc_error_packets,
        )
    }

    /// LLR消去第2パスによる救済率
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

    /// 送信信号の平均電力
    pub fn tx_signal_power(&self) -> Option<f32> {
        if self.total_tx_signal_samples == 0 {
            None
        } else {
            Some((self.total_tx_signal_energy / self.total_tx_signal_samples as f64) as f32)
        }
    }

    /// サンプルあたりの平均処理時間 [ns/sample]
    pub fn avg_process_time_per_sample_ns(&self) -> f32 {
        if self.total_tx_signal_samples == 0 {
            0.0
        } else {
            self.total_process_time_ns as f32 / self.total_tx_signal_samples as f32
        }
    }

    /// FEC前（生）のビットエラーレート
    pub fn raw_ber(&self) -> f32 {
        if self.total_raw_bits_compared == 0 {
            f32::NAN
        } else {
            self.total_raw_bit_errors as f32 / self.total_raw_bits_compared as f32
        }
    }

    /// 生ビットのエラーラン平均長
    pub fn raw_err_run_mean(&self) -> Option<f32> {
        if self.total_raw_error_runs == 0 {
            None
        } else {
            Some(self.total_raw_error_run_bits as f32 / self.total_raw_error_runs as f32)
        }
    }

    /// 生ビットの最大エラーラン長
    pub fn raw_err_run_max(&self) -> Option<usize> {
        if self.max_raw_error_run_len == 0 {
            None
        } else {
            Some(self.max_raw_error_run_len)
        }
    }

    /// コードワードあたりの平均エラービット数
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

    /// FEC適用後のビットエラーレート
    pub fn post_ber(&self) -> f32 {
        if self.total_post_bits_compared == 0 {
            f32::NAN
        } else {
            self.total_post_bit_errors as f32 / self.total_post_bits_compared as f32
        }
    }

    /// Post-FECデコード結果が送信データと一致した割合
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

    /// 平均推定SNR [dB]
    pub fn avg_last_est_snr_db(&self) -> Option<f32> {
        if self.count_last_est_snr_db == 0 {
            None
        } else {
            Some((self.sum_last_est_snr_db / self.count_last_est_snr_db as f64) as f32)
        }
    }

    /// 設定値 sigma から計算した理論上のAWGN SNR [dB]
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

    /// 位相ゲート有効率
    pub fn phase_gate_on_ratio(&self) -> f32 {
        ratio(
            self.total_phase_gate_on_symbols,
            self.total_phase_gate_on_symbols + self.total_phase_gate_off_symbols,
        )
    }

    /// 位相変化棄却率
    pub fn phase_innovation_reject_ratio(&self) -> f32 {
        ratio(
            self.total_phase_innovation_reject_symbols,
            self.total_phase_gate_on_symbols + self.total_phase_gate_off_symbols,
        )
    }

    /// 平均絶対位相誤差 [rad]
    pub fn phase_err_abs_mean_rad(&self) -> Option<f32> {
        if self.total_phase_err_abs_count == 0 {
            None
        } else {
            Some((self.total_phase_err_abs_sum_rad / self.total_phase_err_abs_count as f64) as f32)
        }
    }

    /// 0.5 rad 以上の位相誤差が発生した割合
    pub fn phase_err_abs_ge_0p5_ratio(&self) -> f32 {
        ratio(
            self.total_phase_err_abs_ge_0p5_symbols,
            self.total_phase_err_abs_count,
        )
    }

    /// 1.0 rad 以上の位相誤差が発生した割合
    pub fn phase_err_abs_ge_1p0_ratio(&self) -> f32 {
        ratio(
            self.total_phase_err_abs_ge_1p0_symbols,
            self.total_phase_err_abs_count,
        )
    }
}

pub struct MetricContext<'a> {
    pub scenario: &'a str,
    pub phy: &'a str,
    pub mary_fde_mode: &'a str,
    pub payload_bits: usize,
    pub sigma: f32,
    pub multipath_name: &'a str,
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

        assert_eq!(m.p_complete(), 200.0 / 300.0); // 200 / (100 * 3)
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
        // goodput_effective_bps = (512 * 3) / 10.0 = 153.6
        assert_eq!(m.goodput_effective_bps(payload_bits), 153.6);
        // goodput_success_mean_bps = (512/2 + 512/4 + 512/6) / 3 = (256 + 128 + 85.33) / 3 = 156.444
        let expected_mean_bps = (512.0 / 2.0 + 512.0 / 4.0 + 512.0 / 6.0) / 3.0;
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
        assert_eq!(m.raw_ber(), 100.0 / 1000.0);

        // 6. LLR Rescue
        m.total_llr_second_pass_attempts = 20;
        m.total_llr_second_pass_rescued = 5;
        assert_eq!(m.llr_second_pass_trigger_ratio(), 20.0 / 240.0);
        assert_eq!(m.llr_second_pass_rescue_ratio(), 5.0 / 20.0);

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
}
