use clap::{builder::PossibleValuesParser, Parser, ValueEnum};
use dsp::coding::fec;
use dsp::dsss::encoder::{Encoder as DsssEncoder, EncoderConfig as DsssEncoderConfig};
use dsp::frame::packet::{Packet, PACKET_BYTES};
use dsp::mary::decoder::{CirNormalizationMode, Decoder as MaryDecoder};
use dsp::mary::encoder::Encoder as MaryEncoder;
use dsp::params::PAYLOAD_SIZE;
use dsp::{dsss::decoder::Decoder as DsssDecoder, DspConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
struct MultipathProfile {
    name: String,
    taps: Vec<(usize, f32)>,
}

impl MultipathProfile {
    fn none() -> Self {
        Self {
            name: "none".to_string(),
            taps: vec![(0, 1.0)],
        }
    }

    fn preset(name: &str) -> Option<Self> {
        match name {
            "none" => Some(Self::none()),
            "mild" => Some(Self {
                name: "mild".to_string(),
                taps: vec![(0, 1.0), (9, 0.40), (23, 0.22)],
            }),
            "medium" => Some(Self {
                name: "medium".to_string(),
                taps: vec![(0, 1.0), (9, 0.45), (23, 0.28), (49, 0.18)],
            }),
            "harsh" => Some(Self {
                name: "harsh".to_string(),
                taps: vec![(0, 1.0), (9, 0.55), (23, 0.35), (49, 0.25), (87, 0.18)],
            }),
            _ => None,
        }
    }

    fn parse_custom(spec: &str) -> Option<Self> {
        let mut taps = Vec::new();
        for pair in spec.split(',') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }
            let mut it = pair.split(':');
            let d = it.next()?.trim().parse::<usize>().ok()?;
            let g = it.next()?.trim().parse::<f32>().ok()?;
            if it.next().is_some() {
                return None;
            }
            taps.push((d, g));
        }
        if taps.is_empty() {
            return None;
        }
        taps.sort_by_key(|(d, _)| *d);
        Some(Self {
            name: "custom".to_string(),
            taps,
        })
    }

    fn max_delay(&self) -> usize {
        self.taps.iter().map(|(d, _)| *d).max().unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
struct ChannelImpairment {
    sigma: f32,
    cfo_hz: f32,
    ppm: f32,
    burst_loss: f32,
    fading_depth: f32,
    multipath: MultipathProfile,
}

impl FromStr for MultipathProfile {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let normalized = value.trim().to_ascii_lowercase();
        if let Some(preset) = Self::preset(&normalized) {
            return Ok(preset);
        }
        Self::parse_custom(value).ok_or_else(|| {
            format!(
                "invalid multipath: {value} (preset: none|mild|medium|harsh or taps: 0:1.0,9:0.4)"
            )
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Phy {
    Dsss,
    Mary,
}

impl Phy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Dsss => "dsss",
            Self::Mary => "mary",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum EvalMode {
    Point,
    SweepAwgn,
    SweepPpm,
    SweepLoss,
    SweepFading,
    SweepMultipath,
    SweepBand,
    SweepAll,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    Csv,
    Json,
    Table,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum MaryFdeMode {
    On,
    Off,
    Auto,
}

impl MaryFdeMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::On => "on",
            Self::Off => "off",
            Self::Auto => "auto",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "snake_case")]
enum CirNormArg {
    None,
    UnitEnergy,
    Peak,
}

impl From<CirNormArg> for CirNormalizationMode {
    fn from(value: CirNormArg) -> Self {
        match value {
            CirNormArg::None => CirNormalizationMode::None,
            CirNormArg::UnitEnergy => CirNormalizationMode::UnitEnergy,
            CirNormArg::Peak => CirNormalizationMode::Peak,
        }
    }
}

const DEFAULT_COLUMNS: &[&str] = &[
    "scenario",
    "phy",
    "mary_fde_mode",
    "trials",
    "awgn_snr_db",
    "p_complete",
    "ber",
    "raw_ber",
    "goodput_effective_bps",
    "goodput_success_mean_bps",
    "p95_complete_s",
    "mean_complete_s",
    "avg_proc_ns_sample",
    "synced_frame_ratio",
    "crc_pass_ratio",
    "llr_second_pass_trigger_ratio",
    "llr_second_pass_rescue_ratio",
    "phase_gate_on_ratio",
    "phase_innovation_reject_ratio",
    "phase_err_abs_mean_rad",
    "phase_err_abs_ge_0p5_ratio",
    "phase_err_abs_ge_1p0_ratio",
    "avg_last_est_snr_db",
    "multipath",
];
const ALL_COLUMNS: &[&str] = &[
    "scenario",
    "phy",
    "mary_fde_mode",
    "trials",
    "awgn_snr_db",
    "p_complete",
    "ber",
    "raw_ber",
    "goodput_effective_bps",
    "goodput_success_mean_bps",
    "p95_complete_s",
    "mean_complete_s",
    "avg_proc_ns_sample",
    "synced_frame_ratio",
    "crc_pass_ratio",
    "llr_second_pass_trigger_ratio",
    "llr_second_pass_rescue_ratio",
    "phase_gate_on_ratio",
    "phase_innovation_reject_ratio",
    "phase_err_abs_mean_rad",
    "phase_err_abs_ge_0p5_ratio",
    "phase_err_abs_ge_1p0_ratio",
    "avg_last_est_snr_db",
    "multipath",
    "raw_err_run_mean",
    "raw_err_run_max",
    "err_w_cw_mean",
    "err_w_cw_p50",
    "err_w_cw_p90",
    "err_w_cw_p99",
    "err_w_cw_max",
    "err_w_cw_hist",
    "post_ber",
    "post_decode_match_ratio",
    "post_err_run_mean",
    "post_err_run_max",
    "post_err_w_cw_mean",
    "post_err_w_cw_p50",
    "post_err_w_cw_p90",
    "post_err_w_cw_p99",
    "post_err_w_cw_max",
    "post_err_w_cw_hist",
];

fn parse_positive_f32(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("invalid float: {value}"))?;
    if parsed > 0.0 {
        Ok(parsed)
    } else {
        Err(format!("value must be > 0: {value}"))
    }
}

fn parse_nonnegative_f32(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("invalid float: {value}"))?;
    if parsed >= 0.0 {
        Ok(parsed)
    } else {
        Err(format!("value must be >= 0: {value}"))
    }
}

fn parse_unit_interval_f32(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("invalid float: {value}"))?;
    if (0.0..=1.0).contains(&parsed) {
        Ok(parsed)
    } else {
        Err(format!("value must be in [0,1]: {value}"))
    }
}

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("invalid integer: {value}"))?;
    if parsed > 0 {
        Ok(parsed)
    } else {
        Err(format!("value must be > 0: {value}"))
    }
}

#[derive(Parser, Clone, Debug)]
#[command(author, version, about = "DSSS/Mary E2E評価ツール", long_about = None)]
struct Cli {
    #[arg(long, value_enum, default_value_t = Phy::Dsss)]
    phy: Phy,
    #[arg(long, value_enum, default_value_t = EvalMode::Point)]
    mode: EvalMode,
    #[arg(long, value_parser = parse_positive_usize, default_value_t = 40)]
    trials: usize,
    #[arg(long = "payload-bytes", value_parser = parse_positive_usize, default_value_t = 64)]
    payload_bytes: usize,
    #[arg(long = "max-sec", value_parser = parse_positive_f32, default_value_t = 2.0)]
    max_sec: f32,
    #[arg(long = "chunk-samples", value_parser = parse_positive_usize, default_value_t = 16_384)]
    chunk_samples: usize,
    #[arg(long = "gap-samples", default_value_t = 64)]
    gap_samples: usize,
    #[arg(long, default_value_t = 0xA11CE_u64)]
    seed: u64,
    #[arg(long = "target-p", value_parser = parse_unit_interval_f32, default_value_t = 0.95)]
    target_p_complete: f32,
    #[arg(long, value_parser = parse_nonnegative_f32, default_value_t = 0.0)]
    sigma: f32,
    #[arg(long = "cfo-hz", default_value_t = 0.0)]
    cfo_hz: f32,
    #[arg(long, default_value_t = 0.0)]
    ppm: f32,
    #[arg(long = "burst-loss", value_parser = parse_unit_interval_f32, default_value_t = 0.0)]
    burst_loss: f32,
    #[arg(long = "fading-depth", value_parser = parse_unit_interval_f32, default_value_t = 0.0)]
    fading_depth: f32,
    #[arg(long, default_value = "none")]
    multipath: MultipathProfile,
    #[arg(long = "sweep-awgn", value_delimiter = ',', default_values_t = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])]
    sweep_awgn: Vec<f32>,
    #[arg(long = "sweep-ppm", value_delimiter = ',', default_values_t = [-120.0, -80.0, -40.0, 0.0, 40.0, 80.0, 120.0])]
    sweep_ppm: Vec<f32>,
    #[arg(long = "sweep-loss", value_delimiter = ',', default_values_t = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4])]
    sweep_loss: Vec<f32>,
    #[arg(long = "sweep-fading", value_delimiter = ',', default_values_t = [0.0, 0.2, 0.4, 0.6, 0.8])]
    sweep_fading: Vec<f32>,
    #[arg(long = "sweep-chip-rate", value_delimiter = ',', default_values_t = [6000.0, 8000.0, 10000.0, 12000.0, 14000.0, 15000.0])]
    sweep_chip_rate: Vec<f32>,
    #[arg(long = "sweep-carrier-freq", value_delimiter = ',', default_values_t = [8000.0, 10000.0, 12000.0, 14000.0, 15000.0])]
    sweep_carrier_freq: Vec<f32>,
    #[arg(long = "sample-rate", default_value_t = dsp::params::DEFAULT_SAMPLE_RATE)]
    sample_rate: f32,
    #[arg(long = "chip-rate", default_value_t = dsp::params::CHIP_RATE)]
    chip_rate: f32,
    #[arg(long = "carrier-freq", default_value_t = dsp::params::CARRIER_FREQ)]
    carrier_freq: f32,
    #[arg(long = "mseq-order", default_value_t = dsp::params::MSEQ_ORDER)]
    mseq_order: usize,
    #[arg(long = "rrc-alpha", default_value_t = dsp::params::RRC_ALPHA)]
    rrc_alpha: f32,
    #[arg(long = "sync-word-bits", default_value_t = dsp::params::SYNC_WORD_BITS)]
    sync_word_bits: usize,
    #[arg(long = "preamble-repeat", default_value_t = dsp::params::PREAMBLE_REPEAT)]
    preamble_repeat: usize,
    #[arg(long = "packets-per-burst", default_value_t = dsp::params::PACKETS_PER_SYNC_BURST)]
    packets_per_burst: usize,
    #[arg(long = "preamble-sf", default_value_t = dsp::params::PREAMBLE_SF)]
    preamble_sf: usize,
    #[arg(long = "mary-fde", value_enum, default_value_t = MaryFdeMode::On)]
    mary_fde_mode: MaryFdeMode,
    #[arg(long = "mary-fde-snr-db", default_value_t = 15.0)]
    mary_fde_snr_db: f32,
    #[arg(long = "mary-fde-k", default_value_t = 1.0)]
    mary_fde_lambda_scale: f32,
    #[arg(long = "mary-fde-lambda-floor", default_value_t = 0.0)]
    mary_fde_lambda_floor: f32,
    #[arg(long = "mary-fde-max-inv-gain", value_parser = parse_positive_f32)]
    mary_fde_max_inv_gain: Option<f32>,
    #[arg(long = "mary-cir-norm", value_enum, default_value_t = CirNormArg::None)]
    mary_cir_norm: CirNormArg,
    #[arg(long = "mary-cir-tap-alpha", default_value_t = 0.0)]
    mary_cir_tap_alpha: f32,
    #[arg(
        long = "mary-viterbi-list",
        value_parser = parse_positive_usize,
        default_value_t = 1
    )]
    mary_viterbi_list: usize,
    #[arg(long = "mary-llr-erasure-second-pass")]
    mary_llr_erasure_second_pass: bool,
    #[arg(
        long = "mary-llr-erasure-q",
        value_parser = parse_unit_interval_f32,
        default_value_t = 0.2
    )]
    mary_llr_erasure_q: f32,
    #[arg(
        long = "mary-llr-erasure-list",
        value_parser = parse_positive_usize,
        default_value_t = 8
    )]
    mary_llr_erasure_list: usize,
    #[arg(
        long = "columns",
        value_delimiter = ',',
        value_parser = PossibleValuesParser::new(ALL_COLUMNS)
    )]
    columns: Option<Vec<String>>,
    #[arg(long = "output", value_enum, default_value_t = OutputFormat::Csv)]
    output: OutputFormat,
}

impl Cli {
    fn base_impairment(&self) -> ChannelImpairment {
        ChannelImpairment {
            sigma: self.sigma,
            cfo_hz: self.cfo_hz,
            ppm: self.ppm,
            burst_loss: self.burst_loss,
            fading_depth: self.fading_depth,
            multipath: self.multipath.clone(),
        }
    }
}

#[derive(Default, Clone, Debug)]
struct TrialResult {
    success: bool,
    completion_sec: Option<f32>,
    elapsed_sec: f32,
    attempts: usize,
    /// 同期成立したフレーム数
    synced_frames: usize,
    /// CRC通過フレーム数（Accepted）
    accepted_frames: usize,
    /// CRCエラーフレーム数
    crc_error_frames: usize,
    first_attempt_success: bool,
    bit_errors: usize,
    bits_compared: usize,
    dropped_attempts: usize,
    tx_signal_energy_sum: f64,
    tx_signal_samples: usize,
    process_time_ns: u64,
    /// FEC符号化ビットのハード判定BER（Fountain符号の外側の生BER）
    raw_bit_errors: usize,
    raw_bits_compared: usize,
    raw_error_runs: usize,
    raw_error_run_bits: usize,
    raw_error_run_max: usize,
    codeword_count: usize,
    codeword_error_sum: usize,
    codeword_error_max: usize,
    codeword_error_weights: Vec<usize>,
    post_bit_errors: usize,
    post_bits_compared: usize,
    post_error_runs: usize,
    post_error_run_bits: usize,
    post_error_run_max: usize,
    post_codeword_count: usize,
    post_codeword_error_sum: usize,
    post_codeword_error_max: usize,
    post_codeword_error_weights: Vec<usize>,
    post_decode_attempts: usize,
    post_decode_matched: usize,
    last_est_snr_db: f32,
    phase_gate_on_symbols: usize,
    phase_gate_off_symbols: usize,
    phase_innovation_reject_symbols: usize,
    phase_err_abs_sum_rad: f64,
    phase_err_abs_count: usize,
    phase_err_abs_ge_0p5_symbols: usize,
    phase_err_abs_ge_1p0_symbols: usize,
    llr_second_pass_attempts: usize,
    llr_second_pass_rescued: usize,
}

#[derive(Default, Clone, Debug)]
struct Metrics {
    trials: usize,
    successes: usize,
    first_attempt_successes: usize,
    total_attempts: usize,
    total_synced_frames: usize,
    total_accepted_frames: usize,
    total_crc_error_frames: usize,
    total_bit_errors: usize,
    total_bits_compared: usize,
    total_elapsed_sec: f32,
    dropped_attempts: usize,
    total_tx_signal_energy: f64,
    total_tx_signal_samples: usize,
    total_process_time_ns: u64,
    completion_secs: Vec<f32>,
    total_raw_bit_errors: usize,
    total_raw_bits_compared: usize,
    total_raw_error_runs: usize,
    total_raw_error_run_bits: usize,
    max_raw_error_run_len: usize,
    total_codewords: usize,
    total_codeword_error_sum: usize,
    max_codeword_error: usize,
    codeword_error_weights: Vec<usize>,
    total_post_bit_errors: usize,
    total_post_bits_compared: usize,
    total_post_error_runs: usize,
    total_post_error_run_bits: usize,
    max_post_error_run_len: usize,
    total_post_codewords: usize,
    total_post_codeword_error_sum: usize,
    max_post_codeword_error: usize,
    post_codeword_error_weights: Vec<usize>,
    total_post_decode_attempts: usize,
    total_post_decode_matched: usize,
    sum_last_est_snr_db: f64,
    count_last_est_snr_db: usize,
    total_phase_gate_on_symbols: usize,
    total_phase_gate_off_symbols: usize,
    total_phase_innovation_reject_symbols: usize,
    total_phase_err_abs_sum_rad: f64,
    total_phase_err_abs_count: usize,
    total_phase_err_abs_ge_0p5_symbols: usize,
    total_phase_err_abs_ge_1p0_symbols: usize,
    total_llr_second_pass_attempts: usize,
    total_llr_second_pass_rescued: usize,
}

const TX_WARMUP_SAMPLES: usize = 4096;
const TX_TAIL_SAMPLES: usize = 4096;

impl Metrics {
    fn push(&mut self, t: TrialResult) {
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

    fn p_complete(&self) -> f32 {
        ratio(self.successes, self.trials)
    }

    fn synced_frame_ratio(&self) -> f32 {
        ratio(self.total_synced_frames, self.total_attempts)
    }

    fn crc_pass_ratio(&self) -> f32 {
        ratio(
            self.total_accepted_frames,
            self.total_accepted_frames + self.total_crc_error_frames,
        )
    }

    fn llr_second_pass_trigger_ratio(&self) -> f32 {
        ratio(
            self.total_llr_second_pass_attempts,
            self.total_accepted_frames + self.total_crc_error_frames,
        )
    }

    fn llr_second_pass_rescue_ratio(&self) -> f32 {
        ratio(
            self.total_llr_second_pass_rescued,
            self.total_llr_second_pass_attempts,
        )
    }

    fn ber(&self) -> f32 {
        if self.total_bits_compared == 0 {
            0.0
        } else {
            self.total_bit_errors as f32 / self.total_bits_compared as f32
        }
    }

    fn p95_completion_sec(&self) -> Option<f32> {
        quantile(&self.completion_secs, 0.95)
    }

    fn mean_completion_sec(&self) -> Option<f32> {
        if self.completion_secs.is_empty() {
            None
        } else {
            Some(self.completion_secs.iter().sum::<f32>() / self.completion_secs.len() as f32)
        }
    }

    fn goodput_effective_bps(&self, payload_bits: usize) -> f32 {
        if self.total_elapsed_sec <= 0.0 {
            0.0
        } else {
            (payload_bits * self.successes) as f32 / self.total_elapsed_sec
        }
    }

    fn goodput_success_mean_bps(&self, payload_bits: usize) -> Option<f32> {
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

    fn tx_signal_power(&self) -> Option<f32> {
        if self.total_tx_signal_samples == 0 {
            None
        } else {
            Some((self.total_tx_signal_energy / self.total_tx_signal_samples as f64) as f32)
        }
    }

    fn avg_process_time_per_sample_ns(&self) -> f32 {
        if self.total_tx_signal_samples == 0 {
            0.0
        } else {
            self.total_process_time_ns as f32 / self.total_tx_signal_samples as f32
        }
    }

    fn raw_ber(&self) -> f32 {
        if self.total_raw_bits_compared == 0 {
            f32::NAN
        } else {
            self.total_raw_bit_errors as f32 / self.total_raw_bits_compared as f32
        }
    }

    fn raw_err_run_mean(&self) -> Option<f32> {
        if self.total_raw_error_runs == 0 {
            None
        } else {
            Some(self.total_raw_error_run_bits as f32 / self.total_raw_error_runs as f32)
        }
    }

    fn raw_err_run_max(&self) -> Option<usize> {
        if self.max_raw_error_run_len == 0 {
            None
        } else {
            Some(self.max_raw_error_run_len)
        }
    }

    fn err_w_cw_mean(&self) -> Option<f32> {
        if self.total_codewords == 0 {
            None
        } else {
            Some(self.total_codeword_error_sum as f32 / self.total_codewords as f32)
        }
    }

    fn err_w_cw_p50(&self) -> Option<f32> {
        quantile_usize(&self.codeword_error_weights, 0.5)
    }

    fn err_w_cw_p90(&self) -> Option<f32> {
        quantile_usize(&self.codeword_error_weights, 0.9)
    }

    fn err_w_cw_p99(&self) -> Option<f32> {
        quantile_usize(&self.codeword_error_weights, 0.99)
    }

    fn err_w_cw_max(&self) -> Option<usize> {
        if self.max_codeword_error == 0 && self.total_codewords == 0 {
            None
        } else {
            Some(self.max_codeword_error)
        }
    }

    fn err_w_cw_hist(&self) -> Option<String> {
        error_weight_hist(&self.codeword_error_weights)
    }

    fn post_ber(&self) -> f32 {
        if self.total_post_bits_compared == 0 {
            f32::NAN
        } else {
            self.total_post_bit_errors as f32 / self.total_post_bits_compared as f32
        }
    }

    fn post_decode_match_ratio(&self) -> f32 {
        ratio(
            self.total_post_decode_matched,
            self.total_post_decode_attempts,
        )
    }

    fn post_err_run_mean(&self) -> Option<f32> {
        if self.total_post_error_runs == 0 {
            None
        } else {
            Some(self.total_post_error_run_bits as f32 / self.total_post_error_runs as f32)
        }
    }

    fn post_err_run_max(&self) -> Option<usize> {
        if self.max_post_error_run_len == 0 {
            None
        } else {
            Some(self.max_post_error_run_len)
        }
    }

    fn post_err_w_cw_mean(&self) -> Option<f32> {
        if self.total_post_codewords == 0 {
            None
        } else {
            Some(self.total_post_codeword_error_sum as f32 / self.total_post_codewords as f32)
        }
    }

    fn post_err_w_cw_p50(&self) -> Option<f32> {
        quantile_usize(&self.post_codeword_error_weights, 0.5)
    }

    fn post_err_w_cw_p90(&self) -> Option<f32> {
        quantile_usize(&self.post_codeword_error_weights, 0.9)
    }

    fn post_err_w_cw_p99(&self) -> Option<f32> {
        quantile_usize(&self.post_codeword_error_weights, 0.99)
    }

    fn post_err_w_cw_max(&self) -> Option<usize> {
        if self.max_post_codeword_error == 0 && self.total_post_codewords == 0 {
            None
        } else {
            Some(self.max_post_codeword_error)
        }
    }

    fn post_err_w_cw_hist(&self) -> Option<String> {
        error_weight_hist(&self.post_codeword_error_weights)
    }

    fn avg_last_est_snr_db(&self) -> Option<f32> {
        if self.count_last_est_snr_db == 0 {
            None
        } else {
            Some((self.sum_last_est_snr_db / self.count_last_est_snr_db as f64) as f32)
        }
    }

    fn awgn_snr_db(&self, sigma: f32) -> Option<f32> {
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

    fn phase_gate_on_ratio(&self) -> f32 {
        ratio(
            self.total_phase_gate_on_symbols,
            self.total_phase_gate_on_symbols + self.total_phase_gate_off_symbols,
        )
    }

    fn phase_innovation_reject_ratio(&self) -> f32 {
        ratio(
            self.total_phase_innovation_reject_symbols,
            self.total_phase_gate_on_symbols + self.total_phase_gate_off_symbols,
        )
    }

    fn phase_err_abs_mean_rad(&self) -> Option<f32> {
        if self.total_phase_err_abs_count == 0 {
            None
        } else {
            Some((self.total_phase_err_abs_sum_rad / self.total_phase_err_abs_count as f64) as f32)
        }
    }

    fn phase_err_abs_ge_0p5_ratio(&self) -> f32 {
        ratio(
            self.total_phase_err_abs_ge_0p5_symbols,
            self.total_phase_err_abs_count,
        )
    }

    fn phase_err_abs_ge_1p0_ratio(&self) -> f32 {
        ratio(
            self.total_phase_err_abs_ge_1p0_symbols,
            self.total_phase_err_abs_count,
        )
    }
}

fn ratio(num: usize, den: usize) -> f32 {
    if den == 0 {
        0.0
    } else {
        num as f32 / den as f32
    }
}

fn quantile(values: &[f32], q: f32) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(f32::total_cmp);
    let q = q.clamp(0.0, 1.0);
    let idx = (((v.len() - 1) as f32) * q).round() as usize;
    v.get(idx).copied()
}

fn quantile_usize(values: &[usize], q: f32) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_unstable();
    let q = q.clamp(0.0, 1.0);
    let idx = (((v.len() - 1) as f32) * q).round() as usize;
    v.get(idx).map(|&x| x as f32)
}

fn error_weight_hist(weights: &[usize]) -> Option<String> {
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

fn apply_mary_fde_mode(decoder: &mut MaryDecoder, mode: MaryFdeMode) {
    match mode {
        MaryFdeMode::On => {
            decoder.set_fde_enabled(true);
            decoder.set_fde_auto_path_select(false);
        }
        MaryFdeMode::Off => {
            decoder.set_fde_enabled(false);
            decoder.set_fde_auto_path_select(false);
        }
        MaryFdeMode::Auto => {
            decoder.set_fde_enabled(true);
            decoder.set_fde_auto_path_select(true);
        }
    }
}

fn selected_columns(cli: &Cli) -> Vec<&str> {
    match &cli.columns {
        Some(columns) => columns.iter().map(String::as_str).collect(),
        None => DEFAULT_COLUMNS.to_vec(),
    }
}

fn parse_cli() -> Cli {
    Cli::parse()
}

fn make_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.gen::<u8>()).collect()
}

fn count_bit_errors_bytes(tx: &[u8], rx: Option<&[u8]>) -> usize {
    let Some(rx) = rx else {
        return tx.len() * 8;
    };
    let mut errs = 0usize;
    for (idx, &b) in tx.iter().enumerate() {
        let rb = *rx.get(idx).unwrap_or(&0u8);
        errs += (b ^ rb).count_ones() as usize;
    }
    errs
}

fn bit_errors_and_runs_from_llr(
    expected: &[u8],
    llrs: &[f32],
) -> (usize, usize, usize, usize, usize) {
    let compare_len = expected.len().min(llrs.len());
    let mut errors = 0usize;
    let mut runs = 0usize;
    let mut run_bits = 0usize;
    let mut run_max = 0usize;
    let mut cur_run = 0usize;
    for j in 0..compare_len {
        let bit = expected[j];
        let llr = llrs[j];
        let is_err = (bit == 0 && llr <= 0.0) || (bit == 1 && llr >= 0.0);
        if is_err {
            errors += 1;
            cur_run += 1;
        } else if cur_run > 0 {
            runs += 1;
            run_bits += cur_run;
            run_max = run_max.max(cur_run);
            cur_run = 0;
        }
    }
    if cur_run > 0 {
        runs += 1;
        run_bits += cur_run;
        run_max = run_max.max(cur_run);
    }
    (errors, compare_len, runs, run_bits, run_max)
}

fn bit_errors_and_runs_from_bits(
    expected: &[u8],
    observed: &[u8],
) -> (usize, usize, usize, usize, usize) {
    let compare_len = expected.len().min(observed.len());
    let mut errors = 0usize;
    let mut runs = 0usize;
    let mut run_bits = 0usize;
    let mut run_max = 0usize;
    let mut cur_run = 0usize;
    for j in 0..compare_len {
        let is_err = expected[j] != observed[j];
        if is_err {
            errors += 1;
            cur_run += 1;
        } else if cur_run > 0 {
            runs += 1;
            run_bits += cur_run;
            run_max = run_max.max(cur_run);
            cur_run = 0;
        }
    }
    if cur_run > 0 {
        runs += 1;
        run_bits += cur_run;
        run_max = run_max.max(cur_run);
    }
    (errors, compare_len, runs, run_bits, run_max)
}

fn add_awgn_with_rng(samples: &mut [f32], sigma: f32, rng: &mut StdRng) {
    if sigma <= 0.0 {
        return;
    }
    let normal = Normal::new(0.0, sigma).expect("normal distribution");
    for s in samples {
        *s += normal.sample(rng);
    }
}

fn apply_multipath(input: &[f32], mp: &MultipathProfile) -> Vec<f32> {
    if mp.taps.len() == 1 && mp.taps[0].0 == 0 && (mp.taps[0].1 - 1.0).abs() < 1e-6 {
        return input.to_vec();
    }
    let max_delay = mp.max_delay();
    let mut out = vec![0.0f32; input.len() + max_delay];
    for &(delay, gain) in &mp.taps {
        for (i, &x) in input.iter().enumerate() {
            out[i + delay] += gain * x;
        }
    }
    let norm = mp
        .taps
        .iter()
        .map(|(_, g)| g * g)
        .sum::<f32>()
        .sqrt()
        .max(1e-6);
    for s in &mut out {
        *s /= norm;
    }
    out
}

fn apply_clock_drift_ppm(input: &[f32], ppm: f32) -> Vec<f32> {
    if input.is_empty() || ppm.abs() < 1.0 {
        return input.to_vec();
    }
    let period = (1_000_000.0 / ppm.abs()).round() as usize;
    if period < 2 {
        return input.to_vec();
    }
    if ppm > 0.0 {
        let mut out = Vec::with_capacity(input.len() + input.len() / period + 8);
        for (i, &s) in input.iter().enumerate() {
            out.push(s);
            if (i + 1) % period == 0 {
                out.push(s);
            }
        }
        out
    } else {
        let mut out = Vec::with_capacity(input.len().saturating_sub(input.len() / period));
        for (i, &s) in input.iter().enumerate() {
            if (i + 1) % period == 0 {
                continue;
            }
            out.push(s);
        }
        out
    }
}

fn apply_fading(sig: &mut [f32], depth: f32, rng: &mut StdRng) {
    if sig.is_empty() || depth <= 0.0 {
        return;
    }
    let d = depth.clamp(0.0, 1.0);
    let g0 = 1.0 - d * rng.gen::<f32>();
    let g1 = 1.0 - d * rng.gen::<f32>();
    let n = sig.len().max(2);
    for (i, s) in sig.iter_mut().enumerate() {
        let t = i as f32 / (n - 1) as f32;
        let g = g0 + (g1 - g0) * t;
        *s *= g;
    }
}

fn apply_channel(
    tx: &[f32],
    imp: &ChannelImpairment,
    rng: &mut StdRng,
    drop_burst: bool,
) -> Vec<f32> {
    let mut sig = if drop_burst {
        vec![0.0f32; tx.len()]
    } else {
        apply_multipath(tx, &imp.multipath)
    };
    if imp.ppm.abs() >= 1.0 {
        sig = apply_clock_drift_ppm(&sig, imp.ppm);
    }
    apply_fading(&mut sig, imp.fading_depth, rng);
    add_awgn_with_rng(&mut sig, imp.sigma, rng);
    sig
}

fn signal_energy(samples: &[f32]) -> f64 {
    samples
        .iter()
        .map(|&x| {
            let v = x as f64;
            v * v
        })
        .sum()
}

fn run_trial_dsss_e2e(imp: &ChannelImpairment, cli: &Cli, seed: u64) -> TrialResult {
    let mut tx_cfg = DspConfig::new(cli.sample_rate);
    tx_cfg.chip_rate = cli.chip_rate;
    tx_cfg.carrier_freq = cli.carrier_freq;
    tx_cfg.mseq_order = cli.mseq_order;
    tx_cfg.rrc_alpha = cli.rrc_alpha;
    tx_cfg.sync_word_bits = cli.sync_word_bits;
    tx_cfg.preamble_repeat = cli.preamble_repeat;
    tx_cfg.packets_per_burst = cli.packets_per_burst;
    tx_cfg.preamble_sf = cli.preamble_sf;

    let mut rx_cfg = tx_cfg.clone();
    rx_cfg.carrier_freq += imp.cfo_hz;

    let payload = make_bytes(cli.payload_bytes, seed ^ 0x1234_5678);
    let k = payload.len().div_ceil(PAYLOAD_SIZE).max(1);
    let mut enc_cfg = DsssEncoderConfig::new(tx_cfg.clone());
    enc_cfg.fountain_k = k;
    enc_cfg.packets_per_sync_burst = cli.packets_per_burst;
    let mut encoder = DsssEncoder::new(enc_cfg);
    let mut decoder = DsssDecoder::new(payload.len(), k, rx_cfg);
    decoder.config.packets_per_burst = cli.packets_per_burst;

    let mut rng = StdRng::seed_from_u64(seed ^ 0xD55A_0001);
    let mut elapsed_sec = 0.0f32;
    let mut attempts = 0usize;
    let mut dropped_attempts = 0usize;
    let mut tx_signal_energy_sum = 0.0f64;
    let mut tx_signal_samples = 0usize;
    let mut total_process_ns = 0u64;

    let chunk = cli.chunk_samples.max(1);
    let gap = cli.gap_samples;
    let warmup = encoder.modulate_silence(TX_WARMUP_SAMPLES);
    tx_signal_energy_sum += signal_energy(&warmup);
    tx_signal_samples += warmup.len();
    let warmup_rx = apply_channel(&warmup, imp, &mut rng, false);
    elapsed_sec += warmup_rx.len() as f32 / tx_cfg.sample_rate;
    for piece in warmup_rx.chunks(chunk) {
        let start_time = std::time::Instant::now();
        let _ = decoder.process_samples(piece);
        total_process_ns += start_time.elapsed().as_nanos() as u64;
    }
    {
        let mut stream = encoder.encode_stream(&payload);
        loop {
            if elapsed_sec >= cli.max_sec {
                break;
            }
            let Some(frame) = stream.next() else {
                break;
            };
            attempts += 1;
            tx_signal_energy_sum += signal_energy(&frame);
            tx_signal_samples += frame.len();

            let drop_burst = rng.gen::<f32>() < imp.burst_loss;
            if drop_burst {
                dropped_attempts += 1;
            }
            let rx_frame = apply_channel(&frame, imp, &mut rng, drop_burst);
            elapsed_sec += rx_frame.len() as f32 / tx_cfg.sample_rate;
            for piece in rx_frame.chunks(chunk) {
                let start_time = std::time::Instant::now();
                let progress = decoder.process_samples(piece);
                total_process_ns += start_time.elapsed().as_nanos() as u64;

                if progress.complete {
                    let recovered = decoder.recovered_data();
                    let errs = count_bit_errors_bytes(&payload, recovered);
                    let bits_compared = payload.len() * 8;
                    return TrialResult {
                        success: errs == 0,
                        completion_sec: Some(elapsed_sec),
                        elapsed_sec,
                        attempts,
                        synced_frames: progress.synced_frames,
                        accepted_frames: progress.received_packets,
                        crc_error_frames: progress.crc_error_packets,
                        first_attempt_success: attempts == 1 && errs == 0,
                        bit_errors: errs,
                        bits_compared,
                        dropped_attempts,
                        tx_signal_energy_sum,
                        tx_signal_samples,
                        process_time_ns: total_process_ns,
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
                    };
                }
            }

            if gap > 0 {
                let mut gap_sig = vec![0.0f32; gap];
                add_awgn_with_rng(&mut gap_sig, imp.sigma, &mut rng);
                if imp.ppm.abs() >= 1.0 {
                    gap_sig = apply_clock_drift_ppm(&gap_sig, imp.ppm);
                }
                elapsed_sec += gap_sig.len() as f32 / tx_cfg.sample_rate;
                for piece in gap_sig.chunks(chunk) {
                    let start_time = std::time::Instant::now();
                    let progress = decoder.process_samples(piece);
                    total_process_ns += start_time.elapsed().as_nanos() as u64;

                    if progress.complete {
                        let recovered = decoder.recovered_data();
                        let errs = count_bit_errors_bytes(&payload, recovered);
                        let bits_compared = payload.len() * 8;
                        return TrialResult {
                            success: errs == 0,
                            completion_sec: Some(elapsed_sec),
                            elapsed_sec,
                            attempts,
                            synced_frames: progress.synced_frames,
                            accepted_frames: progress.received_packets,
                            crc_error_frames: progress.crc_error_packets,
                            first_attempt_success: attempts == 1 && errs == 0,
                            bit_errors: errs,
                            bits_compared,
                            dropped_attempts,
                            tx_signal_energy_sum,
                            tx_signal_samples,
                            process_time_ns: total_process_ns,
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
                        };
                    }
                }
            }
        }
    }

    let flush = encoder.flush();
    if !flush.is_empty() {
        tx_signal_energy_sum += signal_energy(&flush);
        tx_signal_samples += flush.len();
        let flush_rx = apply_channel(&flush, imp, &mut rng, false);
        elapsed_sec += flush_rx.len() as f32 / tx_cfg.sample_rate;
        for piece in flush_rx.chunks(chunk) {
            let start_time = std::time::Instant::now();
            let _ = decoder.process_samples(piece);
            total_process_ns += start_time.elapsed().as_nanos() as u64;
        }
    }

    let tail = encoder.modulate_silence(TX_TAIL_SAMPLES);
    if !tail.is_empty() {
        tx_signal_energy_sum += signal_energy(&tail);
        tx_signal_samples += tail.len();
        let tail_rx = apply_channel(&tail, imp, &mut rng, false);
        elapsed_sec += tail_rx.len() as f32 / tx_cfg.sample_rate;
        for piece in tail_rx.chunks(chunk) {
            let start_time = std::time::Instant::now();
            let _ = decoder.process_samples(piece);
            total_process_ns += start_time.elapsed().as_nanos() as u64;
        }
    }

    let bits_compared = payload.len() * 8;
    let bit_errors = count_bit_errors_bytes(&payload, decoder.recovered_data());
    let final_progress = decoder.process_samples(&[]);
    TrialResult {
        success: false,
        completion_sec: None,
        elapsed_sec,
        attempts,
        synced_frames: final_progress.synced_frames,
        accepted_frames: final_progress.received_packets,
        crc_error_frames: final_progress.crc_error_packets,
        first_attempt_success: false,
        bit_errors,
        bits_compared,
        dropped_attempts,
        tx_signal_energy_sum,
        tx_signal_samples,
        process_time_ns: total_process_ns,
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

fn run_trial_mary_e2e(imp: &ChannelImpairment, cli: &Cli, seed: u64) -> TrialResult {
    let mut tx_cfg = dsp::mary::params::dsp_config(cli.sample_rate);
    tx_cfg.chip_rate = cli.chip_rate;
    tx_cfg.carrier_freq = cli.carrier_freq;
    tx_cfg.mseq_order = cli.mseq_order;
    tx_cfg.rrc_alpha = cli.rrc_alpha;
    tx_cfg.sync_word_bits = cli.sync_word_bits;
    tx_cfg.preamble_repeat = cli.preamble_repeat;
    tx_cfg.packets_per_burst = cli.packets_per_burst;
    tx_cfg.preamble_sf = cli.preamble_sf;

    let mut rx_cfg = tx_cfg.clone();
    rx_cfg.carrier_freq += imp.cfo_hz;

    let payload = make_bytes(cli.payload_bytes, seed ^ 0x1234_5678);
    let k = payload.len().div_ceil(PAYLOAD_SIZE).max(1);

    let mut encoder = MaryEncoder::new(tx_cfg.clone());
    encoder.set_data(&payload);

    let mut decoder = MaryDecoder::new(payload.len(), k, rx_cfg);
    decoder.config.packets_per_burst = cli.packets_per_burst;
    apply_mary_fde_mode(&mut decoder, cli.mary_fde_mode);
    decoder.set_fde_mmse_settings(
        cli.mary_fde_snr_db,
        cli.mary_fde_lambda_scale,
        cli.mary_fde_lambda_floor,
        cli.mary_fde_max_inv_gain,
    );
    decoder.set_cir_postprocess(cli.mary_cir_norm.into(), cli.mary_cir_tap_alpha);
    decoder.set_viterbi_list_size(cli.mary_viterbi_list);
    decoder.set_llr_erasure_second_pass(
        cli.mary_llr_erasure_second_pass,
        cli.mary_llr_erasure_q,
        cli.mary_llr_erasure_list,
    );

    let mut rng = StdRng::seed_from_u64(seed ^ 0xD55A_0001);
    let mut elapsed_sec = 0.0f32;
    let mut attempts = 0usize;
    let mut dropped_attempts = 0usize;
    let mut tx_signal_energy_sum = 0.0f64;
    let mut tx_signal_samples = 0usize;
    let mut total_process_ns = 0u64;

    let chunk = cli.chunk_samples.max(1);
    let gap = cli.gap_samples;

    let fountain_params = dsp::coding::fountain::FountainParams::new(k, PAYLOAD_SIZE);
    let mut fountain_encoder =
        dsp::coding::fountain::FountainEncoder::new(&payload, fountain_params);

    // FEC前/後のBER計測用: 送信時の期待ビット列を seq -> bits で記録し、
    // 受信側LLRから復元した seq と照合して到着順ズレの影響を避けて集計する。
    let expected_fec_bits: Arc<Mutex<HashMap<u16, Vec<u8>>>> = Arc::new(Mutex::new(HashMap::new()));
    let expected_packet_bits: Arc<Mutex<HashMap<u16, Vec<u8>>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let raw_bit_errors_acc: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let raw_bits_compared_acc: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let raw_error_runs_acc: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let raw_error_run_bits_acc: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let raw_error_run_max_acc: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let codeword_error_weights_acc: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let post_bit_errors_acc: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let post_bits_compared_acc: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let post_error_runs_acc: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let post_error_run_bits_acc: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let post_error_run_max_acc: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let post_codeword_error_weights_acc: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let post_decode_attempts_acc: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let post_decode_matched_acc: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    {
        let efb = Arc::clone(&expected_fec_bits);
        let epb = Arc::clone(&expected_packet_bits);
        let rbe = Arc::clone(&raw_bit_errors_acc);
        let rbc = Arc::clone(&raw_bits_compared_acc);
        let rer = Arc::clone(&raw_error_runs_acc);
        let rrb = Arc::clone(&raw_error_run_bits_acc);
        let rrm = Arc::clone(&raw_error_run_max_acc);
        let cew = Arc::clone(&codeword_error_weights_acc);
        let pbe = Arc::clone(&post_bit_errors_acc);
        let pbc = Arc::clone(&post_bits_compared_acc);
        let per = Arc::clone(&post_error_runs_acc);
        let prb = Arc::clone(&post_error_run_bits_acc);
        let prm = Arc::clone(&post_error_run_max_acc);
        let pcew = Arc::clone(&post_codeword_error_weights_acc);
        let pda = Arc::clone(&post_decode_attempts_acc);
        let pdm = Arc::clone(&post_decode_matched_acc);
        decoder.llr_callback = Some(Box::new(move |llrs: &[f32]| {
            let decoded_bits = fec::decode_soft(llrs);
            let p_bits_len = PACKET_BYTES * 8;
            if decoded_bits.len() < p_bits_len {
                return;
            }
            let decoded_bytes = fec::bits_to_bytes(&decoded_bits[..p_bits_len]);
            if decoded_bytes.len() < 3 {
                return;
            }
            *pda.lock().unwrap() += 1;
            // CRC検証は通さず、ヘッダ先頭3byteから seq を直接復元して照合する。
            let seq = u16::from_be_bytes([decoded_bytes[1], decoded_bytes[2]]);
            let expected_raw = match efb.lock().unwrap().get(&seq) {
                Some(bits) => bits.clone(),
                None => return,
            };
            let (errors, compare_len, runs, run_bits, run_max) =
                bit_errors_and_runs_from_llr(&expected_raw, llrs);
            *rbe.lock().unwrap() += errors;
            *rbc.lock().unwrap() += compare_len;
            *rer.lock().unwrap() += runs;
            *rrb.lock().unwrap() += run_bits;
            let mut max_guard = rrm.lock().unwrap();
            *max_guard = (*max_guard).max(run_max);
            cew.lock().unwrap().push(errors);

            let expected_post = match epb.lock().unwrap().get(&seq) {
                Some(bits) => bits.clone(),
                None => return,
            };
            *pdm.lock().unwrap() += 1;
            let (post_errors, post_compare_len, post_runs, post_run_bits, post_run_max) =
                bit_errors_and_runs_from_bits(&expected_post, &decoded_bits[..p_bits_len]);
            *pbe.lock().unwrap() += post_errors;
            *pbc.lock().unwrap() += post_compare_len;
            *per.lock().unwrap() += post_runs;
            *prb.lock().unwrap() += post_run_bits;
            let mut post_max_guard = prm.lock().unwrap();
            *post_max_guard = (*post_max_guard).max(post_run_max);
            pcew.lock().unwrap().push(post_errors);
        }));
    }

    let warmup = encoder.modulate_silence(TX_WARMUP_SAMPLES);
    tx_signal_energy_sum += signal_energy(&warmup);
    tx_signal_samples += warmup.len();
    let warmup_rx = apply_channel(&warmup, imp, &mut rng, false);
    elapsed_sec += warmup_rx.len() as f32 / tx_cfg.sample_rate;
    for piece in warmup_rx.chunks(chunk) {
        let start_time = std::time::Instant::now();
        let _ = decoder.process_samples(piece);
        total_process_ns += start_time.elapsed().as_nanos() as u64;
    }

    loop {
        if elapsed_sec >= cli.max_sec {
            break;
        }

        let burst_count = cli.packets_per_burst.max(1);
        let mut packets = Vec::with_capacity(burst_count);
        for _ in 0..burst_count {
            let fp = fountain_encoder.next_packet();
            // raw BER用: このパケットのFEC符号化結果を記録
            {
                let seq = (fp.seq % (u32::from(u16::MAX) + 1)) as u16;
                let pkt = Packet::new(seq, k, &fp.data);
                let bits = fec::bytes_to_bits(&pkt.serialize());
                let fec_encoded = fec::encode(&bits);
                expected_packet_bits.lock().unwrap().insert(seq, bits);
                expected_fec_bits.lock().unwrap().insert(seq, fec_encoded);
            }
            packets.push(fp);
        }
        let frame = encoder.encode_burst(&packets);

        attempts += 1;
        tx_signal_energy_sum += signal_energy(&frame);
        tx_signal_samples += frame.len();

        let drop_burst = rng.gen::<f32>() < imp.burst_loss;
        if drop_burst {
            dropped_attempts += 1;
        }
        let rx_frame = apply_channel(&frame, imp, &mut rng, drop_burst);
        elapsed_sec += rx_frame.len() as f32 / tx_cfg.sample_rate;

        for piece in rx_frame.chunks(chunk) {
            let start_time = std::time::Instant::now();
            let progress = decoder.process_samples(piece);
            total_process_ns += start_time.elapsed().as_nanos() as u64;
            if progress.complete {
                let recovered = decoder.recovered_data();
                let errs = count_bit_errors_bytes(&payload, recovered);
                let bits_compared = payload.len() * 8;
                let raw_bit_errors = *raw_bit_errors_acc.lock().unwrap();
                let raw_bits_compared = *raw_bits_compared_acc.lock().unwrap();
                let raw_error_runs = *raw_error_runs_acc.lock().unwrap();
                let raw_error_run_bits = *raw_error_run_bits_acc.lock().unwrap();
                let raw_error_run_max = *raw_error_run_max_acc.lock().unwrap();
                let codeword_error_weights = codeword_error_weights_acc.lock().unwrap().clone();
                let codeword_count = codeword_error_weights.len();
                let codeword_error_sum = codeword_error_weights.iter().sum::<usize>();
                let codeword_error_max = codeword_error_weights.iter().copied().max().unwrap_or(0);
                let post_bit_errors = *post_bit_errors_acc.lock().unwrap();
                let post_bits_compared = *post_bits_compared_acc.lock().unwrap();
                let post_error_runs = *post_error_runs_acc.lock().unwrap();
                let post_error_run_bits = *post_error_run_bits_acc.lock().unwrap();
                let post_error_run_max = *post_error_run_max_acc.lock().unwrap();
                let post_codeword_error_weights =
                    post_codeword_error_weights_acc.lock().unwrap().clone();
                let post_codeword_count = post_codeword_error_weights.len();
                let post_codeword_error_sum = post_codeword_error_weights.iter().sum::<usize>();
                let post_codeword_error_max = post_codeword_error_weights
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0);
                let post_decode_attempts = *post_decode_attempts_acc.lock().unwrap();
                let post_decode_matched = *post_decode_matched_acc.lock().unwrap();
                return TrialResult {
                    success: errs == 0,
                    completion_sec: Some(elapsed_sec),
                    elapsed_sec,
                    attempts,
                    synced_frames: progress.fde_selected_frames + progress.raw_selected_frames,
                    accepted_frames: progress.received_packets,
                    crc_error_frames: progress.crc_error_packets,
                    first_attempt_success: attempts == 1 && errs == 0,
                    bit_errors: errs,
                    bits_compared,
                    dropped_attempts,
                    tx_signal_energy_sum,
                    tx_signal_samples,
                    process_time_ns: total_process_ns,
                    raw_bit_errors,
                    raw_bits_compared,
                    raw_error_runs,
                    raw_error_run_bits,
                    raw_error_run_max,
                    codeword_count,
                    codeword_error_sum,
                    codeword_error_max,
                    codeword_error_weights,
                    post_bit_errors,
                    post_bits_compared,
                    post_error_runs,
                    post_error_run_bits,
                    post_error_run_max,
                    post_codeword_count,
                    post_codeword_error_sum,
                    post_codeword_error_max,
                    post_codeword_error_weights,
                    post_decode_attempts,
                    post_decode_matched,
                    last_est_snr_db: progress.last_est_snr_db,
                    phase_gate_on_symbols: progress.phase_gate_on_symbols,
                    phase_gate_off_symbols: progress.phase_gate_off_symbols,
                    phase_innovation_reject_symbols: progress.phase_innovation_reject_symbols,
                    phase_err_abs_sum_rad: progress.phase_err_abs_sum_rad,
                    phase_err_abs_count: progress.phase_err_abs_count,
                    phase_err_abs_ge_0p5_symbols: progress.phase_err_abs_ge_0p5_symbols,
                    phase_err_abs_ge_1p0_symbols: progress.phase_err_abs_ge_1p0_symbols,
                    llr_second_pass_attempts: progress.llr_second_pass_attempts,
                    llr_second_pass_rescued: progress.llr_second_pass_rescued,
                };
            }
        }

        if gap > 0 {
            let mut gap_sig = vec![0.0f32; gap];
            add_awgn_with_rng(&mut gap_sig, imp.sigma, &mut rng);
            if imp.ppm.abs() >= 1.0 {
                gap_sig = apply_clock_drift_ppm(&gap_sig, imp.ppm);
            }
            elapsed_sec += gap_sig.len() as f32 / tx_cfg.sample_rate;
            for piece in gap_sig.chunks(chunk) {
                let start_time = std::time::Instant::now();
                let progress = decoder.process_samples(piece);
                total_process_ns += start_time.elapsed().as_nanos() as u64;

                if progress.complete {
                    let recovered = decoder.recovered_data();
                    let errs = count_bit_errors_bytes(&payload, recovered);
                    let bits_compared = payload.len() * 8;
                    let raw_bit_errors = *raw_bit_errors_acc.lock().unwrap();
                    let raw_bits_compared = *raw_bits_compared_acc.lock().unwrap();
                    let raw_error_runs = *raw_error_runs_acc.lock().unwrap();
                    let raw_error_run_bits = *raw_error_run_bits_acc.lock().unwrap();
                    let raw_error_run_max = *raw_error_run_max_acc.lock().unwrap();
                    let codeword_error_weights = codeword_error_weights_acc.lock().unwrap().clone();
                    let codeword_count = codeword_error_weights.len();
                    let codeword_error_sum = codeword_error_weights.iter().sum::<usize>();
                    let codeword_error_max =
                        codeword_error_weights.iter().copied().max().unwrap_or(0);
                    let post_bit_errors = *post_bit_errors_acc.lock().unwrap();
                    let post_bits_compared = *post_bits_compared_acc.lock().unwrap();
                    let post_error_runs = *post_error_runs_acc.lock().unwrap();
                    let post_error_run_bits = *post_error_run_bits_acc.lock().unwrap();
                    let post_error_run_max = *post_error_run_max_acc.lock().unwrap();
                    let post_codeword_error_weights =
                        post_codeword_error_weights_acc.lock().unwrap().clone();
                    let post_codeword_count = post_codeword_error_weights.len();
                    let post_codeword_error_sum = post_codeword_error_weights.iter().sum::<usize>();
                    let post_codeword_error_max = post_codeword_error_weights
                        .iter()
                        .copied()
                        .max()
                        .unwrap_or(0);
                    let post_decode_attempts = *post_decode_attempts_acc.lock().unwrap();
                    let post_decode_matched = *post_decode_matched_acc.lock().unwrap();
                    return TrialResult {
                        success: errs == 0,
                        completion_sec: Some(elapsed_sec),
                        elapsed_sec,
                        attempts,
                        synced_frames: progress.fde_selected_frames + progress.raw_selected_frames,
                        accepted_frames: progress.received_packets,
                        crc_error_frames: progress.crc_error_packets,
                        first_attempt_success: attempts == 1 && errs == 0,
                        bit_errors: errs,
                        bits_compared,
                        dropped_attempts,
                        tx_signal_energy_sum,
                        tx_signal_samples,
                        process_time_ns: total_process_ns,
                        raw_bit_errors,
                        raw_bits_compared,
                        raw_error_runs,
                        raw_error_run_bits,
                        raw_error_run_max,
                        codeword_count,
                        codeword_error_sum,
                        codeword_error_max,
                        codeword_error_weights,
                        post_bit_errors,
                        post_bits_compared,
                        post_error_runs,
                        post_error_run_bits,
                        post_error_run_max,
                        post_codeword_count,
                        post_codeword_error_sum,
                        post_codeword_error_max,
                        post_codeword_error_weights,
                        post_decode_attempts,
                        post_decode_matched,
                        last_est_snr_db: progress.last_est_snr_db,
                        phase_gate_on_symbols: progress.phase_gate_on_symbols,
                        phase_gate_off_symbols: progress.phase_gate_off_symbols,
                        phase_innovation_reject_symbols: progress.phase_innovation_reject_symbols,
                        phase_err_abs_sum_rad: progress.phase_err_abs_sum_rad,
                        phase_err_abs_count: progress.phase_err_abs_count,
                        phase_err_abs_ge_0p5_symbols: progress.phase_err_abs_ge_0p5_symbols,
                        phase_err_abs_ge_1p0_symbols: progress.phase_err_abs_ge_1p0_symbols,
                        llr_second_pass_attempts: progress.llr_second_pass_attempts,
                        llr_second_pass_rescued: progress.llr_second_pass_rescued,
                    };
                }
            }
        }
    }

    let flush = encoder.flush();
    if !flush.is_empty() {
        tx_signal_energy_sum += signal_energy(&flush);
        tx_signal_samples += flush.len();
        let flush_rx = apply_channel(&flush, imp, &mut rng, false);
        elapsed_sec += flush_rx.len() as f32 / tx_cfg.sample_rate;
        for piece in flush_rx.chunks(chunk) {
            let start_time = std::time::Instant::now();
            let _ = decoder.process_samples(piece);
            total_process_ns += start_time.elapsed().as_nanos() as u64;
        }
    }

    let tail = encoder.modulate_silence(TX_TAIL_SAMPLES);
    if !tail.is_empty() {
        tx_signal_energy_sum += signal_energy(&tail);
        tx_signal_samples += tail.len();
        let tail_rx = apply_channel(&tail, imp, &mut rng, false);
        elapsed_sec += tail_rx.len() as f32 / tx_cfg.sample_rate;
        for piece in tail_rx.chunks(chunk) {
            let start_time = std::time::Instant::now();
            let _ = decoder.process_samples(piece);
            total_process_ns += start_time.elapsed().as_nanos() as u64;
        }
    }

    let bits_compared = payload.len() * 8;
    let bit_errors = count_bit_errors_bytes(&payload, decoder.recovered_data());
    let raw_bit_errors = *raw_bit_errors_acc.lock().unwrap();
    let raw_bits_compared = *raw_bits_compared_acc.lock().unwrap();
    let raw_error_runs = *raw_error_runs_acc.lock().unwrap();
    let raw_error_run_bits = *raw_error_run_bits_acc.lock().unwrap();
    let raw_error_run_max = *raw_error_run_max_acc.lock().unwrap();
    let codeword_error_weights = codeword_error_weights_acc.lock().unwrap().clone();
    let codeword_count = codeword_error_weights.len();
    let codeword_error_sum = codeword_error_weights.iter().sum::<usize>();
    let codeword_error_max = codeword_error_weights.iter().copied().max().unwrap_or(0);
    let post_bit_errors = *post_bit_errors_acc.lock().unwrap();
    let post_bits_compared = *post_bits_compared_acc.lock().unwrap();
    let post_error_runs = *post_error_runs_acc.lock().unwrap();
    let post_error_run_bits = *post_error_run_bits_acc.lock().unwrap();
    let post_error_run_max = *post_error_run_max_acc.lock().unwrap();
    let post_codeword_error_weights = post_codeword_error_weights_acc.lock().unwrap().clone();
    let post_codeword_count = post_codeword_error_weights.len();
    let post_codeword_error_sum = post_codeword_error_weights.iter().sum::<usize>();
    let post_codeword_error_max = post_codeword_error_weights
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let post_decode_attempts = *post_decode_attempts_acc.lock().unwrap();
    let post_decode_matched = *post_decode_matched_acc.lock().unwrap();
    let final_progress = decoder.process_samples(&[]);
    TrialResult {
        success: false,
        completion_sec: None,
        elapsed_sec,
        attempts,
        synced_frames: final_progress.synced_frames,
        accepted_frames: final_progress.received_packets,
        crc_error_frames: final_progress.crc_error_packets,
        first_attempt_success: false,
        bit_errors,
        bits_compared,
        dropped_attempts,
        tx_signal_energy_sum,
        tx_signal_samples,
        process_time_ns: total_process_ns,
        raw_bit_errors,
        raw_bits_compared,
        raw_error_runs,
        raw_error_run_bits,
        raw_error_run_max,
        codeword_count,
        codeword_error_sum,
        codeword_error_max,
        codeword_error_weights,
        post_bit_errors,
        post_bits_compared,
        post_error_runs,
        post_error_run_bits,
        post_error_run_max,
        post_codeword_count,
        post_codeword_error_sum,
        post_codeword_error_max,
        post_codeword_error_weights,
        post_decode_attempts,
        post_decode_matched,
        last_est_snr_db: final_progress.last_est_snr_db,
        phase_gate_on_symbols: final_progress.phase_gate_on_symbols,
        phase_gate_off_symbols: final_progress.phase_gate_off_symbols,
        phase_innovation_reject_symbols: final_progress.phase_innovation_reject_symbols,
        phase_err_abs_sum_rad: final_progress.phase_err_abs_sum_rad,
        phase_err_abs_count: final_progress.phase_err_abs_count,
        phase_err_abs_ge_0p5_symbols: final_progress.phase_err_abs_ge_0p5_symbols,
        phase_err_abs_ge_1p0_symbols: final_progress.phase_err_abs_ge_1p0_symbols,
        llr_second_pass_attempts: final_progress.llr_second_pass_attempts,
        llr_second_pass_rescued: final_progress.llr_second_pass_rescued,
    }
}

fn print_header(cli: &Cli) {
    let columns = selected_columns(cli);
    match cli.output {
        OutputFormat::Csv => println!("{}", columns.join(",")),
        OutputFormat::Json => {}
        OutputFormat::Table => {
            println!("| {} |", columns.join(" | "));
            println!("| {} |", vec!["---"; columns.len()].join(" | "));
        }
    }
}

fn fmt_opt(v: Option<f32>) -> String {
    v.map(|x| format!("{x:.6}"))
        .unwrap_or_else(|| "NaN".to_string())
}

fn render_column(
    name: &str,
    scenario: &str,
    cli: &Cli,
    imp: &ChannelImpairment,
    m: &Metrics,
) -> String {
    let raw_ber = m.raw_ber();
    match name {
        "scenario" => scenario.to_string(),
        "phy" => cli.phy.as_str().to_string(),
        "mary_fde_mode" => cli.mary_fde_mode.as_str().to_string(),
        "trials" => m.trials.to_string(),
        "awgn_snr_db" => fmt_opt(m.awgn_snr_db(imp.sigma)),
        "p_complete" => format!("{:.6}", m.p_complete()),
        "ber" => format!("{:.6}", m.ber()),
        "raw_ber" => {
            if raw_ber.is_nan() {
                "NaN".to_string()
            } else {
                format!("{raw_ber:.6}")
            }
        }
        "raw_err_run_mean" => fmt_opt(m.raw_err_run_mean()),
        "raw_err_run_max" => m
            .raw_err_run_max()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "NaN".to_string()),
        "err_w_cw_mean" => fmt_opt(m.err_w_cw_mean()),
        "err_w_cw_p50" => fmt_opt(m.err_w_cw_p50()),
        "err_w_cw_p90" => fmt_opt(m.err_w_cw_p90()),
        "err_w_cw_p99" => fmt_opt(m.err_w_cw_p99()),
        "err_w_cw_max" => m
            .err_w_cw_max()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "NaN".to_string()),
        "err_w_cw_hist" => m.err_w_cw_hist().unwrap_or_else(|| "NaN".to_string()),
        "post_ber" => {
            let post_ber = m.post_ber();
            if post_ber.is_nan() {
                "NaN".to_string()
            } else {
                format!("{post_ber:.6}")
            }
        }
        "post_decode_match_ratio" => format!("{:.6}", m.post_decode_match_ratio()),
        "post_err_run_mean" => fmt_opt(m.post_err_run_mean()),
        "post_err_run_max" => m
            .post_err_run_max()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "NaN".to_string()),
        "post_err_w_cw_mean" => fmt_opt(m.post_err_w_cw_mean()),
        "post_err_w_cw_p50" => fmt_opt(m.post_err_w_cw_p50()),
        "post_err_w_cw_p90" => fmt_opt(m.post_err_w_cw_p90()),
        "post_err_w_cw_p99" => fmt_opt(m.post_err_w_cw_p99()),
        "post_err_w_cw_max" => m
            .post_err_w_cw_max()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "NaN".to_string()),
        "post_err_w_cw_hist" => m.post_err_w_cw_hist().unwrap_or_else(|| "NaN".to_string()),
        "goodput_effective_bps" => format!("{:.3}", m.goodput_effective_bps(cli.payload_bytes * 8)),
        "goodput_success_mean_bps" => fmt_opt(m.goodput_success_mean_bps(cli.payload_bytes * 8)),
        "p95_complete_s" => fmt_opt(m.p95_completion_sec()),
        "mean_complete_s" => fmt_opt(m.mean_completion_sec()),
        "avg_proc_ns_sample" => format!("{:.2}", m.avg_process_time_per_sample_ns()),
        "synced_frame_ratio" => format!("{:.6}", m.synced_frame_ratio()),
        "crc_pass_ratio" => format!("{:.6}", m.crc_pass_ratio()),
        "llr_second_pass_trigger_ratio" => format!("{:.6}", m.llr_second_pass_trigger_ratio()),
        "llr_second_pass_rescue_ratio" => format!("{:.6}", m.llr_second_pass_rescue_ratio()),
        "phase_gate_on_ratio" => format!("{:.6}", m.phase_gate_on_ratio()),
        "phase_innovation_reject_ratio" => format!("{:.6}", m.phase_innovation_reject_ratio()),
        "phase_err_abs_mean_rad" => fmt_opt(m.phase_err_abs_mean_rad()),
        "phase_err_abs_ge_0p5_ratio" => format!("{:.6}", m.phase_err_abs_ge_0p5_ratio()),
        "phase_err_abs_ge_1p0_ratio" => format!("{:.6}", m.phase_err_abs_ge_1p0_ratio()),
        "avg_last_est_snr_db" => fmt_opt(m.avg_last_est_snr_db()),
        "multipath" => imp.multipath.name.clone(),
        _ => unreachable!("column should be validated before rendering"),
    }
}

fn escape_json(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn escape_table_cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

fn json_value_literal(value: &str) -> String {
    if value == "NaN" {
        return "null".to_string();
    }
    if let Ok(parsed) = value.parse::<f64>() {
        if parsed.is_finite() {
            return value.to_string();
        }
    }
    format!("\"{}\"", escape_json(value))
}

fn print_awgn_limit(cli: &Cli, phy_limit: Option<f32>) {
    match cli.output {
        OutputFormat::Csv => {
            println!(
                "# awgn_limit(target_p_complete>={:.2}, phy={}) result={}",
                cli.target_p_complete,
                cli.phy.as_str(),
                phy_limit
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "none".to_string()),
            );
        }
        OutputFormat::Json => {
            let result_sigma = phy_limit
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "null".to_string());
            println!(
                "{{\"type\":\"awgn_limit\",\"target_p_complete\":{:.6},\"phy\":\"{}\",\"result_sigma\":{}}}",
                cli.target_p_complete,
                escape_json(cli.phy.as_str()),
                result_sigma
            );
        }
        OutputFormat::Table => {
            println!(
                "awgn_limit: target_p_complete>={:.2}, phy={}, result={}",
                cli.target_p_complete,
                cli.phy.as_str(),
                phy_limit
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "none".to_string()),
            );
        }
    }
}

fn print_row(scenario: &str, cli: &Cli, imp: &ChannelImpairment, m: &Metrics) {
    let columns = selected_columns(cli);
    let values = columns
        .iter()
        .copied()
        .map(|col| render_column(col, scenario, cli, imp, m))
        .collect::<Vec<_>>();
    match cli.output {
        OutputFormat::Csv => println!("{}", values.join(",")),
        OutputFormat::Json => {
            let fields = columns
                .iter()
                .copied()
                .zip(values.iter())
                .map(|(k, v)| format!("\"{}\":{}", escape_json(k), json_value_literal(v)))
                .collect::<Vec<_>>()
                .join(",");
            println!("{{{fields}}}");
        }
        OutputFormat::Table => {
            let cells = values
                .iter()
                .map(|v| escape_table_cell(v))
                .collect::<Vec<_>>()
                .join(" | ");
            println!("| {} |", cells);
        }
    }
}

fn evaluate(cli: &Cli, imp: &ChannelImpairment, scenario: &str) -> Metrics {
    let mut metrics = Metrics::default();
    for trial_idx in 0..cli.trials {
        let trial_seed = cli
            .seed
            .wrapping_add((trial_idx as u64).wrapping_mul(0x9E37_79B9));
        let tr = if matches!(cli.phy, Phy::Mary) {
            run_trial_mary_e2e(imp, cli, trial_seed)
        } else {
            run_trial_dsss_e2e(imp, cli, trial_seed)
        };
        metrics.push(tr);
    }
    print_row(scenario, cli, imp, &metrics);
    metrics
}

fn run_point(cli: &Cli) {
    print_header(cli);
    let base = cli.base_impairment();
    let scenario = format!(
        "point(bytes={},sigma={:.3},cfo={:.1},ppm={:.1},loss={:.2},fade={:.2})",
        cli.payload_bytes, base.sigma, base.cfo_hz, base.ppm, base.burst_loss, base.fading_depth
    );
    evaluate(cli, &base, &scenario);
}

fn run_sweep_awgn(cli: &Cli) {
    print_header(cli);
    let mut phy_limit = None;
    for &sigma in &cli.sweep_awgn {
        let mut imp = cli.base_impairment();
        imp.sigma = sigma;
        let scenario = format!(
            "awgn(bytes={},sigma={sigma:.3},cfo={:.1},ppm={:.1},loss={:.2},fade={:.2})",
            cli.payload_bytes, imp.cfo_hz, imp.ppm, imp.burst_loss, imp.fading_depth
        );
        let d = evaluate(cli, &imp, &scenario);
        if d.p_complete() >= cli.target_p_complete {
            phy_limit = Some(sigma);
        }
    }
    print_awgn_limit(cli, phy_limit);
}

fn run_sweep_ppm(cli: &Cli) {
    print_header(cli);
    for &ppm in &cli.sweep_ppm {
        let mut imp = cli.base_impairment();
        imp.ppm = ppm;
        let scenario = format!(
            "ppm(bytes={},ppm={ppm:.1},sigma={:.3},cfo={:.1},loss={:.2},fade={:.2})",
            cli.payload_bytes, imp.sigma, imp.cfo_hz, imp.burst_loss, imp.fading_depth
        );
        evaluate(cli, &imp, &scenario);
    }
}

fn run_sweep_loss(cli: &Cli) {
    print_header(cli);
    for &loss in &cli.sweep_loss {
        let mut imp = cli.base_impairment();
        imp.burst_loss = loss.clamp(0.0, 1.0);
        let scenario = format!(
            "loss(bytes={},loss={:.2},sigma={:.3},cfo={:.1},ppm={:.1},fade={:.2})",
            cli.payload_bytes, imp.burst_loss, imp.sigma, imp.cfo_hz, imp.ppm, imp.fading_depth
        );
        evaluate(cli, &imp, &scenario);
    }
}

fn run_sweep_multipath(cli: &Cli) {
    print_header(cli);
    for name in ["none", "mild", "medium", "harsh"] {
        let mut imp = cli.base_impairment();
        imp.multipath = MultipathProfile::preset(name).unwrap_or_else(MultipathProfile::none);
        let scenario = format!(
            "multipath(bytes={},profile={name},sigma={:.3},cfo={:.1},ppm={:.1},loss={:.2},fade={:.2})",
            cli.payload_bytes, imp.sigma, imp.cfo_hz, imp.ppm, imp.burst_loss, imp.fading_depth
        );
        evaluate(cli, &imp, &scenario);
    }
}

fn run_sweep_fading(cli: &Cli) {
    print_header(cli);
    for &fade in &cli.sweep_fading {
        let mut imp = cli.base_impairment();
        imp.fading_depth = fade.clamp(0.0, 1.0);
        let scenario = format!(
            "fading(bytes={},fade={:.2},sigma={:.3},ppm={:.1},loss={:.2})",
            cli.payload_bytes, imp.fading_depth, imp.sigma, imp.ppm, imp.burst_loss
        );
        evaluate(cli, &imp, &scenario);
    }
}

fn run_sweep_band(cli: &Cli) {
    print_header(cli);
    for &chip_rate in &cli.sweep_chip_rate {
        for &carrier_freq in &cli.sweep_carrier_freq {
            let mut cfg = cli.clone();
            cfg.chip_rate = chip_rate.max(1.0);
            cfg.carrier_freq = carrier_freq.max(1.0);
            let rrc_bw = cfg.chip_rate * (1.0 + cfg.rrc_alpha) * 0.5;
            let band_lo = (cfg.carrier_freq - rrc_bw).max(0.0);
            let band_hi = cfg.carrier_freq + rrc_bw;
            let scenario = format!(
                "band(bytes={},chip_rate={:.1},carrier={:.1},bw_lo={:.1},bw_hi={:.1},sigma={:.3},cfo={:.1},ppm={:.1},loss={:.2},fade={:.2})",
                cfg.payload_bytes,
                cfg.chip_rate,
                cfg.carrier_freq,
                band_lo,
                band_hi,
                cfg.sigma,
                cfg.cfo_hz,
                cfg.ppm,
                cfg.burst_loss,
                cfg.fading_depth
            );
            evaluate(&cfg, &cfg.base_impairment(), &scenario);
        }
    }
}

fn main() {
    let cli = parse_cli();

    match cli.mode {
        EvalMode::Point => run_point(&cli),
        EvalMode::SweepAwgn => run_sweep_awgn(&cli),
        EvalMode::SweepPpm => run_sweep_ppm(&cli),
        EvalMode::SweepLoss => run_sweep_loss(&cli),
        EvalMode::SweepFading => run_sweep_fading(&cli),
        EvalMode::SweepMultipath => run_sweep_multipath(&cli),
        EvalMode::SweepBand => run_sweep_band(&cli),
        EvalMode::SweepAll => {
            run_sweep_awgn(&cli);
            run_sweep_ppm(&cli);
            run_sweep_loss(&cli);
            run_sweep_fading(&cli);
            run_sweep_multipath(&cli);
            run_sweep_band(&cli);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multipath_profile_parsing() {
        let p = MultipathProfile::preset("mild").unwrap();
        assert_eq!(p.name, "mild");
        assert_eq!(p.taps.len(), 3);

        let p = MultipathProfile::parse_custom("0:1.0, 10:0.5, 20:0.25").unwrap();
        assert_eq!(p.name, "custom");
        assert_eq!(p.taps.len(), 3);
        assert_eq!(p.taps[1], (10, 0.5));
        assert_eq!(p.max_delay(), 20);

        assert!(MultipathProfile::parse_custom("invalid").is_none());
    }

    #[test]
    fn test_metrics_calculations() {
        let mut m = Metrics::default();

        m.push(TrialResult {
            success: true,
            completion_sec: Some(0.5),
            elapsed_sec: 0.5,
            attempts: 1,
            first_attempt_success: true,
            bit_errors: 0,
            bits_compared: 128,
            raw_error_runs: 0,
            raw_error_run_bits: 0,
            raw_error_run_max: 0,
            ..Default::default()
        });

        m.push(TrialResult {
            success: false,
            completion_sec: None,
            elapsed_sec: 1.2,
            attempts: 3,
            first_attempt_success: false,
            bit_errors: 10,
            bits_compared: 128,
            raw_error_runs: 0,
            raw_error_run_bits: 0,
            raw_error_run_max: 0,
            ..Default::default()
        });

        assert_eq!(m.trials, 2);
        assert_eq!(m.successes, 1);
        assert_eq!(m.p_complete(), 0.5);
        assert_eq!(m.ber(), 10.0 / 256.0);
    }

    #[test]
    fn test_e2e_dsss_smoke() {
        let cli = Cli {
            phy: Phy::Dsss,
            mode: EvalMode::Point,
            trials: 1,
            payload_bytes: 16,
            max_sec: 5.0,
            chunk_samples: 1024,
            gap_samples: 0,
            seed: 123,
            target_p_complete: 0.95,
            sigma: 0.0,
            cfo_hz: 0.0,
            ppm: 0.0,
            burst_loss: 0.0,
            fading_depth: 0.0,
            multipath: MultipathProfile::none(),
            sweep_awgn: vec![],
            sweep_ppm: vec![],
            sweep_loss: vec![],
            sweep_fading: vec![],
            sweep_chip_rate: vec![],
            sweep_carrier_freq: vec![],
            sample_rate: 48000.0,
            chip_rate: 8000.0,
            carrier_freq: 15000.0,
            mseq_order: 4,
            rrc_alpha: 0.3,
            sync_word_bits: 16,
            preamble_repeat: 2,
            packets_per_burst: 1,
            preamble_sf: 13,
            mary_fde_mode: MaryFdeMode::On,
            mary_fde_snr_db: 15.0,
            mary_fde_lambda_scale: 1.0,
            mary_fde_lambda_floor: 0.0,
            mary_fde_max_inv_gain: None,
            mary_cir_norm: CirNormArg::None,
            mary_cir_tap_alpha: 0.0,
            mary_viterbi_list: 1,
            mary_llr_erasure_second_pass: false,
            mary_llr_erasure_q: 0.2,
            mary_llr_erasure_list: 8,
            columns: None,
            output: OutputFormat::Csv,
        };

        let res = run_trial_dsss_e2e(&cli.base_impairment(), &cli, cli.seed);
        assert!(res.success);
    }

    #[test]
    fn test_e2e_mary_smoke() {
        let cli = Cli {
            phy: Phy::Mary,
            mode: EvalMode::Point,
            trials: 1,
            payload_bytes: 16,
            max_sec: 5.0,
            chunk_samples: 1024,
            gap_samples: 0,
            seed: 456,
            target_p_complete: 0.95,
            sigma: 0.0,
            cfo_hz: 0.0,
            ppm: 0.0,
            burst_loss: 0.0,
            fading_depth: 0.0,
            multipath: MultipathProfile::none(),
            sweep_awgn: vec![],
            sweep_ppm: vec![],
            sweep_loss: vec![],
            sweep_fading: vec![],
            sweep_chip_rate: vec![],
            sweep_carrier_freq: vec![],
            sample_rate: 48000.0,
            chip_rate: 8000.0,
            carrier_freq: 15000.0,
            mseq_order: 4,
            rrc_alpha: 0.3,
            sync_word_bits: 16,
            preamble_repeat: 2,
            packets_per_burst: 1,
            preamble_sf: 13,
            mary_fde_mode: MaryFdeMode::On,
            mary_fde_snr_db: 15.0,
            mary_fde_lambda_scale: 1.0,
            mary_fde_lambda_floor: 0.0,
            mary_fde_max_inv_gain: None,
            mary_cir_norm: CirNormArg::None,
            mary_cir_tap_alpha: 0.0,
            mary_viterbi_list: 1,
            mary_llr_erasure_second_pass: false,
            mary_llr_erasure_q: 0.2,
            mary_llr_erasure_list: 8,
            columns: None,
            output: OutputFormat::Csv,
        };

        let res = run_trial_mary_e2e(&cli.base_impairment(), &cli, cli.seed);
        assert!(res.success);
    }
}
