use crate::channel::{ChannelImpairment, MultipathProfile};
use clap::{builder::PossibleValuesParser, Parser, ValueEnum};
use dsp::mary::decoder::CirNormalizationMode;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Phy {
    Dsss,
    Mary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum EvalMode {
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
pub enum OutputFormat {
    Csv,
    Json,
    Table,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum MaryFdeMode {
    On,
    Off,
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "snake_case")]
pub enum CirNormArg {
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

pub const DEFAULT_COLUMNS: &[&str] = &[
    "scenario",
    "phy",
    "mary_fde_mode",
    "total_sim_sec",
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

pub const ALL_COLUMNS: &[&str] = &[
    "scenario",
    "phy",
    "mary_fde_mode",
    "total_sim_sec",
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

pub fn parse_positive_f32(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("invalid float: {value}"))?;
    if parsed > 0.0 {
        Ok(parsed)
    } else {
        Err(format!("value must be > 0: {value}"))
    }
}

pub fn parse_nonnegative_f32(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("invalid float: {value}"))?;
    if parsed >= 0.0 {
        Ok(parsed)
    } else {
        Err(format!("value must be >= 0: {value}"))
    }
}

pub fn parse_unit_interval_f32(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("invalid float: {value}"))?;
    if (0.0..=1.0).contains(&parsed) {
        Ok(parsed)
    } else {
        Err(format!("value must be in [0,1]: {value}"))
    }
}

pub fn parse_positive_usize(value: &str) -> Result<usize, String> {
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
pub struct Cli {
    #[arg(long, value_enum, default_value_t = Phy::Dsss)]
    pub phy: Phy,
    #[arg(long, value_enum, default_value_t = EvalMode::Point)]
    pub mode: EvalMode,
    #[arg(long = "total-sim-sec", value_parser = parse_positive_f32, default_value_t = 60.0)]
    pub total_sim_sec: f32,
    #[arg(long = "payload-bytes", value_parser = parse_positive_usize, default_value_t = 64)]
    pub payload_bytes: usize,
    #[arg(long = "chunk-samples", value_parser = parse_positive_usize, default_value_t = 2048)]
    pub chunk_samples: usize,
    #[arg(long, default_value_t = 0xA11CE_u64)]
    pub seed: u64,
    #[arg(long = "target-p", value_parser = parse_unit_interval_f32, default_value_t = 0.95)]
    pub target_p_complete: f32,
    #[arg(long, value_parser = parse_nonnegative_f32, default_value_t = 0.0)]
    pub sigma: f32,
    #[arg(long = "cfo-hz", default_value_t = 0.0)]
    pub cfo_hz: f32,
    #[arg(long, default_value_t = 0.0)]
    pub ppm: f32,
    #[arg(long = "frame-loss", value_parser = parse_unit_interval_f32, default_value_t = 0.0)]
    pub frame_loss: f32,
    #[arg(long = "fading-depth", value_parser = parse_unit_interval_f32, default_value_t = 0.0)]
    pub fading_depth: f32,
    #[arg(long, default_value = "none")]
    pub multipath: MultipathProfile,
    #[arg(long = "sweep-awgn", value_delimiter = ',', default_values_t = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])]
    pub sweep_awgn: Vec<f32>,
    #[arg(long = "sweep-ppm", value_delimiter = ',', default_values_t = [-120.0, -80.0, -40.0, 0.0, 40.0, 80.0, 120.0])]
    pub sweep_ppm: Vec<f32>,
    #[arg(long = "sweep-loss", value_delimiter = ',', default_values_t = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4])]
    pub sweep_loss: Vec<f32>,
    #[arg(long = "sweep-fading", value_delimiter = ',', default_values_t = [0.0, 0.2, 0.4, 0.6, 0.8])]
    pub sweep_fading: Vec<f32>,
    #[arg(long = "sweep-chip-rate", value_delimiter = ',', default_values_t = [6000.0, 8000.0, 10000.0, 12000.0, 14000.0, 15000.0])]
    pub sweep_chip_rate: Vec<f32>,
    #[arg(long = "sweep-carrier-freq", value_delimiter = ',', default_values_t = [8000.0, 10000.0, 12000.0, 14000.0, 15000.0])]
    pub sweep_carrier_freq: Vec<f32>,
    #[arg(long = "sample-rate", default_value_t = dsp::params::DEFAULT_SAMPLE_RATE)]
    pub sample_rate: f32,
    #[arg(long = "chip-rate", default_value_t = dsp::params::CHIP_RATE)]
    pub chip_rate: f32,
    #[arg(long = "carrier-freq", default_value_t = dsp::params::CARRIER_FREQ)]
    pub carrier_freq: f32,
    #[arg(long = "mseq-order", default_value_t = dsp::params::MSEQ_ORDER)]
    pub mseq_order: usize,
    #[arg(long = "rrc-alpha", default_value_t = dsp::params::RRC_ALPHA)]
    pub rrc_alpha: f32,
    #[arg(long = "sync-word-bits", default_value_t = dsp::params::SYNC_WORD_BITS)]
    pub sync_word_bits: usize,
    #[arg(long = "preamble-repeat", default_value_t = dsp::params::PREAMBLE_REPEAT)]
    pub preamble_repeat: usize,
    #[arg(long = "packets-per-frame", default_value_t = dsp::params::PACKETS_PER_SYNC_BURST)]
    pub packets_per_frame: usize,
    #[arg(long = "preamble-sf", default_value_t = dsp::params::PREAMBLE_SF)]
    pub preamble_sf: usize,
    #[arg(long = "mary-fde", value_enum, default_value_t = MaryFdeMode::On)]
    pub mary_fde_mode: MaryFdeMode,
    #[arg(long = "mary-fde-snr-db", default_value_t = 15.0)]
    pub mary_fde_snr_db: f32,
    #[arg(long = "mary-fde-k", default_value_t = 1.0)]
    pub mary_fde_lambda_scale: f32,
    #[arg(long = "mary-fde-lambda-floor", default_value_t = 0.0)]
    pub mary_fde_lambda_floor: f32,
    #[arg(long = "mary-fde-max-inv-gain", value_parser = parse_positive_f32)]
    pub mary_fde_max_inv_gain: Option<f32>,
    #[arg(long = "mary-cir-norm", value_enum, default_value_t = CirNormArg::None)]
    pub mary_cir_norm: CirNormArg,
    #[arg(long = "mary-cir-tap-alpha", default_value_t = 0.0)]
    pub mary_cir_tap_alpha: f32,
    #[arg(
        long = "mary-viterbi-list",
        value_parser = parse_positive_usize,
        default_value_t = 1
    )]
    pub mary_viterbi_list: usize,
    #[arg(long = "mary-llr-erasure-second-pass")]
    pub mary_llr_erasure_second_pass: bool,
    #[arg(
        long = "mary-llr-erasure-q",
        value_parser = parse_unit_interval_f32,
        default_value_t = 0.2
    )]
    pub mary_llr_erasure_q: f32,
    #[arg(
        long = "mary-llr-erasure-list",
        value_parser = parse_positive_usize,
        default_value_t = 8
    )]
    pub mary_llr_erasure_list: usize,
    #[arg(
        long = "columns",
        value_delimiter = ',',
        value_parser = PossibleValuesParser::new(ALL_COLUMNS)
    )]
    pub columns: Option<Vec<String>>,
    #[arg(long = "output", value_enum, default_value_t = OutputFormat::Csv)]
    pub output: OutputFormat,
}

impl Cli {
    pub fn base_impairment(&self) -> ChannelImpairment {
        ChannelImpairment {
            sigma: self.sigma,
            cfo_hz: self.cfo_hz,
            ppm: self.ppm,
            frame_loss: self.frame_loss,
            fading_depth: self.fading_depth,
            multipath: self.multipath.clone(),
        }
    }
}

pub fn selected_columns(cli: &Cli) -> Vec<&str> {
    match &cli.columns {
        Some(columns) => columns.iter().map(String::as_str).collect(),
        None => DEFAULT_COLUMNS.to_vec(),
    }
}

pub fn parse_cli() -> Cli {
    Cli::parse()
}
