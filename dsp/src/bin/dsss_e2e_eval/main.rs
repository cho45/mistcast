mod channel;
mod metrics;
mod report;
mod runner;
mod utils;

use crate::channel::{ChannelImpairment, MultipathProfile};
use crate::runner::run_by_mode;
use clap::{builder::PossibleValuesParser, Parser, ValueEnum};
use dsp::mary::decoder::CirNormalizationMode;

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
    #[arg(long = "show-metrics-desc")]
    pub show_metrics_desc: bool,
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
        None => get_default_columns(),
    }
}

pub fn parse_cli() -> Cli {
    Cli::parse()
}

fn main() {
    let cli = parse_cli();

    if cli.show_metrics_desc {
        report::print_metrics_desc();
        return;
    }

    run_by_mode(&cli);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::{run_trial_dsss_e2e, run_trial_mary_e2e};

    #[test]
    fn test_e2e_dsss_smoke() {
        let cli = Cli {
            phy: Phy::Dsss,
            mode: EvalMode::Point,
            total_sim_sec: 1.0,
            payload_bytes: 16,
            chunk_samples: 1024,
            seed: 123,
            target_p_complete: 0.95,
            sigma: 0.0,
            cfo_hz: 0.0,
            ppm: 0.0,
            frame_loss: 0.0,
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
            packets_per_frame: 1,
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
            show_metrics_desc: false,
        };

        let res = run_trial_dsss_e2e(&cli.base_impairment(), &cli, cli.seed);
        assert!(!res.completion_secs.is_empty());
    }

    #[test]
    fn test_e2e_mary_smoke() {
        let cli = Cli {
            phy: Phy::Mary,
            mode: EvalMode::Point,
            total_sim_sec: 1.0,
            payload_bytes: 16,
            chunk_samples: 1024,
            seed: 456,
            target_p_complete: 0.95,
            sigma: 0.0,
            cfo_hz: 0.0,
            ppm: 0.0,
            frame_loss: 0.0,
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
            packets_per_frame: 1,
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
            show_metrics_desc: false,
        };

        let res = run_trial_mary_e2e(&cli.base_impairment(), &cli, cli.seed);
        assert!(!res.completion_secs.is_empty());
    }
}
