mod alloc_profiler;
mod channel;
mod metrics;
mod report;
mod runner;
mod utils;

#[cfg(feature = "alloc-prof-dhat")]
#[global_allocator]
static DHAT_ALLOC: dhat::Alloc = dhat::Alloc;

use crate::channel::{ChannelImpairment, MultipathProfile};
use crate::runner::run_by_mode;
use crate::utils::{
    parse_nonnegative_f32, parse_positive_f32, parse_positive_usize, parse_unit_interval_f32,
};
use clap::{builder::PossibleValuesParser, Parser, ValueEnum};
use dsp::mary::decoder::{
    CirNormalizationMode, LLR_ERASURE_LIST_SIZE_DEFAULT, LLR_ERASURE_QUANTILE_DEFAULT,
    LLR_ERASURE_SECOND_PASS_DEFAULT, VITERBI_LIST_SIZE_DEFAULT,
};

define_metrics! {
    scenario,                      "評価シナリオ名",                               |ctx, _| ctx.scenario.into(), default;
    phy,                           "物理方式",                                     |ctx, _| ctx.phy.into(), default;
    mary_fde_mode,                 "Mary の FDE モード",                           |ctx, _| ctx.mary_fde_mode.into(), default;
    total_sim_sec,                 "評価に含めた受信サンプルの総時間 [sec]",        |_, m| m.total_sim_sec.into(), default;
    ebn0_db,                       "AWGN 仮定から換算した理論 Eb/N0 [dB]",         |ctx, m| m.awgn_snr_db(ctx.sigma).map(|snr| dsp::common::channel::snr_db_to_ebn0_db(snr, ctx.sample_rate, ctx.bit_rate)).into(), default;
    cn0_db,                        "AWGN 仮定から換算した理論 C/N0 [dB-Hz]",        |ctx, m| m.awgn_snr_db(ctx.sigma).map(|snr| snr + 10.0 * ctx.sample_rate.log10()).into(), default;
    snr_wideband_db,               "AWGN 仮定での受信全帯域 SNR [dB]",             |ctx, m| m.awgn_snr_db(ctx.sigma).into(), default;
    packet_accept_ratio,           "受理パケット率 = accepted_packets / 全送信packet数",|_, m| m.packet_accept_ratio().into(), default;
    ber,                           "復元成功 payload に対する事後 BER",             |_, m| m.ber().into(), default;
    raw_ber,                       "PHY デコーダの FEC 復号前 codeword BER",        |_, m| m.raw_ber().into(), default;
    goodput_effective_bps,         "総シミュレーション時間あたりの有効 payload bitrate [bps]",|ctx, m| m.goodput_effective_bps(ctx.payload_bits).into(), default;
    goodput_success_mean_bps,      "復元成功イベントごとの payload bitrate の平均 [bps]",|ctx, m| m.goodput_success_mean_bps(ctx.payload_bits).into(), default;
    p95_complete_s,                "復元成功イベントの完了時間 95% 点 [sec]",       |_, m| m.p95_completion_sec().into(), default;
    mean_complete_s,               "復元成功イベントの平均完了時間 [sec]",          |_, m| m.mean_completion_sec().into(), default;
    avg_proc_ns_sample,            "入力 1 sample あたりの平均処理時間 [ns/sample]",|_, m| m.avg_process_time_per_sample_ns().into(), default;
    synced_frame_ratio,            "フレーム同期率 = synced_frames / 送信frame数",   |_, m| m.synced_frame_ratio().into(), default;
    crc_pass_ratio,                "CRC 通過率 = accepted_packets / (accepted + crc_error)",|_, m| m.crc_pass_ratio().into(), default;
    llr_second_pass_trigger_ratio, "LLR 2nd pass 起動率 = 2nd pass試行 / (accepted + crc_error)",|_, m| m.llr_second_pass_trigger_ratio().into(), default;
    llr_second_pass_rescue_ratio,  "LLR 2nd pass 救済率 = rescued / 2nd pass試行",  |_, m| m.llr_second_pass_rescue_ratio().into(), default;
    phase_gate_on_ratio,           "位相更新ゲート ON 比率",                        |_, m| m.phase_gate_on_ratio().into(), default;
    phase_innovation_reject_ratio, "位相更新候補の棄却比率",                        |_, m| m.phase_innovation_reject_ratio().into(), default;
    phase_err_abs_mean_rad,        "位相誤差絶対値の平均 [rad]",                    |_, m| m.phase_err_abs_mean_rad().into(), default;
    phase_err_abs_ge_0p5_ratio,    "|phase error| >= 0.5 rad の比率",               |_, m| m.phase_err_abs_ge_0p5_ratio().into(), default;
    phase_err_abs_ge_1p0_ratio,    "|phase error| >= 1.0 rad の比率",               |_, m| m.phase_err_abs_ge_1p0_ratio().into(), default;
    avg_last_est_snr_db,           "受信器内部の推定 SNR の平均 [dB]",              |_, m| m.avg_last_est_snr_db().into(), default;
    multipath,                     "マルチパスプロファイル名",                      |ctx, _| ctx.multipath_name.into(), default;
    raw_err_run_mean,              "生 BER 系列でのエラーラン平均長",               |_, m| m.raw_err_run_mean().into();
    raw_err_run_max,               "生 BER 系列でのエラーラン最大長",               |_, m| m.raw_err_run_max().into();
    err_w_cw_mean,                 "codeword あたり平均エラービット数",             |_, m| m.err_w_cw_mean().into();
    err_w_cw_p50,                  "codeword あたりエラービット数の中央値",         |_, m| m.err_w_cw_p50().into();
    err_w_cw_p90,                  "codeword あたりエラービット数の 90% 点",        |_, m| m.err_w_cw_p90().into();
    err_w_cw_p99,                  "codeword あたりエラービット数の 99% 点",        |_, m| m.err_w_cw_p99().into();
    err_w_cw_max,                  "codeword あたり最大エラービット数",             |_, m| m.err_w_cw_max().into();
    err_w_cw_hist,                 "codeword エラー重みヒストグラム",               |_, m| m.err_w_cw_hist().into();
    post_ber,                      "PHY デコーダの FEC 復号後 BER",                 |_, m| m.post_ber().into();
    post_decode_match_ratio,       "後段 decode 試行に対する payload 一致率",       |_, m| m.post_decode_match_ratio().into();
    post_err_run_mean,             "後段 BER 系列でのエラーラン平均長",             |_, m| m.post_err_run_mean().into();
    post_err_run_max,              "後段 BER 系列でのエラーラン最大長",             |_, m| m.post_err_run_max().into();
    post_err_w_cw_mean,            "後段 codeword あたり平均エラービット数",        |_, m| m.post_err_w_cw_mean().into();
    post_err_w_cw_p50,             "後段 codeword あたりエラービット数の中央値",    |_, m| m.post_err_w_cw_p50().into();
    post_err_w_cw_p90,             "後段 codeword あたりエラービット数の 90% 点",   |_, m| m.post_err_w_cw_p90().into();
    post_err_w_cw_p99,             "後段 codeword あたりエラービット数の 99% 点",   |_, m| m.post_err_w_cw_p99().into();
    post_err_w_cw_max,             "後段 codeword あたり最大エラービット数",        |_, m| m.post_err_w_cw_max().into();
    post_err_w_cw_hist,            "後段 codeword エラー重みヒストグラム",          |_, m| m.post_err_w_cw_hist().into();
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
    #[arg(long = "mary-fde", value_enum, default_value_t = MaryFdeMode::Auto)]
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
        default_value_t = VITERBI_LIST_SIZE_DEFAULT
    )]
    pub mary_viterbi_list: usize,
    #[arg(
        long = "mary-llr-erasure-second-pass",
        default_value_t = LLR_ERASURE_SECOND_PASS_DEFAULT
    )]
    pub mary_llr_erasure_second_pass: bool,
    #[arg(
        long = "mary-llr-erasure-q",
        value_parser = parse_unit_interval_f32,
        default_value_t = LLR_ERASURE_QUANTILE_DEFAULT
    )]
    pub mary_llr_erasure_q: f32,
    #[arg(
        long = "mary-llr-erasure-list",
        value_parser = parse_positive_usize,
        default_value_t = LLR_ERASURE_LIST_SIZE_DEFAULT
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
    #[arg(long = "alloc-profile")]
    pub alloc_profile: bool,
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
    #[cfg(feature = "alloc-prof-dhat")]
    let _dhat_profiler = dhat::Profiler::new_heap();

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
    use crate::metrics::{MetricContext, MetricValue, Metrics};
    use crate::runner::{run_trial_dsss_e2e, run_trial_mary_e2e};

    fn metric_float(def_id: &str, ctx: &MetricContext, metrics: &Metrics) -> Option<f32> {
        let def = METRICS_DEFS.iter().find(|d| d.id == def_id)?;
        match (def.extractor)(ctx, metrics) {
            MetricValue::Float(v) => Some(v),
            _ => None,
        }
    }

    #[test]
    fn test_cn0_db_formula_and_consistency() {
        let sigma = 0.2f32;
        let sample_rate = 48_000.0f32;
        let bit_rate = 1_000.0f32;

        let mut m = Metrics::new(1);
        // tx_signal_power = 1000 / 2000 = 0.5
        m.total_tx_signal_energy = 1000.0;
        m.total_tx_signal_samples = 2000;

        let ctx = MetricContext {
            scenario: "test",
            phy: "dsss",
            mary_fde_mode: "on",
            payload_bits: 128,
            sigma,
            multipath_name: "none",
            sample_rate,
            bit_rate,
        };

        let snr_db = metric_float("snr_wideband_db", &ctx, &m).expect("snr_wideband_db");
        let cn0_db = metric_float("cn0_db", &ctx, &m).expect("cn0_db");
        let ebn0_db = metric_float("ebn0_db", &ctx, &m).expect("ebn0_db");

        let expected_cn0 = snr_db + 10.0 * sample_rate.log10();
        assert!((cn0_db - expected_cn0).abs() < 1e-4);

        let expected_ebn0 = snr_db + 10.0 * (sample_rate / bit_rate).log10();
        assert!((ebn0_db - expected_ebn0).abs() < 1e-4);

        // C/N0 = Eb/N0 + 10log10(Rb)
        let expected_delta = 10.0 * bit_rate.log10();
        assert!(((cn0_db - ebn0_db) - expected_delta).abs() < 1e-4);
    }

    #[test]
    fn test_cli_default_mary_fde_mode_is_auto() {
        let cli = Cli::parse_from(["dsss_e2e_eval"]);
        assert_eq!(cli.mary_fde_mode, MaryFdeMode::Auto);
    }

    #[test]
    fn test_cli_default_mary_list_viterbi_settings_match_decoder_defaults() {
        let cli = Cli::parse_from(["dsss_e2e_eval"]);
        assert_eq!(cli.mary_viterbi_list, VITERBI_LIST_SIZE_DEFAULT);
        assert_eq!(
            cli.mary_llr_erasure_second_pass,
            LLR_ERASURE_SECOND_PASS_DEFAULT
        );
        assert!((cli.mary_llr_erasure_q - LLR_ERASURE_QUANTILE_DEFAULT).abs() < 1e-7);
        assert_eq!(cli.mary_llr_erasure_list, LLR_ERASURE_LIST_SIZE_DEFAULT);
    }

    #[test]
    fn test_e2e_dsss_smoke() {
        let cli = Cli {
            phy: Phy::Dsss,
            mode: EvalMode::Point,
            total_sim_sec: 1.0,
            payload_bytes: 16,
            chunk_samples: 1024,
            seed: 123,
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
            mary_fde_mode: MaryFdeMode::Auto,
            mary_fde_snr_db: 15.0,
            mary_fde_lambda_scale: 1.0,
            mary_fde_lambda_floor: 0.0,
            mary_fde_max_inv_gain: None,
            mary_cir_norm: CirNormArg::None,
            mary_cir_tap_alpha: 0.0,
            mary_viterbi_list: 1,
            mary_llr_erasure_second_pass: true,
            mary_llr_erasure_q: 0.1,
            mary_llr_erasure_list: 8,
            columns: None,
            output: OutputFormat::Csv,
            show_metrics_desc: false,
            alloc_profile: false,
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
            mary_fde_mode: MaryFdeMode::Auto,
            mary_fde_snr_db: 15.0,
            mary_fde_lambda_scale: 1.0,
            mary_fde_lambda_floor: 0.0,
            mary_fde_max_inv_gain: None,
            mary_cir_norm: CirNormArg::None,
            mary_cir_tap_alpha: 0.0,
            mary_viterbi_list: 1,
            mary_llr_erasure_second_pass: true,
            mary_llr_erasure_q: 0.1,
            mary_llr_erasure_list: 8,
            columns: None,
            output: OutputFormat::Csv,
            show_metrics_desc: false,
            alloc_profile: false,
        };

        let res = run_trial_mary_e2e(&cli.base_impairment(), &cli, cli.seed);
        assert!(!res.completion_secs.is_empty());
    }
}
