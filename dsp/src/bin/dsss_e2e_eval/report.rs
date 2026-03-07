use crate::channel::ChannelImpairment;
use crate::config::{selected_columns, Cli, OutputFormat};
use crate::metrics::Metrics;
use serde::Serialize;
use serde_json::json;

#[derive(Serialize)]
struct ReportRow<'a> {
    scenario: &'a str,
    phy: String,
    mary_fde_mode: String,
    trials: usize,
    awgn_snr_db: f32,
    p_complete: f32,
    ber: f32,
    raw_ber: f32,
    goodput_effective_bps: f32,
    goodput_success_mean_bps: f32,
    p95_complete_s: f32,
    mean_complete_s: f32,
    avg_proc_ns_sample: f32,
    synced_frame_ratio: f32,
    crc_pass_ratio: f32,
    llr_second_pass_trigger_ratio: f32,
    llr_second_pass_rescue_ratio: f32,
    phase_gate_on_ratio: f32,
    phase_innovation_reject_ratio: f32,
    phase_err_abs_mean_rad: f32,
    phase_err_abs_ge_0p5_ratio: f32,
    phase_err_abs_ge_1p0_ratio: f32,
    avg_last_est_snr_db: f32,
    multipath: String,
    raw_err_run_mean: Option<f32>,
    raw_err_run_max: Option<usize>,
    err_w_cw_mean: Option<f32>,
    err_w_cw_p50: Option<f32>,
    err_w_cw_p90: Option<f32>,
    err_w_cw_p99: Option<f32>,
    err_w_cw_max: Option<usize>,
    err_w_cw_hist: Option<String>,
    post_ber: Option<f32>,
    post_decode_match_ratio: Option<f32>,
    post_err_run_mean: Option<f32>,
    post_err_run_max: Option<usize>,
    post_err_w_cw_mean: Option<f32>,
    post_err_w_cw_p50: Option<f32>,
    post_err_w_cw_p90: Option<f32>,
    post_err_w_cw_p99: Option<f32>,
    post_err_w_cw_max: Option<usize>,
    post_err_w_cw_hist: Option<String>,
}

pub fn print_header(cli: &Cli) {
    if matches!(cli.output, OutputFormat::Csv) {
        let cols = selected_columns(cli);
        println!("{}", cols.join(","));
    }
}

pub fn print_row(scenario: &str, cli: &Cli, imp: &ChannelImpairment, m: &Metrics) {
    let row = ReportRow {
        scenario,
        phy: format!("{:?}", cli.phy).to_lowercase(),
        mary_fde_mode: format!("{:?}", cli.mary_fde_mode).to_lowercase(),
        trials: m.trials,
        awgn_snr_db: m.awgn_snr_db(imp.sigma).unwrap_or(f32::NAN),
        p_complete: m.p_complete(),
        ber: m.ber(),
        raw_ber: m.raw_ber(),
        goodput_effective_bps: m.goodput_effective_bps(cli.payload_bytes * 8),
        goodput_success_mean_bps: m.goodput_success_mean_bps(cli.payload_bytes * 8).unwrap_or(f32::NAN),
        p95_complete_s: m.p95_completion_sec().unwrap_or(f32::NAN),
        mean_complete_s: m.mean_completion_sec().unwrap_or(f32::NAN),
        avg_proc_ns_sample: m.avg_process_time_per_sample_ns(),
        synced_frame_ratio: m.synced_frame_ratio(),
        crc_pass_ratio: m.crc_pass_ratio(),
        llr_second_pass_trigger_ratio: m.llr_second_pass_trigger_ratio(),
        llr_second_pass_rescue_ratio: m.llr_second_pass_rescue_ratio(),
        phase_gate_on_ratio: m.phase_gate_on_ratio(),
        phase_innovation_reject_ratio: m.phase_innovation_reject_ratio(),
        phase_err_abs_mean_rad: m.phase_err_abs_mean_rad().unwrap_or(f32::NAN),
        phase_err_abs_ge_0p5_ratio: m.phase_err_abs_ge_0p5_ratio(),
        phase_err_abs_ge_1p0_ratio: m.phase_err_abs_ge_1p0_ratio(),
        avg_last_est_snr_db: m.avg_last_est_snr_db().unwrap_or(f32::NAN),
        multipath: imp.multipath.name.clone(),
        raw_err_run_mean: m.raw_err_run_mean(),
        raw_err_run_max: m.raw_err_run_max(),
        err_w_cw_mean: m.err_w_cw_mean(),
        err_w_cw_p50: m.err_w_cw_p50(),
        err_w_cw_p90: m.err_w_cw_p90(),
        err_w_cw_p99: m.err_w_cw_p99(),
        err_w_cw_max: m.err_w_cw_max(),
        err_w_cw_hist: m.err_w_cw_hist(),
        post_ber: Some(m.post_ber()),
        post_decode_match_ratio: Some(m.post_decode_match_ratio()),
        post_err_run_mean: m.post_err_run_mean(),
        post_err_run_max: m.post_err_run_max(),
        post_err_w_cw_mean: m.post_err_w_cw_mean(),
        post_err_w_cw_p50: m.post_err_w_cw_p50(),
        post_err_w_cw_p90: m.post_err_w_cw_p90(),
        post_err_w_cw_p99: m.post_err_w_cw_p99(),
        post_err_w_cw_max: m.post_err_w_cw_max(),
        post_err_w_cw_hist: m.post_err_w_cw_hist(),
    };

    match cli.output {
        OutputFormat::Csv => {
            let cols = selected_columns(cli);
            let mut writer = csv::WriterBuilder::new()
                .has_headers(false)
                .from_writer(Vec::new());
            
            // selected_columnsの順序に従って値を抽出して書き込む
            let json_val = serde_json::to_value(&row).unwrap();
            let mut values = Vec::new();
            for col in cols {
                let val = &json_val[col];
                if val.is_null() {
                    values.push("".to_string());
                } else if val.is_string() {
                    values.push(val.as_str().unwrap().to_string());
                } else {
                    values.push(val.to_string());
                }
            }
            writer.write_record(&values).unwrap();
            let csv_output = String::from_utf8(writer.into_inner().unwrap()).unwrap();
            print!("{}", csv_output);
        }
        OutputFormat::Json => {
            let cols = selected_columns(cli);
            let json_val = serde_json::to_value(&row).unwrap();
            let mut filtered = serde_json::Map::new();
            for col in cols {
                if let Some(val) = json_val.get(col) {
                    filtered.insert(col.to_string(), val.clone());
                }
            }
            println!("{}", serde_json::to_string(&json!(filtered)).unwrap());
        }
        OutputFormat::Table => {
            println!(
                "{:<40} | P={:.3} | BER={:.2e} | T={:.2}s | Goodput={:.1}bps",
                scenario,
                row.p_complete,
                row.ber,
                m.mean_completion_sec().unwrap_or(f32::NAN),
                row.goodput_effective_bps
            );
        }
    }
}

pub fn print_awgn_limit(cli: &Cli, limit_sigma: Option<f32>) {
    if matches!(cli.output, OutputFormat::Table) {
        if let Some(s) = limit_sigma {
            println!("--- Target P_complete AWGN Limit: sigma = {:.4} ---", s);
        } else {
            println!("--- Target P_complete AWGN Limit: Not reached ---");
        }
    }
}
