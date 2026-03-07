use crate::channel::ChannelImpairment;
use crate::config::{Cli, OutputFormat, selected_columns};
use crate::metrics::Metrics;

pub fn fmt_opt(v: Option<f32>) -> String {
    v.map(|x| format!("{x:.6}"))
        .unwrap_or_else(|| "NaN".to_string())
}

pub fn render_column(
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

pub fn escape_json(value: &str) -> String {
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

pub fn escape_table_cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

pub fn json_value_literal(value: &str) -> String {
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

pub fn print_header(cli: &Cli) {
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

pub fn print_awgn_limit(cli: &Cli, phy_limit: Option<f32>) {
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

pub fn print_row(scenario: &str, cli: &Cli, imp: &ChannelImpairment, m: &Metrics) {
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
