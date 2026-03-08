use crate::channel::ChannelImpairment;
use crate::metrics::{MetricContext, MetricValue, Metrics};
use crate::{selected_columns, Cli, OutputFormat, METRICS_DEFS};
use serde_json::json;

pub fn print_metrics_desc() {
    println!("{:<30} | Description", "Column Name");
    println!("{:-<30}-|-{:-<40}", ":", ":");
    for def in METRICS_DEFS {
        println!("{:<30} | {}", def.id, def.description);
    }
}

pub fn print_header(cli: &Cli) {
    match cli.output {
        OutputFormat::Csv => {
            let cols = selected_columns(cli);
            println!("{}", cols.join(","));
        }
        OutputFormat::Table => {
            let cols = selected_columns(cli);
            let mut header_parts = vec![format!("{:<30}", "scenario")];
            let mut sep_parts = vec![format!("{:-<30}", ":")];
            for col in cols {
                if col == "scenario" {
                    continue;
                }
                header_parts.push(format!("{:<15}", col));
                sep_parts.push(format!("{:-<15}", ":"));
            }
            println!("| {} |", header_parts.join(" | "));
            println!("| {} |", sep_parts.join(" | "));
        }
        OutputFormat::Json => {}
    }
}

fn value_to_json(val: &MetricValue) -> serde_json::Value {
    match val {
        MetricValue::Float(f) => {
            if f.is_nan() {
                json!("NaN")
            } else {
                json!(f)
            }
        }
        MetricValue::Int(i) => json!(i),
        MetricValue::Text(s) => json!(s),
        MetricValue::Null => serde_json::Value::Null,
    }
}

fn value_to_string(val: &MetricValue, format: OutputFormat) -> String {
    match val {
        MetricValue::Float(f) => {
            if f.is_nan() {
                "NaN".to_string()
            } else if matches!(format, OutputFormat::Table) {
                if f.abs() < 1e-3 && *f != 0.0 {
                    format!("{:.2e}", f)
                } else {
                    format!("{:.4}", f)
                }
            } else {
                f.to_string()
            }
        }
        MetricValue::Int(i) => i.to_string(),
        MetricValue::Text(s) => s.clone(),
        MetricValue::Null => match format {
            OutputFormat::Table => "-".to_string(),
            _ => "".to_string(),
        },
    }
}

pub fn print_row(scenario: &str, cli: &Cli, imp: &ChannelImpairment, m: &Metrics) {
    let phy_str = format!("{:?}", cli.phy).to_lowercase();
    let fde_str = format!("{:?}", cli.mary_fde_mode).to_lowercase();
    let ctx = MetricContext {
        scenario,
        phy: &phy_str,
        mary_fde_mode: &fde_str,
        payload_bits: cli.payload_bytes * 8,
        sigma: imp.sigma,
        multipath_name: &imp.multipath.name,
    };

    let cols = selected_columns(cli);
    let mut row_values = Vec::new();

    for col_id in &cols {
        if let Some(def) = METRICS_DEFS.iter().find(|d| d.id == *col_id) {
            row_values.push((col_id, (def.extractor)(&ctx, m)));
        }
    }

    match cli.output {
        OutputFormat::Csv => {
            let mut writer = csv::WriterBuilder::new()
                .has_headers(false)
                .from_writer(Vec::new());

            let mut values = Vec::new();
            for (_, val) in &row_values {
                values.push(value_to_string(val, OutputFormat::Csv));
            }
            writer.write_record(&values).unwrap();
            let csv_output = String::from_utf8(writer.into_inner().unwrap()).unwrap();
            print!("{}", csv_output);
        }
        OutputFormat::Json => {
            let mut map = serde_json::Map::new();
            for (id, val) in &row_values {
                map.insert(id.to_string(), value_to_json(val));
            }
            println!("{}", serde_json::to_string(&map).unwrap());
        }
        OutputFormat::Table => {
            let mut parts = vec![format!("{:<30}", scenario)];
            for (id, val) in &row_values {
                if **id == "scenario" {
                    continue;
                }
                parts.push(format!("{:<15}", value_to_string(val, OutputFormat::Table)));
            }
            println!("| {} |", parts.join(" | "));
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
