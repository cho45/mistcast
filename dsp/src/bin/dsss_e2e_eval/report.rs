use crate::channel::ChannelImpairment;
use crate::config::{selected_columns, Cli, OutputFormat};
use crate::metrics::{MetricContext, Metrics, METRICS_DEFS};

pub fn print_metrics_desc() {
    println!("{:<30} | {}", "Column Name", "Description");
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

fn value_to_string(val: &serde_json::Value) -> String {
    if val.is_null() {
        "-".to_string()
    } else if val.is_string() {
        val.as_str().unwrap().to_string()
    } else if val.is_f64() {
        let f = val.as_f64().unwrap();
        if f.abs() < 1e-3 && f != 0.0 {
            format!("{:.2e}", f)
        } else {
            format!("{:.4}", f)
        }
    } else {
        val.to_string()
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
    let mut row_data = serde_json::Map::new();

    for col_id in &cols {
        if let Some(def) = METRICS_DEFS.iter().find(|d| d.id == *col_id) {
            row_data.insert(col_id.to_string(), (def.extractor)(&ctx, m));
        }
    }

    match cli.output {
        OutputFormat::Csv => {
            let mut writer = csv::WriterBuilder::new()
                .has_headers(false)
                .from_writer(Vec::new());

            let mut values = Vec::new();
            for col in &cols {
                let val = row_data.get(*col).unwrap_or(&serde_json::Value::Null);
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
            println!("{}", serde_json::to_string(&row_data).unwrap());
        }
        OutputFormat::Table => {
            let mut parts = vec![format!("{:<30}", scenario)];
            for col in &cols {
                if *col == "scenario" {
                    continue;
                }
                let val = row_data.get(*col).unwrap_or(&serde_json::Value::Null);
                parts.push(format!("{:<15}", value_to_string(val)));
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
