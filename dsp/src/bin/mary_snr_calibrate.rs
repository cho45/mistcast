use dsp::frame::packet::PACKET_BYTES;
use dsp::mary::interleaver_config;
use dsp::DspConfig;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

const PAYLOAD_SPREAD_FACTOR: f32 = 16.0;

#[derive(Debug, Clone)]
struct SamplePoint {
    sigma: f32,
    awgn_snr_db: f32,
    est_snr_db: f32,
    ebn0_ref_db: f32,
    ebn0_approx_db: f32,
}

#[derive(Debug, Clone)]
struct FitPoint {
    est_snr_db: f32,
    ebn0_fit_db: f32,
    ebn0_approx_db: f32,
}

fn parse_csv_rows(csv_text: &str) -> Vec<HashMap<String, String>> {
    let mut rows = Vec::new();
    let mut header: Option<Vec<String>> = None;

    for line in csv_text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        if l.starts_with("scenario,") {
            header = Some(l.split(',').map(|s| s.to_string()).collect());
            continue;
        }

        let Some(h) = header.as_ref() else {
            continue;
        };

        let parts: Vec<&str> = l.split(',').collect();
        if parts.len() < 7 {
            continue;
        }
        let mut values = Vec::with_capacity(h.len());
        values.push(parts[..6].join(","));
        values.extend(parts[6..].iter().map(|s| s.to_string()));
        if values.len() != h.len() {
            continue;
        }

        let row = h
            .iter()
            .cloned()
            .zip(values.into_iter())
            .collect::<HashMap<_, _>>();
        rows.push(row);
    }

    rows
}

fn parse_sigma_from_scenario(s: &str) -> Option<f32> {
    let key = "sigma=";
    let start = s.find(key)? + key.len();
    let tail = &s[start..];
    let end = tail.find(',').unwrap_or(tail.len());
    tail[..end].parse::<f32>().ok()
}

fn finite(v: &str) -> Option<f32> {
    let x = v.parse::<f32>().ok()?;
    if x.is_finite() {
        Some(x)
    } else {
        None
    }
}

fn isotonic_regression(xs: &[f32], ys: &[f32]) -> Vec<f32> {
    assert_eq!(xs.len(), ys.len());
    if xs.is_empty() {
        return Vec::new();
    }

    #[derive(Clone, Copy)]
    struct Block {
        start: usize,
        end: usize,
        sum_w: f32,
        sum_wy: f32,
    }

    let mut blocks: Vec<Block> = Vec::with_capacity(ys.len());
    for (i, &y) in ys.iter().enumerate() {
        blocks.push(Block {
            start: i,
            end: i,
            sum_w: 1.0,
            sum_wy: y,
        });

        while blocks.len() >= 2 {
            let n = blocks.len();
            let a = blocks[n - 2];
            let b = blocks[n - 1];
            let avg_a = a.sum_wy / a.sum_w;
            let avg_b = b.sum_wy / b.sum_w;
            if avg_a <= avg_b {
                break;
            }
            blocks.pop();
            blocks.pop();
            blocks.push(Block {
                start: a.start,
                end: b.end,
                sum_w: a.sum_w + b.sum_w,
                sum_wy: a.sum_wy + b.sum_wy,
            });
        }
    }

    let mut fit = vec![0.0f32; ys.len()];
    for b in blocks {
        let avg = b.sum_wy / b.sum_w;
        for v in fit.iter_mut().take(b.end + 1).skip(b.start) {
            *v = avg;
        }
    }
    fit
}

fn mean_abs(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().map(|v| v.abs()).sum::<f32>() / xs.len() as f32
}

fn run_eval_sweep() -> Result<String, String> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let eval_bin = manifest_dir
        .join("target")
        .join("release")
        .join("dsss_e2e_eval");

    if !eval_bin.exists() {
        let status = Command::new("cargo")
            .current_dir(&manifest_dir)
            .args(["build", "--release", "--bin", "dsss_e2e_eval"])
            .status()
            .map_err(|e| format!("failed to build dsss_e2e_eval: {e}"))?;
        if !status.success() {
            return Err("build failed for dsss_e2e_eval".to_string());
        }
    }

    let output = Command::new(&eval_bin)
        .current_dir(&manifest_dir)
        .args([
            "--phy",
            "mary",
            "--mode",
            "sweep-awgn",
            "--trials",
            "120",
            "--sweep-awgn",
            "0.03,0.04,0.05,0.07,0.10,0.14,0.20,0.28,0.40,0.56",
        ])
        .output()
        .map_err(|e| format!("failed to run dsss_e2e_eval: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "dsss_e2e_eval failed: status={} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|e| format!("utf8 decode error: {e}"))
}

fn main() -> Result<(), String> {
    let csv = run_eval_sweep()?;
    let rows = parse_csv_rows(&csv);
    if rows.is_empty() {
        return Err("no rows parsed from dsss_e2e_eval output".to_string());
    }

    let cfg = DspConfig::default_48k();
    let symbol_rate = cfg.chip_rate / PAYLOAD_SPREAD_FACTOR;
    let coded_bit_rate = symbol_rate * 6.0;
    let code_rate = (PACKET_BYTES as f32 * 8.0) / interleaver_config::interleaved_bits() as f32;
    let rb_info = coded_bit_rate * code_rate;
    let beq = cfg.chip_rate; // 近似: Beq ≈ Rc
    let offset_db = dsp::common::channel::snr_db_to_ebn0_db(0.0, beq, rb_info);

    let mut points = Vec::<SamplePoint>::new();
    for r in &rows {
        let Some(scenario) = r.get("scenario") else {
            continue;
        };
        let Some(sigma) = parse_sigma_from_scenario(scenario) else {
            continue;
        };
        let Some(awgn_snr_db) = r.get("awgn_snr_db").and_then(|v| finite(v)) else {
            continue;
        };
        let Some(est_snr_db) = r.get("avg_last_est_snr_db").and_then(|v| finite(v)) else {
            continue;
        };

        let ebn0_ref_db = dsp::common::channel::snr_db_to_ebn0_db(awgn_snr_db, beq, rb_info);
        let ebn0_approx_db = dsp::common::channel::snr_db_to_ebn0_db(est_snr_db, beq, rb_info);
        points.push(SamplePoint {
            sigma,
            awgn_snr_db,
            est_snr_db,
            ebn0_ref_db,
            ebn0_approx_db,
        });
    }

    if points.len() < 4 {
        return Err(format!(
            "insufficient valid points for regression: {}",
            points.len()
        ));
    }

    points.sort_by(|a, b| {
        match a
            .est_snr_db
            .partial_cmp(&b.est_snr_db)
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Equal => a.sigma.partial_cmp(&b.sigma).unwrap_or(Ordering::Equal),
            ord => ord,
        }
    });

    let xs: Vec<f32> = points.iter().map(|p| p.est_snr_db).collect();
    let ys: Vec<f32> = points.iter().map(|p| p.ebn0_ref_db).collect();
    let fit = isotonic_regression(&xs, &ys);

    let mut out = Vec::<FitPoint>::with_capacity(points.len());
    let mut diff_reg_approx = Vec::<f32>::with_capacity(points.len());
    let mut err_reg = Vec::<f32>::with_capacity(points.len());
    let mut err_approx = Vec::<f32>::with_capacity(points.len());
    for (p, y_fit) in points.iter().zip(fit.iter().copied()) {
        out.push(FitPoint {
            est_snr_db: p.est_snr_db,
            ebn0_fit_db: y_fit,
            ebn0_approx_db: p.ebn0_approx_db,
        });
        diff_reg_approx.push(y_fit - p.ebn0_approx_db);
        err_reg.push(y_fit - p.ebn0_ref_db);
        err_approx.push(p.ebn0_approx_db - p.ebn0_ref_db);
    }

    println!("# mary_snr_calibrate (default config)");
    println!(
        "# cfg: chip_rate={:.1}Hz, payload_sf=16, interleaved_bits={}, packet_bits={}",
        cfg.chip_rate,
        interleaver_config::interleaved_bits(),
        PACKET_BYTES * 8
    );
    println!(
        "# approx: Eb/N0 ≈ est_snr_internal + 10log10(Beq/Rb), Beq=Rc={:.1}Hz, Rb_info={:.3}bps, offset={:+.3}dB",
        beq, rb_info, offset_db
    );
    println!(
        "# fit quality: MAE(reg-vs-ref)={:.3}dB, MAE(approx-vs-ref)={:.3}dB, MAE(reg-vs-approx)={:.3}dB",
        mean_abs(&err_reg),
        mean_abs(&err_approx),
        mean_abs(&diff_reg_approx)
    );
    println!("#");
    println!("sigma,awgn_snr_db,est_snr_db,ebn0_ref_db,ebn0_db_awgn_equiv,ebn0_approx_db,diff(reg-approx)_db");
    for (p, f) in points.iter().zip(out.iter()) {
        println!(
            "{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:+.3}",
            p.sigma,
            p.awgn_snr_db,
            f.est_snr_db,
            p.ebn0_ref_db,
            f.ebn0_fit_db,
            f.ebn0_approx_db,
            f.ebn0_fit_db - f.ebn0_approx_db
        );
    }

    println!("# calibration_knots (est_snr_db -> ebn0_db_awgn_equiv)");
    let mut first = true;
    let mut last_y = 0.0f32;
    for f in &out {
        if first || (f.ebn0_fit_db - last_y).abs() > 1e-6 {
            println!("{:.6},{:.6}", f.est_snr_db, f.ebn0_fit_db);
            last_y = f.ebn0_fit_db;
            first = false;
        }
    }

    Ok(())
}
