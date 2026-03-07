mod channel;
mod config;
mod engine;
mod metrics;
mod report;
mod runner;
mod utils;

use channel::{ChannelImpairment, MultipathProfile};
use config::{parse_cli, Cli, EvalMode, Phy};
use metrics::Metrics;
use report::{print_awgn_limit, print_header, print_row};
use runner::{run_trial_dsss_e2e, run_trial_mary_e2e};

fn evaluate(cli: &Cli, imp: &ChannelImpairment, scenario: &str) -> Metrics {
    let metrics = if matches!(cli.phy, Phy::Mary) {
        run_trial_mary_e2e(imp, cli, cli.seed)
    } else {
        run_trial_dsss_e2e(imp, cli, cli.seed)
    };
    print_row(scenario, cli, imp, &metrics);
    metrics
}

fn run_point(cli: &Cli) {
    print_header(cli);
    let base = cli.base_impairment();
    let scenario = format!(
        "point(bytes={},sigma={:.3},cfo={:.1},ppm={:.1},loss={:.2},fade={:.2})",
        cli.payload_bytes, base.sigma, base.cfo_hz, base.ppm, base.frame_loss, base.fading_depth
    );
    evaluate(cli, &base, &scenario);
}

fn run_sweep_awgn(cli: &Cli) {
    print_header(cli);
    let mut base = cli.base_impairment();
    let mut last_p = 1.0;
    let mut phy_limit = None;

    for &sigma in &cli.sweep_awgn {
        base.sigma = sigma;
        let scenario = format!("sweep_awgn(sigma={sigma:.3})");
        let m = evaluate(cli, &base, &scenario);
        let p = m.p_complete();
        if last_p >= cli.target_p_complete && p < cli.target_p_complete {
            phy_limit = Some(sigma);
        }
        last_p = p;
    }
    print_awgn_limit(cli, phy_limit);
}

fn run_sweep_ppm(cli: &Cli) {
    print_header(cli);
    let mut base = cli.base_impairment();
    for &ppm in &cli.sweep_ppm {
        base.ppm = ppm;
        let scenario = format!("sweep_ppm(ppm={ppm:.1})");
        evaluate(cli, &base, &scenario);
    }
}

fn run_sweep_loss(cli: &Cli) {
    print_header(cli);
    let mut base = cli.base_impairment();
    for &loss in &cli.sweep_loss {
        base.frame_loss = loss;
        let scenario = format!("sweep_loss(loss={loss:.2})");
        evaluate(cli, &base, &scenario);
    }
}

fn run_sweep_fading(cli: &Cli) {
    print_header(cli);
    let mut base = cli.base_impairment();
    for &depth in &cli.sweep_fading {
        base.fading_depth = depth;
        let scenario = format!("sweep_fading(depth={depth:.2})");
        evaluate(cli, &base, &scenario);
    }
}

fn run_sweep_multipath(cli: &Cli) {
    print_header(cli);
    let mut base = cli.base_impairment();
    let profiles = ["none", "mild", "medium", "harsh"];
    for name in profiles {
        base.multipath = MultipathProfile::preset(name).unwrap_or_else(MultipathProfile::none);
        let scenario = format!("sweep_multipath(profile={name})");
        evaluate(cli, &base, &scenario);
    }
}

fn run_sweep_band(cli: &Cli) {
    print_header(cli);
    let mut current_cli = cli.clone();
    let base = cli.base_impairment();

    for &chip_rate in &cli.sweep_chip_rate {
        current_cli.chip_rate = chip_rate;
        for &carrier_freq in &cli.sweep_carrier_freq {
            current_cli.carrier_freq = carrier_freq;
            let scenario = format!("sweep_band(chip={chip_rate:.0},carrier={carrier_freq:.0})");
            evaluate(&current_cli, &base, &scenario);
        }
    }
}

fn run_sweep_all(cli: &Cli) {
    run_point(cli);
    run_sweep_awgn(cli);
    run_sweep_ppm(cli);
    run_sweep_loss(cli);
    run_sweep_fading(cli);
    run_sweep_multipath(cli);
    run_sweep_band(cli);
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
        EvalMode::SweepAll => run_sweep_all(&cli),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use config::{CirNormArg, MaryFdeMode, OutputFormat};

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
        };

        let res = run_trial_mary_e2e(&cli.base_impairment(), &cli, cli.seed);
        assert!(!res.completion_secs.is_empty());
    }
}
