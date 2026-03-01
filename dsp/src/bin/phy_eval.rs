use dsp::decoder::Decoder;
use dsp::encoder::{Encoder, EncoderConfig};
use dsp::params::PAYLOAD_SIZE;
use dsp::phy::fsk::{BfskConfig, BfskDemodulator, BfskModulator};
use dsp::DspConfig;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::env;

const CHUNK_SAMPLES: usize = 8192;
const GAP_SAMPLES_DSSS: usize = 64;

#[derive(Clone, Copy, Debug)]
enum PhyKind {
    Dsss,
    Fsk,
}

impl PhyKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Dsss => "dsss",
            Self::Fsk => "fsk",
        }
    }
}

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
        // format: "0:1.0,9:0.4,23:0.2"
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
}

#[derive(Clone, Debug)]
struct ChannelImpairment {
    sigma: f32,
    cfo_hz: f32,
    ppm: f32,
    burst_loss: f32,
    multipath: MultipathProfile,
}

#[derive(Clone, Debug)]
struct Cli {
    mode: String,
    trials: usize,
    payload_len: usize,
    deadline_sec: f32,
    max_sec: f32,
    seed: u64,
    target_p_complete: f32,
    base: ChannelImpairment,
    sweep_awgn: Vec<f32>,
    sweep_cfo: Vec<f32>,
    sweep_ppm: Vec<f32>,
    sweep_loss: Vec<f32>,
}

#[derive(Default, Clone, Debug)]
struct TrialResult {
    success: bool,
    completion_sec: Option<f32>,
    elapsed_sec: f32,
    tx_frames: usize,
    frame_successes: usize,
    tx_packets: usize,
    packet_successes: usize,
    dropped_bursts: usize,
}

#[derive(Default, Clone, Debug)]
struct Metrics {
    trials: usize,
    successes: usize,
    deadline_hits: usize,
    completion_secs: Vec<f32>,
    total_elapsed_sec: f32,
    tx_frames: usize,
    frame_successes: usize,
    tx_packets: usize,
    packet_successes: usize,
    dropped_bursts: usize,
}

impl Metrics {
    fn push(&mut self, trial: TrialResult, deadline_sec: f32) {
        self.trials += 1;
        self.total_elapsed_sec += trial.elapsed_sec;
        self.tx_frames += trial.tx_frames;
        self.frame_successes += trial.frame_successes;
        self.tx_packets += trial.tx_packets;
        self.packet_successes += trial.packet_successes;
        self.dropped_bursts += trial.dropped_bursts;

        if trial.success {
            self.successes += 1;
            if let Some(t) = trial.completion_sec {
                self.completion_secs.push(t);
                if t <= deadline_sec {
                    self.deadline_hits += 1;
                }
            }
        }
    }

    fn p_complete(&self) -> f32 {
        ratio(self.successes, self.trials)
    }

    fn p_complete_deadline(&self) -> f32 {
        ratio(self.deadline_hits, self.trials)
    }

    fn per(&self) -> f32 {
        if self.tx_packets == 0 {
            0.0
        } else {
            1.0 - (self.packet_successes as f32 / self.tx_packets as f32)
        }
    }

    fn fer(&self) -> f32 {
        if self.tx_frames == 0 {
            0.0
        } else {
            1.0 - (self.frame_successes as f32 / self.tx_frames as f32)
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

    fn goodput_bps_effective(&self, payload_bits: usize) -> f32 {
        if self.total_elapsed_sec <= 0.0 {
            0.0
        } else {
            (payload_bits * self.successes) as f32 / self.total_elapsed_sec
        }
    }

    fn goodput_bps_success_mean(&self, payload_bits: usize) -> Option<f32> {
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

fn parse_list(arg: Option<String>, fallback: &[f32]) -> Vec<f32> {
    let Some(raw) = arg else {
        return fallback.to_vec();
    };
    let mut out = Vec::new();
    for part in raw.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        if let Ok(v) = p.parse::<f32>() {
            out.push(v);
        }
    }
    if out.is_empty() {
        fallback.to_vec()
    } else {
        out
    }
}

fn parse_cli() -> Cli {
    let mut kv = std::collections::HashMap::<String, String>::new();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(stripped) = arg.strip_prefix("--") {
            if let Some((k, v)) = stripped.split_once('=') {
                kv.insert(k.to_string(), v.to_string());
            } else {
                let key = stripped.to_string();
                if let Some(next) = args.next() {
                    if next.starts_with("--") {
                        kv.insert(key, "true".to_string());
                        let stripped_next = next.trim_start_matches("--");
                        if let Some((k2, v2)) = stripped_next.split_once('=') {
                            kv.insert(k2.to_string(), v2.to_string());
                        } else {
                            kv.insert(stripped_next.to_string(), "true".to_string());
                        }
                    } else {
                        kv.insert(key, next);
                    }
                } else {
                    kv.insert(key, "true".to_string());
                }
            }
        }
    }

    let mode = kv
        .remove("mode")
        .unwrap_or_else(|| "point".to_string())
        .to_lowercase();
    let trials = kv
        .remove("trials")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(40)
        .max(1);
    let payload_len = kv
        .remove("payload-len")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(32)
        .max(1);
    let deadline_sec = kv
        .remove("deadline-sec")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(2.0)
        .max(0.05);
    let max_sec = kv
        .remove("max-sec")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(4.0)
        .max(deadline_sec);
    let seed = kv
        .remove("seed")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0xBADC0FFE_u64);
    let target_p_complete = kv
        .remove("target-p")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.95)
        .clamp(0.0, 1.0);
    let sigma = kv
        .remove("sigma")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0)
        .max(0.0);
    let cfo_hz = kv
        .remove("cfo-hz")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0);
    let ppm = kv
        .remove("ppm")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0);
    let burst_loss = kv
        .remove("burst-loss")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);

    let multipath = if let Some(spec) = kv.remove("multipath") {
        if let Some(p) = MultipathProfile::preset(&spec.to_lowercase()) {
            p
        } else {
            MultipathProfile::parse_custom(&spec).unwrap_or_else(MultipathProfile::none)
        }
    } else {
        MultipathProfile::none()
    };

    let sweep_awgn = parse_list(
        kv.remove("sweep-awgn"),
        &[0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04],
    );
    let sweep_cfo = parse_list(
        kv.remove("sweep-cfo"),
        &[0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 120.0],
    );
    let sweep_ppm = parse_list(
        kv.remove("sweep-ppm"),
        &[-200.0, -120.0, -80.0, -40.0, 0.0, 40.0, 80.0, 120.0, 200.0],
    );
    let sweep_loss = parse_list(kv.remove("sweep-loss"), &[0.0, 0.05, 0.1, 0.2, 0.3, 0.4]);

    Cli {
        mode,
        trials,
        payload_len,
        deadline_sec,
        max_sec,
        seed,
        target_p_complete,
        base: ChannelImpairment {
            sigma,
            cfo_hz,
            ppm,
            burst_loss,
            multipath,
        },
        sweep_awgn,
        sweep_cfo,
        sweep_ppm,
        sweep_loss,
    }
}

fn make_payload(len: usize, seed: u64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = vec![0u8; len];
    rng.fill(data.as_mut_slice());
    data
}

fn add_awgn_with_rng(samples: &mut [f32], sigma: f32, rng: &mut StdRng) {
    if sigma <= 0.0 {
        return;
    }
    let normal = Normal::new(0.0, sigma).expect("normal distribution");
    for s in samples.iter_mut() {
        *s += normal.sample(rng);
    }
}

fn apply_multipath(input: &[f32], mp: &MultipathProfile) -> Vec<f32> {
    if mp.taps.len() == 1 && mp.taps[0].0 == 0 && (mp.taps[0].1 - 1.0).abs() < 1e-6 {
        return input.to_vec();
    }

    let max_delay = mp.taps.iter().map(|(d, _)| *d).max().unwrap_or(0);
    let mut out = vec![0.0f32; input.len() + max_delay];
    for &(delay, gain) in &mp.taps {
        for (i, &x) in input.iter().enumerate() {
            out[i + delay] += gain * x;
        }
    }

    // チャネル利得で正規化（平均電力の極端な増減を避ける）
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

    add_awgn_with_rng(&mut sig, imp.sigma, rng);
    sig
}

fn run_dsss_trial(payload: &[u8], imp: &ChannelImpairment, max_sec: f32, seed: u64) -> TrialResult {
    let dsp_cfg = DspConfig::default_48k();
    let sample_rate = dsp_cfg.sample_rate;
    let max_samples = (max_sec * sample_rate).round() as usize;
    let k = payload.len().div_ceil(PAYLOAD_SIZE).max(2);

    let mut enc_cfg = EncoderConfig::new(dsp_cfg.clone());
    enc_cfg.fountain_k = k;
    let packets_per_burst = enc_cfg.packets_per_sync_burst.max(1);

    let mut encoder = Encoder::new(enc_cfg);
    let mut stream = encoder.encode_stream(payload);

    let mut rx_cfg = dsp_cfg.clone();
    rx_cfg.carrier_freq += imp.cfo_hz;
    let mut decoder = Decoder::new(payload.len(), k, rx_cfg);

    let gap = vec![0.0f32; GAP_SAMPLES_DSSS];
    let mut rng = StdRng::seed_from_u64(seed ^ 0xD55A_0001);

    let mut elapsed_samples = 0usize;
    let mut tx_frames = 0usize;
    let mut frame_successes = 0usize;
    let mut tx_packets = 0usize;
    let mut packet_successes = 0usize;
    let mut dropped_bursts = 0usize;
    let mut received_prev = 0usize;

    while elapsed_samples < max_samples {
        let mut burst = stream.next().expect("encoder stream should be infinite");
        burst.extend_from_slice(&gap);

        tx_frames += 1;
        tx_packets += packets_per_burst;

        let drop_burst = rng.gen::<f32>() < imp.burst_loss;
        if drop_burst {
            dropped_bursts += 1;
        }

        let burst = apply_channel(&burst, imp, &mut rng, drop_burst);

        let recv_before = received_prev;
        let mut recv_after = received_prev;

        for chunk in burst.chunks(CHUNK_SAMPLES) {
            elapsed_samples += chunk.len();
            let progress = decoder.process_samples(chunk);
            recv_after = progress.received_packets;

            if progress.complete {
                let delta = recv_after.saturating_sub(recv_before);
                packet_successes += delta;
                if delta > 0 {
                    frame_successes += 1;
                }
                let ok = decoder
                    .recovered_data()
                    .is_some_and(|r| &r[..payload.len()] == payload);
                return TrialResult {
                    success: ok,
                    completion_sec: Some(elapsed_samples as f32 / sample_rate),
                    elapsed_sec: elapsed_samples as f32 / sample_rate,
                    tx_frames,
                    frame_successes,
                    tx_packets,
                    packet_successes,
                    dropped_bursts,
                };
            }

            if elapsed_samples >= max_samples {
                break;
            }
        }

        let delta = recv_after.saturating_sub(recv_before);
        packet_successes += delta;
        if delta > 0 {
            frame_successes += 1;
        }
        received_prev = recv_after;
    }

    TrialResult {
        success: false,
        completion_sec: None,
        elapsed_sec: elapsed_samples as f32 / sample_rate,
        tx_frames,
        frame_successes,
        tx_packets,
        packet_successes,
        dropped_bursts,
    }
}

fn run_fsk_trial(payload: &[u8], imp: &ChannelImpairment, max_sec: f32, seed: u64) -> TrialResult {
    let mut tx_cfg = BfskConfig::default_48k();
    tx_cfg.freq0 += imp.cfo_hz;
    tx_cfg.freq1 += imp.cfo_hz;

    let rx_cfg = BfskConfig::default_48k();
    let sample_rate = rx_cfg.sample_rate;
    let max_samples = (max_sec * sample_rate).round() as usize;

    let tx = BfskModulator::new(tx_cfg);
    let rx = BfskDemodulator::new(rx_cfg.clone());

    let mut rng = StdRng::seed_from_u64(seed ^ 0xF5F5_0001);
    let mut elapsed_samples = 0usize;
    let mut tx_frames = 0usize;
    let mut dropped_bursts = 0usize;

    while elapsed_samples < max_samples {
        let frame = tx.modulate_frame(payload);
        let lead = rng.gen_range(0..rx_cfg.samples_per_bit());

        let mut burst = vec![0.0f32; lead];
        burst.extend_from_slice(&frame);
        burst.extend(std::iter::repeat_n(0.0f32, rx_cfg.samples_per_bit() * 2));

        tx_frames += 1;
        let drop_burst = rng.gen::<f32>() < imp.burst_loss;
        if drop_burst {
            dropped_bursts += 1;
        }

        let burst = apply_channel(&burst, imp, &mut rng, drop_burst);
        elapsed_samples += burst.len();

        let decoded = rx.find_and_decode(&burst);
        if decoded.as_deref() == Some(payload) {
            return TrialResult {
                success: true,
                completion_sec: Some(elapsed_samples as f32 / sample_rate),
                elapsed_sec: elapsed_samples as f32 / sample_rate,
                tx_frames,
                frame_successes: 1,
                tx_packets: tx_frames,
                packet_successes: 1,
                dropped_bursts,
            };
        }
    }

    TrialResult {
        success: false,
        completion_sec: None,
        elapsed_sec: elapsed_samples as f32 / sample_rate,
        tx_frames,
        frame_successes: 0,
        tx_packets: tx_frames,
        packet_successes: 0,
        dropped_bursts,
    }
}

fn evaluate(phy: PhyKind, cli: &Cli, imp: &ChannelImpairment, scenario: &str) -> Metrics {
    let mut metrics = Metrics::default();

    for trial_idx in 0..cli.trials {
        let trial_seed = cli
            .seed
            .wrapping_add((trial_idx as u64).wrapping_mul(0x9E37_79B9));
        let payload = make_payload(cli.payload_len, trial_seed ^ 0xABCDEF01);

        let result = match phy {
            PhyKind::Dsss => run_dsss_trial(&payload, imp, cli.max_sec, trial_seed),
            PhyKind::Fsk => run_fsk_trial(&payload, imp, cli.max_sec, trial_seed),
        };

        metrics.push(result, cli.deadline_sec);
    }

    print_metrics_row(scenario, phy, cli, imp, &metrics);
    metrics
}

fn print_metrics_header() {
    println!(
        "scenario,phy,trials,success,p_complete,p_complete_deadline,deadline_s,per,fer,goodput_effective_bps,goodput_success_mean_bps,p95_complete_s,mean_complete_s,drop_bursts,multipath"
    );
}

fn fmt_opt(v: Option<f32>) -> String {
    v.map(|x| format!("{x:.6}"))
        .unwrap_or_else(|| "NaN".to_string())
}

fn print_metrics_row(
    scenario: &str,
    phy: PhyKind,
    cli: &Cli,
    imp: &ChannelImpairment,
    m: &Metrics,
) {
    let payload_bits = cli.payload_len * 8;
    println!(
        "{scenario},{},{},{},{:.6},{:.6},{:.3},{:.6},{:.6},{:.3},{},{},{},{},{}",
        phy.as_str(),
        m.trials,
        m.successes,
        m.p_complete(),
        m.p_complete_deadline(),
        cli.deadline_sec,
        m.per(),
        m.fer(),
        m.goodput_bps_effective(payload_bits),
        fmt_opt(m.goodput_bps_success_mean(payload_bits)),
        fmt_opt(m.p95_completion_sec()),
        fmt_opt(m.mean_completion_sec()),
        m.dropped_bursts,
        imp.multipath.name,
    );
}

fn run_point(cli: &Cli) {
    print_metrics_header();
    let scenario = format!(
        "point(sigma={:.3},cfo={:.1},ppm={:.1},loss={:.2})",
        cli.base.sigma, cli.base.cfo_hz, cli.base.ppm, cli.base.burst_loss
    );
    evaluate(PhyKind::Dsss, cli, &cli.base, &scenario);
    evaluate(PhyKind::Fsk, cli, &cli.base, &scenario);
}

fn run_sweep_awgn(cli: &Cli) {
    print_metrics_header();

    let mut dsss_limit = None;
    let mut fsk_limit = None;

    for &sigma in &cli.sweep_awgn {
        let mut imp = cli.base.clone();
        imp.sigma = sigma;
        let scenario = format!(
            "awgn(sigma={sigma:.3},cfo={:.1},ppm={:.1},loss={:.2})",
            imp.cfo_hz, imp.ppm, imp.burst_loss
        );

        let d = evaluate(PhyKind::Dsss, cli, &imp, &scenario);
        let f = evaluate(PhyKind::Fsk, cli, &imp, &scenario);

        if d.p_complete_deadline() >= cli.target_p_complete {
            dsss_limit = Some(sigma);
        }
        if f.p_complete_deadline() >= cli.target_p_complete {
            fsk_limit = Some(sigma);
        }
    }

    println!(
        "# awgn_limit(target_p_complete_deadline>={:.2}, deadline_s={:.2}) dsss={} fsk={}",
        cli.target_p_complete,
        cli.deadline_sec,
        dsss_limit
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "none".to_string()),
        fsk_limit
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "none".to_string()),
    );
}

fn run_sweep_cfo(cli: &Cli) {
    print_metrics_header();
    for &cfo in &cli.sweep_cfo {
        let mut imp = cli.base.clone();
        imp.cfo_hz = cfo;
        let scenario = format!(
            "cfo(cfo={cfo:.1},sigma={:.3},ppm={:.1},loss={:.2})",
            imp.sigma, imp.ppm, imp.burst_loss
        );
        evaluate(PhyKind::Dsss, cli, &imp, &scenario);
        evaluate(PhyKind::Fsk, cli, &imp, &scenario);
    }
}

fn run_sweep_ppm(cli: &Cli) {
    print_metrics_header();
    for &ppm in &cli.sweep_ppm {
        let mut imp = cli.base.clone();
        imp.ppm = ppm;
        let scenario = format!(
            "ppm(ppm={ppm:.1},sigma={:.3},cfo={:.1},loss={:.2})",
            imp.sigma, imp.cfo_hz, imp.burst_loss
        );
        evaluate(PhyKind::Dsss, cli, &imp, &scenario);
        evaluate(PhyKind::Fsk, cli, &imp, &scenario);
    }
}

fn run_sweep_loss(cli: &Cli) {
    print_metrics_header();
    for &loss in &cli.sweep_loss {
        let mut imp = cli.base.clone();
        imp.burst_loss = loss.clamp(0.0, 1.0);
        let scenario = format!(
            "loss(loss={:.2},sigma={:.3},cfo={:.1},ppm={:.1})",
            imp.burst_loss, imp.sigma, imp.cfo_hz, imp.ppm
        );
        evaluate(PhyKind::Dsss, cli, &imp, &scenario);
        evaluate(PhyKind::Fsk, cli, &imp, &scenario);
    }
}

fn run_sweep_multipath(cli: &Cli) {
    print_metrics_header();
    for name in ["none", "mild", "medium", "harsh"] {
        let mut imp = cli.base.clone();
        imp.multipath = MultipathProfile::preset(name).unwrap_or_else(MultipathProfile::none);
        let scenario = format!(
            "multipath(profile={name},sigma={:.3},cfo={:.1},ppm={:.1},loss={:.2})",
            imp.sigma, imp.cfo_hz, imp.ppm, imp.burst_loss
        );
        evaluate(PhyKind::Dsss, cli, &imp, &scenario);
        evaluate(PhyKind::Fsk, cli, &imp, &scenario);
    }
}

fn print_help() {
    println!(
        "usage: cargo run --release --bin phy_eval -- [options]\n\
         modes:\n\
         - --mode point           単一点評価 (default)\n\
         - --mode sweep-awgn      AWGN sweep + awgn_limit推定\n\
         - --mode sweep-cfo       CFO sweep\n\
         - --mode sweep-ppm       ppm sweep\n\
         - --mode sweep-loss      バースト欠落率 sweep\n\
         - --mode sweep-multipath マルチパス profile sweep\n\
         - --mode sweep-all       上記 sweep を一括実行\n\n\
         common options:\n\
         --trials N\n\
         --payload-len N\n\
         --deadline-sec F\n\
         --max-sec F\n\
         --seed N\n\
         --sigma F\n\
         --cfo-hz F\n\
         --ppm F\n\
         --burst-loss F      (0..1)\n\
         --multipath NAME_OR_TAPS\n\
           NAME: none|mild|medium|harsh\n\
           TAPS: \"0:1.0,9:0.4,23:0.2\"\n\
         --target-p F        (awgn_limit判定用, default 0.95)\n\n\
         sweep lists (comma separated):\n\
         --sweep-awgn \"0,0.005,0.01\"\n\
         --sweep-cfo  \"0,2,5,10\"\n\
         --sweep-ppm  \"-120,-80,0,80,120\"\n\
         --sweep-loss \"0,0.1,0.2,0.3\"\n"
    );
}

fn main() {
    let cli = parse_cli();
    if cli.mode == "help" || cli.mode == "--help" || cli.mode == "-h" {
        print_help();
        return;
    }

    match cli.mode.as_str() {
        "point" => run_point(&cli),
        "sweep-awgn" => run_sweep_awgn(&cli),
        "sweep-cfo" => run_sweep_cfo(&cli),
        "sweep-ppm" => run_sweep_ppm(&cli),
        "sweep-loss" => run_sweep_loss(&cli),
        "sweep-multipath" => run_sweep_multipath(&cli),
        "sweep-all" => {
            run_sweep_awgn(&cli);
            run_sweep_cfo(&cli);
            run_sweep_ppm(&cli);
            run_sweep_loss(&cli);
            run_sweep_multipath(&cli);
        }
        _ => {
            eprintln!("unknown mode: {}", cli.mode);
            print_help();
            std::process::exit(2);
        }
    }
}
