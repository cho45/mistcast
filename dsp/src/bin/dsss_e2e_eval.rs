use dsp::encoder::{Encoder, EncoderConfig};
use dsp::params::PAYLOAD_SIZE;
use dsp::{decoder::Decoder, DspConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::env;

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

    fn max_delay(&self) -> usize {
        self.taps.iter().map(|(d, _)| *d).max().unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
struct ChannelImpairment {
    sigma: f32,
    cfo_hz: f32,
    ppm: f32,
    burst_loss: f32,
    fading_depth: f32,
    multipath: MultipathProfile,
}

#[derive(Clone, Debug)]
struct Cli {
    mode: String,
    trials: usize,
    payload_bytes: usize,
    deadline_sec: f32,
    max_sec: f32,
    chunk_samples: usize,
    gap_samples: usize,
    seed: u64,
    target_p_complete: f32,
    base: ChannelImpairment,
    sweep_awgn: Vec<f32>,
    sweep_ppm: Vec<f32>,
    sweep_loss: Vec<f32>,
    sweep_fading: Vec<f32>,
    sample_rate: f32,
    chip_rate: f32,
    carrier_freq: f32,
    mseq_order: usize,
    rrc_alpha: f32,
    sync_word_bits: usize,
    preamble_repeat: usize,
    packets_per_burst: usize,
}

#[derive(Default, Clone, Debug)]
struct TrialResult {
    success: bool,
    completion_sec: Option<f32>,
    elapsed_sec: f32,
    attempts: usize,
    first_attempt_success: bool,
    bit_errors: usize,
    bits_compared: usize,
    dropped_attempts: usize,
    tx_signal_energy_sum: f64,
    tx_signal_samples: usize,
    process_time_ns: u64,
}

#[derive(Default, Clone, Debug)]
struct Metrics {
    trials: usize,
    successes: usize,
    deadline_hits: usize,
    first_attempt_successes: usize,
    total_attempts: usize,
    total_bit_errors: usize,
    total_bits_compared: usize,
    total_elapsed_sec: f32,
    dropped_attempts: usize,
    total_tx_signal_energy: f64,
    total_tx_signal_samples: usize,
    total_process_time_ns: u64,
    completion_secs: Vec<f32>,
}

impl Metrics {
    fn push(&mut self, t: TrialResult, deadline_sec: f32) {
        self.trials += 1;
        self.total_elapsed_sec += t.elapsed_sec;
        self.total_attempts += t.attempts;
        self.total_bit_errors += t.bit_errors;
        self.total_bits_compared += t.bits_compared;
        self.dropped_attempts += t.dropped_attempts;
        self.total_tx_signal_energy += t.tx_signal_energy_sum;
        self.total_tx_signal_samples += t.tx_signal_samples;
        self.total_process_time_ns += t.process_time_ns;

        if t.first_attempt_success {
            self.first_attempt_successes += 1;
        }
        if t.success {
            self.successes += 1;
            if let Some(c) = t.completion_sec {
                self.completion_secs.push(c);
                if c <= deadline_sec {
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

    fn ber(&self) -> f32 {
        if self.total_bits_compared == 0 {
            0.0
        } else {
            self.total_bit_errors as f32 / self.total_bits_compared as f32
        }
    }

    fn fer(&self) -> f32 {
        1.0 - ratio(self.first_attempt_successes, self.trials)
    }

    fn per(&self) -> f32 {
        self.fer()
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

    fn goodput_effective_bps(&self, payload_bits: usize) -> f32 {
        if self.total_elapsed_sec <= 0.0 {
            0.0
        } else {
            (payload_bits * self.successes) as f32 / self.total_elapsed_sec
        }
    }

    fn goodput_success_mean_bps(&self, payload_bits: usize) -> Option<f32> {
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

    fn tx_signal_power(&self) -> Option<f32> {
        if self.total_tx_signal_samples == 0 {
            None
        } else {
            Some((self.total_tx_signal_energy / self.total_tx_signal_samples as f64) as f32)
        }
    }

    fn avg_process_time_per_sample_ns(&self) -> f32 {
        if self.total_tx_signal_samples == 0 {
            0.0
        } else {
            self.total_process_time_ns as f32 / self.total_tx_signal_samples as f32
        }
    }

    fn awgn_snr_db(&self, sigma: f32) -> Option<f32> {
        if sigma <= 0.0 {
            return None;
        }
        let p_sig = self.tx_signal_power()?;
        if p_sig <= 0.0 {
            return None;
        }
        let p_noise = sigma * sigma;
        Some(10.0 * (p_sig / p_noise).log10())
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
    let mut kv = HashMap::<String, String>::new();
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
    let payload_bytes = kv
        .remove("payload-bytes")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64)
        .max(1);
    let deadline_sec = kv
        .remove("deadline-sec")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.8)
        .max(0.01);
    let max_sec = kv
        .remove("max-sec")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(2.0)
        .max(deadline_sec);
    let chunk_samples = kv
        .remove("chunk-samples")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(16_384)
        .max(1);
    let gap_samples = kv
        .remove("gap-samples")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64);
    let seed = kv
        .remove("seed")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0xA11CE_u64);
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
    let fading_depth = kv
        .remove("fading-depth")
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
        &[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    );
    let sweep_ppm = parse_list(
        kv.remove("sweep-ppm"),
        &[-120.0, -80.0, -40.0, 0.0, 40.0, 80.0, 120.0],
    );
    let sweep_loss = parse_list(kv.remove("sweep-loss"), &[0.0, 0.05, 0.1, 0.2, 0.3, 0.4]);
    let sweep_fading = parse_list(kv.remove("sweep-fading"), &[0.0, 0.2, 0.4, 0.6, 0.8]);

    let sample_rate = kv
        .remove("sample-rate")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(dsp::params::DEFAULT_SAMPLE_RATE);
    let chip_rate = kv
        .remove("chip-rate")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(dsp::params::CHIP_RATE);
    let carrier_freq = kv
        .remove("carrier-freq")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(dsp::params::CARRIER_FREQ);
    let mseq_order = kv
        .remove("mseq-order")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(dsp::params::MSEQ_ORDER);
    let rrc_alpha = kv
        .remove("rrc-alpha")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(dsp::params::RRC_ALPHA);

    let sync_word_bits = kv
        .remove("sync-word-bits")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(dsp::params::SYNC_WORD_BITS);

    let preamble_repeat = kv
        .remove("preamble-repeat")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(dsp::params::PREAMBLE_REPEAT);

    let packets_per_burst = kv
        .remove("packets-per-burst")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(dsp::params::PACKETS_PER_SYNC_BURST);

    Cli {
        mode,
        trials,
        payload_bytes,
        deadline_sec,
        max_sec,
        chunk_samples,
        gap_samples,
        seed,
        target_p_complete,
        base: ChannelImpairment {
            sigma,
            cfo_hz,
            ppm,
            burst_loss,
            fading_depth,
            multipath,
        },
        sweep_awgn,
        sweep_ppm,
        sweep_loss,
        sweep_fading,
        sample_rate,
        chip_rate,
        carrier_freq,
        mseq_order,
        rrc_alpha,
        sync_word_bits,
        preamble_repeat,
        packets_per_burst,
    }
}

fn make_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.gen::<u8>()).collect()
}

fn count_bit_errors_bytes(tx: &[u8], rx: Option<&[u8]>) -> usize {
    let Some(rx) = rx else {
        return tx.len() * 8;
    };
    let mut errs = 0usize;
    for (idx, &b) in tx.iter().enumerate() {
        let rb = *rx.get(idx).unwrap_or(&0u8);
        errs += (b ^ rb).count_ones() as usize;
    }
    errs
}

fn add_awgn_with_rng(samples: &mut [f32], sigma: f32, rng: &mut StdRng) {
    if sigma <= 0.0 {
        return;
    }
    let normal = Normal::new(0.0, sigma).expect("normal distribution");
    for s in samples {
        *s += normal.sample(rng);
    }
}

fn apply_multipath(input: &[f32], mp: &MultipathProfile) -> Vec<f32> {
    if mp.taps.len() == 1 && mp.taps[0].0 == 0 && (mp.taps[0].1 - 1.0).abs() < 1e-6 {
        return input.to_vec();
    }
    let max_delay = mp.max_delay();
    let mut out = vec![0.0f32; input.len() + max_delay];
    for &(delay, gain) in &mp.taps {
        for (i, &x) in input.iter().enumerate() {
            out[i + delay] += gain * x;
        }
    }
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

fn apply_fading(sig: &mut [f32], depth: f32, rng: &mut StdRng) {
    if sig.is_empty() || depth <= 0.0 {
        return;
    }
    let d = depth.clamp(0.0, 1.0);
    let g0 = 1.0 - d * rng.gen::<f32>();
    let g1 = 1.0 - d * rng.gen::<f32>();
    let n = sig.len().max(2);
    for (i, s) in sig.iter_mut().enumerate() {
        let t = i as f32 / (n - 1) as f32;
        let g = g0 + (g1 - g0) * t;
        *s *= g;
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
    apply_fading(&mut sig, imp.fading_depth, rng);
    add_awgn_with_rng(&mut sig, imp.sigma, rng);
    sig
}

fn signal_energy(samples: &[f32]) -> f64 {
    samples
        .iter()
        .map(|&x| {
            let v = x as f64;
            v * v
        })
        .sum()
}

fn run_trial_dsss_e2e(imp: &ChannelImpairment, cli: &Cli, seed: u64) -> TrialResult {
    let mut tx_cfg = DspConfig::new(cli.sample_rate);
    tx_cfg.chip_rate = cli.chip_rate;
    tx_cfg.carrier_freq = cli.carrier_freq;
    tx_cfg.mseq_order = cli.mseq_order;
    tx_cfg.rrc_alpha = cli.rrc_alpha;
    tx_cfg.sync_word_bits = cli.sync_word_bits;
    tx_cfg.preamble_repeat = cli.preamble_repeat;

    let mut rx_cfg = tx_cfg.clone();
    rx_cfg.carrier_freq += imp.cfo_hz;

    let payload = make_bytes(cli.payload_bytes, seed ^ 0x1234_5678);
    let k = payload.len().div_ceil(PAYLOAD_SIZE).max(1);
    let mut enc_cfg = EncoderConfig::new(tx_cfg.clone());
    enc_cfg.fountain_k = k;
    enc_cfg.packets_per_sync_burst = cli.packets_per_burst;
    let mut encoder = Encoder::new(enc_cfg);
    let mut stream = encoder.encode_stream(&payload);
    let mut decoder = Decoder::new(payload.len(), k, rx_cfg);
    decoder.set_packets_per_sync_burst(cli.packets_per_burst);

    let mut rng = StdRng::seed_from_u64(seed ^ 0xD55A_0001);
    let mut elapsed_sec = 0.0f32;
    let mut attempts = 0usize;
    let mut dropped_attempts = 0usize;
    let mut tx_signal_energy_sum = 0.0f64;
    let mut tx_signal_samples = 0usize;
    let mut total_process_ns = 0u64;

    let chunk = cli.chunk_samples.max(1);
    let gap = cli.gap_samples;
    loop {
        if elapsed_sec >= cli.max_sec {
            break;
        }
        let Some(frame) = stream.next() else {
            break;
        };
        attempts += 1;
        tx_signal_energy_sum += signal_energy(&frame);
        tx_signal_samples += frame.len();

        let drop_burst = rng.gen::<f32>() < imp.burst_loss;
        if drop_burst {
            dropped_attempts += 1;
        }
        let rx_frame = apply_channel(&frame, imp, &mut rng, drop_burst);
        elapsed_sec += rx_frame.len() as f32 / tx_cfg.sample_rate;
        for piece in rx_frame.chunks(chunk) {
            let start_time = std::time::Instant::now();
            let progress = decoder.process_samples(piece);
            total_process_ns += start_time.elapsed().as_nanos() as u64;

            if progress.complete {
                let recovered = decoder.recovered_data();
                let errs = count_bit_errors_bytes(&payload, recovered);
                let bits_compared = payload.len() * 8;
                return TrialResult {
                    success: errs == 0,
                    completion_sec: Some(elapsed_sec),
                    elapsed_sec,
                    attempts,
                    first_attempt_success: attempts == 1 && errs == 0,
                    bit_errors: errs,
                    bits_compared,
                    dropped_attempts,
                    tx_signal_energy_sum,
                    tx_signal_samples,
                    process_time_ns: total_process_ns,
                };
            }
        }

        if gap > 0 {
            let mut gap_sig = vec![0.0f32; gap];
            add_awgn_with_rng(&mut gap_sig, imp.sigma, &mut rng);
            if imp.ppm.abs() >= 1.0 {
                gap_sig = apply_clock_drift_ppm(&gap_sig, imp.ppm);
            }
            elapsed_sec += gap_sig.len() as f32 / tx_cfg.sample_rate;
            for piece in gap_sig.chunks(chunk) {
                let start_time = std::time::Instant::now();
                let progress = decoder.process_samples(piece);
                total_process_ns += start_time.elapsed().as_nanos() as u64;

                if progress.complete {
                    let recovered = decoder.recovered_data();
                    let errs = count_bit_errors_bytes(&payload, recovered);
                    let bits_compared = payload.len() * 8;
                    return TrialResult {
                        success: errs == 0,
                        completion_sec: Some(elapsed_sec),
                        elapsed_sec,
                        attempts,
                        first_attempt_success: attempts == 1 && errs == 0,
                        bit_errors: errs,
                        bits_compared,
                        dropped_attempts,
                        tx_signal_energy_sum,
                        tx_signal_samples,
                        process_time_ns: total_process_ns,
                    };
                }
            }
        }
    }

    let bits_compared = payload.len() * 8;
    let bit_errors = count_bit_errors_bytes(&payload, decoder.recovered_data());
    TrialResult {
        success: false,
        completion_sec: None,
        elapsed_sec,
        attempts,
        first_attempt_success: false,
        bit_errors,
        bits_compared,
        dropped_attempts,
        tx_signal_energy_sum,
        tx_signal_samples,
        process_time_ns: total_process_ns,
    }
}

fn print_header() {
    println!(
        "scenario,phy,trials,success,deadline_hits,first_attempt_successes,total_bits_compared,total_bit_errors,tx_signal_power,awgn_noise_power,awgn_snr_db,p_complete,p_complete_deadline,deadline_s,ber,per,fer,goodput_effective_bps,goodput_success_mean_bps,p95_complete_s,mean_complete_s,total_attempts,dropped_attempts,avg_proc_ns_sample,multipath"
    );
}

fn fmt_opt(v: Option<f32>) -> String {
    v.map(|x| format!("{x:.6}"))
        .unwrap_or_else(|| "NaN".to_string())
}

fn print_row(scenario: &str, cli: &Cli, imp: &ChannelImpairment, m: &Metrics) {
    let noise_power = if imp.sigma > 0.0 {
        Some(imp.sigma * imp.sigma)
    } else {
        None
    };
    println!(
        "{scenario},dsss-e2e,{},{},{},{},{},{},{},{},{:.6},{:.6},{:.3},{:.6},{:.6},{:.6},{:.3},{},{},{},{},{},{},{:.2},{}",
        m.trials,
        m.successes,
        m.deadline_hits,
        m.first_attempt_successes,
        m.total_bits_compared,
        m.total_bit_errors,
        fmt_opt(m.tx_signal_power()),
        fmt_opt(noise_power),
        fmt_opt(m.awgn_snr_db(imp.sigma)),
        m.p_complete(),
        m.p_complete_deadline(),
        cli.deadline_sec,
        m.ber(),
        m.per(),
        m.fer(),
        m.goodput_effective_bps(cli.payload_bytes * 8),
        fmt_opt(m.goodput_success_mean_bps(cli.payload_bytes * 8)),
        fmt_opt(m.p95_completion_sec()),
        fmt_opt(m.mean_completion_sec()),
        m.total_attempts,
        m.dropped_attempts,
        m.avg_process_time_per_sample_ns(),
        imp.multipath.name,
    );
}

fn evaluate(cli: &Cli, imp: &ChannelImpairment, scenario: &str) -> Metrics {
    let mut metrics = Metrics::default();
    for trial_idx in 0..cli.trials {
        let trial_seed = cli
            .seed
            .wrapping_add((trial_idx as u64).wrapping_mul(0x9E37_79B9));
        let tr = run_trial_dsss_e2e(imp, cli, trial_seed);
        metrics.push(tr, cli.deadline_sec);
    }
    print_row(scenario, cli, imp, &metrics);
    metrics
}

fn run_point(cli: &Cli) {
    print_header();
    let scenario = format!(
        "point(bytes={},sigma={:.3},cfo={:.1},ppm={:.1},loss={:.2},fade={:.2})",
        cli.payload_bytes,
        cli.base.sigma,
        cli.base.cfo_hz,
        cli.base.ppm,
        cli.base.burst_loss,
        cli.base.fading_depth
    );
    evaluate(cli, &cli.base, &scenario);
}

fn run_sweep_awgn(cli: &Cli) {
    print_header();
    let mut dsss_limit = None;
    for &sigma in &cli.sweep_awgn {
        let mut imp = cli.base.clone();
        imp.sigma = sigma;
        let scenario = format!(
            "awgn(bytes={},sigma={sigma:.3},cfo={:.1},ppm={:.1},loss={:.2},fade={:.2})",
            cli.payload_bytes, imp.cfo_hz, imp.ppm, imp.burst_loss, imp.fading_depth
        );
        let d = evaluate(cli, &imp, &scenario);
        if d.p_complete_deadline() >= cli.target_p_complete {
            dsss_limit = Some(sigma);
        }
    }
    println!(
        "# awgn_limit(target_p_complete_deadline>={:.2}, deadline_s={:.2}) dsss={}",
        cli.target_p_complete,
        cli.deadline_sec,
        dsss_limit
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "none".to_string()),
    );
}

fn run_sweep_ppm(cli: &Cli) {
    print_header();
    for &ppm in &cli.sweep_ppm {
        let mut imp = cli.base.clone();
        imp.ppm = ppm;
        let scenario = format!(
            "ppm(bytes={},ppm={ppm:.1},sigma={:.3},cfo={:.1},loss={:.2},fade={:.2})",
            cli.payload_bytes, imp.sigma, imp.cfo_hz, imp.burst_loss, imp.fading_depth
        );
        evaluate(cli, &imp, &scenario);
    }
}

fn run_sweep_loss(cli: &Cli) {
    print_header();
    for &loss in &cli.sweep_loss {
        let mut imp = cli.base.clone();
        imp.burst_loss = loss.clamp(0.0, 1.0);
        let scenario = format!(
            "loss(bytes={},loss={:.2},sigma={:.3},cfo={:.1},ppm={:.1},fade={:.2})",
            cli.payload_bytes, imp.burst_loss, imp.sigma, imp.cfo_hz, imp.ppm, imp.fading_depth
        );
        evaluate(cli, &imp, &scenario);
    }
}

fn run_sweep_multipath(cli: &Cli) {
    print_header();
    for name in ["none", "mild", "medium", "harsh"] {
        let mut imp = cli.base.clone();
        imp.multipath = MultipathProfile::preset(name).unwrap_or_else(MultipathProfile::none);
        let scenario = format!(
            "multipath(bytes={},profile={name},sigma={:.3},cfo={:.1},ppm={:.1},loss={:.2},fade={:.2})",
            cli.payload_bytes, imp.sigma, imp.cfo_hz, imp.ppm, imp.burst_loss, imp.fading_depth
        );
        evaluate(cli, &imp, &scenario);
    }
}

fn run_sweep_fading(cli: &Cli) {
    print_header();
    for &fade in &cli.sweep_fading {
        let mut imp = cli.base.clone();
        imp.fading_depth = fade.clamp(0.0, 1.0);
        let scenario = format!(
            "fading(bytes={},fade={:.2},sigma={:.3},ppm={:.1},loss={:.2})",
            cli.payload_bytes, imp.fading_depth, imp.sigma, imp.ppm, imp.burst_loss
        );
        evaluate(cli, &imp, &scenario);
    }
}

fn print_help() {
    println!(
        "usage: cargo run --release --bin dsss_e2e_eval -- [options]\n\
         modes:\n\
         - --mode point             単点評価 (default)\n\
         - --mode sweep-awgn        AWGN sweep + awgn_limit推定\n\
         - --mode sweep-ppm         ppm sweep\n\
         - --mode sweep-loss        バースト欠落率 sweep\n\
         - --mode sweep-fading      振幅フェージング深さ sweep\n\
         - --mode sweep-multipath   マルチパス profile sweep\n\
         - --mode sweep-all         全sweep\n\n\
         common options:\n\
         --trials N\n\
         --payload-bytes N          送信バイト長 (default: 64)\n\
         --deadline-sec F\n\
         --max-sec F\n\
         --chunk-samples N\n\
         --gap-samples N\n\
         --seed N\n\
         --sigma F\n\
         --ppm F\n\
         --burst-loss F             (0..1)\n\
         --fading-depth F           (0..1)\n\
         --multipath NAME_OR_TAPS\n\
             NAME: none|mild|medium|harsh\n\
             TAPS: \"0:1.0,9:0.4,23:0.2\"\n\
         --target-p F               (awgn_limit判定用)\n\n\
         sweep lists:\n\
         --sweep-awgn \"0,0.005,0.01\"\n\
         --sweep-ppm  \"-120,-80,0,80,120\"\n\
         --sweep-loss \"0,0.1,0.2,0.3\"\n\
         --sweep-fading \"0,0.2,0.4,0.6,0.8\"\n"
    );
}

fn main() {
    let cli = parse_cli();

    if matches!(cli.mode.as_str(), "help" | "--help" | "-h") {
        print_help();
        return;
    }

    match cli.mode.as_str() {
        "point" => run_point(&cli),
        "sweep-awgn" => run_sweep_awgn(&cli),
        "sweep-ppm" => run_sweep_ppm(&cli),
        "sweep-loss" => run_sweep_loss(&cli),
        "sweep-fading" => run_sweep_fading(&cli),
        "sweep-multipath" => run_sweep_multipath(&cli),
        "sweep-all" => {
            run_sweep_awgn(&cli);
            run_sweep_ppm(&cli);
            run_sweep_loss(&cli);
            run_sweep_fading(&cli);
            run_sweep_multipath(&cli);
        }
        _ => {
            eprintln!("unknown mode: {}", cli.mode);
            print_help();
            std::process::exit(2);
        }
    }
}
