use dsp::common::msequence::MSequence;
use dsp::common::rrc_filter::RrcFilter;
use dsp::params::MODULATION;
use dsp::phy::fsk::{BfskConfig, BfskDemodulator, BfskModulator};
use dsp::phy::modulator::Modulator;
use dsp::phy::sync::downconvert;
use dsp::DspConfig;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::env;

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
    multipath: MultipathProfile,
}

#[derive(Clone, Debug)]
struct Cli {
    mode: String,
    trials: usize,
    payload_bits: usize,
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
    attempts: usize,
    first_attempt_success: bool,
    bit_errors: usize,
    bits_compared: usize,
    dropped_attempts: usize,
    tx_signal_energy_sum: f64,
    tx_signal_samples: usize,
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
        // PHY-only評価では1 frame = 1 packetとして扱う
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

    let payload_bits = kv
        .remove("payload-bits")
        .or_else(|| {
            kv.remove("payload-len")
                .map(|s| (s.parse::<usize>().unwrap_or(32) * 8).to_string())
        })
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(256)
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
        payload_bits,
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

fn make_bits(len: usize, seed: u64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len)
        .map(|_| if rng.gen::<bool>() { 1 } else { 0 })
        .collect()
}

fn count_bit_errors(tx: &[u8], rx: Option<&[u8]>) -> usize {
    let Some(rx) = rx else {
        return tx.len();
    };

    let mut errs = 0usize;
    for (idx, &b) in tx.iter().enumerate() {
        let rb = rx.get(idx).copied().unwrap_or(1 - b);
        if rb != b {
            errs += 1;
        }
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

fn dsss_modulate_bits(bits: &[u8], cfg: &DspConfig) -> Vec<f32> {
    let mut modulator = Modulator::new(cfg.clone());
    modulator.reset();
    let mut samples = modulator.modulate(bits);
    // 受信側マッチドフィルタ遅延ぶんの末尾サンプルを確保する。
    samples.extend(std::iter::repeat_n(0.0f32, cfg.rrc_num_taps() / 2));
    samples
}

fn decode_diff_to_bits_mode(diff_re: f32, diff_im: f32, out: &mut Vec<u8>) {
    if diff_re.abs() >= diff_im.abs() {
        if diff_re >= 0.0 {
            out.extend_from_slice(&[0, 0]);
        } else {
            out.extend_from_slice(&[1, 1]);
        }
    } else if diff_im >= 0.0 {
        out.extend_from_slice(&[0, 1]);
    } else {
        out.extend_from_slice(&[1, 0]);
    }
}

fn dsss_demodulate_bits(samples: &[f32], rx_cfg: &DspConfig, bit_len: usize) -> Option<Vec<u8>> {
    let (i_raw, q_raw) = downconvert(samples, 0, rx_cfg);

    let mut rrc_i = RrcFilter::from_config(rx_cfg);
    let mut rrc_q = RrcFilter::from_config(rx_cfg);

    let mut i_f = Vec::with_capacity(i_raw.len());
    let mut q_f = Vec::with_capacity(q_raw.len());
    for (&i, &q) in i_raw.iter().zip(q_raw.iter()) {
        i_f.push(rrc_i.process(i));
        q_f.push(rrc_q.process(q));
    }

    let spc = rx_cfg.samples_per_chip().max(1);
    let sf = rx_cfg.spread_factor();
    let sym_len = sf * spc;
    let payload_symbols = bit_len.div_ceil(MODULATION.bits_per_symbol());
    let total_symbols = payload_symbols;
    let total_delay = rx_cfg.rrc_num_taps().saturating_sub(1);

    let mut mseq = MSequence::new(rx_cfg.mseq_order);
    let pn: Vec<f32> = mseq.generate(sf).into_iter().map(|v| v as f32).collect();

    let correlate_symbol = |sym_idx: usize, phase: usize, delay: usize| -> Option<(f32, f32)> {
        let base = total_delay + delay + phase + sym_idx * sym_len;
        if base + (sf - 1) * spc >= i_f.len() {
            return None;
        }
        let mut ci = 0.0f32;
        let mut cq = 0.0f32;
        for (chip_idx, &pn_val) in pn.iter().enumerate() {
            let p = base + chip_idx * spc;
            ci += i_f[p] * pn_val;
            cq += q_f[p] * pn_val;
        }
        Some((ci, cq))
    };

    let mut best_phase = 0usize;
    let mut best_delay = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for delay in 0..=128usize {
        for phase in 0..spc {
            if total_symbols == 0 {
                continue;
            }

            let mut total_power = 0.0f32;
            let mut valid = true;
            for sym_idx in 0..total_symbols {
                let Some(cur) = correlate_symbol(sym_idx, phase, delay) else {
                    valid = false;
                    break;
                };
                total_power += cur.0 * cur.0 + cur.1 * cur.1;
            }
            if !valid {
                continue;
            }

            if total_power > best_score {
                best_score = total_power;
                best_phase = phase;
                best_delay = delay;
            }
        }
    }

    let mut payload_bits = Vec::with_capacity(payload_symbols * MODULATION.bits_per_symbol());
    let mut prev = (1.0f32, 0.0f32);
    for sym_idx in 0..payload_symbols {
        let cur = correlate_symbol(sym_idx, best_phase, best_delay)?;
        let diff_re = prev.0 * cur.0 + prev.1 * cur.1;
        let diff_im = prev.0 * cur.1 - prev.1 * cur.0;
        decode_diff_to_bits_mode(diff_re, diff_im, &mut payload_bits);
        let norm = (cur.0 * cur.0 + cur.1 * cur.1).sqrt().max(1e-6);
        prev = (cur.0 / norm, cur.1 / norm);
    }
    payload_bits.truncate(bit_len);
    Some(payload_bits)
}

fn fsk_modulate_bits(bits: &[u8], tx_cfg: &BfskConfig) -> Vec<f32> {
    let tx = BfskModulator::new(tx_cfg.clone());
    tx.modulate_raw_bits(bits)
}

fn fsk_demodulate_bits(samples: &[f32], rx_cfg: &BfskConfig, bit_len: usize) -> Option<Vec<u8>> {
    let rx = BfskDemodulator::new(rx_cfg.clone());
    rx.demodulate_aligned_bits(samples, bit_len)
}

fn run_trial(
    phy: PhyKind,
    bits: &[u8],
    imp: &ChannelImpairment,
    max_sec: f32,
    seed: u64,
) -> TrialResult {
    match phy {
        PhyKind::Dsss => run_trial_dsss(bits, imp, max_sec, seed),
        PhyKind::Fsk => run_trial_fsk(bits, imp, max_sec, seed),
    }
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

fn run_trial_dsss(bits: &[u8], imp: &ChannelImpairment, max_sec: f32, seed: u64) -> TrialResult {
    let tx_cfg = DspConfig::default_48k();
    let mut rx_cfg = tx_cfg.clone();
    rx_cfg.carrier_freq += imp.cfo_hz;

    let mut rng = StdRng::seed_from_u64(seed ^ 0xD55A_0001);
    let mut elapsed_sec = 0.0f32;
    let mut attempts = 0usize;
    let mut bit_errors = 0usize;
    let mut bits_compared = 0usize;
    let mut dropped_attempts = 0usize;
    let mut first_attempt_success = false;
    let mut tx_signal_energy_sum = 0.0f64;
    let mut tx_signal_samples = 0usize;

    loop {
        if elapsed_sec >= max_sec {
            break;
        }

        attempts += 1;
        let tx = dsss_modulate_bits(bits, &tx_cfg);
        tx_signal_energy_sum += signal_energy(&tx);
        tx_signal_samples += tx.len();
        let drop = rng.gen::<f32>() < imp.burst_loss;
        if drop {
            dropped_attempts += 1;
        }
        let rx = apply_channel(&tx, imp, &mut rng, drop);

        elapsed_sec += rx.len() as f32 / tx_cfg.sample_rate;
        let decoded = dsss_demodulate_bits(&rx, &rx_cfg, bits.len());
        let errs = count_bit_errors(bits, decoded.as_deref());
        bit_errors += errs;
        bits_compared += bits.len();

        if errs == 0 {
            if attempts == 1 {
                first_attempt_success = true;
            }
            return TrialResult {
                success: true,
                completion_sec: Some(elapsed_sec),
                elapsed_sec,
                attempts,
                first_attempt_success,
                bit_errors,
                bits_compared,
                dropped_attempts,
                tx_signal_energy_sum,
                tx_signal_samples,
            };
        }
    }

    TrialResult {
        success: false,
        completion_sec: None,
        elapsed_sec,
        attempts,
        first_attempt_success,
        bit_errors,
        bits_compared,
        dropped_attempts,
        tx_signal_energy_sum,
        tx_signal_samples,
    }
}

fn run_trial_fsk(bits: &[u8], imp: &ChannelImpairment, max_sec: f32, seed: u64) -> TrialResult {
    let mut tx_cfg = BfskConfig::default_48k();
    tx_cfg.freq0 += imp.cfo_hz;
    tx_cfg.freq1 += imp.cfo_hz;
    let rx_cfg = BfskConfig::default_48k();

    let mut rng = StdRng::seed_from_u64(seed ^ 0xF5F5_0001);
    let mut elapsed_sec = 0.0f32;
    let mut attempts = 0usize;
    let mut bit_errors = 0usize;
    let mut bits_compared = 0usize;
    let mut dropped_attempts = 0usize;
    let mut first_attempt_success = false;
    let mut tx_signal_energy_sum = 0.0f64;
    let mut tx_signal_samples = 0usize;

    loop {
        if elapsed_sec >= max_sec {
            break;
        }

        attempts += 1;
        let tx = fsk_modulate_bits(bits, &tx_cfg);
        tx_signal_energy_sum += signal_energy(&tx);
        tx_signal_samples += tx.len();
        let drop = rng.gen::<f32>() < imp.burst_loss;
        if drop {
            dropped_attempts += 1;
        }
        let rx = apply_channel(&tx, imp, &mut rng, drop);

        elapsed_sec += rx.len() as f32 / rx_cfg.sample_rate;
        let decoded = fsk_demodulate_bits(&rx, &rx_cfg, bits.len());
        let errs = count_bit_errors(bits, decoded.as_deref());
        bit_errors += errs;
        bits_compared += bits.len();

        if errs == 0 {
            if attempts == 1 {
                first_attempt_success = true;
            }
            return TrialResult {
                success: true,
                completion_sec: Some(elapsed_sec),
                elapsed_sec,
                attempts,
                first_attempt_success,
                bit_errors,
                bits_compared,
                dropped_attempts,
                tx_signal_energy_sum,
                tx_signal_samples,
            };
        }
    }

    TrialResult {
        success: false,
        completion_sec: None,
        elapsed_sec,
        attempts,
        first_attempt_success,
        bit_errors,
        bits_compared,
        dropped_attempts,
        tx_signal_energy_sum,
        tx_signal_samples,
    }
}

fn print_header() {
    println!(
        "scenario,phy,trials,success,deadline_hits,first_attempt_successes,total_bits_compared,total_bit_errors,tx_signal_power,awgn_noise_power,awgn_snr_db,p_complete,p_complete_deadline,deadline_s,ber,per,fer,goodput_effective_bps,goodput_success_mean_bps,p95_complete_s,mean_complete_s,total_attempts,dropped_attempts,multipath"
    );
}

fn fmt_opt(v: Option<f32>) -> String {
    v.map(|x| format!("{x:.6}"))
        .unwrap_or_else(|| "NaN".to_string())
}

fn print_row(scenario: &str, phy: PhyKind, cli: &Cli, imp: &ChannelImpairment, m: &Metrics) {
    let noise_power = if imp.sigma > 0.0 {
        Some(imp.sigma * imp.sigma)
    } else {
        None
    };
    println!(
        "{scenario},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.3},{:.6},{:.6},{:.6},{:.3},{},{},{},{},{},{},{}",
        phy.as_str(),
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
        m.goodput_effective_bps(cli.payload_bits),
        fmt_opt(m.goodput_success_mean_bps(cli.payload_bits)),
        fmt_opt(m.p95_completion_sec()),
        fmt_opt(m.mean_completion_sec()),
        m.total_attempts,
        m.dropped_attempts,
        imp.multipath.name,
    );
}

fn evaluate(phy: PhyKind, cli: &Cli, imp: &ChannelImpairment, scenario: &str) -> Metrics {
    let mut metrics = Metrics::default();
    for trial_idx in 0..cli.trials {
        let trial_seed = cli
            .seed
            .wrapping_add((trial_idx as u64).wrapping_mul(0x9E37_79B9));
        let bits = make_bits(cli.payload_bits, trial_seed ^ 0x1234_5678);
        let tr = run_trial(phy, &bits, imp, cli.max_sec, trial_seed);
        metrics.push(tr, cli.deadline_sec);
    }

    print_row(scenario, phy, cli, imp, &metrics);
    metrics
}

fn run_point(cli: &Cli) {
    print_header();
    let scenario = format!(
        "point(bits={},sigma={:.3},cfo={:.1},ppm={:.1},loss={:.2})",
        cli.payload_bits, cli.base.sigma, cli.base.cfo_hz, cli.base.ppm, cli.base.burst_loss
    );
    evaluate(PhyKind::Dsss, cli, &cli.base, &scenario);
    evaluate(PhyKind::Fsk, cli, &cli.base, &scenario);
}

fn run_sweep_awgn(cli: &Cli) {
    print_header();
    let mut dsss_limit = None;
    let mut fsk_limit = None;

    for &sigma in &cli.sweep_awgn {
        let mut imp = cli.base.clone();
        imp.sigma = sigma;
        let scenario = format!(
            "awgn(bits={},sigma={sigma:.3},cfo={:.1},ppm={:.1},loss={:.2})",
            cli.payload_bits, imp.cfo_hz, imp.ppm, imp.burst_loss
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
    print_header();
    for &cfo in &cli.sweep_cfo {
        let mut imp = cli.base.clone();
        imp.cfo_hz = cfo;
        let scenario = format!(
            "cfo(bits={},cfo={cfo:.1},sigma={:.3},ppm={:.1},loss={:.2})",
            cli.payload_bits, imp.sigma, imp.ppm, imp.burst_loss
        );
        evaluate(PhyKind::Dsss, cli, &imp, &scenario);
        evaluate(PhyKind::Fsk, cli, &imp, &scenario);
    }
}

fn run_sweep_ppm(cli: &Cli) {
    print_header();
    for &ppm in &cli.sweep_ppm {
        let mut imp = cli.base.clone();
        imp.ppm = ppm;
        let scenario = format!(
            "ppm(bits={},ppm={ppm:.1},sigma={:.3},cfo={:.1},loss={:.2})",
            cli.payload_bits, imp.sigma, imp.cfo_hz, imp.burst_loss
        );
        evaluate(PhyKind::Dsss, cli, &imp, &scenario);
        evaluate(PhyKind::Fsk, cli, &imp, &scenario);
    }
}

fn run_sweep_loss(cli: &Cli) {
    print_header();
    for &loss in &cli.sweep_loss {
        let mut imp = cli.base.clone();
        imp.burst_loss = loss.clamp(0.0, 1.0);
        let scenario = format!(
            "loss(bits={},loss={:.2},sigma={:.3},cfo={:.1},ppm={:.1})",
            cli.payload_bits, imp.burst_loss, imp.sigma, imp.cfo_hz, imp.ppm
        );
        evaluate(PhyKind::Dsss, cli, &imp, &scenario);
        evaluate(PhyKind::Fsk, cli, &imp, &scenario);
    }
}

fn run_sweep_multipath(cli: &Cli) {
    print_header();
    for name in ["none", "mild", "medium", "harsh"] {
        let mut imp = cli.base.clone();
        imp.multipath = MultipathProfile::preset(name).unwrap_or_else(MultipathProfile::none);
        let scenario = format!(
            "multipath(bits={},profile={name},sigma={:.3},cfo={:.1},ppm={:.1},loss={:.2})",
            cli.payload_bits, imp.sigma, imp.cfo_hz, imp.ppm, imp.burst_loss
        );
        evaluate(PhyKind::Dsss, cli, &imp, &scenario);
        evaluate(PhyKind::Fsk, cli, &imp, &scenario);
    }
}

fn print_help() {
    println!(
        "usage: cargo run --release --bin phy_eval -- [options]\n\
         modes:\n\
         - --mode point             単点評価 (default)\n\
         - --mode sweep-awgn        AWGN sweep + awgn_limit推定\n\
         - --mode sweep-cfo         CFO sweep\n\
         - --mode sweep-ppm         ppm sweep\n\
         - --mode sweep-loss        バースト欠落率 sweep\n\
         - --mode sweep-multipath   マルチパス profile sweep\n\
         - --mode sweep-all         全sweep\n\n\
         common options:\n\
         --trials N\n\
         --payload-bits N           送信ビット長 (default: 256)\n\
         --deadline-sec F\n\
         --max-sec F\n\
         --seed N\n\
         --sigma F\n\
         --cfo-hz F\n\
         --ppm F\n\
         --burst-loss F             (0..1)\n\
         --multipath NAME_OR_TAPS\n\
             NAME: none|mild|medium|harsh\n\
             TAPS: \"0:1.0,9:0.4,23:0.2\"\n\
         --target-p F               (awgn_limit判定用)\n\n\
         sweep lists:\n\
         --sweep-awgn \"0,0.005,0.01\"\n\
         --sweep-cfo  \"0,2,5,10\"\n\
         --sweep-ppm  \"-120,-80,0,80,120\"\n\
         --sweep-loss \"0,0.1,0.2,0.3\"\n"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_bit_errors_none_is_all_error() {
        let bits = vec![0, 1, 0, 1, 1, 0];
        assert_eq!(count_bit_errors(&bits, None), bits.len());
    }

    #[test]
    fn test_count_bit_errors_partial_len() {
        let tx = vec![0, 1, 0, 1, 1, 0];
        let rx = vec![0, 1, 1];
        // idx2 mismatch + idx3..5 missing扱いでerror
        assert_eq!(count_bit_errors(&tx, Some(&rx)), 4);
    }

    #[test]
    fn test_apply_clock_drift_ppm_expand_and_shrink() {
        let input = vec![1.0f32; 1000];
        let expanded = apply_clock_drift_ppm(&input, 1000.0);
        let shrunken = apply_clock_drift_ppm(&input, -1000.0);
        assert!(expanded.len() > input.len());
        assert!(shrunken.len() < input.len());
    }
}
