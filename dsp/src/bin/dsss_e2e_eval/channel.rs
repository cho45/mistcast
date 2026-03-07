use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::str::FromStr;

#[derive(Clone, Debug)]
pub struct MultipathProfile {
    pub name: String,
    pub taps: Vec<(usize, f32)>,
}

impl MultipathProfile {
    pub fn none() -> Self {
        Self {
            name: "none".to_string(),
            taps: vec![(0, 1.0)],
        }
    }

    pub fn preset(name: &str) -> Option<Self> {
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

    pub fn parse_custom(spec: &str) -> Option<Self> {
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

    pub fn max_delay(&self) -> usize {
        self.taps.iter().map(|(d, _)| *d).max().unwrap_or(0)
    }
}

impl FromStr for MultipathProfile {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let normalized = value.trim().to_ascii_lowercase();
        if let Some(preset) = Self::preset(&normalized) {
            return Ok(preset);
        }
        Self::parse_custom(value).ok_or_else(|| {
            format!(
                "invalid multipath: {value} (preset: none|mild|medium|harsh or taps: 0:1.0,9:0.4)"
            )
        })
    }
}

#[derive(Clone, Debug)]
pub struct ChannelImpairment {
    pub sigma: f32,
    pub cfo_hz: f32,
    pub ppm: f32,
    pub burst_loss: f32,
    pub fading_depth: f32,
    pub multipath: MultipathProfile,
}

pub fn add_awgn_with_rng(samples: &mut [f32], sigma: f32, rng: &mut StdRng) {
    if sigma <= 0.0 {
        return;
    }
    let normal = Normal::new(0.0, sigma).expect("normal distribution");
    for s in samples {
        *s += normal.sample(rng);
    }
}

pub fn apply_multipath(input: &[f32], mp: &MultipathProfile) -> Vec<f32> {
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

pub fn apply_clock_drift_ppm(input: &[f32], ppm: f32) -> Vec<f32> {
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

pub fn apply_fading(sig: &mut [f32], depth: f32, rng: &mut StdRng) {
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

pub fn apply_channel(
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
