use dsp::common::channel;
use rand::rngs::StdRng;
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
    pub frame_loss: f32,
    pub fading_depth: f32,
    pub multipath: MultipathProfile,
}

pub fn apply_channel(
    tx: &[f32],
    imp: &ChannelImpairment,
    rng: &mut StdRng,
    drop_frame: bool,
) -> Vec<f32> {
    let mut sig = if drop_frame {
        vec![0.0f32; tx.len()]
    } else {
        channel::apply_multipath(tx, &imp.multipath.taps)
    };
    if imp.ppm.abs() >= 0.1 {
        sig = channel::apply_clock_drift_ppm(&sig, imp.ppm);
    }
    channel::apply_fading(&mut sig, imp.fading_depth, rng);
    channel::add_awgn(&mut sig, imp.sigma, rng);
    sig
}
