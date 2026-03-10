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
            // 48kHz 기준:
            // 4,11,19 samples = 0.083,0.229,0.396ms (約2.9,7.9,13.6cm path diff)
            // デバイス近傍・机面の短遅延反射を想定した高域コムノッチ向け。
            "desk" => Some(Self {
                name: "desk".to_string(),
                taps: vec![(0, 1.0), (4, -0.45), (11, 0.28), (19, -0.18)],
            }),
            // 48kHz 기준:
            // 42,87,154,233 samples = 0.875,1.813,3.208,4.854ms
            // (約0.30,0.62,1.10,1.66m path diff)
            // 家庭内の中距離反射を想定。
            "room" => Some(Self {
                name: "room".to_string(),
                taps: vec![(0, 1.0), (42, 0.52), (87, -0.34), (154, 0.22), (233, -0.15)],
            }),
            // 48kHz 기준:
            // 70,138,236,356 samples = 1.458,2.875,4.917,7.417ms
            // (約0.50,0.99,1.69,2.54m path diff)
            // 廊下/ホールの長遅延反射を想定。
            "hall" => Some(Self {
                name: "hall".to_string(),
                taps: vec![(0, 1.0), (70, 0.55), (138, -0.40), (236, 0.30), (356, -0.22)],
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
                "invalid multipath: {value} (preset: none|mild|medium|harsh|desk|room|hall or taps: 0:1.0,9:0.4)"
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

#[inline]
fn apply_multipath_into(input: &[f32], taps: &[(usize, f32)], out: &mut Vec<f32>) {
    if taps.is_empty() || (taps.len() == 1 && taps[0].0 == 0 && (taps[0].1 - 1.0).abs() < 1e-6) {
        out.clear();
        out.extend_from_slice(input);
        return;
    }

    let max_delay = taps.iter().map(|(d, _)| *d).max().unwrap_or(0);
    let out_len = input.len() + max_delay;
    out.clear();
    out.resize(out_len, 0.0);

    for &(delay, gain) in taps {
        for (i, &x) in input.iter().enumerate() {
            out[i + delay] += gain * x;
        }
    }

    let norm = taps
        .iter()
        .map(|(_, g)| g * g)
        .sum::<f32>()
        .sqrt()
        .max(1e-6);
    for s in out.iter_mut() {
        *s /= norm;
    }
}

#[inline]
fn apply_clock_drift_ppm_into(input: &[f32], ppm: f32, out: &mut Vec<f32>) {
    if input.is_empty() || ppm.abs() < 0.1 {
        out.clear();
        out.extend_from_slice(input);
        return;
    }

    let ratio = 1.0 + ppm / 1_000_000.0;
    let out_len = (input.len() as f32 / ratio).floor() as usize;
    out.clear();
    out.reserve(out_len);

    for i in 0..out_len {
        let pos = i as f32 * ratio;
        let i0 = pos.floor() as usize;
        let frac = pos - i0 as f32;
        if i0 + 1 < input.len() {
            let a = input[i0];
            let b = input[i0 + 1];
            out.push(a + (b - a) * frac);
        } else if i0 < input.len() {
            out.push(input[i0]);
        } else {
            break;
        }
    }
}

pub fn apply_channel_into(
    tx: &[f32],
    imp: &ChannelImpairment,
    rng: &mut StdRng,
    drop_frame: bool,
    out: &mut Vec<f32>,
    scratch: &mut Vec<f32>,
) {
    if drop_frame {
        out.clear();
        out.resize(tx.len(), 0.0);
    } else {
        apply_multipath_into(tx, &imp.multipath.taps, out);
    }

    if imp.ppm.abs() >= 0.1 {
        apply_clock_drift_ppm_into(out, imp.ppm, scratch);
        std::mem::swap(out, scratch);
    }

    channel::apply_fading(out, imp.fading_depth, rng);
    channel::add_awgn(out, imp.sigma, rng);
}

#[cfg(test)]
mod tests {
    use super::MultipathProfile;

    #[test]
    fn test_realistic_presets_exist_and_are_sorted() {
        for &name in &["desk", "room", "hall"] {
            let p = MultipathProfile::preset(name).expect("preset must exist");
            assert_eq!(p.name, name);
            assert_eq!(p.taps.first().copied(), Some((0, 1.0)));
            assert!(p.taps.len() >= 4, "preset {} should have enough taps", name);
            assert!(
                p.taps.windows(2).all(|w| w[0].0 < w[1].0),
                "preset {} taps must be strictly sorted by delay",
                name
            );
        }
    }

    #[test]
    fn test_unknown_profile_error_mentions_realistic_presets() {
        let err = "not-a-profile".parse::<MultipathProfile>().unwrap_err();
        assert!(err.contains("desk|room|hall"), "error={}", err);
    }
}
