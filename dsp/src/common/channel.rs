use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// 加法的白色ガウスノイズ(AWGN)を付与する
pub fn add_awgn(samples: &mut [f32], sigma: f32, rng: &mut StdRng) {
    if sigma <= 0.0 {
        return;
    }
    let normal = Normal::new(0.0, sigma).expect("invalid sigma for AWGN");
    for s in samples {
        *s += normal.sample(rng);
    }
}

/// クロックドリフト(PPM)を模擬するための線形補完によるリサンプリング
pub fn apply_clock_drift_ppm(input: &[f32], ppm: f32) -> Vec<f32> {
    if input.is_empty() || ppm.abs() < 0.1 {
        return input.to_vec();
    }

    // ratio > 1.0 (ppm > 0): 送信クロックが速い -> サンプル間隔が広がる -> 信号が伸びる
    // ratio < 1.0 (ppm < 0): 送信クロックが遅い -> サンプル間隔が狭まる -> 信号が縮む
    let ratio = 1.0 + ppm / 1_000_000.0;
    let out_len = (input.len() as f32 / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);

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
    out
}

/// 単純な線形フェージング（振幅変動）を模擬する
pub fn apply_fading(sig: &mut [f32], depth: f32, rng: &mut StdRng) {
    if sig.is_empty() || depth <= 0.0 {
        return;
    }
    let d = depth.clamp(0.0, 1.0);
    // サンプルの開始と終了でランダムな利得を生成し、その間を線形補完する
    let g0 = 1.0 - d * rng.gen::<f32>();
    let g1 = 1.0 - d * rng.gen::<f32>();
    let n = sig.len().max(2);
    for (i, s) in sig.iter_mut().enumerate() {
        let t = i as f32 / (n - 1) as f32;
        let g = g0 + (g1 - g0) * t;
        *s *= g;
    }
}

/// マルチパス・フェージング（タップ遅延線モデル）を模擬する
pub fn apply_multipath(input: &[f32], taps: &[(usize, f32)]) -> Vec<f32> {
    if taps.is_empty() {
        return input.to_vec();
    }
    if taps.len() == 1 && taps[0].0 == 0 && (taps[0].1 - 1.0).abs() < 1e-6 {
        return input.to_vec();
    }

    let max_delay = taps.iter().map(|(d, _)| *d).max().unwrap_or(0);
    let mut out = vec![0.0f32; input.len() + max_delay];

    for &(delay, gain) in taps {
        for (i, &x) in input.iter().enumerate() {
            out[i + delay] += gain * x;
        }
    }

    // エネルギー正規化（ゲインの2乗和の平方根で割る）
    let norm = taps
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_clock_drift_ppm_expansion() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // 100,000 ppm = 10% drift (ratio = 1.1)
        // out_len = floor(5 / 1.1) = 4
        let drifted = apply_clock_drift_ppm(&input, 100_000.0);
        assert_eq!(drifted.len(), 4);
        // pos=0: i0=0, frac=0 -> input[0]=1.0
        // pos=1.1: i0=1, frac=0.1 -> 2.0 + (3.0-2.0)*0.1 = 2.1
        assert!((drifted[0] - 1.0).abs() < 1e-6);
        assert!((drifted[1] - 2.1).abs() < 1e-6);
    }

    #[test]
    fn test_apply_multipath_normalization() {
        let input = vec![1.0; 100];
        let taps = vec![(0, 1.0), (10, 1.0)]; // Two equal taps
        let output = apply_multipath(&input, &taps);

        // norm = sqrt(1^2 + 1^2) = sqrt(2)
        // gain per tap = 1 / sqrt(2) = 0.707
        // Mid-section value should be 1*0.707 + 1*0.707 = 1.414
        let mid_val = output[50];
        assert!((mid_val - 2.0f32.sqrt()).abs() < 1e-5);
    }
}
