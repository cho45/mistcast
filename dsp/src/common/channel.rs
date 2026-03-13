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

/// 信号のアクティブ区間の平均パワーを算出する。
/// 0.01（実信号の場合は0.001）以下のエネルギーのサンプルは無音（ガードインターバル等）として無視する。
/// 複素信号の場合は I^2 + Q^2、実信号の場合は s^2 を電力とする。
pub fn estimate_signal_power(samples: &[f32]) -> f32 {
    let mut energy_sum = 0.0f32;
    let mut count = 0;
    for &s in samples {
        let en = s * s;
        if en > 0.001 {
            // real-valued samples なので少し閾値を下げる
            energy_sum += en;
            count += 1;
        }
    }
    if count > 0 {
        energy_sum / count as f32
    } else {
        0.0
    }
}

/// 指定されたSNR(dB)に基づき、実信号サンプル列にAWGNを付与する。
/// ここでのSNRは、サンプリング帯域内の信号電力(E[s^2])対雑音電力(sigma^2)の比として定義される。
pub fn add_awgn_snr(samples: &mut [f32], snr_db: f32, rng: &mut StdRng) {
    let p_sig = estimate_signal_power(samples);
    if p_sig <= 0.0 {
        return;
    }
    let snr_linear = 10.0f32.powf(snr_db / 10.0);
    let p_noise = p_sig / snr_linear;
    let sigma = p_noise.sqrt();
    add_awgn(samples, sigma, rng);
}

/// 指定されたSNR(dB)に基づき、複素信号（I/Q分離）サンプル列にAWGNを付与する。
/// ここでのSNRは、複素信号電力(E[I^2 + Q^2])対雑音電力(E[Ni^2 + Nq^2])の比として定義される。
/// 雑音電力はIとQの各パスに均等に分配される。
pub fn add_awgn_snr_iq(
    i_samples: &mut [f32],
    q_samples: &mut [f32],
    snr_db: f32,
    rng: &mut StdRng,
) {
    let len = i_samples.len().min(q_samples.len());
    let mut energy_sum = 0.0f32;
    let mut count = 0;
    for k in 0..len {
        let i = i_samples[k];
        let q = q_samples[k];
        let en = i * i + q * q;
        if en > 0.01 {
            energy_sum += en;
            count += 1;
        }
    }
    let p_sig = if count > 0 {
        energy_sum / count as f32
    } else {
        0.0
    };
    if p_sig <= 0.0 {
        return;
    }

    let snr_linear = 10.0f32.powf(snr_db / 10.0);
    let p_noise = p_sig / snr_linear;
    // IとQにノイズパワーを等分配するため、それぞれの sigma は sqrt(p_noise / 2)
    let sigma = (p_noise / 2.0).sqrt();

    add_awgn(&mut i_samples[..len], sigma, rng);
    add_awgn(&mut q_samples[..len], sigma, rng);
}

/// Eb/N0 (ビットエネルギー対雑音電力密度比) [dB] を内部的な SNR [dB] に変換する。
/// ここでの内部SNRは、指定された帯域幅 (bandwidth) 全体での信号電力対雑音電力比。
/// 離散時間信号において雑音電力をサンプリングレート fs 全域で定義する場合、bandwidth には fs を指定する。
/// 重要: `bandwidth` は「その SNR が定義されている帯域 B」を指定すること。
/// 例:
/// - 波形電力ベースSNRなら `bandwidth = sample_rate`
/// - チップ領域/相関器内部で定義したSNRなら `bandwidth = chip_rate`（または等価帯域）
/// rb は情報ビットレート (bps)。
pub fn ebn0_db_to_snr_db(ebn0_db: f32, bandwidth: f32, rb: f32) -> f32 {
    // SNR = (Eb/N0) * (Rb/B)
    ebn0_db + 10.0 * (rb / bandwidth).log10()
}

/// 指定された帯域幅 (bandwidth) における SNR [dB] を Eb/N0 [dB] に変換する。
/// 重要: `bandwidth` は `snr_db` を定義したときの帯域 B と一致させること。
/// rb は情報ビットレート (bps)。
pub fn snr_db_to_ebn0_db(snr_db: f32, bandwidth: f32, rb: f32) -> f32 {
    // Eb/N0 = SNR * (B/Rb)
    snr_db + 10.0 * (bandwidth / rb).log10()
}

/// 指定された帯域幅 (bandwidth) における SNR [dB] を C/N0 [dB-Hz] に変換する。
///
/// 重要: `bandwidth` は `snr_db` を定義したときの帯域 B と一致させること。
pub fn snr_db_to_cn0_db(snr_db: f32, bandwidth: f32) -> f32 {
    snr_db + 10.0 * bandwidth.log10()
}

/// 指定された帯域幅 (bandwidth) における C/N0 [dB-Hz] を SNR [dB] に変換する。
pub fn cn0_db_to_snr_db(cn0_db: f32, bandwidth: f32) -> f32 {
    cn0_db - 10.0 * bandwidth.log10()
}
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_add_awgn_statistics() {
        let mut rng = StdRng::seed_from_u64(42);
        let sigma = 0.5;
        let mut samples = vec![0.0f32; 10000];
        add_awgn(&mut samples, sigma, &mut rng);

        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let variance =
            samples.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32;

        // 統計的なばらつきを考慮し、余裕を持った範囲で検証
        assert!(mean.abs() < 0.05, "Mean should be near 0, got {}", mean);
        assert!(
            (variance - sigma.powi(2)).abs() < 0.05,
            "Variance should be near sigma^2, got {}",
            variance
        );
    }

    #[test]
    fn test_apply_clock_drift_ppm_values() {
        // PPM=0: 変化なし
        let input = vec![1.0, 2.0, 3.0];
        assert_eq!(apply_clock_drift_ppm(&input, 0.0), input);

        // Positive PPM (信号が伸びる/低速リサンプリング)
        // 100,000 ppm = 10% drift (ratio = 1.1)
        // 5 samples / 1.1 = 4.54 -> 4 samples
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let drifted = apply_clock_drift_ppm(&input, 100_000.0);
        assert_eq!(drifted.len(), 4);
        assert!((drifted[0] - 1.0).abs() < 1e-6); // pos=0.0
        assert!((drifted[1] - 2.1).abs() < 1e-6); // pos=1.1: 2.0 + (3.0-2.0)*0.1

        // Negative PPM (信号が縮む/高速リサンプリング)
        // -100,000 ppm = ratio = 0.9
        // 5 samples / 0.9 = 5.55 -> 5 samples
        let drifted = apply_clock_drift_ppm(&input, -100_000.0);
        assert_eq!(drifted.len(), 5);
        assert!((drifted[1] - 1.9).abs() < 1e-6); // pos=0.9: 1.0 + (2.0-1.0)*0.9
    }

    #[test]
    fn test_apply_fading_range() {
        let mut rng = StdRng::seed_from_u64(123);
        let depth = 0.5;
        let mut samples = vec![1.0f32; 1000];
        apply_fading(&mut samples, depth, &mut rng);

        for &s in &samples {
            // 元が 1.0 なので、結果は 1.0 - depth (0.5) から 1.0 の範囲にあるはず
            assert!((0.5..=1.0).contains(&s), "Fading value out of range: {}", s);
        }
    }

    #[test]
    fn test_apply_multipath_behavior() {
        let input = vec![1.0, 0.0, 0.0];
        // 2つの等しいタップ: (0, 1.0), (2, 1.0)
        // norm = sqrt(1^2 + 1^2) = sqrt(2)
        let taps = vec![(0, 1.0), (2, 1.0)];
        let out = apply_multipath(&input, &taps);

        assert_eq!(out.len(), 5); // 3 + 2 delay
        let val = 1.0 / 2.0f32.sqrt();
        assert!((out[0] - val).abs() < 1e-6); // 1st tap at t=0
        assert!((out[1] - 0.0).abs() < 1e-6);
        assert!((out[2] - val).abs() < 1e-6); // 2nd tap at t=2
    }

    #[test]
    fn test_estimate_signal_power() {
        // 純粋なDC
        assert!((estimate_signal_power(&[1.0, 1.0, -1.0, -1.0]) - 1.0).abs() < 1e-6);

        // バースト信号（無音区間を含む）
        let samples = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        // 0.0 は 0.001 以下なので除外されるはず。power は 1.0 になるべき。
        assert!((estimate_signal_power(&samples) - 1.0).abs() < 1e-6);

        // 全て無音
        assert_eq!(estimate_signal_power(&[0.0, 0.0001]), 0.0);
    }

    #[test]
    fn test_add_awgn_snr_real() {
        let mut rng = StdRng::seed_from_u64(777);
        let snr_db = 10.0; // Power ratio = 10
        let mut samples = vec![1.0f32; 10000]; // Power = 1.0
        add_awgn_snr(&mut samples, snr_db, &mut rng);

        // Result = Signal(1.0) + Noise
        // Noise Power should be SignalPower / 10 = 0.1
        // sigma should be sqrt(0.1) = 0.316
        let noise_variance =
            samples.iter().map(|&x| (x - 1.0).powi(2)).sum::<f32>() / samples.len() as f32;
        assert!(
            (noise_variance - 0.1).abs() < 0.02,
            "Measured noise power {} should be near 0.1",
            noise_variance
        );
    }

    #[test]
    fn test_add_awgn_snr_iq() {
        let mut rng = StdRng::seed_from_u64(888);
        let snr_db = 20.0; // Power ratio = 100
        let mut i_samples = vec![1.0f32; 10000];
        let mut q_samples = vec![0.0f32; 10000]; // Complex Power = 1^2 + 0^2 = 1.0

        add_awgn_snr_iq(&mut i_samples, &mut q_samples, snr_db, &mut rng);

        // Total Noise Power = SignalPower / 100 = 0.01
        // Measured variance sum (I + Q) should be 0.01
        let i_var =
            i_samples.iter().map(|&x| (x - 1.0).powi(2)).sum::<f32>() / i_samples.len() as f32;
        let q_var =
            q_samples.iter().map(|&x| (x - 0.0).powi(2)).sum::<f32>() / q_samples.len() as f32;
        let total_noise = i_var + q_var;

        assert!(
            (total_noise - 0.01).abs() < 0.002,
            "Total noise power {} should be near 0.01",
            total_noise
        );
        assert!(
            (i_var - q_var).abs() < 0.002,
            "Noise should be equally distributed between I and Q"
        );
    }

    #[test]
    fn test_ebn0_snr_conversion() {
        let bandwidth = 48000.0;
        let rb = 1200.0; // 1/40 bit rate
                         // bandwidth/rb = 40. 10*log10(40) = 16.02 dB

        let ebn0 = 10.0;
        let snr = ebn0_db_to_snr_db(ebn0, bandwidth, rb);
        // snr = 10.0 + 10*log10(1200/48000) = 10.0 - 16.02 = -6.02 dB
        assert!((snr - (-6.0206)).abs() < 0.01);

        let ebn0_back = snr_db_to_ebn0_db(snr, bandwidth, rb);
        assert!((ebn0_back - ebn0).abs() < 1e-5);
    }

    #[test]
    fn test_cn0_snr_conversion() {
        let bandwidth = 48_000.0f32;
        let snr_db = -6.0206f32;
        let cn0_db = snr_db_to_cn0_db(snr_db, bandwidth);
        let snr_back = cn0_db_to_snr_db(cn0_db, bandwidth);
        assert!((snr_back - snr_db).abs() < 1e-5);
    }
}
