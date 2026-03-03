use dsp::mary::decoder::Decoder;
use dsp::mary::encoder::Encoder;
use dsp::DspConfig;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::Normal;
use std::time::{Duration, Instant};

const QUICK_NO_NOISE_BUDGET: Duration = Duration::from_secs(4);
const QUICK_MARGIN_BUDGET: Duration = Duration::from_secs(7);
const MARGIN_SIGMA: f32 = 0.025;
const QUICK_MAX_PACKETS: usize = 20;
const QUICK_CHUNK_SAMPLES: usize = 16384;

/// AWGN (加法性ホワイトガウスノイズ) を付与する
fn add_awgn<R: Rng + ?Sized>(samples: &mut [f32], sigma: f32, rng: &mut R) {
    if sigma <= 0.0 {
        return;
    }
    let normal = Normal::new(0.0, sigma).unwrap();
    for s in samples.iter_mut() {
        *s += normal.sample(rng);
    }
}

/// 指定されたシグマ（ノイズ強度）で E2E 通信テストを行う (Mary版)。
fn test_transmission_quick(sigma: f32, seed: u64, sample_rate: f32) -> bool {
    let data = b"Mary E2E noise test payload";
    let lt_k = 10usize;
    let dsp_config = DspConfig::new(sample_rate);
    let mut encoder = Encoder::new(dsp_config.clone());
    encoder.set_data(data);

    let mut decoder = Decoder::new(data.len(), lt_k, dsp_config.clone());
    let mut rng = StdRng::seed_from_u64(seed);

    // Mary版はバースト全体ではなく、1パケット(シンボル)ずつ、または一括で流す
    // ここでは1パケットずつエンコードして、ノイズを乗せてデコーダへ
    for _ in 0..QUICK_MAX_PACKETS {
        let Some(mut frame) = encoder.encode_frame() else {
            break;
        };
        add_awgn(&mut frame, sigma, &mut rng);

        for chunk in frame.chunks(QUICK_CHUNK_SAMPLES) {
            let progress = decoder.process_samples(chunk);
            if progress.complete {
                if let Some(recovered) = decoder.recovered_data() {
                    return &recovered[..data.len()] == data;
                }
                return false;
            }
        }
    }

    // Flush分
    let mut flush_samples = encoder.flush();
    add_awgn(&mut flush_samples, sigma, &mut rng);
    let progress = decoder.process_samples(&flush_samples);
    if progress.complete {
        if let Some(recovered) = decoder.recovered_data() {
            return &recovered[..data.len()] == data;
        }
    }

    false
}

fn assert_quick_awgn_case(sigma: f32, seed: u64, time_budget: Duration, sample_rate: f32) {
    let start = Instant::now();
    assert!(
        test_transmission_quick(sigma, seed, sample_rate),
        "Sigma={sigma} / seed={seed:#x} / rate={sample_rate} で復号できませんでした (Mary)"
    );
    assert!(
        start.elapsed() < time_budget,
        "mary e2e quick test is too slow (sigma={sigma}, rate={sample_rate}): {:?}",
        start.elapsed()
    );
}

fn success_count_for_sigma(sigma: f32, seeds: &[u64], sample_rate: f32) -> usize {
    seeds
        .iter()
        .copied()
        .filter(|&seed| test_transmission_quick(sigma, seed, sample_rate))
        .count()
}

#[test]
fn test_mary_awgn_e2e_quick_no_noise_48k() {
    assert_quick_awgn_case(0.0, 0xC0FFEE, QUICK_NO_NOISE_BUDGET, 48000.0);
}

#[test]
fn test_mary_awgn_e2e_quick_no_noise_44k() {
    assert_quick_awgn_case(0.0, 0xC0FFEE, QUICK_NO_NOISE_BUDGET, 44100.0);
}

#[test]
fn test_mary_awgn_e2e_quick_margin_noise_48k() {
    let seeds = [0xBAD5EED, 0xC0FFEE];
    let start = Instant::now();
    let success = success_count_for_sigma(MARGIN_SIGMA, &seeds, 48000.0);
    assert_eq!(
        success,
        seeds.len(),
        "Sigma={MARGIN_SIGMA:.3} は 48k Mary境界テストとして弱すぎるか強すぎます。"
    );
    assert!(
        start.elapsed() < QUICK_MARGIN_BUDGET,
        "mary margin e2e test is too slow: {:?}",
        start.elapsed()
    );
}

#[test]
fn test_mary_awgn_e2e_quick_margin_noise_44k() {
    let seeds = [0xBAD5EED, 0xC0FFEE];
    let success = success_count_for_sigma(MARGIN_SIGMA, &seeds, 44100.0);
    assert_eq!(
        success,
        seeds.len(),
        "Sigma={MARGIN_SIGMA:.3} は 44.1k Mary境界テストとして弱すぎるか強すぎます。"
    );
}

#[test]
#[ignore = "境界探索用: sigma上限の再調整時のみ実行"]
fn test_mary_awgn_e2e_sigma_margin_sweep() {
    let seeds = [0xBAD5EED, 0xC0FFEE, 0xA11CE, 0x5EED1234, 0xDEADBEEF];
    let candidates = [0.01f32, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045];
    let mut best_sigma = 0.0f32;

    println!(
        "
--- Mary AWGN sigma margin sweep ---"
    );
    for sigma in candidates {
        let start = Instant::now();
        let success = success_count_for_sigma(sigma, &seeds, 48000.0);
        let elapsed = start.elapsed();
        println!(
            "sigma={sigma:.3} success={success}/{} elapsed={elapsed:?}",
            seeds.len()
        );
        if success == seeds.len() {
            best_sigma = sigma;
        } else {
            break;
        }
    }
    println!("recommended always-on margin sigma: {best_sigma:.3}");
    println!("configured MARGIN_SIGMA: {MARGIN_SIGMA:.3}");
    println!(
        "--------------------------------
"
    );
    assert!(
        best_sigma > 0.0,
        "どの候補 sigma でも全seed成功しませんでした。Mary復調性能の劣化が疑われます。"
    );
}
