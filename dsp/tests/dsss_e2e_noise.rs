use dsp::common::channel::add_awgn;
use dsp::dsss::decoder::Decoder;
use dsp::dsss::encoder::{Encoder, EncoderConfig};
use dsp::dsss::params;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::time::{Duration, Instant};

const QUICK_NO_NOISE_BUDGET: Duration = Duration::from_secs(4);
const QUICK_MARGIN_BUDGET: Duration = Duration::from_secs(7);
const MARGIN_SIGMA: f32 = 0.025;
const QUICK_MAX_FRAMES: usize = 10;
const QUICK_GAP_SAMPLES: usize = 64;
const QUICK_CHUNK_SAMPLES: usize = 16384;

/// 指定されたシグマ（ノイズ強度）で E2E 通信テストを行う。
/// 送信フレームを逐次デコーダへ流し、復号完了時点で即終了する。
fn test_transmission_quick(sigma: f32, seed: u64, sample_rate: f32) -> bool {
    let data = b"E2E quick test payload";
    let lt_k = 4usize;
    let dsp_config = params::dsp_config(sample_rate);
    let mut enc_config = EncoderConfig::new(dsp_config.clone());
    enc_config.fountain_k = lt_k;
    let mut encoder = Encoder::new(enc_config);
    let mut stream = encoder.encode_stream(data);
    let mut decoder = Decoder::new(data.len(), lt_k, dsp_config);
    let mut rng = StdRng::seed_from_u64(seed);
    let gap = vec![0.0f32; QUICK_GAP_SAMPLES];

    // フレームを逐次処理し、復号できた時点で即終了する。
    for _ in 0..QUICK_MAX_FRAMES {
        let Some(mut frame) = stream.next() else {
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

        for chunk in gap.chunks(QUICK_CHUNK_SAMPLES) {
            let progress = decoder.process_samples(chunk);
            if progress.complete {
                if let Some(recovered) = decoder.recovered_data() {
                    return &recovered[..data.len()] == data;
                }
                return false;
            }
        }
    }

    false
}

fn assert_quick_awgn_case(sigma: f32, seed: u64, time_budget: Duration, sample_rate: f32) {
    let start = Instant::now();
    assert!(
        test_transmission_quick(sigma, seed, sample_rate),
        "Sigma={sigma} / seed={seed:#x} / rate={sample_rate} で復号できませんでした"
    );
    assert!(
        start.elapsed() < time_budget,
        "e2e quick test is too slow (sigma={sigma}, rate={sample_rate}): {:?}",
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
fn test_awgn_e2e_quick_no_noise_48k() {
    assert_quick_awgn_case(0.0, 0xC0FFEE, QUICK_NO_NOISE_BUDGET, 48000.0);
}

#[test]
fn test_awgn_e2e_quick_no_noise_44k() {
    assert_quick_awgn_case(0.0, 0xC0FFEE, QUICK_NO_NOISE_BUDGET, 44100.0);
}

#[test]
fn test_awgn_e2e_quick_margin_noise_48k() {
    // 常時実行の境界テスト。複数seedで通る最大帯に近い sigma を維持する。
    let seeds = [0xBAD5EED, 0xC0FFEE];
    let start = Instant::now();
    let success = success_count_for_sigma(MARGIN_SIGMA, &seeds, 48000.0);
    assert_eq!(
        success,
        seeds.len(),
        "Sigma={MARGIN_SIGMA:.3} は 48k 境界テストとして弱すぎるか強すぎます。"
    );
    assert!(
        start.elapsed() < QUICK_MARGIN_BUDGET,
        "margin e2e test is too slow: {:?}",
        start.elapsed()
    );
}

#[test]
fn test_awgn_e2e_quick_margin_noise_44k() {
    let seeds = [0xBAD5EED, 0xC0FFEE];
    let success = success_count_for_sigma(MARGIN_SIGMA, &seeds, 44100.0);
    assert_eq!(
        success,
        seeds.len(),
        "Sigma={MARGIN_SIGMA:.3} は 44.1k 境界テストとして弱すぎるか強すぎます。"
    );
}

#[test]
#[ignore = "境界探索用: sigma上限の再調整時のみ実行"]
fn test_awgn_e2e_sigma_margin_sweep() {
    let seeds = [0xBAD5EED, 0xC0FFEE, 0xA11CE, 0x5EED1234, 0xDEADBEEF];
    let candidates = [0.01f32, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045];
    let mut best_sigma = 0.0f32;

    println!("\n--- AWGN sigma margin sweep ---");
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
    println!("--------------------------------\n");
    assert!(
        best_sigma > 0.0,
        "どの候補 sigma でも全seed成功しませんでした。復調性能の劣化が疑われます。"
    );
}
