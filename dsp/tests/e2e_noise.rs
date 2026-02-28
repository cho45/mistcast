use dsp::decoder::Decoder;
use dsp::encoder::{Encoder, EncoderConfig};
use dsp::DspConfig;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::Normal;
use std::time::{Duration, Instant};

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

/// 指定されたシグマ（ノイズ強度）で E2E 通信テストを行う。
/// 送信フレームを逐次デコーダへ流し、復号完了時点で即終了する。
fn test_transmission_quick(sigma: f32, seed: u64) -> bool {
    let data = b"E2E quick test payload";
    let lt_k = 4usize;
    let dsp_config = DspConfig::default_48k();
    let mut enc_config = EncoderConfig::new(dsp_config.clone());
    enc_config.lt_k = lt_k;
    let mut encoder = Encoder::new(enc_config);
    let mut stream = encoder.encode_stream(data);
    let mut decoder = Decoder::new(data.len(), lt_k, dsp_config.clone());
    let mut rng = StdRng::seed_from_u64(seed);
    let gap = vec![0.0f32; 128];
    let mut tx_signal = Vec::new();

    // K=4 に対して十分な冗長を持たせる。
    for _ in 0..8 {
        let Some(mut frame) = stream.next() else {
            break;
        };
        add_awgn(&mut frame, sigma, &mut rng);
        tx_signal.extend_from_slice(&frame);
        tx_signal.extend_from_slice(&gap);
    }

    for chunk in tx_signal.chunks(4096) {
        let progress = decoder.process_samples(chunk);
        if progress.complete {
            if let Some(recovered) = decoder.recovered_data() {
                return &recovered[..data.len()] == data;
            }
            return false;
        }
    }

    false
}

#[test]
fn test_awgn_e2e_quick() {
    let start = Instant::now();

    assert!(
        test_transmission_quick(0.0, 0xC0FFEE),
        "Sigma=0.0 で復号できませんでした"
    );

    // デバッグビルドでも短時間で終わることを保証する。
    assert!(
        start.elapsed() < Duration::from_secs(8),
        "e2e quick test is too slow: {:?}",
        start.elapsed()
    );
}
