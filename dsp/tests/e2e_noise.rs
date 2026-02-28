use dsp::encoder::{Encoder, EncoderConfig};
use dsp::decoder::Decoder;
use dsp::DspConfig;
use rand::prelude::*;
use rand_distr::Normal;

/// AWGN (加法性ホワイトガウスノイズ) を付与する
fn add_awgn(samples: &mut [f32], sigma: f32) {
    if sigma <= 0.0 { return; }
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, sigma).unwrap();
    for s in samples.iter_mut() {
        *s += normal.sample(&mut rng);
    }
}

/// 指定されたシグマ（ノイズ強度）で E2E 通信テストを行う
fn test_transmission(sigma: f32) -> bool {
    let data = b"Acoustic E2E Noise Test Data.";
    let dsp_config = DspConfig::default_48k();
    let mut encoder = Encoder::new(EncoderConfig::new(dsp_config.clone()));
    
    // 送信信号生成
    let mut stream = encoder.encode_stream(data);
    let mut tx_signal = Vec::new();
    tx_signal.extend(vec![0.0; 9600]);

    for _ in 0..15 {
        if let Some(mut frame) = stream.next() {
            add_awgn(&mut frame, sigma);
            tx_signal.extend(frame);
            tx_signal.extend(vec![0.0; 4800]);
        }
    }

    // デコード
    let mut decoder = Decoder::new(data.len(), 8, dsp_config.clone());
    let chunk_size = 2048; // WebAudio標準
    let mut tx_offset = 0;

    while tx_offset < tx_signal.len() {
        let end = (tx_offset + chunk_size).min(tx_signal.len());
        let chunk = &tx_signal[tx_offset..end];
        tx_offset = end;

        let progress = decoder.process_samples(chunk);
        
        if progress.complete {
            return true;
        }
    }

    // デコードされたデータが一致するか確認
    if let Some(recovered) = decoder.recovered_data() {
        if recovered == data {
            return true;
        } else {
            println!("Decoded data mismatch!");
            return false;
        }
    }

    false
}

#[test]
fn test_awgn_tolerance_search() {
    println!("\n--- AWGN Tolerance Measurement ---");
    let sigmas = [0.0, 0.05, 0.1, 0.2];
    
    for &sigma in &sigmas {
        let mut success_count = 0;
        let trials = 10; // 1 trial is enough for quick regression
        for _ in 0..trials {
            if test_transmission(sigma) {
                success_count += 1;
            }
        }
        let rate = (success_count as f32 / trials as f32) * 100.0;
        println!("Sigma: {:.2}, Success Rate: {:.1}% ({}/{})", sigma, rate, success_count, trials);
        
        if sigma == 0.0 && success_count == 0 {
            panic!("ノイズなし(Sigma=0.0)で通信に失敗しました。各モジュールの不整合が疑われます。");
        }
        
        if rate == 0.0 && sigma > 0.0 {
            break; // これ以上のノイズ耐性テストは無意味なので打ち切り
        }
    }
    println!("-----------------------------------------------------------\n");
}
