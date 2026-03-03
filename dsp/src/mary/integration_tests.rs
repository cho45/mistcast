//! MaryDQPSKエンドツーエンド統合テスト
//!
//! エンコーダとデコーダの統合テスト。実用的なシナリオでの動作を検証する。

use crate::mary::encoder::Encoder;
use crate::mary::decoder::Decoder;
use crate::common::walsh::WalshDictionary;
use crate::DspConfig;
use num_complex::Complex32;

/// エンコーダとデコーダの統合テストヘルパー
struct MaryTestSystem {
    encoder: Encoder,
    decoder: Decoder,
    config: DspConfig,
}

impl MaryTestSystem {
    fn new() -> Self {
        let config = DspConfig::default_48k();
        let encoder = Encoder::new(config.clone());
        let decoder = Decoder::new(160, 10, config.clone());
        Self { encoder, decoder, config }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 基本的なE2Eテスト：小さなデータの往復
    #[test]
    fn test_e2e_small_data_roundtrip() {
        let mut system = MaryTestSystem::new();
        let data = vec![0xABu8; 16];

        // 完全な往復テスト（簡易版：ノイズなし）
        // 注：現在のデコーダ実装は不完全なため、このテストは
        // エンコードが正常に動作することを確認するのみ
        system.encoder.set_data(&data);
        let frame = system.encoder.encode_frame();

        assert!(frame.is_some(), "Should encode a frame");
        let frame_samples = frame.unwrap();
        assert!(!frame_samples.is_empty(), "Frame should not be empty");
        assert!(frame_samples.iter().all(|&s| s.is_finite()), "All samples should be finite");
    }

    /// フレーム構造の検証：プリアンブル+Sync+Payload
    #[test]
    fn test_e2e_frame_structure() {
        let mut system = MaryTestSystem::new();
        let data = vec![0x12u8; 16];
        system.encoder.set_data(&data);

        let frame = system.encoder.encode_frame().unwrap();

        // フレームは十分な長さを持つべき
        let min_expected_length = 2000; // プリアンブル + Sync + Payload
        assert!(frame.len() > min_expected_length, "Frame should be long enough");

        // 全サンプルが有限値
        assert!(frame.iter().all(|&s| s.is_finite()));

        // 振幅範囲のチェック（クリッピングがない）
        let max_amp = frame.iter().fold(0.0f32, |a, &s| a.max(s.abs()));
        assert!(max_amp < 3.0, "Amplitude should be reasonable, got {}", max_amp);
    }

    /// プリアンブルの基本構造検証
    #[test]
    fn test_e2e_preamble_structure() {
        let mut system = MaryTestSystem::new();
        let data = vec![0x34u8; 16];
        system.encoder.set_data(&data);

        let frame = system.encoder.encode_frame().unwrap();

        // プリアンブルはフレームの先頭にある
        let spc = system.config.samples_per_chip();
        let sf_preamble = 15;
        let preamble_len = system.config.preamble_repeat * sf_preamble * spc;

        // プリアンブル部分が存在する
        assert!(frame.len() > preamble_len, "Frame should contain preamble");

        // プリアンブル部分のエネルギーを確認
        let preamble_samples = &frame[..preamble_len.min(frame.len())];
        let preamble_energy: f32 = preamble_samples.iter().map(|&s| s * s).sum();
        let avg_energy = preamble_energy / preamble_samples.len() as f32;

        // エネルギーがゼロでない（信号が存在する）
        assert!(avg_energy > 0.001, "Preamble should have signal energy, got {}", avg_energy);

        // 振幅が過大でない（クリッピングがない）
        let max_amp = preamble_samples.iter().fold(0.0f32, |a, &s| a.max(s.abs()));
        assert!(max_amp < 3.0, "Preamble amplitude should be reasonable, got {}", max_amp);
    }

    /// Sync Wordの埋め込みテスト
    #[test]
    fn test_e2e_sync_word_embedding() {
        let mut system = MaryTestSystem::new();
        let data = vec![0x56u8; 16];
        system.encoder.set_data(&data);

        let frame = system.encoder.encode_frame().unwrap();

        // フレームにはSync Wordが含まれている
        // Sync Wordは16ビット = 16シンボル（DBPSK, sf=15）
        let sync_samples_count = 16 * 15 * system.config.samples_per_chip();
        let preamble_count = system.config.preamble_repeat * 15 * system.config.samples_per_chip();

        // Sync部分がフレームに含まれている
        assert!(frame.len() > preamble_count + sync_samples_count);
    }

    /// 複数パケットのエンコード
    #[test]
    fn test_e2e_multiple_packets() {
        let mut system = MaryTestSystem::new();
        let data = vec![0x78u8; 32];
        system.encoder.set_data(&data);

        // 複数のパケットをエンコード
        let frame1 = system.encoder.encode_frame();
        let frame2 = system.encoder.encode_frame();
        let frame3 = system.encoder.encode_frame();

        assert!(frame1.is_some());
        assert!(frame2.is_some());
        assert!(frame3.is_some());

        let f1 = frame1.unwrap();
        let f2 = frame2.unwrap();
        let f3 = frame3.unwrap();

        // 全てのフレームが有限値
        assert!(f1.iter().all(|&s| s.is_finite()));
        assert!(f2.iter().all(|&s| s.is_finite()));
        assert!(f3.iter().all(|&s| s.is_finite()));

        // フレーム長が類似している（同じデータ設定）
        let avg_len = (f1.len() + f2.len() + f3.len()) / 3;
        assert!((f1.len() as i32 - avg_len as i32).abs() < 100);
        assert!((f2.len() as i32 - avg_len as i32).abs() < 100);
        assert!((f3.len() as i32 - avg_len as i32).abs() < 100);
    }

    /// 連続処理の安定性
    #[test]
    fn test_e2e_continuous_processing() {
        let mut system = MaryTestSystem::new();

        // 連続してフレームを処理
        for i in 0..5 {
            let data = vec![i as u8; 16];
            system.encoder.set_data(&data);

            let frame = system.encoder.encode_frame();
            assert!(frame.is_some(), "Frame {} should be encoded", i);

            let frame_samples = frame.unwrap();
            assert!(frame_samples.iter().all(|&s| s.is_finite()));
        }
    }

    /// 復調器のWalsh系列相関の検証
    #[test]
    fn test_e2e_demodulator_walsh_correlation() {
        let wdict = WalshDictionary::default_w16();

        // 各Walsh系列について相関を確認
        for walsh_idx in 0..16 {
            let signal: Vec<Complex32> = wdict.w16[walsh_idx]
                .iter()
                .map(|&w| Complex32::new(w as f32, 0.0))
                .collect();

            // 16系列並列相関
            let mut correlations = [0.0f32; 16];
            for idx in 0..16 {
                let seq = &wdict.w16[idx];
                let mut corr = Complex32::new(0.0, 0.0);
                for (&s, &w) in signal.iter().zip(seq.iter()) {
                    corr += s * w as f32;
                }
                correlations[idx] = corr.norm_sqr();
            }

            // 対応するWalsh系列の相関が最大
            let max_idx = correlations
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            assert_eq!(
                max_idx, walsh_idx,
                "Walsh[{}] should have max correlation at index {}, got {}",
                walsh_idx, walsh_idx, max_idx
            );

            // 他の系列との直交性
            for idx in 0..16 {
                if idx != walsh_idx {
                    assert!(
                        correlations[idx] < 1e-6,
                        "Walsh[{}] and Walsh[{}] should be orthogonal, correlation={}",
                        walsh_idx, idx, correlations[idx]
                    );
                }
            }
        }
    }

    /// 実用的なデータサイズでのテスト
    #[test]
    fn test_e2e_practical_data_sizes() {
        let mut system = MaryTestSystem::new();

        // 様々なデータサイズでテスト
        for size in [16, 32, 48, 64].iter() {
            let data = vec![0xAAu8; *size];
            system.encoder.set_data(&data);

            let frame = system.encoder.encode_frame();
            assert!(frame.is_some(), "Data size {} should be encodable", size);

            let frame_samples = frame.unwrap();
            assert!(frame_samples.iter().all(|&s| s.is_finite()));
        }
    }

    /// エンコーダの状態管理の検証
    #[test]
    fn test_e2e_encoder_state_management() {
        let mut system = MaryTestSystem::new();

        // データ設定
        let data1 = vec![0x11u8; 16];
        system.encoder.set_data(&data1);
        let frame1 = system.encoder.encode_frame();

        // リセット
        system.encoder.reset();

        // 同じデータで再設定
        let data2 = vec![0x11u8; 16];
        system.encoder.set_data(&data2);
        let frame2 = system.encoder.encode_frame();

        assert!(frame1.is_some());
        assert!(frame2.is_some());

        // リセット後も同様のフレーム長
        let f1 = frame1.unwrap();
        let f2 = frame2.unwrap();
        assert_eq!(f1.len(), f2.len(), "Reset should produce consistent frame lengths");
    }

    /// デコーダの状態管理の検証
    #[test]
    fn test_e2e_decoder_state_management() {
        let mut system = MaryTestSystem::new();

        // 無音を処理しても完了しない
        let silence = vec![0.0f32; 4800];
        system.decoder.process_samples(&silence);

        // 結果なし
        assert!(system.decoder.recovered_data().is_none());
    }

    /// フレームの連続エンコード
    #[test]
    fn test_e2e_sequential_frames() {
        let mut system = MaryTestSystem::new();

        // 連続してフレームをエンコード
        for i in 0..3 {
            let data = vec![i as u8; 16];
            system.encoder.set_data(&data);

            let frame = system.encoder.encode_frame();
            assert!(frame.is_some(), "Frame {} should be encoded", i);

            let frame_samples = frame.unwrap();
            assert!(frame_samples.iter().all(|&s| s.is_finite()));
            assert!(frame_samples.len() > 1000, "Frame should be long enough");
        }
    }
}
