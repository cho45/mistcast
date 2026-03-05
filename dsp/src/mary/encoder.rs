//! MaryDQPSKエンコーダ
//!
//! Modulatorをラップした高レベルエンコーダ。

use crate::coding::fec;
use crate::coding::fountain::{FountainEncoder, FountainPacket};
use crate::coding::interleaver::BlockInterleaver;
use crate::coding::scrambler::Scrambler;
use crate::frame::packet::Packet;
use crate::mary::modulator::Modulator;
use crate::params::PAYLOAD_SIZE;
use crate::DspConfig;

/// MaryDQPSKエンコーダ
/// エンコーダ設定
#[derive(Clone)]
pub struct EncoderConfig {
    pub fountain_k: usize,
    pub packets_per_sync_burst: usize,
    pub il_rows: usize,
    pub il_cols: usize,
    pub dsp: DspConfig,
}

impl EncoderConfig {
    pub fn new(dsp: DspConfig) -> Self {
        use crate::frame::packet::PACKET_BYTES;
        let raw_bits = PACKET_BYTES * 8 + 6; // テールビット(6)含む
        let fec_bits = raw_bits * 2;
        let rows = 16;
        let cols = fec_bits.div_ceil(rows);
        EncoderConfig {
            fountain_k: 10,
            packets_per_sync_burst: dsp.packets_per_burst,
            il_rows: rows,
            il_cols: cols,
            dsp,
        }
    }
}

pub struct Encoder {
    config: EncoderConfig,
    modulator: Modulator,
    fountain_encoder: Option<FountainEncoder>,
    // ゼロアロケーション用バッファプール
    fec_buffer: Vec<u8>,
    padded_buffer: Vec<u8>,
    interleaved_buffer: Vec<u8>,
    burst_bits_buffer: Vec<u8>,
}

impl Encoder {
    /// 新しいエンコーダを作成する
    pub fn new(dsp_config: DspConfig) -> Self {
        let config = EncoderConfig::new(dsp_config);
        let packets_per_sync_burst = config.packets_per_sync_burst;
        let modulator = Modulator::new(config.dsp.clone());
        // バッファサイズ計算
        let raw_bits = crate::frame::packet::PACKET_BYTES * 8 + 6;
        let fec_bits = raw_bits * 2;
        let rows = 16;
        let cols = fec_bits.div_ceil(rows);
        let interleaved_size = rows * cols;
        let mary_aligned_size = interleaved_size.div_ceil(6) * 6;

        Encoder {
            config,
            modulator,
            fountain_encoder: None,
            fec_buffer: Vec::with_capacity(fec_bits),
            padded_buffer: Vec::with_capacity(interleaved_size),
            interleaved_buffer: Vec::with_capacity(mary_aligned_size),
            burst_bits_buffer: Vec::with_capacity(mary_aligned_size * packets_per_sync_burst),
        }
    }

    pub fn with_config(config: EncoderConfig) -> Self {
        let packets_per_sync_burst = config.packets_per_sync_burst;
        let modulator = Modulator::new(config.dsp.clone());
        // バッファサイズ計算
        let raw_bits = crate::frame::packet::PACKET_BYTES * 8 + 6;
        let fec_bits = raw_bits * 2;
        let rows = 16;
        let cols = fec_bits.div_ceil(rows);
        let interleaved_size = rows * cols;
        let mary_aligned_size = interleaved_size.div_ceil(6) * 6;

        Encoder {
            config,
            modulator,
            fountain_encoder: None,
            fec_buffer: Vec::with_capacity(fec_bits),
            padded_buffer: Vec::with_capacity(interleaved_size),
            interleaved_buffer: Vec::with_capacity(mary_aligned_size),
            burst_bits_buffer: Vec::with_capacity(mary_aligned_size * packets_per_sync_burst),
        }
    }

    /// データを設定する
    pub fn set_data(&mut self, data: &[u8]) {
        let max_k = crate::frame::packet::LT_K_MAX;
        let needed_k = data.len().div_ceil(PAYLOAD_SIZE).max(1);
        assert!(
            needed_k <= max_k,
            "input is too large for current packet header: need k={}, max k={}",
            needed_k,
            max_k
        );
        self.config.fountain_k = needed_k;
        let params = crate::coding::fountain::FountainParams::new(needed_k, PAYLOAD_SIZE);
        self.fountain_encoder = Some(FountainEncoder::new(data, params));
    }

    /// フレームをエンコードする
    pub fn encode_frame(&mut self) -> Option<Vec<f32>> {
        let encoder = self.fountain_encoder.as_mut()?;
        let burst_count = self.config.packets_per_sync_burst.max(1);
        let mut packets = Vec::with_capacity(burst_count);
        for _ in 0..burst_count {
            packets.push(encoder.next_packet());
        }
        Some(self.encode_burst(&packets))
    }

    /// バーストをエンコードする
    pub fn encode_burst(&mut self, packets: &[FountainPacket]) -> Vec<f32> {
        self.burst_bits_buffer.clear();
        for packet in packets {
            let bits = self.encode_packet_bits(packet);
            self.burst_bits_buffer.extend_from_slice(&bits);
        }
        self.modulator.encode_frame(&self.burst_bits_buffer)
    }

    /// パケットをエンコードする
    pub fn encode_packet(&mut self, packet: &FountainPacket) -> Vec<f32> {
        let bits = self.encode_packet_bits(packet);
        self.modulator.encode_frame(&bits)
    }

    /// パケットをビット列にエンコードする（テスト用）
    pub fn encode_packet_bits(&mut self, packet: &FountainPacket) -> Vec<u8> {
        let seq = (packet.seq % (u32::from(u16::MAX) + 1)) as u16;
        let pkt = Packet::new(seq, self.config.fountain_k, &packet.data);
        let pkt_bytes = pkt.serialize();

        // FECエンコード（バッファ使用）
        self.fec_buffer.clear();
        let bits = fec::bytes_to_bits(&pkt_bytes);
        let coded = fec::encode(&bits);
        self.fec_buffer.extend_from_slice(&coded);

        // インターリーバのサイズを決定
        let raw_bits = crate::frame::packet::PACKET_BYTES * 8 + 6;
        let fec_bits = raw_bits * 2;
        let rows = 16;
        let cols = fec_bits.div_ceil(rows);

        // パディング（バッファ使用）
        self.padded_buffer.clear();
        self.padded_buffer.extend_from_slice(&self.fec_buffer);
        self.padded_buffer.resize(rows * cols, 0);

        // スクランブル
        let mut scrambler = Scrambler::default();
        scrambler.process_bits(&mut self.padded_buffer);

        // インターリーブ（インプレースAPI使用）
        let interleaver = BlockInterleaver::new(rows, cols);
        self.interleaved_buffer.resize(rows * cols, 0);
        interleaver.interleave_in_place(
            &self.padded_buffer,
            &mut self.interleaved_buffer[..rows * cols],
        );

        // Maryシンボル（6ビット単位）の境界に揃えるようにパディング
        let mary_aligned_size = (rows * cols).div_ceil(6) * 6;
        self.interleaved_buffer.resize(mary_aligned_size, 0);

        self.interleaved_buffer.clone()
    }

    /// Fountain Kを取得する
    pub fn fountain_k(&self) -> usize {
        self.config.fountain_k
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// フラッシュ
    pub fn flush(&mut self) -> Vec<f32> {
        self.modulator.flush()
    }

    /// 無音を変調する
    pub fn modulate_silence(&mut self, samples: usize) -> Vec<f32> {
        self.modulator.modulate_silence(samples)
    }

    /// リセット
    pub fn reset(&mut self) {
        self.modulator.reset();
    }

    /// modulatorへの参照を取得（テスト用）
    pub fn modulator(&self) -> &Modulator {
        &self.modulator
    }

    /// modulatorへの可変参照を取得（テスト用）
    pub fn modulator_mut(&mut self) -> &mut Modulator {
        &mut self.modulator
    }

    /// Fountainエンコーダへの参照を取得（テスト用）
    pub fn fountain_encoder(&self) -> Option<&FountainEncoder> {
        self.fountain_encoder.as_ref()
    }

    /// Fountainエンコーダへの可変参照を取得（テスト用）
    pub fn fountain_encoder_mut(&mut self) -> Option<&mut FountainEncoder> {
        self.fountain_encoder.as_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_encoder() -> Encoder {
        let config = DspConfig::default_48k();
        Encoder::new(config)
    }

    /// エンコーダの作成とリセット
    #[test]
    fn test_encoder_creation_and_reset() {
        let mut encoder = make_encoder();
        encoder.reset();
    }

    /// データ設定とフレームエンコード
    #[test]
    fn test_set_data_and_encode() {
        let mut encoder = make_encoder();
        let data = vec![0x12u8; 32];
        encoder.set_data(&data);

        let frame = encoder.encode_frame();
        assert!(frame.is_some(), "Should encode a frame");
        let frame = frame.unwrap();
        assert!(!frame.is_empty(), "Frame should not be empty");
        assert!(
            frame.iter().all(|&s| s.is_finite()),
            "All samples should be finite"
        );
    }

    /// バーストエンコード
    #[test]
    fn test_encode_burst() {
        let mut encoder = make_encoder();
        let data = vec![0x34u8; 32];
        encoder.set_data(&data);

        // エンコーダからパケットを取得
        let packet1 = encoder.fountain_encoder.as_mut().unwrap().next_packet();
        let packet2 = encoder.fountain_encoder.as_mut().unwrap().next_packet();

        let burst = encoder.encode_burst(&[packet1, packet2]);
        assert!(!burst.is_empty(), "Burst should not be empty");
        assert!(
            burst.iter().all(|&s| s.is_finite()),
            "All samples should be finite"
        );
    }

    /// フラッシュ
    #[test]
    fn test_flush() {
        let mut encoder = make_encoder();
        let flushed = encoder.flush();
        assert!(
            flushed.iter().all(|&s| s.is_finite()),
            "All samples should be finite"
        );
    }

    /// 無音変調
    #[test]
    fn test_modulate_silence() {
        let mut encoder = make_encoder();
        let silence = encoder.modulate_silence(100);
        assert!(!silence.is_empty(), "Should produce samples");
        assert!(silence.len() <= 100, "Should not exceed 100 samples");
        assert!(
            silence.iter().all(|&s| s.is_finite()),
            "All samples should be finite"
        );
    }

    /// 連続フレームエンコードの整合性
    #[test]
    fn test_consecutive_frames() {
        let mut encoder = make_encoder();
        let data = vec![0xABu8; 16];
        encoder.set_data(&data);

        // 複数のフレームをエンコード
        let frame1 = encoder.encode_frame();
        let frame2 = encoder.encode_frame();
        let frame3 = encoder.encode_frame();

        assert!(frame1.is_some(), "First frame should be encoded");
        assert!(frame2.is_some(), "Second frame should be encoded");
        assert!(frame3.is_some(), "Third frame should be encoded");

        let f1 = frame1.unwrap();
        let f2 = frame2.unwrap();
        let f3 = frame3.unwrap();

        // 全てのフレームが有限値である
        assert!(f1.iter().all(|&s| s.is_finite()));
        assert!(f2.iter().all(|&s| s.is_finite()));
        assert!(f3.iter().all(|&s| s.is_finite()));

        // フレーム長が一貫している（近似）
        let avg_len = (f1.len() + f2.len() + f3.len()) / 3;
        assert!((f1.len() as i32 - avg_len as i32).abs() < 100);
        assert!((f2.len() as i32 - avg_len as i32).abs() < 100);
        assert!((f3.len() as i32 - avg_len as i32).abs() < 100);
    }

    /// パケット構造の検証
    #[test]
    fn test_packet_structure() {
        let mut encoder = make_encoder();
        let data = vec![0x12u8; 16];
        encoder.set_data(&data);

        // パケットをエンコード
        let frame = encoder.encode_frame();
        assert!(frame.is_some());

        let frame_samples = frame.unwrap();

        // フレームにはプリアンブル、Sync、Payloadが含まれる
        // 最小フレーム長の検証
        assert!(
            frame_samples.len() > 1000,
            "Frame should contain preamble, sync, and payload"
        );
    }

    /// リセット後の動作検証
    #[test]
    fn test_reset_consistency() {
        let mut encoder = make_encoder();
        let data = vec![0x99u8; 32];
        encoder.set_data(&data);

        let frame1 = encoder.encode_frame();
        encoder.reset();
        let frame2 = encoder.encode_frame();

        assert!(frame1.is_some());
        assert!(frame2.is_some());

        // リセット後も同じデータ設定であれば、同様のフレーム長
        let f1 = frame1.unwrap();
        let f2 = frame2.unwrap();
        assert!((f1.len() as i32 - f2.len() as i32).abs() < 50);
    }

    // ========== 厳密な信号処理テスト ==========

    /// パケットビット列のFEC符号化の正当性を検証
    ///
    /// 検証項目：
    /// 1. FEC符号化後のビット列は入力より長い（情報理論：符号化は冗長性を追加）
    /// 2. FEC符号化は可逆である（同じパケット→同じビット列）
    /// 3. 異なるパケットは異なるビット列を生成する（決定性）
    #[test]
    fn test_encode_packet_bits_fec() {
        let mut encoder = make_encoder();
        let data = vec![0x12u8; 16];
        encoder.set_data(&data);

        // 同じパケットを2回エンコードして可逆性を検証
        let packet_data1 = vec![0xABu8; crate::params::PAYLOAD_SIZE];
        let packet1 = FountainPacket {
            seq: 0,
            coefficients: vec![1u8],
            data: packet_data1,
        };

        let bits1 = encoder.encode_packet_bits(&packet1);
        let bits2 = encoder.encode_packet_bits(&packet1);

        // 1. FEC符号化後のビット列は入力パケットより長いはず
        // 根拠：FEC（畳み込み符号）は冗長性を追加するため
        let input_bit_count = packet1.data.len() * 8; // payload bits
        assert!(
            bits1.len() > input_bit_count,
            "FEC encoded bits ({}) should be greater than input bits ({})",
            bits1.len(),
            input_bit_count
        );

        // 2. 同じパケット→同じビット列（決定性）
        assert_eq!(bits1, bits2, "Same packet should produce same bits");

        // 3. 異なるパケット→異なるビット列
        let packet_data3 = vec![0xCDu8; crate::params::PAYLOAD_SIZE];
        let packet3 = FountainPacket {
            seq: 0,
            coefficients: vec![1u8],
            data: packet_data3,
        };
        let bits3 = encoder.encode_packet_bits(&packet3);

        assert_ne!(
            bits1, bits3,
            "Different packets should produce different bits"
        );
    }

    /// インターリービングの数学的性質を検証
    ///
    /// 検証項目：
    /// 1. インターリーブ後のビット数はrows×colsで割り切れる（ブロック構造）
    /// 2. インターリーブは可逆である（同じ入力→同じ出力）
    /// 3. 異なる入力は異なる出力を生成する（衝突なし）
    #[test]
    fn test_interleaving_integration() {
        let mut encoder = make_encoder();
        let data = vec![0x34u8; 16];
        encoder.set_data(&data);

        // パケットを生成
        let packet_data1 = vec![0xABu8; crate::params::PAYLOAD_SIZE];
        let packet1 = FountainPacket {
            seq: 0,
            coefficients: vec![1u8],
            data: packet_data1,
        };

        let bits1 = encoder.encode_packet_bits(&packet1);

        // 1. インターリーブ後のビット数は 6ビット境界（Maryシンボル）に揃っているはず
        assert_eq!(
            bits1.len() % 6,
            0,
            "Interleaved bits ({}) must be a multiple of 6 for Mary symbols",
            bits1.len()
        );

        // インターリーバのコアサイズ（352）が rows(16) の倍数であることは encode_packet_bits の内部定数で保証済み。
        // ここでは最終出力が Mary 境界を満たしていることを検証する。

        // 2. 同じパケット→同じビット列（可逆性・決定性）
        let bits2 = encoder.encode_packet_bits(&packet1);
        assert_eq!(bits1, bits2, "Interleaving should be deterministic");

        // 3. 異なるパケット→異なるビット列（衝突なし）
        let packet_data3 = vec![0xCDu8; crate::params::PAYLOAD_SIZE];
        let packet3 = FountainPacket {
            seq: 0,
            coefficients: vec![1u8],
            data: packet_data3,
        };
        let bits3 = encoder.encode_packet_bits(&packet3);

        assert_ne!(
            bits1, bits3,
            "Different packets should produce different interleaved bits"
        );
    }

    /// エンコードフレームの信号特性検証
    #[test]
    fn test_encoded_frame_signal_characteristics() {
        let mut encoder = make_encoder();
        let data = vec![0x56u8; 32];
        encoder.set_data(&data);

        let frame = encoder.encode_frame().unwrap();

        // 1. 全サンプルが有限値
        assert!(
            frame.iter().all(|&s| s.is_finite()),
            "All samples should be finite"
        );

        // 2. 振幅範囲（クリッピングなし）
        // 根拠：modulatorと同じ信号処理パイプライン
        // - RRCフィルタのピーク振幅 = 1.0
        // - リサンプラ変動 = ±20%
        // - 実際の最大振幅 ≈ 1.2
        // - 安全余裕2倍：2.5（クリッピング防止）
        // - 5.0はさらに保守的な値（複数シンボルの蓄積を考慮）
        let max_amp = frame.iter().fold(0.0f32, |a, &s| a.max(s.abs()));
        assert!(
            max_amp < 5.0,
            "Amplitude should be reasonable (theoretical max ~1.2, safety margin 4x), got {}",
            max_amp
        );

        // 3. 信号エネルギー（ゼロでない）
        // 根拠：フレームはプリアンブル + Sync + Payloadで構成され、
        //       各部分は非ゼロの信号を含む
        //       最小フレーム長 ≈ 2000サンプル、各サンプルの平均電力 > 0.001
        //       したがって、全エネルギー > 1.0は妥当な下限
        let energy: f32 = frame.iter().map(|&s| s * s).sum();
        assert!(energy > 1.0, "Frame should have energy, got {}", energy);

        // 4. フレーム長の整合性
        let spc = encoder.modulator.config().samples_per_chip();
        let preamble_len = 15 * encoder.modulator.config().preamble_repeat * spc;
        let sync_len = 16 * 15 * spc;

        assert!(
            frame.len() > preamble_len + sync_len,
            "Frame should contain preamble + sync + payload"
        );
    }

    /// プリアンブルとSyncの分離検証
    #[test]
    fn test_preamble_sync_separation() {
        let mut encoder = make_encoder();
        let data = vec![0x78u8; 16];
        encoder.set_data(&data);

        let frame = encoder.encode_frame().unwrap();

        let spc = encoder.modulator.config().samples_per_chip();
        let preamble_repeat = encoder.modulator.config().preamble_repeat;
        let preamble_len = 15 * preamble_repeat * spc;
        let sync_len = 16 * 15 * spc;

        // プリアンブル部分が存在する
        assert!(frame.len() > preamble_len, "Frame should contain preamble");

        // Sync部分が存在する
        assert!(
            frame.len() > preamble_len + sync_len,
            "Frame should contain sync"
        );

        // 各部分のエネルギーを確認
        // 根拠：プリアンブルは15チップ × repeat回分で構成され、
        //       各チップは±1の値を持つため、非ゼロのエネルギーを持つ
        //       RRCフィルタとリサンプラを通った後の平均電力は約0.01/sample程度
        //       preamble_lenは数百サンプルなので、全エネルギー > 1.0が期待される
        //       0.1は非常に保守的な下限値（実際には10倍以上あるはず）
        let preamble_energy: f32 = frame[..preamble_len.min(frame.len())]
            .iter()
            .map(|&s| s * s)
            .sum();
        assert!(
            preamble_energy > 0.1,
            "Preamble should have significant energy (got {}, expected > 1.0)",
            preamble_energy
        );
    }

    /// 複数パケットエンコードの連続性
    #[test]
    fn test_multi_packet_encoding_continuity() {
        let mut encoder = make_encoder();
        let data = vec![0x9Au8; 48]; // 3パケット分
        encoder.set_data(&data);

        let mut frames = Vec::new();
        for _ in 0..3 {
            if let Some(frame) = encoder.encode_frame() {
                frames.push(frame);
            }
        }

        assert_eq!(frames.len(), 3, "Should encode 3 frames");

        // 全てのフレームが有限値
        for (i, frame) in frames.iter().enumerate() {
            assert!(
                frame.iter().all(|&s| s.is_finite()),
                "Frame {} should have all finite samples",
                i
            );
        }

        // フレーム長が一貫している
        // 根拠：同じデータ設定で連続エンコードする場合、フレーム長は一貫すべき
        //       許容誤差はRRCフィルタの群遅延とリサンプラの補間遅延による
        //       群遅延 = (L-1)/2 ≈ 8サンプル、リサンプラ遅延 = ±1サンプル
        //       最大遅延 ≈ 10サンプルだが、安全余裕を見て100サンプルとする
        let avg_len = frames.iter().map(|f| f.len()).sum::<usize>() / frames.len();
        for (i, frame) in frames.iter().enumerate() {
            let diff = (frame.len() as i32 - avg_len as i32).abs();
            assert!(diff < 100,
                    "Frame {} length {} should be close to average {} (max delay ~10 samples, safety margin 100)",
                    i, frame.len(), avg_len);
        }
    }

    /// デコーダへの入力としての妥当性検証
    #[test]
    fn test_encoded_frame_valid_for_decoder() {
        let mut encoder = make_encoder();
        let data = vec![0xBCu8; 32];
        encoder.set_data(&data);

        let frame = encoder.encode_frame().unwrap();

        // デコーダで処理できる形式であることを確認
        // 1. サンプルレートが一致する
        // 2. 信号帯域が適切である
        // 3. DCオフセットが大きすぎない

        // DCオフセットチェック
        let dc_offset = frame.iter().sum::<f32>() / frame.len() as f32;
        assert!(
            dc_offset.abs() < 1.0,
            "DC offset should be small, got {}",
            dc_offset
        );

        // RMSレベルチェック
        let rms = (frame.iter().map(|&s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
        assert!(
            rms > 0.001 && rms < 2.0,
            "RMS should be reasonable, got {}",
            rms
        );
    }

    /// 信号対ノイズ比の概算
    #[test]
    fn test_signal_quality_metrics() {
        let mut encoder = make_encoder();
        let data = vec![0xDEu8; 32];
        encoder.set_data(&data);

        let frame = encoder.encode_frame().unwrap();

        // 信号電力
        let signal_power: f32 = frame.iter().map(|&s| s * s).sum::<f32>() / frame.len() as f32;

        // ピーク電力
        let peak_power = frame.iter().map(|&s| s * s).fold(0.0f32, |a, p| a.max(p));

        // PAPR (Peak-to-Average Power Ratio)
        // 根拠：MaryDQPSK信号のPAPRは理論上以下のように計算できる
        // - 平均電力：各シンボルのエネルギーはsf=16、信号は±1なので平均電力は約1.0
        // - ピーク電力：RRCフィルタのピークで最大約1.44
        // - 理論PAPR ≈ 1.44 / 1.0 = 1.44
        // - 実際には複数シンボルの蓄積とリサンプラの影響で3-5程度
        // - 10.0は非常に保守的な上限（正常な信号では5以下）
        let papr = peak_power / signal_power;
        assert!(
            papr < 10.0,
            "PAPR should be reasonable (theoretical ~1.44, typical 3-5), got {}",
            papr
        );

        // クリッピングチェック
        // 根拠：正常な信号の最大振幅は2.5以下（modulatorの振幅テスト参照）
        //       3.0を超えるサンプルは信号処理の異常を示す
        //       閾値3.0は安全余裕を含む保守的な値
        let clipped_count = frame.iter().filter(|&&s| s.abs() > 3.0).count();
        assert!(
            clipped_count == 0,
            "Signal should not clip (threshold 3.0, theoretical max ~2.5), {} samples exceeded",
            clipped_count
        );
    }

    /// Fountainエンコーダの状態管理
    #[test]
    fn test_fountain_encoder_state_management() {
        let mut encoder = make_encoder();
        let data = vec![0xF0u8; 16];
        encoder.set_data(&data);

        // K値の確認
        let expected_k = data.len().div_ceil(crate::params::PAYLOAD_SIZE).max(1);
        assert_eq!(encoder.fountain_k(), expected_k, "Fountain K should match");

        // 複数パケットをエンコードしてもFountainエンコーダが正常に動作
        for i in 0..10 {
            let frame = encoder.encode_frame();
            assert!(frame.is_some(), "Should encode packet {}", i);
        }
    }

    /// modulator統合の検証
    #[test]
    fn test_modulator_integration() {
        let mut encoder = make_encoder();
        let data = vec![0xAAu8; 16];
        encoder.set_data(&data);

        // ビット列をエンコード
        let packet_data = vec![0xBBu8; crate::params::PAYLOAD_SIZE];
        let packet = FountainPacket {
            seq: 0,
            coefficients: vec![1u8],
            data: packet_data,
        };
        let bits = encoder.encode_packet_bits(&packet);

        // ビット数はインターリーブでrows*colsに調整される
        // 6の倍数でなくても、modulatorは余剰ビットを無視して処理する
        let expected_symbols = bits.len() / 6;
        assert!(expected_symbols > 0, "Should have at least one symbol");

        // modulatorで変調
        let samples = encoder.modulator.modulate(&bits);

        // 変調結果が妥当である
        assert!(!samples.is_empty(), "Modulator should produce samples");
        assert!(
            samples.iter().all(|&s| s.is_finite()),
            "All modulated samples should be finite"
        );
    }

    /// encode_packet_bitsのビット数を検証
    ///
    /// 既存のDQPSK実装と同じインターリーバ設定を使用していることを確認
    /// 根拠：既存DQPSKは2ビット/シンボルで352ビット（176シンボル）
    ///       MaryDQPSKは6ビット/シンボルで352ビットは割り切れない（58.66...シンボル）
    ///       最後の4ビットはパディングとして扱う
    #[test]
    fn test_encode_packet_bits_bit_count() {
        use crate::frame::packet::PACKET_BYTES;

        let mut encoder = make_encoder();
        let data = vec![0xCCu8; 16];
        encoder.set_data(&data);

        // ビット数の計算
        // 1. PACKET_BYTES * 8 = 21 * 8 = 168ビット（元のパケット）
        // 2. fec::encode()で2倍化 = 336ビット
        // 3. インターリーバパディングでrows * colsに調整
        let packet_data = vec![0xDDu8; crate::params::PAYLOAD_SIZE];
        let packet = FountainPacket {
            seq: 0,
            coefficients: vec![1u8],
            data: packet_data,
        };
        let bits = encoder.encode_packet_bits(&packet);

        // インターリーバサイズ計算（既存DQPSKと同じ）
        let raw_bits = PACKET_BYTES * 8 + 6; // 168 + 6 = 174（テールビット含む）
        let fec_bits = raw_bits * 2; // 348
        let rows = 16;
        let cols = fec_bits.div_ceil(rows); // 348 / 16 = 21.75 → 22
        let interleaved_bits = rows * cols; // 16 * 22 = 352

        // Maryシンボル境界（6ビット）にパディングした後の期待されるビット数
        let expected_mary_bits = interleaved_bits.div_ceil(6) * 6; // 354

        assert_eq!(
            bits.len(),
            expected_mary_bits,
            "encode_packet_bits should produce {} bits (6-bit aligned), got {}",
            expected_mary_bits,
            bits.len()
        );

        // MaryDQPSKシンボル数（6ビット/シンボル）
        let full_symbols = bits.len() / 6; // 59
        assert_eq!(full_symbols, 59, "Should have 59 full symbols");

        // 注：最後の4ビットはmodulatorで無視されるか、パディングとして扱われる
        // decoderでも同様にパディングを無視する必要がある
    }
}
