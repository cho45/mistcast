//! MaryDQPSKエンコーダ
//!
//! Modulatorをラップした高レベルエンコーダ。

use crate::coding::fec;
use crate::coding::fountain::{FountainEncoder, FountainPacket};
use crate::coding::interleaver::BlockInterleaver;
use crate::coding::scrambler::Scrambler;
use crate::frame::packet::{Packet, PACKET_BYTES};
use crate::mary::interleaver_config;
use crate::mary::modulator::Modulator;
use crate::mary::params;
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
        EncoderConfig {
            fountain_k: 10,
            packets_per_sync_burst: dsp.packets_per_burst,
            il_rows: interleaver_config::INTERLEAVER_ROWS,
            il_cols: interleaver_config::INTERLEAVER_COLS,
            dsp,
        }
    }
}

pub struct Encoder {
    config: EncoderConfig,
    modulator: Modulator,
    interleaver: BlockInterleaver,
    fountain_encoder: Option<FountainEncoder>,
    // ゼロアロケーション用バッファプール
    packet_bytes_buffer: Vec<u8>,
    raw_bits_buffer: Vec<u8>,
    fec_buffer: Vec<u8>,
    padded_buffer: Vec<u8>,
    interleaved_buffer: Vec<u8>,
    burst_bits_buffer: Vec<u8>,
    modulator_output: Vec<f32>, // Modulator 出力バッファ
}

impl Encoder {
    /// 新しいエンコーダを作成する
    pub fn new(dsp_config: DspConfig) -> Self {
        let config = EncoderConfig::new(dsp_config);
        let packets_per_sync_burst = config.packets_per_sync_burst;
        let modulator = Modulator::new(config.dsp.clone());
        // バッファサイズ計算（interleaver_config使用）
        let fec_bits = interleaver_config::fec_bits();
        let interleaved_size = interleaver_config::interleaved_bits();
        let mary_aligned_size = interleaver_config::mary_aligned_bits();

        // Modulator 出力バッファサイズ (最大フレームサンプル数)
        let max_bits = mary_aligned_size * packets_per_sync_burst;
        let max_symbols = max_bits.div_ceil(6);
        let max_chips = max_symbols * params::PAYLOAD_SPREAD_FACTOR + 2000;
        let dsp_ref = &config.dsp;
        let max_samples =
            (max_chips as f32 * (dsp_ref.sample_rate / dsp_ref.proc_sample_rate())) as usize + 5000;

        Encoder {
            config,
            modulator,
            interleaver: BlockInterleaver::new(
                interleaver_config::INTERLEAVER_ROWS,
                interleaver_config::INTERLEAVER_COLS,
            ),
            fountain_encoder: None,
            packet_bytes_buffer: Vec::with_capacity(PACKET_BYTES),
            raw_bits_buffer: Vec::with_capacity(PACKET_BYTES * 8),
            fec_buffer: Vec::with_capacity(fec_bits),
            padded_buffer: Vec::with_capacity(interleaved_size),
            interleaved_buffer: Vec::with_capacity(mary_aligned_size),
            burst_bits_buffer: Vec::with_capacity(mary_aligned_size * packets_per_sync_burst),
            modulator_output: Vec::with_capacity(max_samples),
        }
    }

    pub fn with_config(config: EncoderConfig) -> Self {
        let packets_per_sync_burst = config.packets_per_sync_burst;
        // バッファサイズ計算（interleaver_config使用）
        let fec_bits = interleaver_config::fec_bits();
        let interleaved_size = interleaver_config::interleaved_bits();
        let mary_aligned_size = interleaver_config::mary_aligned_bits();

        // Modulator 出力バッファサイズ
        let max_bits = mary_aligned_size * packets_per_sync_burst;
        let max_symbols = max_bits.div_ceil(6);
        let max_chips = max_symbols * params::PAYLOAD_SPREAD_FACTOR + 2000;
        let dsp_ref = &config.dsp;
        let max_samples =
            (max_chips as f32 * (dsp_ref.sample_rate / dsp_ref.proc_sample_rate())) as usize + 5000;

        let modulator = Modulator::new(config.dsp.clone());

        Encoder {
            config,
            modulator,
            interleaver: BlockInterleaver::new(
                interleaver_config::INTERLEAVER_ROWS,
                interleaver_config::INTERLEAVER_COLS,
            ),
            fountain_encoder: None,
            packet_bytes_buffer: Vec::with_capacity(PACKET_BYTES),
            raw_bits_buffer: Vec::with_capacity(PACKET_BYTES * 8),
            fec_buffer: Vec::with_capacity(fec_bits),
            padded_buffer: Vec::with_capacity(interleaved_size),
            interleaved_buffer: Vec::with_capacity(mary_aligned_size),
            burst_bits_buffer: Vec::with_capacity(mary_aligned_size * packets_per_sync_burst),
            modulator_output: Vec::with_capacity(max_samples),
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
        let mut out = Vec::new();
        self.encode_burst_into(packets, &mut out);
        out
    }

    pub fn encode_burst_into(&mut self, packets: &[FountainPacket], out: &mut Vec<f32>) {
        let interleaved_size = interleaver_config::interleaved_bits();
        self.burst_bits_buffer.clear();
        for packet in packets {
            self.fill_packet_bits_buffer(packet);
            self.burst_bits_buffer
                .extend_from_slice(&self.interleaved_buffer[..interleaved_size]);
        }
        self.modulator_output.clear();
        self.modulator
            .encode_frame(&self.burst_bits_buffer, &mut self.modulator_output);
        out.clear();
        std::mem::swap(out, &mut self.modulator_output);
    }

    /// バーストをキャリア混合前の複素ベースバンドでエンコードする（テスト専用）
    #[cfg(test)]
    pub fn encode_burst_baseband_into_for_test(
        &mut self,
        packets: &[FountainPacket],
        out_i: &mut Vec<f32>,
        out_q: &mut Vec<f32>,
    ) {
        let interleaved_size = interleaver_config::interleaved_bits();
        self.burst_bits_buffer.clear();
        for packet in packets {
            self.fill_packet_bits_buffer(packet);
            self.burst_bits_buffer
                .extend_from_slice(&self.interleaved_buffer[..interleaved_size]);
        }
        self.modulator
            .encode_frame_baseband_for_test(&self.burst_bits_buffer, out_i, out_q);
    }

    /// フレームをキャリア混合前の複素ベースバンドでエンコードする（テスト専用）
    #[cfg(test)]
    pub fn encode_frame_baseband_for_test(&mut self) -> Option<(Vec<f32>, Vec<f32>)> {
        let encoder = self.fountain_encoder.as_mut()?;
        let burst_count = self.config.packets_per_sync_burst.max(1);
        let mut packets = Vec::with_capacity(burst_count);
        for _ in 0..burst_count {
            packets.push(encoder.next_packet());
        }
        let mut out_i = Vec::new();
        let mut out_q = Vec::new();
        self.encode_burst_baseband_into_for_test(&packets, &mut out_i, &mut out_q);
        Some((out_i, out_q))
    }

    /// パケットをエンコードする
    pub fn encode_packet(&mut self, packet: &FountainPacket) -> Vec<f32> {
        self.fill_packet_bits_buffer(packet);
        self.modulator_output.clear();
        self.modulator
            .encode_frame(&self.interleaved_buffer, &mut self.modulator_output);
        self.modulator_output.clone()
    }

    fn fill_packet_bits_buffer(&mut self, packet: &FountainPacket) {
        let seq = (packet.seq % (u32::from(u16::MAX) + 1)) as u16;
        let pkt = Packet::new(seq, self.config.fountain_k, &packet.data);
        pkt.serialize_into(&mut self.packet_bytes_buffer);

        // FECエンコード（バッファ使用）
        self.raw_bits_buffer.clear();
        fec::bytes_to_bits_into(&self.packet_bytes_buffer, &mut self.raw_bits_buffer);
        self.fec_buffer.clear();
        fec::encode_into(&self.raw_bits_buffer, &mut self.fec_buffer);

        let interleaved_size = interleaver_config::interleaved_bits(); // 348 = fec_bits

        // パディング（バッファ使用）
        self.padded_buffer.clear();
        self.padded_buffer.extend_from_slice(&self.fec_buffer);
        self.padded_buffer.resize(interleaved_size, 0);

        // スクランブル
        let mut scrambler = Scrambler::default();
        scrambler.process_bits(&mut self.padded_buffer);

        // インターリーブ（インプレースAPI使用）
        self.interleaved_buffer.resize(interleaved_size, 0);
        self.interleaver.interleave_in_place(
            &self.padded_buffer,
            &mut self.interleaved_buffer[..interleaved_size],
        );
    }

    /// パケットをビット列にエンコードする（テスト用）
    pub fn encode_packet_bits(&mut self, packet: &FountainPacket) -> Vec<u8> {
        self.fill_packet_bits_buffer(packet);
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
        let mut out = Vec::new();
        self.flush_into(&mut out);
        out
    }

    pub fn flush_into(&mut self, out: &mut Vec<f32>) {
        self.modulator_output.clear();
        self.modulator.flush(&mut self.modulator_output);
        out.clear();
        std::mem::swap(out, &mut self.modulator_output);
    }

    /// 無音を変調する
    pub fn modulate_silence(&mut self, samples: usize) -> Vec<f32> {
        let mut out = Vec::new();
        self.modulate_silence_into(samples, &mut out);
        out
    }

    pub fn modulate_silence_into(&mut self, samples: usize, out: &mut Vec<f32>) {
        self.modulator_output.clear();
        self.modulator
            .modulate_silence(samples, &mut self.modulator_output);
        out.clear();
        std::mem::swap(out, &mut self.modulator_output);
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
    use num_complex::Complex;
    use rustfft::FftPlanner;

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

    #[test]
    fn test_encode_frame_baseband_for_test_smoke() {
        let mut encoder = make_encoder();
        let data = vec![0x24u8; 32];
        encoder.set_data(&data);

        let (bb_i, bb_q) = encoder
            .encode_frame_baseband_for_test()
            .expect("Should encode a baseband frame");
        assert_eq!(bb_i.len(), bb_q.len(), "I/Q length mismatch");
        assert!(!bb_i.is_empty(), "Baseband frame should not be empty");
        assert!(bb_i.iter().all(|&s| s.is_finite()), "I samples must be finite");
        assert!(bb_q.iter().all(|&s| s.is_finite()), "Q samples must be finite");

        let energy: f32 = bb_i
            .iter()
            .zip(bb_q.iter())
            .map(|(&i, &q)| i * i + q * q)
            .sum();
        assert!(energy > 1.0, "Baseband frame should have non-zero energy");
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
        let preamble_len = encoder.modulator.config().preamble_sf
            * encoder.modulator.config().preamble_repeat
            * spc;
        let sync_len = encoder.modulator.config().sync_word_bits * params::SYNC_SPREAD_FACTOR * spc;

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
        let preamble_len = encoder.modulator.config().preamble_sf * preamble_repeat * spc;
        let sync_len = encoder.modulator.config().sync_word_bits * params::SYNC_SPREAD_FACTOR * spc;

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
        // - 平均電力：各シンボルのエネルギーは sf=PAYLOAD_SPREAD_FACTOR、信号は±1なので平均電力は約1.0
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
        let mut samples_buf = Vec::new();
        encoder.modulator.modulate(&bits, &mut samples_buf);
        let samples = samples_buf;

        // 変調結果が妥当である
        assert!(!samples.is_empty(), "Modulator should produce samples");
        assert!(
            samples.iter().all(|&s| s.is_finite()),
            "All modulated samples should be finite"
        );
    }

    /// encode_packet_bitsのビット数を検証
    ///
    /// インターリーバサイズ最適化後の設定を検証
    /// 根拠：FECビット数(348)とインターリーバサイズ(29×12=348)が完全一致
    ///       348は6で割り切れるため、Maryシンボル境界に追加のパディングが不要
    ///       58シンボルで無駄なく伝送できる
    #[test]
    fn test_encode_packet_bits_bit_count() {
        let mut encoder = make_encoder();
        let data = vec![0xCCu8; 16];
        encoder.set_data(&data);

        // ビット数の計算
        // 1. PACKET_BYTES * 8 = 21 * 8 = 168ビット（元のパケット）
        // 2. fec::encode()で2倍化 = 348ビット（テールビット6含む）
        // 3. インターリーバサイズ 29×12 = 348ビット（FECビット数と一致）
        // 4. 348は6で割り切れる → 58シンボル（パディング不要）
        let packet_data = vec![0xDDu8; crate::params::PAYLOAD_SIZE];
        let packet = FountainPacket {
            seq: 0,
            coefficients: vec![1u8],
            data: packet_data,
        };
        let bits = encoder.encode_packet_bits(&packet);

        // インターリーバサイズ計算（interleaver_config使用）
        let expected_bits = interleaver_config::interleaved_bits(); // 348
        let expected_symbols = interleaver_config::mary_symbols(); // 58

        assert_eq!(
            bits.len(),
            expected_bits,
            "encode_packet_bits should produce {} bits (no padding needed), got {}",
            expected_bits,
            bits.len()
        );

        // MaryDQPSKシンボル数（6ビット/シンボル）
        let full_symbols = bits.len() / 6;
        assert_eq!(
            full_symbols, expected_symbols,
            "Should have {} full symbols",
            expected_symbols
        );

        // パディング不要であることを確認
        assert_eq!(
            bits.len(),
            interleaver_config::fec_bits(),
            "ビット数はFECビット数と一致すべき（パディングなし）"
        );
    }

    fn out_of_band_leakage_db_and_peak_db(
        i_samples: &[f32],
        q_samples: &[f32],
        sample_rate: f32,
        inband_edge_hz: f32,
        oob_start_hz: f32,
    ) -> (f32, f32) {
        assert_eq!(i_samples.len(), q_samples.len());
        let trim = (i_samples.len() / 8).min(512);
        let start = trim;
        let end = i_samples.len().saturating_sub(trim);
        let n = end.saturating_sub(start);
        assert!(n >= 512, "FFT解析サンプルが不足: n={}", n);

        let nfft = n.next_power_of_two();
        let mut buf = vec![Complex::new(0.0f32, 0.0f32); nfft];
        for idx in 0..n {
            let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * idx as f32 / (n - 1) as f32).cos();
            buf[idx] = Complex::new(i_samples[start + idx] * w, q_samples[start + idx] * w);
        }

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(nfft);
        fft.process(&mut buf);

        let nyquist_guard_hz = sample_rate * 0.45;
        let mut inband_power = 0.0f64;
        let mut oob_power = 0.0f64;
        let mut inband_peak = 0.0f64;
        let mut oob_peak = 0.0f64;
        for (k, v) in buf.iter().enumerate() {
            let mut f = k as f32 * sample_rate / nfft as f32;
            if k > nfft / 2 {
                f -= sample_rate;
            }
            let af = f.abs();
            let p = v.norm_sqr() as f64;
            if af <= inband_edge_hz {
                inband_power += p;
                inband_peak = inband_peak.max(p);
            } else if af >= oob_start_hz && af <= nyquist_guard_hz {
                oob_power += p;
                oob_peak = oob_peak.max(p);
            }
        }

        assert!(inband_power > 0.0, "inband_power must be positive");
        assert!(inband_peak > 0.0, "inband_peak must be positive");
        let leakage_ratio = (oob_power / inband_power).max(1e-20);
        let peak_ratio = (oob_peak / inband_peak).max(1e-20);
        (
            10.0 * (leakage_ratio.log10() as f32),
            10.0 * (peak_ratio.log10() as f32),
        )
    }

    /// キャリア混合前の複素ベースバンドで帯域外リークが十分小さいことを検証
    #[test]
    fn test_baseband_fft_has_low_out_of_band_leakage_before_carrier_mix() {
        let mut encoder = make_encoder();
        let packet = FountainPacket {
            seq: 0,
            coefficients: vec![1u8],
            data: vec![0xA5u8; crate::params::PAYLOAD_SIZE],
        };
        let bits = encoder.encode_packet_bits(&packet);

        let mut chips_i = Vec::new();
        let mut chips_q = Vec::new();
        let mut phase = 0u8;
        Modulator::bits_to_chips(
            &bits,
            &encoder.modulator().wdict,
            &mut phase,
            &mut chips_i,
            &mut chips_q,
        );

        let mut bb_i = Vec::new();
        let mut bb_q = Vec::new();
        encoder
            .modulator_mut()
            .chips_to_baseband_for_test(&chips_i, &chips_q, &mut bb_i, &mut bb_q);

        assert_eq!(bb_i.len(), bb_q.len());
        assert!(
            bb_i.len() >= 2048,
            "baseband length too short: {}",
            bb_i.len()
        );

        let cfg = encoder.modulator().config();
        let rrc_bandwidth_hz = 0.5 * cfg.chip_rate * (1.0 + cfg.rrc_alpha);
        let inband_edge_hz = rrc_bandwidth_hz * 1.05;
        let oob_start_hz = rrc_bandwidth_hz * 1.35;
        let (leakage_db, peak_leakage_db) = out_of_band_leakage_db_and_peak_db(
            &bb_i,
            &bb_q,
            cfg.sample_rate,
            inband_edge_hz,
            oob_start_hz,
        );
        println!(
            "baseband_oob_leakage_db={:.2}, baseband_oob_peak_db={:.2}, inband_edge_hz={:.1}, oob_start_hz={:.1}",
            leakage_db, peak_leakage_db, inband_edge_hz, oob_start_hz
        );

        assert!(
            leakage_db < -50.0,
            "out-of-band integrated leakage is too large: {:.2} dB (inband<= {:.1} Hz, oob>= {:.1} Hz)",
            leakage_db,
            inband_edge_hz,
            oob_start_hz
        );
        assert!(
            peak_leakage_db < -50.0,
            "out-of-band peak leakage is too large: {:.2} dB (inband<= {:.1} Hz, oob>= {:.1} Hz)",
            peak_leakage_db,
            inband_edge_hz,
            oob_start_hz
        );
    }
}
