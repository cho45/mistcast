//! 送信パイプライン (統合エンコーダ)
//!
//! ```text
//! 入力データ → LT符号化 → パケット化 → バイト→ビット → 畳み込み符号化
//! → インターリーブ → DBPSK+DSSS変調 → f32 音声サンプル列
//! ```

use crate::{
    coding::fec,
    coding::fountain::{EncodedPacket, LtEncoder, LtParams},
    coding::interleaver::BlockInterleaver,
    phy::modulator::Modulator,
    frame::packet::{Packet, PACKET_BYTES},
    params::PAYLOAD_SIZE,
    DspConfig,
};

/// エンコーダ設定
pub struct EncoderConfig {
    pub lt_k: usize,
    pub il_rows: usize,
    pub il_cols: usize,
    pub dsp: DspConfig,
}

impl EncoderConfig {
    pub fn new(dsp: DspConfig) -> Self {
        // PACKET_BYTES * 8 (bits) + 6 (tail bits)
        let raw_bits = PACKET_BYTES * 8 + 6;
        let fec_bits = raw_bits * 2; // 畳み込み符号 (r=1/2) なので2倍
        EncoderConfig {
            lt_k: 8,
            il_rows: 16,
            il_cols: fec_bits.div_ceil(16),
            dsp,
        }
    }

    pub fn default_48k() -> Self {
        Self::new(DspConfig::default_48k())
    }
}

/// 送信フレームを生成するエンコーダ
pub struct Encoder {
    config: EncoderConfig,
    modulator: Modulator,
    interleaver: BlockInterleaver,
}

impl Encoder {
    pub fn new(config: EncoderConfig) -> Self {
        let il = BlockInterleaver::new(config.il_rows, config.il_cols);
        let modulator = Modulator::new(config.dsp.clone());
        Encoder { config, modulator, interleaver: il }
    }

    pub fn with_default_config() -> Self {
        Self::new(EncoderConfig::default_48k())
    }

    /// データをLT符号化して音声フレームのイテレータを返す
    pub fn encode_stream<'a>(&'a mut self, data: &'a [u8]) -> EncoderStream<'a> {
        let params = LtParams::new(self.config.lt_k, PAYLOAD_SIZE);
        let lt_encoder = LtEncoder::new(data, params);
        EncoderStream { encoder: self, lt_encoder, seq: 0 }
    }

    /// 単一のLT符号化パケットを音声サンプル列に変換する
    pub fn encode_packet(&mut self, packet: &EncodedPacket, seq: u32) -> Vec<f32> {
        let pkt = Packet::new(seq, packet.seq, &packet.data[..PAYLOAD_SIZE.min(packet.data.len())]);
        let pkt_bytes = pkt.serialize();
        let bits = fec::bytes_to_bits(&pkt_bytes);
        let coded = fec::encode(&bits);
        let interleaved = self.interleaver.interleave(&coded);
        self.modulator.encode_frame(&interleaved)
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }
}

pub struct EncoderStream<'a> {
    encoder: &'a mut Encoder,
    lt_encoder: LtEncoder,
    seq: u32,
}

impl<'a> Iterator for EncoderStream<'a> {
    type Item = Vec<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        let lt_pkt = self.lt_encoder.next_packet();
        let samples = self.encoder.encode_packet(&lt_pkt, self.seq);
        self.seq = self.seq.wrapping_add(1);
        Some(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_produces_finite_samples() {
        let data = b"Hello, acoustic air-gap!";
        let mut encoder = Encoder::with_default_config();
        let mut stream = encoder.encode_stream(data);
        let frame = stream.next().unwrap();
        assert!(!frame.is_empty());
        assert!(frame.iter().all(|&s| s.is_finite()));
    }

    #[test]
    fn test_multiple_frames() {
        let data = b"Test data for acoustic transmission!";
        let mut encoder = Encoder::with_default_config();
        let mut stream = encoder.encode_stream(data);
        for i in 0..5 {
            let frame = stream.next();
            assert!(frame.is_some(), "フレーム {} が生成されること", i);
        }
    }

    #[test]
    fn test_44k_encoding() {
        let data = b"Hello 44.1kHz world!";
        let config = EncoderConfig::new(DspConfig::default_44k());
        let mut encoder = Encoder::new(config);
        let mut stream = encoder.encode_stream(data);
        let frame = stream.next().unwrap();
        assert!(!frame.is_empty());
        assert!(frame.iter().all(|&s| s.is_finite()));
    }
}
