//! 送信パイプライン (統合エンコーダ)

use crate::{
    coding::fec,
    coding::fountain::{EncodedPacket, LtEncoder, LtParams},
    coding::interleaver::BlockInterleaver,
    frame::packet::Packet,
    params::PAYLOAD_SIZE,
    phy::modulator::Modulator,
    DspConfig,
};

/// エンコーダ設定
#[derive(Clone)]
pub struct EncoderConfig {
    pub lt_k: usize,
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
            lt_k: 10,
            il_rows: rows,
            il_cols: cols,
            dsp,
        }
    }
}

pub struct Encoder {
    config: EncoderConfig,
    modulator: Modulator,
    interleaver: BlockInterleaver,
}

impl Encoder {
    pub fn new(config: EncoderConfig) -> Self {
        let il = BlockInterleaver::new(config.il_rows, config.il_cols);
        let modulator = Modulator::new(config.dsp.clone());
        Encoder {
            config,
            modulator,
            interleaver: il,
        }
    }

    pub fn encode_packet(&mut self, packet: &EncodedPacket) -> Vec<f32> {
        let pkt = Packet::new(packet.seq as u16, &packet.data);
        let pkt_bytes = pkt.serialize();
        let bits = fec::bytes_to_bits(&pkt_bytes);
        let coded = fec::encode(&bits);

        // インターリーバの全スロットを埋めるようにパディング
        let mut padded = coded;
        padded.resize(self.config.il_rows * self.config.il_cols, 0);

        let interleaved = self.interleaver.interleave(&padded);
        self.modulator.encode_frame(&interleaved)
    }

    pub fn encode_stream<'a>(&'a mut self, data: &'a [u8]) -> EncoderStream<'a> {
        let params = LtParams::new(self.config.lt_k, PAYLOAD_SIZE);
        let lt_encoder = LtEncoder::new(data, params);
        EncoderStream {
            encoder: self,
            lt_encoder,
        }
    }
}

pub struct EncoderStream<'a> {
    encoder: &'a mut Encoder,
    lt_encoder: LtEncoder,
}

impl<'a> Iterator for EncoderStream<'a> {
    type Item = Vec<f32>;
    fn next(&mut self) -> Option<Self::Item> {
        let lt_pkt = self.lt_encoder.next_packet();
        Some(self.encoder.encode_packet(&lt_pkt))
    }
}
