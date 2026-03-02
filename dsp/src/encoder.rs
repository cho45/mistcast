//! 送信パイプライン (統合エンコーダ)

use crate::{
    coding::fec,
    coding::fountain::{FountainEncoder, FountainPacket, FountainParams},
    coding::interleaver::BlockInterleaver,
    frame::packet::{Packet, LT_K_MAX, LT_SEQ_MAX},
    params::{PACKETS_PER_SYNC_BURST, PAYLOAD_SIZE},
    phy::modulator::Modulator,
    DspConfig,
};

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
            packets_per_sync_burst: PACKETS_PER_SYNC_BURST,
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

    fn encode_packet_bits(&mut self, packet: &FountainPacket) -> Vec<u8> {
        let seq = (packet.seq % (u32::from(LT_SEQ_MAX) + 1)) as u16;
        let pkt = Packet::new(seq, self.config.fountain_k, &packet.data);
        let pkt_bytes = pkt.serialize();
        let bits = fec::bytes_to_bits(&pkt_bytes);
        let coded = fec::encode(&bits);

        // インターリーバの全スロットを埋めるようにパディング
        let mut padded = coded;
        padded.resize(self.config.il_rows * self.config.il_cols, 0);

        // ペイロード全体（パディング含む）をスクランブルしてトーンを抑圧
        let mut scrambler = crate::coding::scrambler::Scrambler::default();
        scrambler.process_bits(&mut padded);

        self.interleaver.interleave(&padded)
    }

    pub fn encode_packet(&mut self, packet: &FountainPacket) -> Vec<f32> {
        let interleaved = self.encode_packet_bits(packet);
        self.modulator.encode_frame(&interleaved)
    }

    pub fn encode_burst(&mut self, packets: &[FountainPacket]) -> Vec<f32> {
        let burst_bits_len = self.config.il_rows * self.config.il_cols * packets.len();
        let mut burst_bits = Vec::with_capacity(burst_bits_len);
        for packet in packets {
            let interleaved = self.encode_packet_bits(packet);
            burst_bits.extend_from_slice(&interleaved);
        }
        self.modulator.encode_frame(&burst_bits)
    }

    pub fn flush(&mut self) -> Vec<f32> {
        self.modulator.flush()
    }

    pub fn modulate_silence(&mut self, samples: usize) -> Vec<f32> {
        self.modulator.modulate_silence(samples)
    }

    pub fn reset(&mut self) {
        self.modulator.reset();
    }

    pub fn encode_stream<'a>(&'a mut self, data: &'a [u8]) -> EncoderStream<'a> {
        let params = FountainParams::new(self.config.fountain_k, PAYLOAD_SIZE);
        let fountain_encoder = FountainEncoder::new(data, params);
        EncoderStream {
            encoder: self,
            fountain_encoder,
        }
    }

    pub fn set_fountain_k(&mut self, fountain_k: usize) {
        assert!(
            (1..=LT_K_MAX).contains(&fountain_k),
            "fountain_k must be in 1..={LT_K_MAX}"
        );
        self.config.fountain_k = fountain_k;
    }
}

pub struct EncoderStream<'a> {
    encoder: &'a mut Encoder,
    fountain_encoder: FountainEncoder,
}

impl<'a> Iterator for EncoderStream<'a> {
    type Item = Vec<f32>;
    fn next(&mut self) -> Option<Self::Item> {
        let burst_count = self.encoder.config.packets_per_sync_burst.max(1);
        let mut packets = Vec::with_capacity(burst_count);
        for _ in 0..burst_count {
            packets.push(self.fountain_encoder.next_packet());
        }
        Some(self.encoder.encode_burst(&packets))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_burst_amortizes_sync_overhead() {
        let config = DspConfig::default_48k();
        let mut enc = Encoder::new(EncoderConfig::new(config));
        let p0 = FountainPacket {
            seq: 0,
            coefficients: vec![],
            data: vec![0x11; PAYLOAD_SIZE],
        };
        let p1 = FountainPacket {
            seq: 1,
            coefficients: vec![],
            data: vec![0x22; PAYLOAD_SIZE],
        };
        let p2 = FountainPacket {
            seq: 2,
            coefficients: vec![],
            data: vec![0x33; PAYLOAD_SIZE],
        };

        let single_total = enc.encode_packet(&p0).len()
            + enc.encode_packet(&p1).len()
            + enc.encode_packet(&p2).len();
        let burst_len = enc.encode_burst(&[p0, p1, p2]).len();

        assert!(
            burst_len < single_total,
            "one sync burst should be shorter than three standalone sync frames"
        );
    }
}
