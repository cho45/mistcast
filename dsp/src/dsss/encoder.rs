//! 送信パイプライン (統合エンコーダ)

use crate::{
    coding::fec,
    coding::fountain::{FountainEncoder, FountainPacket, FountainParams},
    coding::interleaver::BlockInterleaver,
    dsss::modulator::Modulator,
    frame::packet::{Packet, LT_K_MAX, LT_SEQ_MAX, PACKET_BYTES},
    params::PAYLOAD_SIZE,
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
    interleaver: BlockInterleaver,
    raw_bits_buffer: Vec<u8>,
    fec_bits_buffer: Vec<u8>,
    padded_bits_buffer: Vec<u8>,
    interleaved_bits_buffer: Vec<u8>,
    burst_bits_buffer: Vec<u8>,
}

impl Encoder {
    pub fn new(config: EncoderConfig) -> Self {
        let il = BlockInterleaver::new(config.il_rows, config.il_cols);
        let modulator = Modulator::new(config.dsp.clone());
        let packets_per_sync_burst = config.packets_per_sync_burst.max(1);
        let interleaved_bits_per_packet = config.il_rows * config.il_cols;
        let burst_bits_capacity = config.il_rows * config.il_cols * packets_per_sync_burst;
        let raw_bits_capacity = PACKET_BYTES * 8;
        let fec_bits_capacity = (raw_bits_capacity + 6) * 2;
        Encoder {
            config,
            modulator,
            interleaver: il,
            raw_bits_buffer: Vec::with_capacity(raw_bits_capacity),
            fec_bits_buffer: Vec::with_capacity(fec_bits_capacity),
            padded_bits_buffer: Vec::with_capacity(interleaved_bits_per_packet),
            interleaved_bits_buffer: Vec::with_capacity(interleaved_bits_per_packet),
            burst_bits_buffer: Vec::with_capacity(burst_bits_capacity),
        }
    }

    fn fill_packet_bits_buffer(&mut self, packet: &FountainPacket) {
        let seq = (packet.seq % (u32::from(LT_SEQ_MAX) + 1)) as u16;
        let pkt = Packet::new(seq, self.config.fountain_k, &packet.data);
        let pkt_bytes = pkt.serialize();
        fec::bytes_to_bits_into(&pkt_bytes, &mut self.raw_bits_buffer);
        fec::encode_into(&self.raw_bits_buffer, &mut self.fec_bits_buffer);

        // インターリーバの全スロットを埋めるようにパディング
        let interleaved_bits_per_packet = self.config.il_rows * self.config.il_cols;
        self.padded_bits_buffer.clear();
        self.padded_bits_buffer
            .extend_from_slice(&self.fec_bits_buffer);
        self.padded_bits_buffer
            .resize(interleaved_bits_per_packet, 0);

        // ペイロード全体（パディング含む）をスクランブルしてトーンを抑圧
        let mut scrambler = crate::coding::scrambler::Scrambler::default();
        scrambler.process_bits(&mut self.padded_bits_buffer);

        self.interleaved_bits_buffer
            .resize(interleaved_bits_per_packet, 0);
        self.interleaver.interleave_in_place(
            &self.padded_bits_buffer,
            &mut self.interleaved_bits_buffer[..interleaved_bits_per_packet],
        );
    }

    pub fn encode_packet(&mut self, packet: &FountainPacket) -> Vec<f32> {
        self.fill_packet_bits_buffer(packet);
        self.modulator.encode_frame(&self.interleaved_bits_buffer)
    }

    pub fn encode_burst(&mut self, packets: &[FountainPacket]) -> Vec<f32> {
        let mut out = Vec::new();
        self.encode_burst_into(packets, &mut out);
        out
    }

    pub fn encode_burst_into(&mut self, packets: &[FountainPacket], out: &mut Vec<f32>) {
        let interleaved_bits_per_packet = self.config.il_rows * self.config.il_cols;
        self.burst_bits_buffer.clear();
        for packet in packets {
            self.fill_packet_bits_buffer(packet);
            self.burst_bits_buffer
                .extend_from_slice(&self.interleaved_bits_buffer[..interleaved_bits_per_packet]);
        }
        self.modulator
            .encode_frame_into(&self.burst_bits_buffer, out);
    }

    pub fn flush(&mut self) -> Vec<f32> {
        let mut out = Vec::new();
        self.flush_into(&mut out);
        out
    }

    pub fn flush_into(&mut self, out: &mut Vec<f32>) {
        self.modulator.flush_into(out);
    }

    pub fn modulate_silence(&mut self, samples: usize) -> Vec<f32> {
        let mut out = Vec::new();
        self.modulate_silence_into(samples, &mut out);
        out
    }

    pub fn modulate_silence_into(&mut self, samples: usize, out: &mut Vec<f32>) {
        self.modulator.modulate_silence_into(samples, out);
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

    pub fn config(&self) -> &EncoderConfig {
        &self.config
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
        let config = crate::dsss::params::dsp_config_48k();
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
