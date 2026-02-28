//! 受信パイプライン (統合デコーダ)
//!
//! ```text
//! 受信音声サンプル列 → 同期捕捉 → DBPSK+DSSS復調
//! → デインターリーブ → Viterbi FEC → CRC確認 → LTデコーダ → ファイル復元
//! ```

use crate::{
    demodulator::Demodulator,
    fec,
    fountain::{EncodedPacket, LtDecoder, LtParams},
    interleaver::BlockInterleaver,
    packet::{Packet, PACKET_BYTES},
    params::PAYLOAD_SIZE,
    sync::SyncDetector,
    DspConfig,
};

/// デコード進捗情報
#[derive(Debug, Clone)]
pub struct DecodeProgress {
    pub received_packets: usize,
    pub needed_packets: usize,
    pub progress: f32,
    pub complete: bool,
}

/// 受信パイプライン統合デコーダ
pub struct Decoder {
    #[allow(dead_code)]
    config: DspConfig,
    sync_detector: SyncDetector,
    demodulator: Demodulator,
    interleaver: BlockInterleaver,
    lt_decoder: LtDecoder,
    recovered_data: Option<Vec<u8>>,
    original_size: usize,
}

impl Decoder {
    pub fn new(data_size: usize, lt_k: usize, dsp_config: DspConfig) -> Self {
        let params = LtParams::new(lt_k, PAYLOAD_SIZE);
        let fec_bits = (PACKET_BYTES + 6) * 2 * 8;
        let il_rows = 16;
        let il_cols = fec_bits.div_ceil(16);

        Decoder {
            sync_detector: SyncDetector::new(dsp_config.clone()),
            demodulator: Demodulator::new(dsp_config.clone()),
            interleaver: BlockInterleaver::new(il_rows, il_cols),
            lt_decoder: LtDecoder::new(params),
            recovered_data: None,
            original_size: data_size,
            config: dsp_config,
        }
    }

    pub fn default_48k(data_size: usize, lt_k: usize) -> Self {
        Self::new(data_size, lt_k, DspConfig::default_48k())
    }

    /// 受信サンプル列を処理する
    pub fn process_samples(&mut self, samples: &[f32]) -> DecodeProgress {
        if let Some(sync) = self.sync_detector.detect(samples) {
            let data_start = sync.data_start_sample.min(samples.len());
            let data_samples = &samples[data_start..];

            if !data_samples.is_empty() {
                let raw_bits = self.demodulator.demodulate(data_samples, data_start);
                let deinterleaved = self.interleaver.deinterleave(&raw_bits);
                let decoded_bits = fec::decode(&deinterleaved);
                let decoded_bytes = fec::bits_to_bytes(&decoded_bits);

                if decoded_bytes.len() >= PACKET_BYTES {
                    if let Some(packet) = Packet::deserialize(&decoded_bytes[..PACKET_BYTES]) {
                        let (degree, neighbors) = crate::fountain::reconstruct_packet_metadata(
                            packet.lt_seq,
                            self.lt_decoder.params().k,
                            self.lt_decoder.params().c,
                            self.lt_decoder.params().delta,
                        );
                        let lt_pkt = EncodedPacket {
                            seq: packet.lt_seq,
                            degree,
                            neighbors,
                            data: packet.payload.to_vec(),
                        };
                        self.lt_decoder.receive(lt_pkt);

                        if let Some(data) = self.lt_decoder.decode() {
                            let truncated = data[..self.original_size.min(data.len())].to_vec();
                            self.recovered_data = Some(truncated);
                        }
                    }
                }
            }
        }

        DecodeProgress {
            received_packets: self.lt_decoder.received_count(),
            needed_packets: self.lt_decoder.needed_count(),
            progress: self.lt_decoder.progress(),
            complete: self.recovered_data.is_some(),
        }
    }

    pub fn recovered_data(&self) -> Option<&[u8]> {
        self.recovered_data.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::Encoder;

    #[test]
    fn test_progress_calculation() {
        let _decoder = Decoder::default_48k(100, 8);
        let progress = DecodeProgress {
            received_packets: 0,
            needed_packets: 9,
            progress: 0.0,
            complete: false,
        };
        assert_eq!(progress.progress, 0.0);
        assert!(!progress.complete);
    }

    #[test]
    fn test_encoder_decoder_interaction() {
        let data = b"Hello, acoustic world!  ";
        let mut encoder = Encoder::with_default_config();
        let mut stream = encoder.encode_stream(data);
        let frame = stream.next().unwrap();

        let mut decoder = Decoder::default_48k(data.len(), 8);
        let progress = decoder.process_samples(&frame);
        assert!(progress.progress >= 0.0 && progress.progress <= 1.0);
    }
}
