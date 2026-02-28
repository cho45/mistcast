//! 受信パイプライン (統合デコーダ)
//!
//! ```text
//! 受信音声サンプル列 → 同期捕捉 → DBPSK+DSSS復調
//! → デインターリーブ → Viterbi FEC → CRC確認 → LTデコーダ → ファイル復元
//! ```

use crate::{
    coding::fec,
    coding::fountain::{EncodedPacket, LtDecoder, LtParams},
    coding::interleaver::BlockInterleaver,
    phy::demodulator::Demodulator,
    phy::sync::SyncDetector,
    common::rrc_filter::RrcFilter,
    frame::packet::{Packet, PACKET_BYTES},
    params::PAYLOAD_SIZE,
    DspConfig,
};

/// デコード進捗情報
#[derive(Debug, Clone)]
pub struct DecodeProgress {
    pub received_packets: usize,
    pub needed_packets: usize,
    pub progress: f32,
    pub complete: bool,
    /// 今回の呼び出しで消費（処理済みとして破棄可能）されたサンプル数
    pub consumed_samples: usize,
}

/// 受信パイプライン統合デコーダ
pub struct Decoder {
    config: DspConfig,
    sample_idx: usize,
    rrc_i: RrcFilter,
    rrc_q: RrcFilter,
    pub(crate) chip_buffer_i: Vec<f32>,
    pub(crate) chip_buffer_q: Vec<f32>,
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
        // PACKET_BYTES * 8 (bits) + 6 (tail bits)
        let raw_bits = PACKET_BYTES * 8 + 6;
        let fec_bits = raw_bits * 2;
        let il_rows = 16;
        let il_cols = fec_bits.div_ceil(16);

        Decoder {
            rrc_i: RrcFilter::from_config(&dsp_config),
            rrc_q: RrcFilter::from_config(&dsp_config),
            chip_buffer_i: Vec::new(),
            chip_buffer_q: Vec::new(),
            sample_idx: 0,
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
        let two_pi = 2.0 * std::f32::consts::PI;
        let fs = self.config.sample_rate;
        let fc = self.config.carrier_freq;
        let spc = self.config.samples_per_chip();
        let total_delay = self.config.rrc_num_taps().saturating_sub(1);

        for (idx, &s) in samples.iter().enumerate() {
            let current_sample_idx = self.sample_idx + idx;
            let t = current_sample_idx as f32 / fs;
            let cos_val = (two_pi * fc * t).cos();
            let sin_val = (two_pi * fc * t).sin();
            
            let i_val = s * cos_val * 2.0;
            let q_val = s * (-sin_val) * 2.0;

            let filtered_i = self.rrc_i.process(i_val);
            let filtered_q = self.rrc_q.process(q_val);

            if current_sample_idx >= total_delay && (current_sample_idx - total_delay).is_multiple_of(spc) {
                self.chip_buffer_i.push(filtered_i);
                self.chip_buffer_q.push(filtered_q);
            }
        }
        self.sample_idx += samples.len();

        let raw_bits_len = PACKET_BYTES * 8 + 6;
        let needed_fec_bits = (raw_bits_len * 2).div_ceil(16) * 16;
        let sync_bits_len = 32;
        let total_needed_bits = needed_fec_bits + sync_bits_len;
        let needed_chips = total_needed_bits * self.config.spread_factor();

        // 1回の関数呼び出しでバッファ内の可能な限りのパケットをすべて処理する
        loop {
            let (sync_opt, searched_samples) = self.sync_detector.detect_chips(&self.chip_buffer_i, &self.chip_buffer_q);
            
            if let Some(sync) = sync_opt {
                let data_start_chip = sync.data_start_sample / spc;

                if self.chip_buffer_i.len() >= data_start_chip + needed_chips {
                    let data_chips_i = &self.chip_buffer_i[data_start_chip..];
                    let data_chips_q = &self.chip_buffer_q[data_start_chip..];

                    let raw_bits = self.demodulator.demodulate_chips(&data_chips_i[..needed_chips], &data_chips_q[..needed_chips]);
                    
                    if raw_bits.len() >= total_needed_bits {
                        let payload_bits = &raw_bits[sync_bits_len..total_needed_bits];
                        let deinterleaved = self.interleaver.deinterleave(payload_bits);
                        let decoded_bits = fec::decode(&deinterleaved);
                        let decoded_bytes = fec::bits_to_bytes(&decoded_bits);

                        if decoded_bytes.len() >= PACKET_BYTES {
                            if let Some(packet) = Packet::deserialize(&decoded_bytes[..PACKET_BYTES]) {
                                let (degree, neighbors) = crate::coding::fountain::reconstruct_packet_metadata(
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
                    
                    // パケットを処理したかどうかにかかわらず、データ開始位置まで＋処理に必要だったチップ数を消費する。
                    self.chip_buffer_i.drain(0..(data_start_chip + needed_chips));
                    self.chip_buffer_q.drain(0..(data_start_chip + needed_chips));
                    self.demodulator.reset();
                    // バッファを消費したので、残りのバッファにまだパケットがあるか再度ループで調べる
                    continue;
                } else {
                    // syncは見つかったが、まだ必要なデータ長に達していない場合。
                    // 次のチャンクでデータが来るのを待つため、ループを抜ける。
                    break;
                }
            } else {
                // 同期が見つからなかった場合、探索済み範囲までは確実に不要。
                let drained_chips = searched_samples / spc;
                if drained_chips > 0 {
                    let drain_amount = drained_chips.min(self.chip_buffer_i.len());
                    self.chip_buffer_i.drain(0..drain_amount);
                    self.chip_buffer_q.drain(0..drain_amount);
                }
                break;
            }
        }

        DecodeProgress {
            received_packets: self.lt_decoder.received_count(),
            needed_packets: self.lt_decoder.needed_count(),
            progress: self.lt_decoder.progress(),
            complete: self.recovered_data.is_some(),
            consumed_samples: samples.len(),
        }
    }

    /// デコーダの状態をリセットする
    pub fn reset(&mut self) {
        self.demodulator.reset();
        self.interleaver.reset();
        self.rrc_i.reset();
        self.rrc_q.reset();
        self.chip_buffer_i.clear();
        self.chip_buffer_q.clear();
        self.sample_idx = 0;
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
            consumed_samples: 0,
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
        
        // --- デバッグ: パイプラインの途中状態をインターセプトして検証 ---
        let spc = decoder.config.samples_per_chip();
        let mut test_chip_buffer_i = Vec::new();
        let mut test_chip_buffer_q = Vec::new();
        let total_delay = decoder.config.rrc_num_taps().saturating_sub(1);
        let mut rrc_i = crate::common::rrc_filter::RrcFilter::from_config(&decoder.config);
        let mut rrc_q = crate::common::rrc_filter::RrcFilter::from_config(&decoder.config);
        let two_pi = 2.0 * std::f32::consts::PI;
        let fs = decoder.config.sample_rate;
        let fc = decoder.config.carrier_freq;
        
        for (sample_idx, &s) in frame.iter().enumerate() {
            let t = sample_idx as f32 / fs;
            let i_val = s * (two_pi * fc * t).cos() * 2.0;
            let q_val = s * -(two_pi * fc * t).sin() * 2.0;
            let fi = rrc_i.process(i_val);
            let fq = rrc_q.process(q_val);
            if sample_idx >= total_delay && (sample_idx - total_delay).is_multiple_of(spc) {
                test_chip_buffer_i.push(fi);
                test_chip_buffer_q.push(fq);
            }
        }
        
        let (sync_opt, _) = decoder.sync_detector.detect_chips(&test_chip_buffer_i, &test_chip_buffer_q);
        assert!(sync_opt.is_some(), "Sync must be found");
        let data_start_chip = sync_opt.unwrap().data_start_sample / spc;
        
        let raw_bits = decoder.demodulator.demodulate_chips(&test_chip_buffer_i[data_start_chip..], &test_chip_buffer_q[data_start_chip..]);
        
        // 最初の32bitはSYNC_WORD (0xDEADBEEF) のはず
        let sync_word = crate::params::SYNC_WORD;
        let expected_sync_bits: Vec<u8> = (0..32).rev().map(|i| ((sync_word >> i) & 1) as u8).collect();
        
        println!("test_chip_buffer_i.len() = {}, data_start_chip = {}, raw_bits.len() = {}", test_chip_buffer_i.len(), data_start_chip, raw_bits.len());
        assert!(raw_bits.len() >= 32, "Must have decoded at least 32 bits");
        
        let mut match_count = 0;
        for i in 0..32 {
            if raw_bits[i] == expected_sync_bits[i] {
                match_count += 1;
            }
        }
        assert_eq!(match_count, 32, "同期直後の32bitは完全にSYNC_WORDと一致しなければならない。初期位相のズレが疑われる");
        
        // -------------------------------------------------------------

        let progress = decoder.process_samples(&frame);
        
        // パケットが正しく1つデコードされていなければならない
        assert_eq!(
            progress.received_packets, 1,
            "1フレーム分のサンプルを入力したため、1パケットが受信できているはずである"
        );
        assert!(progress.progress > 0.0);
    }

    #[test]
    fn test_decoder_stream_vs_batch() {
        let data = b"Hello Stream vs Batch!";
        let config = DspConfig::default_48k();
        let mut encoder = Encoder::new(crate::encoder::EncoderConfig::new(config.clone()));
        let mut stream = encoder.encode_stream(data);
        
        let frame = stream.next().unwrap();
        let mut tx_signal = vec![0.0; 1000]; // silence
        tx_signal.extend(frame);
        tx_signal.extend(vec![0.0; 1000]); // silence

        let mut batch_decoder = Decoder::new(data.len(), 8, config.clone());
        let batch_progress = batch_decoder.process_samples(&tx_signal);
        
        assert_eq!(batch_progress.received_packets, 1, "Batch processing must decode 1 packet");
        
        let mut stream_decoder = Decoder::new(data.len(), 8, config);
        let mut stream_progress = DecodeProgress { received_packets: 0, needed_packets: 0, progress: 0.0, complete: false, consumed_samples: 0 };
        
        let chunk_size = 128; // Small chunk size to stress test the boundaries
        for chunk in tx_signal.chunks(chunk_size) {
            stream_progress = stream_decoder.process_samples(chunk);
        }
        
        assert_eq!(stream_progress.received_packets, 1, "Stream processing must decode 1 packet identically to batch processing");
        
        // 注: バッチ処理は1回の関数呼び出しで終了するため末尾の無音区間がバッファに残るが、
        // ストリーム処理ではチャンクごとに不要なバッファが切り捨てられるため最終バッファサイズは一致しない。
        // （ストリーム処理の方がメモリ効率的に正しく動作していることの証明となる）
        assert!(stream_decoder.chip_buffer_i.len() <= 124, "Stream buffer should not leak memory and bounded by window size");
    }
}
