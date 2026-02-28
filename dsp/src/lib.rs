//! 音響通信DSPシステム

pub mod coding;
pub mod common;
pub mod frame;
pub mod phy;

pub mod decoder;
pub mod encoder;

use wasm_bindgen::prelude::*;

pub mod params {
    pub const DEFAULT_SAMPLE_RATE: f32 = 48000.0;
    pub const CARRIER_FREQ: f32 = 8000.0;
    pub const MSEQ_ORDER: usize = 5;
    pub const SPREAD_FACTOR: usize = 31;
    pub const RRC_ALPHA: f32 = 0.35;
    pub const CHIP_RATE: f32 = 8000.0;
    pub const PREAMBLE_REPEAT: usize = 4;
    pub const SYNC_WORD: u32 = 0xDEAD_BEEF;
    pub const PAYLOAD_SIZE: usize = 16;
    pub const FIXED_K: usize = 10;
    pub const FOUNTAIN_OVERHEAD: f32 = 0.1;
}

#[derive(Clone, Debug)]
pub struct DspConfig {
    pub sample_rate: f32,
    pub carrier_freq: f32,
    pub mseq_order: usize,
    pub chip_rate: f32,
    pub rrc_alpha: f32,
    pub rrc_taps_per_symbol: usize,
    pub preamble_repeat: usize,
}

impl DspConfig {
    pub fn new(sample_rate: f32) -> Self {
        DspConfig {
            sample_rate,
            carrier_freq: params::CARRIER_FREQ,
            mseq_order: params::MSEQ_ORDER,
            chip_rate: params::CHIP_RATE,
            rrc_alpha: params::RRC_ALPHA,
            rrc_taps_per_symbol: 16,
            preamble_repeat: params::PREAMBLE_REPEAT,
        }
    }
    pub fn default_48k() -> Self {
        Self::new(params::DEFAULT_SAMPLE_RATE)
    }
    pub fn default_44k() -> Self {
        Self::new(44100.0)
    }
    pub fn samples_per_chip(&self) -> usize {
        (self.sample_rate / self.chip_rate) as usize
    }
    pub fn spread_factor(&self) -> usize {
        (1 << self.mseq_order) - 1
    }
    pub fn samples_per_symbol(&self) -> usize {
        self.samples_per_chip() * self.spread_factor()
    }
    pub fn rrc_num_taps(&self) -> usize {
        self.rrc_taps_per_symbol * self.samples_per_chip() + 1
    }
}

#[wasm_bindgen]
pub struct WasmDecodeProgress {
    pub received_packets: usize,
    pub needed_packets: usize,
    pub rank_packets: usize,
    pub stalled_packets: usize,
    pub dependent_packets: usize,
    pub duplicate_packets: usize,
    pub crc_error_packets: usize,
    pub parse_error_packets: usize,
    pub invalid_neighbor_packets: usize,
    pub last_packet_seq: i32,
    pub last_rank_up_seq: i32,
    pub progress: f32,
    pub complete: bool,
}

#[wasm_bindgen]
pub struct WasmDecoder {
    inner: decoder::Decoder,
}

#[wasm_bindgen]
impl WasmDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> Self {
        console_error_panic_hook::set_once();
        let config = DspConfig::new(sample_rate);
        WasmDecoder {
            inner: decoder::Decoder::new(
                params::FIXED_K * params::PAYLOAD_SIZE,
                params::FIXED_K,
                config,
            ),
        }
    }
    pub fn process_samples(&mut self, samples: &[f32]) -> WasmDecodeProgress {
        let progress = self.inner.process_samples(samples);
        WasmDecodeProgress {
            received_packets: progress.received_packets,
            needed_packets: progress.needed_packets,
            rank_packets: progress.rank_packets,
            stalled_packets: progress.stalled_packets,
            dependent_packets: progress.dependent_packets,
            duplicate_packets: progress.duplicate_packets,
            crc_error_packets: progress.crc_error_packets,
            parse_error_packets: progress.parse_error_packets,
            invalid_neighbor_packets: progress.invalid_neighbor_packets,
            last_packet_seq: progress.last_packet_seq,
            last_rank_up_seq: progress.last_rank_up_seq,
            progress: progress.progress,
            complete: progress.complete,
        }
    }
    pub fn recovered_data(&self) -> Option<Vec<u8>> {
        self.inner.recovered_data().map(|d| d.to_vec())
    }
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

#[wasm_bindgen]
pub struct WasmEncoder {
    inner: encoder::Encoder,
    lt_encoder: Option<coding::fountain::LtEncoder>,
}

#[wasm_bindgen]
impl WasmEncoder {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> Self {
        let config = DspConfig::new(sample_rate);
        let enc_config = encoder::EncoderConfig::new(config);
        WasmEncoder {
            inner: encoder::Encoder::new(enc_config),
            lt_encoder: None,
        }
    }
    pub fn set_data(&mut self, data: &[u8]) {
        let max_k = crate::frame::packet::LT_K_MAX;
        let needed_k = data.len().div_ceil(params::PAYLOAD_SIZE).max(1);
        assert!(
            needed_k <= max_k,
            "input is too large for current packet header: need k={}, max k={}",
            needed_k,
            max_k
        );
        self.inner.set_lt_k(needed_k);
        let params = coding::fountain::LtParams::new(needed_k, params::PAYLOAD_SIZE);
        self.lt_encoder = Some(coding::fountain::LtEncoder::new(data, params));
    }
    pub fn pull_frame(&mut self) -> Option<Vec<f32>> {
        let lt_pkt = self.lt_encoder.as_mut()?.next_packet();
        Some(self.inner.encode_packet(&lt_pkt))
    }
}
