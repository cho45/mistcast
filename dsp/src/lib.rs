//! 音響通信DSP system

pub mod coding;
pub mod common;
pub mod dsss;
pub mod frame;
pub mod mary;

use wasm_bindgen::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DifferentialModulation {
    Dbpsk,
    Dqpsk,
}

impl DifferentialModulation {
    pub const fn bits_per_symbol(self) -> usize {
        match self {
            DifferentialModulation::Dbpsk => 1,
            DifferentialModulation::Dqpsk => 2,
        }
    }
}

pub mod params {
    use super::DifferentialModulation;

    pub const DEFAULT_SAMPLE_RATE: f32 = 48000.0;
    pub const CARRIER_FREQ: f32 = 15000.0;
    pub const MSEQ_ORDER: usize = 4;
    pub const SPREAD_FACTOR: usize = 15;
    pub const RRC_ALPHA: f32 = 0.30;
    pub const CHIP_RATE: f32 = 8000.0;
    pub const PREAMBLE_REPEAT: usize = 2;
    pub const SYNC_WORD_BITS: usize = 8;
    pub const SYNC_WORD: u32 = 0xDEAD_BEEF;
    pub const PREAMBLE_SF: usize = 71;
    pub const PACKETS_PER_SYNC_BURST: usize = 1;
    pub const PAYLOAD_SIZE: usize = 16;
    pub const FIXED_K: usize = 10;
    pub const FOUNTAIN_OVERHEAD: f32 = 0.1;
    pub const MODULATION: DifferentialModulation = DifferentialModulation::Dqpsk;

    /// 内部処理で使用するチップあたりのサンプル数 (Samples Per Chip)
    /// 奇数にすることでチップ中央のサンプルを正確に取得可能にする。
    pub const INTERNAL_SPC: usize = 3;
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
    pub sync_word_bits: usize,
    pub packets_per_burst: usize,
    pub preamble_sf: usize,
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
            sync_word_bits: params::SYNC_WORD_BITS,
            packets_per_burst: params::PACKETS_PER_SYNC_BURST,
            preamble_sf: params::PREAMBLE_SF,
        }
    }

    pub fn default_48k() -> Self {
        Self::new(params::DEFAULT_SAMPLE_RATE)
    }

    pub fn default_44k() -> Self {
        Self::new(44100.0)
    }

    /// 内部ベースバンド処理のサンプリングレート
    pub fn proc_sample_rate(&self) -> f32 {
        self.chip_rate * (params::INTERNAL_SPC as f32)
    }

    /// 内部処理における1チップあたりのサンプル数
    pub fn proc_samples_per_chip(&self) -> usize {
        params::INTERNAL_SPC
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
        self.rrc_taps_per_symbol * self.proc_samples_per_chip() + 1
    }
}

// --- DSSS (Standard) WASM Interface ---

#[wasm_bindgen]
pub struct WasmDsssDecodeProgress {
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
    #[wasm_bindgen(skip)]
    pub basis_matrix: Vec<u8>,
}

#[wasm_bindgen]
impl WasmDsssDecodeProgress {
    #[wasm_bindgen(getter)]
    pub fn basis_matrix(&self) -> js_sys::Uint8Array {
        js_sys::Uint8Array::from(&self.basis_matrix[..])
    }
}

#[wasm_bindgen]
pub struct WasmDsssDecoder {
    inner: dsss::decoder::Decoder,
}

#[wasm_bindgen]
impl WasmDsssDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32, packets_per_burst: usize) -> Self {
        console_error_panic_hook::set_once();
        let mut config = dsss::params::dsp_config(sample_rate);
        config.packets_per_burst = packets_per_burst;
        WasmDsssDecoder {
            inner: dsss::decoder::Decoder::new(
                params::FIXED_K * params::PAYLOAD_SIZE,
                params::FIXED_K,
                config,
            ),
        }
    }
    pub fn process_samples(&mut self, samples: &[f32]) -> WasmDsssDecodeProgress {
        let progress = self.inner.process_samples(samples);
        WasmDsssDecodeProgress {
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
            basis_matrix: progress.basis_matrix,
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
pub struct WasmDsssEncoder {
    inner: dsss::encoder::Encoder,
    fountain_encoder: Option<coding::fountain::FountainEncoder>,
}

#[wasm_bindgen]
impl WasmDsssEncoder {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32, packets_per_burst: usize) -> Self {
        let mut config = dsss::params::dsp_config(sample_rate);
        config.packets_per_burst = packets_per_burst;
        let enc_config = dsss::encoder::EncoderConfig::new(config);
        WasmDsssEncoder {
            inner: dsss::encoder::Encoder::new(enc_config),
            fountain_encoder: None,
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
        self.inner.set_fountain_k(needed_k);
        let params = coding::fountain::FountainParams::new(needed_k, params::PAYLOAD_SIZE);
        self.fountain_encoder = Some(coding::fountain::FountainEncoder::new(data, params));
    }
    pub fn pull_frame(&mut self) -> Option<Vec<f32>> {
        let encoder = self.fountain_encoder.as_mut()?;
        let mut packets = Vec::with_capacity(self.inner.config().packets_per_sync_burst);
        for _ in 0..self.inner.config().packets_per_sync_burst {
            packets.push(encoder.next_packet());
        }
        Some(self.inner.encode_burst(&packets))
    }
    pub fn pull_frame_with_seq(&mut self, seq: u32) -> Option<Vec<f32>> {
        let encoder = self.fountain_encoder.as_ref()?;
        let burst_size = self.inner.config().packets_per_sync_burst;
        let mut packets = Vec::with_capacity(burst_size);
        for i in 0..burst_size {
            packets.push(encoder.get_packet(seq.wrapping_add(i as u32)));
        }
        Some(self.inner.encode_burst(&packets))
    }
    pub fn flush(&mut self) -> Vec<f32> {
        self.inner.flush()
    }
    pub fn modulate_silence(&mut self, samples: usize) -> Vec<f32> {
        self.inner.modulate_silence(samples)
    }
    pub fn reset(&mut self) {
        self.inner.reset();
        self.fountain_encoder = None;
    }
}

// --- Mary (Robust) WASM Interface ---

#[wasm_bindgen]
pub struct WasmMaryDecodeProgress {
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
    #[wasm_bindgen(skip)]
    pub basis_matrix: Vec<u8>,
}

#[wasm_bindgen]
impl WasmMaryDecodeProgress {
    #[wasm_bindgen(getter)]
    pub fn basis_matrix(&self) -> js_sys::Uint8Array {
        js_sys::Uint8Array::from(&self.basis_matrix[..])
    }
}

#[wasm_bindgen]
pub struct WasmMaryDecoder {
    inner: mary::decoder::Decoder,
}

#[wasm_bindgen]
impl WasmMaryDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32, packets_per_burst: usize) -> Self {
        console_error_panic_hook::set_once();
        let mut config = DspConfig::new(sample_rate);
        config.packets_per_burst = packets_per_burst;
        WasmMaryDecoder {
            inner: mary::decoder::Decoder::new(
                params::FIXED_K * params::PAYLOAD_SIZE,
                params::FIXED_K,
                config,
            ),
        }
    }
    pub fn process_samples(&mut self, samples: &[f32]) -> WasmMaryDecodeProgress {
        let progress = self.inner.process_samples(samples);
        WasmMaryDecodeProgress {
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
            basis_matrix: progress.basis_matrix,
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
pub struct WasmMaryEncoder {
    inner: mary::encoder::Encoder,
}

#[wasm_bindgen]
impl WasmMaryEncoder {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32, packets_per_burst: usize) -> Self {
        let mut config = DspConfig::new(sample_rate);
        config.packets_per_burst = packets_per_burst;
        WasmMaryEncoder {
            inner: mary::encoder::Encoder::new(config),
        }
    }
    pub fn set_data(&mut self, data: &[u8]) {
        self.inner.set_data(data);
    }
    pub fn pull_frame(&mut self) -> Option<Vec<f32>> {
        self.inner.encode_frame()
    }
    pub fn pull_frame_with_seq(&mut self, seq: u32) -> Option<Vec<f32>> {
        let encoder = self.inner.fountain_encoder()?;
        let burst_size = self.inner.config().packets_per_sync_burst;
        let mut packets = Vec::with_capacity(burst_size);
        for i in 0..burst_size {
            packets.push(encoder.get_packet(seq.wrapping_add(i as u32)));
        }
        Some(self.inner.encode_burst(&packets))
    }
    pub fn flush(&mut self) -> Vec<f32> {
        self.inner.flush()
    }
    pub fn modulate_silence(&mut self, samples: usize) -> Vec<f32> {
        self.inner.modulate_silence(samples)
    }
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}
