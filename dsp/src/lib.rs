//! 音響通信DSPシステム

pub mod coding;
pub mod common;
pub mod frame;
pub mod phy;

pub mod decoder;
pub mod encoder;

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
    pub const CARRIER_FREQ: f32 = 12000.0;
    pub const MSEQ_ORDER: usize = 4;
    pub const SPREAD_FACTOR: usize = 15;
    pub const RRC_ALPHA: f32 = 0.30;
    pub const CHIP_RATE: f32 = 12000.0;
    pub const PREAMBLE_REPEAT: usize = 4;
    pub const SYNC_WORD_BITS: usize = 32;
    pub const SYNC_WORD: u32 = 0xDEAD_BEEF;
    pub const PACKETS_PER_SYNC_BURST: usize = 2;
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
        }
    }

    /// 指定された設定を基に、内部処理用の設定を生成する。
    /// サンプルレートは chip_rate * INTERNAL_SPC に固定され、
    /// ベースバンド処理を前提とするため carrier_freq は 0 となる。
    pub fn new_for_processing(chip_rate: f32) -> Self {
        let mut config = DspConfig::new(params::DEFAULT_SAMPLE_RATE);
        config.chip_rate = chip_rate;
        Self::new_for_processing_from(&config)
    }

    pub fn new_for_processing_from(base: &DspConfig) -> Self {
        let sample_rate = base.chip_rate * (params::INTERNAL_SPC as f32);
        DspConfig {
            sample_rate,
            carrier_freq: 0.0, // ベースバンド
            mseq_order: base.mseq_order,
            chip_rate: base.chip_rate,
            rrc_alpha: base.rrc_alpha,
            rrc_taps_per_symbol: base.rrc_taps_per_symbol,
            preamble_repeat: base.preamble_repeat,
            sync_word_bits: base.sync_word_bits,
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
    #[wasm_bindgen(skip)]
    pub basis_matrix: Vec<u8>,
}

#[wasm_bindgen]
impl WasmDecodeProgress {
    #[wasm_bindgen(getter)]
    pub fn basis_matrix(&self) -> js_sys::Uint8Array {
        js_sys::Uint8Array::from(&self.basis_matrix[..])
    }
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
pub struct WasmEncoder {
    inner: encoder::Encoder,
    fountain_encoder: Option<coding::fountain::FountainEncoder>,
}

#[wasm_bindgen]
impl WasmEncoder {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> Self {
        let config = DspConfig::new(sample_rate);
        let enc_config = encoder::EncoderConfig::new(config);
        WasmEncoder {
            inner: encoder::Encoder::new(enc_config),
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
        let burst_count = params::PACKETS_PER_SYNC_BURST.max(1);
        let mut packets = Vec::with_capacity(burst_count);
        let encoder = self.fountain_encoder.as_mut()?;
        for _ in 0..burst_count {
            packets.push(encoder.next_packet());
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
