//! 音響通信DSPシステム
//!
//! エアギャップ環境での音響通信を実現するDSPコアライブラリ。
//! DSSS + DBPSK 変調、RRCフィルタ、FEC、LT符号(Fountain Code)を実装する。
//!
//! # WASM Interface
//! この `lib.rs` は、JavaScript/WASM に対する唯一の公式インターフェースを提供します。
//! 他の内部モジュールは純粋な Rust 実装として保たれ、WASM 境界の型変換や
//! ラッパー（WasmDecoder, WasmEncoder）はこのファイルに集約されます。

use wasm_bindgen::prelude::*;

pub mod common;
pub mod phy;
pub mod coding;
pub mod frame;

pub mod decoder;
pub mod encoder;

/// システムデフォルト定数
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
    pub const FOUNTAIN_OVERHEAD: f32 = 0.1;
}

/// DSP動作設定
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

    #[inline]
    pub fn samples_per_chip(&self) -> usize {
        (self.sample_rate / self.chip_rate) as usize
    }

    #[inline]
    pub fn spread_factor(&self) -> usize {
        (1 << self.mseq_order) - 1
    }

    #[inline]
    pub fn samples_per_symbol(&self) -> usize {
        self.samples_per_chip() * self.spread_factor()
    }

    #[inline]
    pub fn rrc_num_taps(&self) -> usize {
        self.rrc_taps_per_symbol * self.samples_per_chip() + 1
    }
}

impl Default for DspConfig {
    fn default() -> Self {
        Self::default_48k()
    }
}

// --- WASM Interface Layer ---

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct WasmDecodeProgress {
    pub received_packets: usize,
    pub needed_packets: usize,
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
    pub fn new(data_size: usize, lt_k: usize, sample_rate: f32) -> Self {
        let config = DspConfig::new(sample_rate);
        WasmDecoder {
            inner: decoder::Decoder::new(data_size, lt_k, config),
        }
    }

    pub fn process_samples(&mut self, samples: &[f32]) -> WasmDecodeProgress {
        let progress = self.inner.process_samples(samples);
        WasmDecodeProgress {
            received_packets: progress.received_packets,
            needed_packets: progress.needed_packets,
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
}

#[wasm_bindgen]
impl WasmEncoder {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> Self {
        let config = DspConfig::new(sample_rate);
        let enc_config = encoder::EncoderConfig::new(config);
        WasmEncoder {
            inner: encoder::Encoder::new(enc_config),
        }
    }

    /// 一度にすべてのサンプルを生成する (簡易版)
    pub fn encode_all(&mut self, data: &[u8]) -> Vec<f32> {
        let mut stream = self.inner.encode_stream(data);
        let mut all_samples = Vec::new();
        // LT符号なので十分な数（K*2程度）を生成
        for _ in 0..16 {
            if let Some(frame) = stream.next() {
                all_samples.extend(frame);
                // ギャップを追加
                all_samples.extend(vec![0.0; 4800]);
            }
        }
        all_samples
    }
}
