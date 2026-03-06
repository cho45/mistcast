//! Mary モジュール固有のパラメータ定義

use crate::DspConfig;

const MARY_MSEQ_ORDER: usize = crate::params::MSEQ_ORDER;
const MARY_PREAMBLE_REPEAT: usize = crate::params::PREAMBLE_REPEAT;
const MARY_PREAMBLE_SF: usize = crate::params::PREAMBLE_SF;
const MARY_SYNC_WORD_BITS: usize = crate::params::SYNC_WORD_BITS;

/// Mary 同期語の拡散率（Walsh[0], DBPSK）
pub const SYNC_SPREAD_FACTOR: usize = 16;

/// Mary ペイロードの拡散率（Walsh[0..15], DQPSK）
pub const PAYLOAD_SPREAD_FACTOR: usize = 16;

/// Mary 用の `DspConfig` を作成する
pub fn dsp_config(sample_rate: f32) -> DspConfig {
    let mut config = DspConfig::new(sample_rate);
    config.mseq_order = MARY_MSEQ_ORDER;
    config.preamble_repeat = MARY_PREAMBLE_REPEAT;
    config.preamble_sf = MARY_PREAMBLE_SF;
    config.sync_word_bits = MARY_SYNC_WORD_BITS;
    config
}

/// 48kHz 用の Mary `DspConfig` を作成する
pub fn dsp_config_48k() -> DspConfig {
    dsp_config(crate::params::DEFAULT_SAMPLE_RATE)
}

/// 44.1kHz 用の Mary `DspConfig` を作成する
pub fn dsp_config_44k() -> DspConfig {
    dsp_config(44100.0)
}
