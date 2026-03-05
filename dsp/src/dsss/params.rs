//! DSSS モジュール固有のパラメータ定義

use crate::DspConfig;

const DSSS_MSEQ_ORDER: usize = 4;
const DSSS_PREAMBLE_REPEAT: usize = 2;
const DSSS_PREAMBLE_SF: usize = 15;
const DSSS_SYNC_WORD_BITS: usize = 16;

/// DSSS 用の DspConfig を作成する
pub fn dsp_config(sample_rate: f32) -> DspConfig {
    let mut config = DspConfig::new(sample_rate);
    config.mseq_order = DSSS_MSEQ_ORDER;
    config.preamble_repeat = DSSS_PREAMBLE_REPEAT;
    config.preamble_sf = DSSS_PREAMBLE_SF;
    config.sync_word_bits = DSSS_SYNC_WORD_BITS;
    config
}

/// 48kHz 用の DSSS DspConfig を作成する
pub fn dsp_config_48k() -> DspConfig {
    dsp_config(crate::params::DEFAULT_SAMPLE_RATE)
}

/// 44.1kHz 用の DSSS DspConfig を作成する
pub fn dsp_config_44k() -> DspConfig {
    dsp_config(44100.0)
}
