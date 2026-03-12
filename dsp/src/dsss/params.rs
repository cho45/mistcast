//! DSSS モジュール固有のパラメータ定義

use crate::frame::packet::PACKET_BYTES;
use crate::DspConfig;

const DSSS_MSEQ_ORDER: usize = 4;
const DSSS_PREAMBLE_REPEAT: usize = 2;
const DSSS_PREAMBLE_SF: usize = 15;
const DSSS_SYNC_WORD_BITS: usize = 16;

/// 畳み込み符号のテールビット数
pub const TAIL_BITS: usize = 6;

/// DSSS インターリーバ行数
pub const INTERLEAVER_ROWS: usize = 12;

/// DSSS インターリーバ列数
pub const INTERLEAVER_COLS: usize = fec_bits() / INTERLEAVER_ROWS;

/// FEC入力ビット数（生ビット + テールビット）
#[inline]
pub const fn raw_bits() -> usize {
    PACKET_BYTES * 8 + TAIL_BITS
}

/// FEC出力ビット数（畳み込み符号率 1/2）
#[inline]
pub const fn fec_bits() -> usize {
    raw_bits() * 2
}

/// インターリーバ処理後のビット数
#[inline]
pub const fn interleaved_bits() -> usize {
    INTERLEAVER_ROWS * INTERLEAVER_COLS
}

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
