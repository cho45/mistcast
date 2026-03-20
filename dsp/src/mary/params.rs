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

/// ペイロード内に挿入する位相パイロットの間隔（データシンボル数単位）。
/// 16 の場合、16データシンボルごとに1パイロットを挿入する。
pub const PAYLOAD_PILOT_INTERVAL_SYMBOLS: usize = 16;

/// 位相パイロットで送る既知Walsh index。
pub const PAYLOAD_PILOT_WALSH_INDEX: usize = 0;

/// 位相パイロットで送る既知DQPSKビット（delta=0, つまり位相保持）。
pub const PAYLOAD_PILOT_DQPSK_BITS: (u8, u8) = (0, 0);

/// データシンボル数に対する挿入パイロット数を返す。
#[inline]
pub const fn payload_pilot_count(data_symbols: usize) -> usize {
    if PAYLOAD_PILOT_INTERVAL_SYMBOLS == 0 || data_symbols <= 1 {
        0
    } else {
        (data_symbols - 1) / PAYLOAD_PILOT_INTERVAL_SYMBOLS
    }
}

/// データ＋パイロットの総シンボル数を返す。
#[inline]
pub const fn payload_total_symbols(data_symbols: usize) -> usize {
    data_symbols + payload_pilot_count(data_symbols)
}

/// データシンボル index が、パイロット挿入後に何番目の送信シンボルかを返す。
#[inline]
pub const fn payload_symbol_slot_for_data_index(data_index: usize) -> usize {
    if PAYLOAD_PILOT_INTERVAL_SYMBOLS == 0 {
        data_index
    } else {
        data_index + data_index / PAYLOAD_PILOT_INTERVAL_SYMBOLS
    }
}

/// 送信シンボル index がデータスロットなら、そのデータ index を返す。
/// パイロットスロットの場合は `None` を返す。
#[inline]
pub const fn payload_data_index_for_symbol_slot(slot_index: usize) -> Option<usize> {
    if PAYLOAD_PILOT_INTERVAL_SYMBOLS == 0 {
        Some(slot_index)
    } else if is_payload_pilot_slot(slot_index) {
        None
    } else {
        Some(slot_index - slot_index / (PAYLOAD_PILOT_INTERVAL_SYMBOLS + 1))
    }
}

/// 送信シンボル index がパイロットスロットかどうかを返す。
#[inline]
pub const fn is_payload_pilot_slot(slot_index: usize) -> bool {
    if PAYLOAD_PILOT_INTERVAL_SYMBOLS == 0 {
        false
    } else {
        (slot_index + 1).is_multiple_of(PAYLOAD_PILOT_INTERVAL_SYMBOLS + 1)
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payload_pilot_layout_helpers() {
        assert_eq!(payload_pilot_count(0), 0);
        assert_eq!(payload_pilot_count(1), 0);
        assert_eq!(payload_pilot_count(16), 0);
        assert_eq!(payload_pilot_count(17), 1);
        assert_eq!(payload_pilot_count(32), 1);
        assert_eq!(payload_pilot_count(33), 2);

        assert_eq!(payload_total_symbols(16), 16);
        assert_eq!(payload_total_symbols(17), 18);
        assert_eq!(payload_total_symbols(33), 35);

        assert_eq!(payload_symbol_slot_for_data_index(0), 0);
        assert_eq!(payload_symbol_slot_for_data_index(15), 15);
        assert_eq!(payload_symbol_slot_for_data_index(16), 17);
        assert_eq!(payload_symbol_slot_for_data_index(31), 32);
        assert_eq!(payload_symbol_slot_for_data_index(32), 34);
        assert_eq!(payload_data_index_for_symbol_slot(0), Some(0));
        assert_eq!(payload_data_index_for_symbol_slot(15), Some(15));
        assert_eq!(payload_data_index_for_symbol_slot(16), None);
        assert_eq!(payload_data_index_for_symbol_slot(17), Some(16));
        assert_eq!(payload_data_index_for_symbol_slot(32), Some(31));
        assert_eq!(payload_data_index_for_symbol_slot(33), None);

        assert!(!is_payload_pilot_slot(0));
        assert!(!is_payload_pilot_slot(15));
        assert!(is_payload_pilot_slot(16));
        assert!(!is_payload_pilot_slot(17));
        assert!(is_payload_pilot_slot(33));
    }
}
