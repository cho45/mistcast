//! デコーダ統計収集モジュール
//!
//! 復調パイプラインの統計情報を収集・管理する。

use crate::coding::fountain::FountainDecoder;
use crate::mary::interleaver_config;
use crate::DspConfig;

/// デコード進捗
#[derive(Debug, Clone)]
pub struct DecodeProgress {
    pub synced_frames: usize,
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
    pub fde_selected_frames: usize,
    pub raw_selected_frames: usize,
    pub last_path_used: i32,
    pub last_pred_mse_fde: f32,
    pub last_pred_mse_raw: f32,
    pub last_est_snr_db: f32,
    pub phase_gate_on_symbols: usize,
    pub phase_gate_off_symbols: usize,
    pub phase_gate_on_ratio: f32,
    pub phase_innovation_reject_symbols: usize,
    pub phase_err_abs_sum_rad: f64,
    pub phase_err_abs_count: usize,
    pub phase_err_abs_mean_rad: f32,
    pub phase_err_abs_ge_0p5_symbols: usize,
    pub phase_err_abs_ge_1p0_symbols: usize,
    pub phase_err_abs_ge_0p5_ratio: f32,
    pub phase_err_abs_ge_1p0_ratio: f32,
    pub llr_second_pass_attempts: usize,
    pub llr_second_pass_rescued: usize,
    pub viterbi_packet_decode_attempts: usize,
    pub viterbi_crc_candidate_checks: usize,
    pub ebn0_approx_db: f32,
    pub basis_matrix: Vec<u8>,
}

/// デコーダ統計管理
pub struct DecoderStats {
    // 同期統計
    pub(crate) synced_frames: usize,

    // パケット受信統計
    pub(crate) received_packets: usize,
    pub(crate) stalled_packets: usize,
    pub(crate) dependent_packets: usize,
    pub(crate) duplicate_packets: usize,
    pub(crate) crc_error_packets: usize,
    pub(crate) parse_error_packets: usize,
    pub(crate) invalid_neighbor_packets: usize,
    pub(crate) last_packet_seq: Option<u32>,
    pub(crate) last_rank_up_seq: Option<u32>,

    // フレームパス選択統計
    pub(crate) fde_selected_frames: usize,
    pub(crate) raw_selected_frames: usize,
    pub(crate) last_path_used: i32,
    pub(crate) last_pred_mse_fde: f32,
    pub(crate) last_pred_mse_raw: f32,
    pub(crate) last_est_snr_db: f32,

    // 位相追跡統計
    pub(crate) phase_gate_on_symbols: usize,
    pub(crate) phase_gate_off_symbols: usize,
    pub(crate) phase_innovation_reject_symbols: usize,
    pub(crate) phase_err_abs_sum_rad: f64,
    pub(crate) phase_err_abs_count: usize,
    pub(crate) phase_err_abs_ge_0p5_symbols: usize,
    pub(crate) phase_err_abs_ge_1p0_symbols: usize,

    // LLR第二パス統計
    pub(crate) llr_second_pass_attempts: usize,
    pub(crate) llr_second_pass_rescued: usize,
    pub(crate) viterbi_packet_decode_attempts: usize,
    pub(crate) viterbi_crc_candidate_checks: usize,

    // 全体統計
    pub stats_total_samples: usize,
}

impl DecoderStats {
    pub fn new() -> Self {
        Self {
            synced_frames: 0,
            received_packets: 0,
            stalled_packets: 0,
            dependent_packets: 0,
            duplicate_packets: 0,
            crc_error_packets: 0,
            parse_error_packets: 0,
            invalid_neighbor_packets: 0,
            last_packet_seq: None,
            last_rank_up_seq: None,
            fde_selected_frames: 0,
            raw_selected_frames: 0,
            last_path_used: -1,
            last_pred_mse_fde: f32::NAN,
            last_pred_mse_raw: f32::NAN,
            last_est_snr_db: f32::NAN,
            phase_gate_on_symbols: 0,
            phase_gate_off_symbols: 0,
            phase_innovation_reject_symbols: 0,
            phase_err_abs_sum_rad: 0.0,
            phase_err_abs_count: 0,
            phase_err_abs_ge_0p5_symbols: 0,
            phase_err_abs_ge_1p0_symbols: 0,
            llr_second_pass_attempts: 0,
            llr_second_pass_rescued: 0,
            viterbi_packet_decode_attempts: 0,
            viterbi_crc_candidate_checks: 0,
            stats_total_samples: 0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn reset_fountain_session(&mut self) {
        // セッション境界で同期/受信統計をリセットする。
        self.synced_frames = 0;
        self.received_packets = 0;
        self.stalled_packets = 0;
        self.dependent_packets = 0;
        self.duplicate_packets = 0;
        self.crc_error_packets = 0;
        self.parse_error_packets = 0;
        self.invalid_neighbor_packets = 0;
        self.last_packet_seq = None;
        self.last_rank_up_seq = None;
        self.viterbi_packet_decode_attempts = 0;
        self.viterbi_crc_candidate_checks = 0;
    }

    /// DecodeProgress構造体を生成
    pub fn to_progress(
        &self,
        fountain_decoder: &FountainDecoder,
        config: &DspConfig,
        complete: bool,
    ) -> DecodeProgress {
        let needed = fountain_decoder.params().k;
        let progress = fountain_decoder.progress();
        let ebn0_approx_db = estimate_ebn0_approx_db(config, self.last_est_snr_db);
        let phase_gate_total = self.phase_gate_on_symbols + self.phase_gate_off_symbols;
        let phase_gate_on_ratio = if phase_gate_total == 0 {
            0.0
        } else {
            self.phase_gate_on_symbols as f32 / phase_gate_total as f32
        };
        let phase_err_abs_mean_rad = if self.phase_err_abs_count == 0 {
            0.0
        } else {
            (self.phase_err_abs_sum_rad / self.phase_err_abs_count as f64) as f32
        };
        let phase_err_abs_ge_0p5_ratio = if self.phase_err_abs_count == 0 {
            0.0
        } else {
            self.phase_err_abs_ge_0p5_symbols as f32 / self.phase_err_abs_count as f32
        };
        let phase_err_abs_ge_1p0_ratio = if self.phase_err_abs_count == 0 {
            0.0
        } else {
            self.phase_err_abs_ge_1p0_symbols as f32 / self.phase_err_abs_count as f32
        };

        DecodeProgress {
            synced_frames: self.synced_frames,
            received_packets: self.received_packets,
            needed_packets: needed,
            rank_packets: fountain_decoder.rank(),
            stalled_packets: self.stalled_packets,
            dependent_packets: self.dependent_packets,
            duplicate_packets: self.duplicate_packets,
            crc_error_packets: self.crc_error_packets,
            parse_error_packets: self.parse_error_packets,
            invalid_neighbor_packets: self.invalid_neighbor_packets,
            last_packet_seq: self.last_packet_seq.map(|s| s as i32).unwrap_or(-1),
            last_rank_up_seq: self.last_rank_up_seq.map(|s| s as i32).unwrap_or(-1),
            progress,
            complete,
            fde_selected_frames: self.fde_selected_frames,
            raw_selected_frames: self.raw_selected_frames,
            last_path_used: self.last_path_used,
            last_pred_mse_fde: self.last_pred_mse_fde,
            last_pred_mse_raw: self.last_pred_mse_raw,
            last_est_snr_db: self.last_est_snr_db,
            phase_gate_on_symbols: self.phase_gate_on_symbols,
            phase_gate_off_symbols: self.phase_gate_off_symbols,
            phase_gate_on_ratio,
            phase_innovation_reject_symbols: self.phase_innovation_reject_symbols,
            phase_err_abs_sum_rad: self.phase_err_abs_sum_rad,
            phase_err_abs_count: self.phase_err_abs_count,
            phase_err_abs_mean_rad,
            phase_err_abs_ge_0p5_symbols: self.phase_err_abs_ge_0p5_symbols,
            phase_err_abs_ge_1p0_symbols: self.phase_err_abs_ge_1p0_symbols,
            phase_err_abs_ge_0p5_ratio,
            phase_err_abs_ge_1p0_ratio,
            llr_second_pass_attempts: self.llr_second_pass_attempts,
            llr_second_pass_rescued: self.llr_second_pass_rescued,
            viterbi_packet_decode_attempts: self.viterbi_packet_decode_attempts,
            viterbi_crc_candidate_checks: self.viterbi_crc_candidate_checks,
            ebn0_approx_db,
            basis_matrix: fountain_decoder.get_basis_matrix(),
        }
    }
}

impl Default for DecoderStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Eb/N0の近似推定
fn estimate_ebn0_approx_db(config: &DspConfig, est_snr_db_internal: f32) -> f32 {
    if !est_snr_db_internal.is_finite() {
        return f32::NAN;
    }

    let chip_rate = config.chip_rate.max(1e-6);
    let symbol_rate = chip_rate / crate::mary::params::PAYLOAD_SPREAD_FACTOR as f32;
    let coded_bit_rate = symbol_rate * 6.0;
    let code_rate = (crate::frame::packet::PACKET_BYTES as f32 * 8.0)
        / interleaver_config::interleaved_bits() as f32;
    let rb_info = (coded_bit_rate * code_rate).max(1e-6);
    let beq = chip_rate; // 近似: 等価雑音帯域幅 B ≈ Rc
    est_snr_db_internal + 10.0 * (beq / rb_info).log10()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coding::fountain::FountainParams;

    #[test]
    fn test_decoder_stats_creation() {
        let stats = DecoderStats::new();
        assert_eq!(stats.received_packets, 0);
        assert_eq!(stats.fde_selected_frames, 0);
        assert!(stats.last_est_snr_db.is_nan());
    }

    #[test]
    fn test_decoder_stats_reset() {
        let mut stats = DecoderStats::new();
        stats.received_packets = 10;
        stats.reset();
        assert_eq!(stats.received_packets, 0);
    }

    #[test]
    fn test_decoder_stats_reset_fountain_session_preserves_frame_diagnostics() {
        let mut stats = DecoderStats::new();
        stats.received_packets = 4;
        stats.synced_frames = 9;
        stats.last_packet_seq = Some(7);
        stats.last_rank_up_seq = Some(6);
        stats.last_pred_mse_fde = 0.25;
        stats.last_pred_mse_raw = 0.5;
        stats.last_est_snr_db = 12.0;
        stats.last_path_used = 1;
        stats.fde_selected_frames = 3;
        stats.viterbi_packet_decode_attempts = 5;
        stats.viterbi_crc_candidate_checks = 9;

        stats.reset_fountain_session();

        assert_eq!(stats.synced_frames, 0);
        assert_eq!(stats.received_packets, 0);
        assert_eq!(stats.last_packet_seq, None);
        assert_eq!(stats.last_rank_up_seq, None);
        assert_eq!(stats.last_pred_mse_fde, 0.25);
        assert_eq!(stats.last_pred_mse_raw, 0.5);
        assert_eq!(stats.last_est_snr_db, 12.0);
        assert_eq!(stats.last_path_used, 1);
        assert_eq!(stats.fde_selected_frames, 3);
        assert_eq!(stats.viterbi_packet_decode_attempts, 0);
        assert_eq!(stats.viterbi_crc_candidate_checks, 0);
    }

    #[test]
    fn test_estimate_ebn0_approx_db_returns_nan_for_non_finite_input() {
        let config = DspConfig::default_48k();

        assert!(estimate_ebn0_approx_db(&config, f32::NAN).is_nan());
        assert!(estimate_ebn0_approx_db(&config, f32::INFINITY).is_nan());
        assert!(estimate_ebn0_approx_db(&config, f32::NEG_INFINITY).is_nan());
    }

    #[test]
    fn test_estimate_ebn0_approx_db_adds_expected_rate_offset() {
        let config = DspConfig::default_48k();
        let est_snr_db_internal = 0.0;
        let code_rate = (crate::frame::packet::PACKET_BYTES as f32 * 8.0)
            / interleaver_config::interleaved_bits() as f32;
        let expected_offset_db =
            10.0 * (crate::mary::params::PAYLOAD_SPREAD_FACTOR as f32 / (6.0 * code_rate)).log10();

        let ebn0 = estimate_ebn0_approx_db(&config, est_snr_db_internal);

        assert!((ebn0 - expected_offset_db).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_ebn0_approx_db_preserves_input_delta() {
        let config = DspConfig::default_48k();

        let low = estimate_ebn0_approx_db(&config, -3.5);
        let high = estimate_ebn0_approx_db(&config, 8.25);

        assert!(((high - low) - (8.25 - -3.5)).abs() < 1e-6);
    }

    #[test]
    fn test_to_progress_includes_viterbi_counters() {
        let mut stats = DecoderStats::new();
        stats.viterbi_packet_decode_attempts = 7;
        stats.viterbi_crc_candidate_checks = 11;
        let fountain_decoder = FountainDecoder::new(FountainParams::new(4, 16));
        let config = DspConfig::default_48k();

        let progress = stats.to_progress(&fountain_decoder, &config, false);

        assert_eq!(progress.viterbi_packet_decode_attempts, 7);
        assert_eq!(progress.viterbi_crc_candidate_checks, 11);
    }
}
