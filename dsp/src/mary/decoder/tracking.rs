//! 位相/タイミング追跡モジュール
//!
//! MaryDQPSK復調における位相追跡とタイミング追跡の機能を提供する。

use num_complex::Complex32;

/// 追跡関連定数
pub const TRACKING_TIMING_PROP_GAIN: f32 = 0.18;
pub const TRACKING_TIMING_RATE_GAIN: f32 = 0.01;
pub const TRACKING_PHASE_PROP_GAIN: f32 = 0.35;
pub const TRACKING_PHASE_FREQ_GAIN: f32 = 0.05;
pub const TRACKING_TIMING_LIMIT_CHIP: f32 = 2.0;
pub const TRACKING_TIMING_RATE_LIMIT_CHIP: f32 = 0.25;
pub const TRACKING_PHASE_RATE_LIMIT_RAD: f32 = 2.6;
pub const TRACKING_PHASE_STEP_CLAMP: f32 = 2.8;
pub const TRACKING_PHASE_DQPSK_CONF_ON_MIN: f32 = 1.10;
pub const TRACKING_PHASE_WALSH_CONF_ON_MIN: f32 = 0.12;
pub const TRACKING_PHASE_SNR_PROXY_ON_MIN: f32 = 4.0;
pub const TRACKING_PHASE_DQPSK_CONF_OFF_MIN: f32 = 0.70;
pub const TRACKING_PHASE_WALSH_CONF_OFF_MIN: f32 = 0.08;
pub const TRACKING_PHASE_SNR_PROXY_OFF_MIN: f32 = 2.8;
pub const TRACKING_PHASE_RATE_HOLD_DECAY: f32 = 0.9;
pub const TRACKING_PHASE_PROP_GAIN_OFF: f32 = 0.08;
pub const TRACKING_PHASE_FREQ_GAIN_OFF: f32 = 0.01;
pub const TRACKING_PHASE_OFF_ERR_CLAMP: f32 = 0.35;
pub const TRACKING_PHASE_ERR_GATE_RAD: f32 = 1.00;
pub const TRACKING_PHASE_ERR_GATE_DQPSK_CONF_HIGH: f32 = 1.60;
pub const PHASE_ERR_ABS_THRESH_0P5_RAD: f32 = 0.5;
pub const PHASE_ERR_ABS_THRESH_1P0_RAD: f32 = 1.0;

/// 位相/タイミング追跡状態
#[derive(Clone, Copy, Debug)]
pub struct TrackingState {
    pub phase_ref: Complex32,
    pub phase_rate: f32,
    pub timing_offset: f32,
    pub timing_rate: f32,
    pub phase_gate_enabled: bool,
}

impl TrackingState {
    pub fn new() -> Self {
        Self {
            phase_ref: Complex32::new(1.0, 0.0),
            phase_rate: 0.0,
            timing_offset: 0.0,
            timing_rate: 0.0,
            phase_gate_enabled: false,
        }
    }

    #[cfg(test)]
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for TrackingState {
    fn default() -> Self {
        Self::new()
    }
}

/// Early-Lateタイミング誤差検出
#[inline]
pub fn timing_error_from_early_late(early_mag: f32, late_mag: f32) -> f32 {
    (late_mag - early_mag) / (late_mag + early_mag + 1e-6)
}

/// タイミングレート更新
#[inline]
pub fn update_timing_rate(timing_rate: f32, timing_err: f32, timing_rate_limit: f32) -> f32 {
    (timing_rate + TRACKING_TIMING_RATE_GAIN * timing_err)
        .clamp(-timing_rate_limit, timing_rate_limit)
}

/// タイミングオフセット更新
#[inline]
pub fn update_timing_offset(
    timing_offset: f32,
    timing_rate: f32,
    timing_err: f32,
    timing_limit: f32,
) -> f32 {
    (timing_offset + timing_rate + TRACKING_TIMING_PROP_GAIN * timing_err)
        .clamp(-timing_limit, timing_limit)
}

/// 位相誤差計算
#[inline]
pub fn phase_error_from_diff(diff: Complex32, decided_symbol: Complex32) -> f32 {
    let diff_data_removed = diff * decided_symbol.conj();
    diff_data_removed.im.atan2(diff_data_removed.re)
}

/// 位相レート更新
#[inline]
pub fn update_phase_rate(phase_rate: f32, phase_err: f32) -> f32 {
    (phase_rate + TRACKING_PHASE_FREQ_GAIN * phase_err).clamp(
        -TRACKING_PHASE_RATE_LIMIT_RAD,
        TRACKING_PHASE_RATE_LIMIT_RAD,
    )
}

/// 位相ステップ計算
#[inline]
pub fn phase_step_from_phase_error(phase_err: f32, phase_rate: f32) -> f32 {
    (phase_rate + TRACKING_PHASE_PROP_GAIN * phase_err)
        .clamp(-TRACKING_PHASE_STEP_CLAMP, TRACKING_PHASE_STEP_CLAMP)
}

/// 位相ゲート通過カウント
#[inline]
pub fn phase_gate_pass_count(
    dqpsk_conf: f32,
    walsh_conf: f32,
    snr_proxy: f32,
    dqpsk_thr: f32,
    walsh_thr: f32,
    snr_thr: f32,
) -> u8 {
    u8::from(dqpsk_conf >= dqpsk_thr)
        + u8::from(walsh_conf >= walsh_thr)
        + u8::from(snr_proxy >= snr_thr)
}

/// 次の位相ゲート状態を計算
#[inline]
pub fn next_phase_gate_enabled(
    prev_enabled: bool,
    dqpsk_conf: f32,
    walsh_conf: f32,
    snr_proxy: f32,
) -> bool {
    if prev_enabled {
        // ON→OFF: OFF閾値で2条件以上満たさなければ無効化
        let pass_count = phase_gate_pass_count(
            dqpsk_conf,
            walsh_conf,
            snr_proxy,
            TRACKING_PHASE_DQPSK_CONF_OFF_MIN,
            TRACKING_PHASE_WALSH_CONF_OFF_MIN,
            TRACKING_PHASE_SNR_PROXY_OFF_MIN,
        );
        pass_count >= 2
    } else {
        // OFF→ON: ON閾値で2条件満たせば有効化
        let pass_count = phase_gate_pass_count(
            dqpsk_conf,
            walsh_conf,
            snr_proxy,
            TRACKING_PHASE_DQPSK_CONF_ON_MIN,
            TRACKING_PHASE_WALSH_CONF_ON_MIN,
            TRACKING_PHASE_SNR_PROXY_ON_MIN,
        );
        pass_count >= 2
    }
}

/// 位相レート更新の可否を計算
///
/// 位相ゲートがONでも、各シンボル更新はON判定と同じ閾値で再評価する。
/// これにより低信頼シンボルでの誤更新を抑え、ループ発散を避ける。
#[inline]
pub fn phase_rate_update_enabled(dqpsk_conf: f32, walsh_conf: f32, snr_proxy: f32) -> bool {
    let pass_count = phase_gate_pass_count(
        dqpsk_conf,
        walsh_conf,
        snr_proxy,
        TRACKING_PHASE_DQPSK_CONF_ON_MIN,
        TRACKING_PHASE_WALSH_CONF_ON_MIN,
        TRACKING_PHASE_SNR_PROXY_ON_MIN,
    );
    pass_count >= 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracking_state_creation() {
        let state = TrackingState::new();
        assert_eq!(state.phase_ref, Complex32::new(1.0, 0.0));
        assert_eq!(state.phase_rate, 0.0);
    }

    #[test]
    fn test_tracking_state_reset() {
        let mut state = TrackingState::new();
        state.phase_rate = 1.0;
        state.reset();
        assert_eq!(state.phase_rate, 0.0);
    }

    #[test]
    fn test_timing_error_from_early_late() {
        let err = timing_error_from_early_late(0.9, 1.1);
        assert!(err > 0.0); // late > early, positive error
    }

    #[test]
    fn test_phase_error_from_diff() {
        let diff = Complex32::new(1.0, 0.1);
        let decided = Complex32::new(1.0, 0.0);
        let err = phase_error_from_diff(diff, decided);
        assert!(err.abs() < std::f32::consts::PI);
    }

    #[test]
    fn test_update_timing_rate_clamps_to_limit() {
        let updated = update_timing_rate(0.24, 10.0, 0.25);
        let updated_neg = update_timing_rate(-0.24, -10.0, 0.25);

        assert_eq!(updated, 0.25);
        assert_eq!(updated_neg, -0.25);
    }

    #[test]
    fn test_update_timing_offset_clamps_to_limit() {
        let updated = update_timing_offset(1.95, 0.2, 1.0, 2.0);
        let updated_neg = update_timing_offset(-1.95, -0.2, -1.0, 2.0);

        assert_eq!(updated, 2.0);
        assert_eq!(updated_neg, -2.0);
    }

    #[test]
    fn test_update_phase_rate_clamps_to_limit() {
        let updated = update_phase_rate(2.55, 10.0);
        let updated_neg = update_phase_rate(-2.55, -10.0);

        assert_eq!(updated, TRACKING_PHASE_RATE_LIMIT_RAD);
        assert_eq!(updated_neg, -TRACKING_PHASE_RATE_LIMIT_RAD);
    }

    #[test]
    fn test_phase_step_from_phase_error_clamps_to_limit() {
        let step = phase_step_from_phase_error(100.0, 0.0);
        let step_neg = phase_step_from_phase_error(-100.0, 0.0);

        assert_eq!(step, TRACKING_PHASE_STEP_CLAMP);
        assert_eq!(step_neg, -TRACKING_PHASE_STEP_CLAMP);
    }

    #[test]
    fn test_phase_gate_pass_count_counts_threshold_hits() {
        let pass_count = phase_gate_pass_count(1.2, 0.05, 5.0, 1.1, 0.08, 4.0);
        let zero_count = phase_gate_pass_count(0.1, 0.01, 0.5, 1.1, 0.08, 4.0);

        assert_eq!(pass_count, 2);
        assert_eq!(zero_count, 0);
    }

    #[test]
    fn test_next_phase_gate_enabled_turns_on_with_two_on_threshold_hits() {
        let enabled = next_phase_gate_enabled(
            false,
            TRACKING_PHASE_DQPSK_CONF_ON_MIN + 0.01,
            TRACKING_PHASE_WALSH_CONF_ON_MIN + 0.01,
            TRACKING_PHASE_SNR_PROXY_ON_MIN - 0.5,
        );

        assert!(enabled);
    }

    #[test]
    fn test_next_phase_gate_enabled_turns_off_when_off_threshold_hits_drop_below_two() {
        let enabled = next_phase_gate_enabled(
            true,
            TRACKING_PHASE_DQPSK_CONF_OFF_MIN - 0.01,
            TRACKING_PHASE_WALSH_CONF_OFF_MIN + 0.01,
            TRACKING_PHASE_SNR_PROXY_OFF_MIN - 0.5,
        );

        assert!(!enabled);
    }

    #[test]
    fn test_next_phase_gate_enabled_stays_on_with_two_off_threshold_hits() {
        let enabled = next_phase_gate_enabled(
            true,
            TRACKING_PHASE_DQPSK_CONF_OFF_MIN + 0.01,
            TRACKING_PHASE_WALSH_CONF_OFF_MIN + 0.01,
            TRACKING_PHASE_SNR_PROXY_OFF_MIN - 0.5,
        );

        assert!(enabled);
    }

    #[test]
    fn test_phase_rate_update_enabled_requires_two_on_threshold_hits() {
        let enabled = phase_rate_update_enabled(
            TRACKING_PHASE_DQPSK_CONF_ON_MIN + 0.01,
            TRACKING_PHASE_WALSH_CONF_ON_MIN + 0.01,
            TRACKING_PHASE_SNR_PROXY_ON_MIN - 0.5,
        );
        let disabled = phase_rate_update_enabled(
            TRACKING_PHASE_DQPSK_CONF_ON_MIN - 0.1,
            TRACKING_PHASE_WALSH_CONF_ON_MIN + 0.01,
            TRACKING_PHASE_SNR_PROXY_ON_MIN - 0.5,
        );

        assert!(enabled);
        assert!(!disabled);
    }
}
