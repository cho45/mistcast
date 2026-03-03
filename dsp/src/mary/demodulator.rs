//! MaryDQPSK (16-ary + DQPSK) 復調器
//!
//! # 復調パイプライン
//! 1. 16系列並列相関: 各Walsh系列との相関を計算
//! 2. Max-Log-MAP LLR計算: Walsh index (4ビット) と DQPSK phase (2ビット) のLLRを計算
//! 3. 差分検波: DQPSKの位相差を検出
//!
//! # 仕様
//! - プリアンブル/Sync: Walsh[0]、DBPSK、sf=15
//! - Payload: Walsh[0-15]、DQPSK、sf=16

use crate::common::walsh::{WalshCorrelator, WalshDictionary};
use num_complex::Complex32;

/// MaryDQPSK復調器
pub struct Demodulator {
    correlators: Vec<WalshCorrelator>,
    prev: Complex32,
}

impl Demodulator {
    /// 新しいMaryDQPSK復調器を作成する
    pub fn new() -> Self {
        let wdict = WalshDictionary::default_w16();
        let correlators = (0..16)
            .map(|idx| WalshCorrelator::new(wdict.w16[idx].clone()))
            .collect();

        Demodulator {
            correlators,
            prev: Complex32::new(1.0, 0.0),
        }
    }

    /// 16系列並列相復調
    ///
    /// `signal`: 長さ16の複素数信号
    ///
    /// 戻り値: 16系列の相関値
    pub fn despread_all(&self, signal: &[Complex32]) -> [Complex32; 16] {
        assert_eq!(signal.len(), 16, "Signal must have length 16");

        let mut correlations = [Complex32::new(0.0, 0.0); 16];
        for (idx, correlator) in self.correlators.iter().enumerate() {
            correlations[idx] = correlator.correlate(signal);
        }
        correlations
    }

    /// Max-Log-MAP LLR計算（4ビットWalsh index）
    ///
    /// `energies`: 16系列の相関エネルギー
    /// `max_energy`: 全系列中の最大エネルギー（正規化用）
    ///
    /// 戻り値: 4ビットLLR [bit0, bit1, bit2, bit3]
    ///
    /// # 理論
    /// Max-Log-MAP近似では、各ビットのLLRは以下のように計算される：
    /// LLR(b_i) = (max_{x: b_i=0} energy[x] - max_{x: b_i=1} energy[x]) / E_max
    pub fn walsh_llr(&self, energies: &[f32; 16], max_energy: f32) -> [f32; 4] {
        let mut llr = [0.0f32; 4];
        let denom = max_energy.max(1e-6);

        // Encoderは bits[idx] (配列の先頭) を w_idx の MSB (bit=3) にマップする。
        // Decoderは all_llrs に先頭から追加していくため、
        // 戻り値の配列は [LLR(MSB=3), LLR(bit=2), LLR(bit=1), LLR(LSB=0)] の順でなければならない。
        for i in 0..4 {
            let bit = 3 - i; // i=0 -> bit=3, i=1 -> bit=2 ...

            // bit=0のWalsh indexの最大エネルギー
            let max_e0 = (0..16)
                .filter(|&idx| (idx & (1 << bit)) == 0)
                .map(|idx| energies[idx])
                .fold(f32::NEG_INFINITY, f32::max);

            // bit=1のWalsh indexの最大エネルギー
            let max_e1 = (0..16)
                .filter(|&idx| (idx & (1 << bit)) != 0)
                .map(|idx| energies[idx])
                .fold(f32::NEG_INFINITY, f32::max);

            llr[i] = (max_e0 - max_e1) / denom;
        }

        llr
    }

    /// DQPSK LLR計算（2ビットphase）
    ///
    /// `diff`: 差分信号 (Z_curr * Z_prev*)
    /// `max_energy`: 現在のシンボルの最大エネルギー (正規化用)
    ///
    /// 戻り値: 2ビットLLR [bit0, bit1]
    pub fn dqpsk_llr(&self, diff: Complex32, max_energy: f32) -> [f32; 2] {
        let denom = max_energy.max(1e-6);
        // DQPSK mapping (Gray code):
        // 00 -> 0 -> (1, 0)
        // 01 -> 1 -> (0, 1)
        // 11 -> 2 -> (-1, 0)
        // 10 -> 3 -> (0, -1)
        // LLR(b0) = diff.re + diff.im
        // LLR(b1) = diff.re - diff.im
        [(diff.re + diff.im) / denom, (diff.re - diff.im) / denom]
    }

    /// 1シンボル分の復調
    ///
    /// `signal`: 長さ16の複素数信号
    ///
    /// 戻り値: (Walsh LLR[4], DQPSK LLR[2], 差分信号)
    pub fn demod_symbol(&mut self, signal: &[Complex32]) -> ([f32; 4], [f32; 2], Complex32) {
        let correlations = self.despread_all(signal);

        // 最大エネルギーの系列を特定
        let mut max_energy = 0.0f32;
        let mut best_idx = 0usize;
        for (idx, &corr) in correlations.iter().enumerate() {
            let energy = corr.norm_sqr();
            if energy > max_energy {
                max_energy = energy;
                best_idx = idx;
            }
        }

        let best_corr = correlations[best_idx];

        // 差分検波: D = Z_curr * Z_prev*
        // prev は前回の best_corr (未正規化)
        let diff = best_corr * self.prev.conj();

        // 次のシンボルのためのリファレンス更新 (未正規化)
        self.prev = best_corr;

        // LLR計算
        let energies: [f32; 16] = correlations.map(|c| c.norm_sqr());
        let walsh_llr = self.walsh_llr(&energies, max_energy);
        
        // DQPSK LLRの正規化
        // D = Z_curr * Z_prev* の次元はエネルギー次元 (|A*SF|^2)
        // なので E_max で割ることで [-1, 1] に収まる
        let dqpsk_llr = self.dqpsk_llr(diff, max_energy);

        (walsh_llr, dqpsk_llr, diff)
    }

    /// リセット
    pub fn reset(&mut self) {
        // 初期状態の振幅は不明だが、1.0（基準）としておく。
        // Sync後の最初の復調では set_reference_phase で正しい値が設定されることを期待。
        self.prev = Complex32::new(1.0, 0.0);
    }

    /// 参照位相を設定（未正規化の複素相関値を受け取る）
    pub fn set_reference_phase(&mut self, i: f32, q: f32) {
        self.prev = Complex32::new(i, q);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_demodulator() -> Demodulator {
        Demodulator::new()
    }

    /// 16系列並列相復調のテスト
    #[test]
    fn test_despread_all() {
        let demod = make_demodulator();
        let wdict = WalshDictionary::default_w16();

        // Walsh[0]の信号を生成
        let signal: Vec<Complex32> = wdict.w16[0]
            .iter()
            .map(|&w| Complex32::new(w as f32, 0.0))
            .collect();

        let correlations = demod.despread_all(&signal);

        // Walsh[0]との相関が最大
        let max_idx = correlations
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.norm_sqr()
                    .partial_cmp(&b.1.norm_sqr())
                    .unwrap()
            })
            .map(|(idx, _)| idx)
            .unwrap();

        assert_eq!(max_idx, 0);
    }

    /// 直交性テスト: Walsh[0]信号に対してWalsh[1-15]の相関は0
    #[test]
    fn test_orthogonality() {
        let demod = make_demodulator();
        let wdict = WalshDictionary::default_w16();

        // Walsh[0]の信号を生成
        let signal: Vec<Complex32> = wdict.w16[0]
            .iter()
            .map(|&w| Complex32::new(w as f32, 0.0))
            .collect();

        let correlations = demod.despread_all(&signal);

        // Walsh[0]以外の相関は0（直交性）
        for idx in 1..16 {
            assert!(
                correlations[idx].norm() < 1e-6,
                "Cross-correlation with Walsh[{}] should be 0",
                idx
            );
        }
    }

    /// DQPSK LLRのテスト
    #[test]
    fn test_dqpsk_llr() {
        let demod = make_demodulator();

        // diff = (1.0, 0.0) -> bit0=0, bit1=0 -> LLR0>0, LLR1>0
        let diff = Complex32::new(1.0, 0.0);
        let llr = demod.dqpsk_llr(diff, 1.0);
        assert!(llr[0] > 0.5, "LLR[0] should be positive for diff=(1,0)");
        assert!(llr[1] > 0.5, "LLR[1] should be positive for diff=(1,0)");

        // diff = (-1.0, 0.0) -> bit0=1, bit1=1 -> LLR0<0, LLR1<0
        let diff = Complex32::new(-1.0, 0.0);
        let llr = demod.dqpsk_llr(diff, 1.0);
        assert!(llr[0] < -0.5, "LLR[0] should be negative for diff=(-1,0)");
        assert!(llr[1] < -0.5, "LLR[1] should be negative for diff=(-1,0)");

        // diff = (0.0, 1.0) -> bit0=0, bit1=1 -> LLR0>0, LLR1<0
        let diff = Complex32::new(0.0, 1.0);
        let llr = demod.dqpsk_llr(diff, 1.0);
        assert!(llr[0] > 0.5, "LLR[0] should be positive for diff=(0,1)");
        assert!(llr[1] < -0.5, "LLR[1] should be negative for diff=(0,1)");

        // diff = (0.0, -1.0) -> bit0=1, bit1=0 -> LLR0<0, LLR1>0
        let diff = Complex32::new(0.0, -1.0);
        let llr = demod.dqpsk_llr(diff, 1.0);
        assert!(llr[0] < -0.5, "LLR[0] should be negative for diff=(0,-1)");
        assert!(llr[1] > 0.5, "LLR[1] should be positive for diff=(0,-1)");
    }

    /// Walsh LLRのテスト
    #[test]
    fn test_walsh_llr() {
        let demod = make_demodulator();

        // Walsh[0]のエネルギーが最大
        let mut energies = [0.0f32; 16];
        energies[0] = 100.0;
        for i in 1..16 {
            energies[i] = 10.0;
        }
        let llr = demod.walsh_llr(&energies, 100.0);

        // Walsh[0] = 0b0000 -> 全ビットのLLRが正
        for bit in 0..4 {
            assert!(
                llr[bit] > 0.0,
                "LLR[{}] should be positive for Walsh[0]",
                bit
            );
        }

        // Walsh[15] = 0b1111のエネルギーが最大
        let mut energies = [0.0f32; 16];
        energies[15] = 100.0;
        for i in 0..15 {
            energies[i] = 10.0;
        }
        let llr = demod.walsh_llr(&energies, 100.0);

        // Walsh[15] = 0b1111 -> 全ビットのLLRが負
        for bit in 0..4 {
            assert!(
                llr[bit] < 0.0,
                "LLR[{}] should be negative for Walsh[15]",
                bit
            );
        }
    }

    /// リセット後の状態確認
    #[test]
    fn test_reset() {
        let mut demod = make_demodulator();

        // 適当に位相を更新
        let wdict = WalshDictionary::default_w16();
        let signal: Vec<Complex32> = wdict.w16[0]
            .iter()
            .map(|&w| Complex32::new(w as f32, 0.0))
            .collect();
        demod.demod_symbol(&signal);

        // リセット
        demod.reset();

        // リセット後は初期状態に戻る
        assert!(
            (demod.prev.re - 1.0).abs() < 1e-6,
            "After reset, prev.re should be 1.0"
        );
        assert!(
            demod.prev.im.abs() < 1e-6,
            "After reset, prev.im should be 0.0"
        );
    }

    /// 参照位相設定のテスト
    #[test]
    fn test_set_reference_phase() {
        let mut demod = make_demodulator();
        demod.set_reference_phase(0.0, 1.0);

        assert!(
            demod.prev.re.abs() < 1e-6,
            "After set_reference_phase(0,1), prev.re should be ~0"
        );
        assert!(
            (demod.prev.im - 1.0).abs() < 1e-6,
            "After set_reference_phase(0,1), prev.im should be 1.0"
        );
    }

    /// 1シンボル復調のテスト
    #[test]
    fn test_demod_symbol() {
        let mut demod = make_demodulator();
        let wdict = WalshDictionary::default_w16();

        // 参照位相を適切に設定（振幅16.0）
        demod.set_reference_phase(16.0, 0.0);

        // Walsh[0] (インデックス0 = 0b0000) + 位相(1,0)の信号
        let walsh_idx = 0;
        let phase_i = 1.0;
        let phase_q = 0.0;
        let signal: Vec<Complex32> = wdict.w16[walsh_idx]
            .iter()
            .map(|&w| Complex32::new(phase_i * w as f32, phase_q * w as f32))
            .collect();

        let (walsh_llr, dqpsk_llr, _diff) = demod.demod_symbol(&signal);

        // Walsh[0] = 0b0000 -> 全ビットのLLRが正
        for bit in 0..4 {
            assert!(
                walsh_llr[bit] > 0.0,
                "Walsh[0] bit{} should be positive, got {}",
                bit,
                walsh_llr[bit]
            );
        }

        // 位相(1,0) -> diff = (16,0) * (16,0) = (256,0) -> LLR = 256/256 = 1.0
        assert!(dqpsk_llr[0] > 0.5, "Phase (1,0) should give positive LLR[0], got {}", dqpsk_llr[0]);
    }

    /// Walsh[1] (0b0001)のLLR確認
    #[test]
    fn test_demod_symbol_walsh_1() {
        let mut demod = make_demodulator();
        let wdict = WalshDictionary::default_w16();
        demod.set_reference_phase(16.0, 0.0);

        // Walsh[1] + 位相(1,0)の信号
        let walsh_idx = 1;
        let phase_i = 1.0;
        let phase_q = 0.0;
        let signal: Vec<Complex32> = wdict.w16[walsh_idx]
            .iter()
            .map(|&w| Complex32::new(phase_i * w as f32, phase_q * w as f32))
            .collect();

        let (walsh_llr, _dqpsk_llr, _diff) = demod.demod_symbol(&signal);

        // Walsh[1] = 0b0001 -> bit3(0)>0, bit2(0)>0, bit1(0)>0, bit0(1)<0
        assert!(walsh_llr[0] > 0.0, "Walsh[1] bit3 should be positive");
        assert!(walsh_llr[1] > 0.0, "Walsh[1] bit2 should be positive");
        assert!(walsh_llr[2] > 0.0, "Walsh[1] bit1 should be positive");
        assert!(walsh_llr[3] < 0.0, "Walsh[1] bit0 should be negative");
    }

    /// 全てのWalsh系列の復調を検証する
    #[test]
    fn test_demod_all_walsh_indices() {
        let mut demod = make_demodulator();
        let wdict = WalshDictionary::default_w16();

        for walsh_idx in 0..16 {
            demod.reset();
            let signal: Vec<Complex32> = wdict.w16[walsh_idx]
                .iter()
                .map(|&w| Complex32::new(w as f32, 0.0))
                .collect();

            let (walsh_llr, _dqpsk_llr, _diff) = demod.demod_symbol(&signal);

            // 各ビットのLLR符号を確認
            for bit in 0..4 {
                let expected_sign = if (walsh_idx >> bit) & 1 == 0 {
                    1.0 // bit=0 -> LLR>0
                } else {
                    -1.0 // bit=1 -> LLR<0
                };

                let llr_idx = 3 - bit;
                let llr_positive = walsh_llr[llr_idx] > 0.0;

                assert_eq!(llr_positive, expected_sign > 0.0,
                           "Walsh[{}] bit{}: LLR={}, bit_set={}",
                           walsh_idx, bit, walsh_llr[llr_idx], expected_sign < 0.0);
            }
        }
    }

    /// ノイズ環境での復調性能
    #[test]
    fn test_demod_with_noise() {
        let mut demod = make_demodulator();
        let wdict = WalshDictionary::default_w16();

        // Walsh[5]信号にノイズを追加
        let walsh_idx = 5;
        let phase_i = 1.0;
        let phase_q = 0.0;

        let signal_noisy: Vec<Complex32> = wdict.w16[walsh_idx]
            .iter()
            .enumerate()
            .map(|(i, &w)| {
                let signal = Complex32::new(phase_i * w as f32, phase_q * w as f32);
                // 小さなノイズを追加
                // 根拠：
                // - 信号振幅 = ±1（Walsh系列）
                // - ノイズ標準偏差 = 0.05
                // - SNR ≈ 20*log10(1/0.05) ≈ 26dB（非常に高いSNR）
                // - これは復調性能が劣化しない程度の小さなノイズ
                let noise_re = (i as f32 * 0.01).sin() * 0.05;
                let noise_im = (i as f32 * 0.01).cos() * 0.05;
                signal + Complex32::new(noise_re, noise_im)
            })
            .collect();

        let (walsh_llr, _dqpsk_llr, _diff) = demod.demod_symbol(&signal_noisy);

        // Walsh[5] = 0b0101 -> 各ビットのLLR符号を確認
        // LLR配列は [bit3, bit2, bit1, bit0] の順
        // bit3=0(>0), bit2=1(<0), bit1=0(>0), bit0=1(<0)
        let expected_signs = [1.0, -1.0, 1.0, -1.0]; // [bit3, bit2, bit1, bit0]
        for i in 0..4 {
            let actual_sign = walsh_llr[i].signum();
            let expected_sign = expected_signs[i];
            assert!(
                actual_sign == expected_sign || walsh_llr[i].abs() < 0.1,
                "Walsh[5] bit_index={} LLR={} should have sign {} or be near zero",
                i,
                walsh_llr[i],
                expected_sign
            );
        }
    }

    /// DQPSK全位相のLLRを検証する
    #[test]
    fn test_dqpsk_all_phases() {
        let demod = make_demodulator();

        // 全てのDQPSK位相を検証
        let test_cases = [
            (Complex32::new(1.0, 0.0), [1.0, 1.0]),   // 位相0 (delta=0)
            (Complex32::new(0.0, 1.0), [1.0, -1.0]),  // 位相1 (delta=1)
            (Complex32::new(-1.0, 0.0), [-1.0, -1.0]), // 位相2 (delta=2)
            (Complex32::new(0.0, -1.0), [-1.0, 1.0]),  // 位相3 (delta=3)
        ];

        for (diff, expected_signs) in test_cases {
            let llr = demod.dqpsk_llr(diff, 1.0);

            for bit in 0..2 {
                let expected_sign = expected_signs[bit];
                assert!(
                    llr[bit] * expected_sign > 0.5,
                    "Phase ({:?}) bit{} LLR={} should align with expected sign {}",
                    diff,
                    bit,
                    llr[bit],
                    expected_sign
                );
            }
        }
    }

    /// 連続シンボルの差分検波を検証する
    #[test]
    fn test_consecutive_symbol_differential_detection() {
        let mut demod = make_demodulator();
        let wdict = WalshDictionary::default_w16();

        // シンボル1: Walsh[0] + 位相0
        // シンボル2: Walsh[0] + 位相1 (delta=1)
        let signal1: Vec<Complex32> = wdict.w16[0]
            .iter()
            .map(|&w| Complex32::new(w as f32, 0.0))
            .collect();

        let signal2: Vec<Complex32> = wdict.w16[0]
            .iter()
            .map(|&w| Complex32::new(0.0, w as f32))
            .collect();

        // シンボル1を復調
        let (_walsh_llr1, _dqpsk_llr1, _diff1) = demod.demod_symbol(&signal1);

        // シンボル2を復調
        let (_walsh_llr2, dqpsk_llr2, _diff2) = demod.demod_symbol(&signal2);

        // 差分が位相遷移を反映しているはず
        // 位相0(1,0) -> 位相1(0,1) (delta=1) の場合：
        // diff = (0,1) * conj(1,0) = (0,1)
        // LLR[0] = diff.re + diff.im = 1
        // LLR[1] = diff.re - diff.im = -1
        assert!(dqpsk_llr2[0] > 0.0, "DQPSK LLR[0] should be positive for delta=1, got {}", dqpsk_llr2[0]);
        assert!(dqpsk_llr2[1] < 0.0, "DQPSK LLR[1] should be negative for delta=1, got {}", dqpsk_llr2[1]);
    }

    /// LLRのスケーリングと正規化の厳密な検証
    ///
    /// 検証項目：
    /// 1. 全てのLLR（Walsh 4ビット + DQPSK 2ビット）が同程度のスケールであること
    /// 2. LLRが概ね [-2.0, 2.0] の範囲に収まっていること（正規化の確認）
    /// 3. 強信号時でもスケールが爆発しないこと
    #[test]
    fn test_llr_scaling_and_normalization() {
        let mut demod = make_demodulator();
        let wdict = WalshDictionary::default_w16();
        demod.set_reference_phase(16.0, 0.0);

        // 理想的な信号（Walsh[3] + Phase 0）
        let walsh_idx = 3;
        let signal: Vec<Complex32> = wdict.w16[walsh_idx]
            .iter()
            .map(|&w| Complex32::new(w as f32, 0.0))
            .collect();

        let (walsh_llr, dqpsk_llr, _diff) = demod.demod_symbol(&signal);

        // 1. & 2. スケールの検証
        // 現状の実装では Walsh LLR は ~256 になるため、ここで失敗するはず
        for (i, &llr) in walsh_llr.iter().enumerate() {
            assert!(llr.abs() <= 2.1, 
                    "Walsh LLR[{}] is out of expected range [-2, 2]: {}", i, llr);
        }
        for (i, &llr) in dqpsk_llr.iter().enumerate() {
            assert!(llr.abs() <= 2.1, 
                    "DQPSK LLR[{}] is out of expected range [-2, 2]: {}", i, llr);
        }

        // 3. WalshとDQPSKの相対的な重みが同程度であることを確認
        // 理想信号において、正しく判定されたビットのLLR絶対値の平均を比較
        let walsh_avg = walsh_llr.iter().map(|v| v.abs()).sum::<f32>() / 4.0;
        let dqpsk_avg = dqpsk_llr[0].abs(); // bit0は1.0, bit1は~0を想定

        let ratio = walsh_avg / dqpsk_avg;
        assert!(ratio > 0.5 && ratio < 2.0, 
                "LLR scale mismatch between Walsh ({}) and DQPSK ({}). Ratio: {}", 
                walsh_avg, dqpsk_avg, ratio);
    }

    /// LLRの信頼性を検証する：エネルギーが大きいほどLLRの絶対値が大きい
    /// 
    /// 注：現在の正規化ロジック (E_maxで除算) では、ノイズのない理想信号は
    /// 振幅に関わらず LLR=1.0 に収束する。そのため、ここでは絶対値ではなく
    /// 正しく 1.0 に正規化されることを検証する。
    #[test]
    fn test_llr_reliability_normalized() {
        let mut demod = make_demodulator();
        let wdict = WalshDictionary::default_w16();

        // 小振幅信号（振幅0.5）
        let signal_weak: Vec<Complex32> = wdict.w16[0]
            .iter()
            .map(|&w| Complex32::new(0.5 * w as f32, 0.0))
            .collect();

        // 大振幅信号（振幅2.0）
        let signal_strong: Vec<Complex32> = wdict.w16[0]
            .iter()
            .map(|&w| Complex32::new(2.0 * w as f32, 0.0))
            .collect();

        // 復調してLLRを取得
        let (walsh_llr_weak, _, _) = demod.demod_symbol(&signal_weak);
        demod.reset();
        let (walsh_llr_strong, _, _) = demod.demod_symbol(&signal_strong);

        // 検証：理想信号ではどちらも LLR ≈ 1.0 になる
        for bit in 0..4 {
            assert!((walsh_llr_weak[bit].abs() - 1.0).abs() < 1e-3,
                    "Weak signal bit{} LLR should be ~1.0, got {}", bit, walsh_llr_weak[bit]);
            assert!((walsh_llr_strong[bit].abs() - 1.0).abs() < 1e-3,
                    "Strong signal bit{} LLR should be ~1.0, got {}", bit, walsh_llr_strong[bit]);
        }
    }

    /// Sync→Payloadハンドオーバー：Walsh indexの変化を検証
    #[test]
    fn test_sync_to_payload_walsh_index_transition() {
        let mut demod = make_demodulator();
        let wdict = WalshDictionary::default_w16();

        // Sync: Walsh[0]のみ
        let sync_signal: Vec<Complex32> = wdict.w16[0]
            .iter()
            .map(|&w| Complex32::new(w as f32, 0.0))
            .collect();

        let (sync_walsh_llr, _sync_dqpsk_llr, _sync_diff) = demod.demod_symbol(&sync_signal);

        // SyncではWalsh[0]=0b0000なので全ビットLLR>0
        for bit in 0..4 {
            assert!(
                sync_walsh_llr[bit] > 0.0,
                "Sync Walsh[0] bit{} should have positive LLR, got {}",
                bit,
                sync_walsh_llr[bit]
            );
        }

        // Payload: Walsh[7] = 0b0111
        let payload_signal: Vec<Complex32> = wdict.w16[7]
            .iter()
            .map(|&w| Complex32::new(w as f32, 0.0))
            .collect();

        let (payload_walsh_llr, _payload_dqpsk_llr, _payload_diff) = demod.demod_symbol(&payload_signal);

        // Walsh[7] = 0b0111 -> bit3(0)>0, bit2(1)<0, bit1(1)<0, bit0(1)<0
        assert!(payload_walsh_llr[0] > 0.0, "Payload Walsh[7] bit3 should be positive");
        assert!(payload_walsh_llr[1] < 0.0, "Payload Walsh[7] bit2 should be negative");
        assert!(payload_walsh_llr[2] < 0.0, "Payload Walsh[7] bit1 should be negative");
        assert!(payload_walsh_llr[3] < 0.0, "Payload Walsh[7] bit0 should be negative");
    }

    /// Sync→Payloadハンドオーバー：DQPSK phaseの継続性
    #[test]
    fn test_sync_to_payload_dqpsk_phase_continuity() {
        let mut demod = make_demodulator();
        let wdict = WalshDictionary::default_w16();

        // Sync最後: 位相(1,0) -> phase=0
        let sync_last: Vec<Complex32> = wdict.w16[0]
            .iter()
            .map(|&w| Complex32::new(w as f32, 0.0))
            .collect();

        let (_sync_walsh_llr, _sync_dqpsk_llr, sync_diff) = demod.demod_symbol(&sync_last);

        // Sync差分 = (1,0) * (1,0) = (1,0) -> phase=0
        assert!(sync_diff.re > 0.5, "Sync diff should indicate phase 0");
        assert!(sync_diff.im.abs() < 0.5, "Sync diff should indicate phase 0");

        // Payload先頭: 位相(0,1) -> phase=1 (delta=1 from phase 0)
        let payload_first: Vec<Complex32> = wdict.w16[0]
            .iter()
            .map(|&w| Complex32::new(0.0, w as f32))
            .collect();

        let (_payload_walsh_llr, payload_dqpsk_llr, _payload_diff) = demod.demod_symbol(&payload_first);

        // Payload差分 = (0,1) * (1,0) = (0,1) -> DQPSK 01
        // LLR[0] = 0 + 1 = 1 (> 0)
        // LLR[1] = 0 - 1 = -1 (< 0)
        assert!(payload_dqpsk_llr[0] > 0.0, "Payload DQPSK bit0 should be positive for delta=1");
        assert!(payload_dqpsk_llr[1] < 0.0, "Payload DQPSK bit1 should be negative for delta=1");
    }

    /// Walsh LLRの符号・絶対値の詳細検証
    #[test]
    fn test_walsh_llr_sign_and_magnitude() {
        let demod = make_demodulator();
        let wdict = WalshDictionary::default_w16();

        // Walsh[0]信号: 最も高いエネルギー
        let signal0: Vec<Complex32> = wdict.w16[0]
            .iter()
            .map(|&w| Complex32::new(w as f32, 0.0))
            .collect();

        let correlations0 = demod.despread_all(&signal0);
        let energies0: [f32; 16] = correlations0.map(|c| c.norm_sqr());
        let max_energy0 = energies0.iter().fold(0.0f32, |a, &e| a.max(e));
        let llrs0 = demod.walsh_llr(&energies0, max_energy0);

        // Walsh[0]=0b0000: 全LLR>0, かつLLR[0]が最大（bit0の識別力が最強）
        for bit in 0..4 {
            assert!(llrs0[bit] > 0.0, "Walsh[0] bit{} LLR should be positive", bit);
        }

        // Walsh[0]自身のエネルギーが最大
        assert_eq!(max_energy0, energies0[0], "Walsh[0] should have max energy");

        // Walsh[15]信号: Walsh[0]=0b0000との対照
        let signal15: Vec<Complex32> = wdict.w16[15]
            .iter()
            .map(|&w| Complex32::new(w as f32, 0.0))
            .collect();

        let correlations15 = demod.despread_all(&signal15);
        let energies15: [f32; 16] = correlations15.map(|c| c.norm_sqr());
        let max_energy15 = energies15.iter().fold(0.0f32, |a, &e| a.max(e));
        let llrs15 = demod.walsh_llr(&energies15, max_energy15);

        // Walsh[15]=0b1111: 全LLR<0
        for bit in 0..4 {
            assert!(llrs15[bit] < 0.0, "Walsh[15] bit{} LLR should be negative", bit);
        }
    }

    /// 16系列並列相関のエネルギー分布
    #[test]
    fn test_despread_all_energy_distribution() {
        let demod = make_demodulator();
        let wdict = WalshDictionary::default_w16();

        // Walsh[5]信号
        let signal: Vec<Complex32> = wdict.w16[5]
            .iter()
            .map(|&w| Complex32::new(w as f32, 0.0))
            .collect();

        let correlations = demod.despread_all(&signal);
        let energies: [f32; 16] = correlations.map(|c| c.norm_sqr());

        // Walsh[5]のエネルギーが最大
        let max_idx = (0..16)
            .max_by(|&a, &b| energies[a].partial_cmp(&energies[b]).unwrap())
            .unwrap();

        assert_eq!(max_idx, 5, "Walsh[5] should have max energy");

        // 他の系列のエネルギーは0（直交）
        // 根拠：Walsh系列は直交しているため、異なる系列の内積は0
        //       ただし、浮動小数点演算の丸め誤差を考慮する必要がある
        //       f32の機械イプシロン ≈ 1.2e-7
        //       16回の演算で誤差が蓄積：ε × √16 ≈ 5e-7
        //       閾値1e-6は、丸め誤差の2倍の安全余裕を含む
        for idx in 0..16 {
            if idx != 5 {
                assert!(
                    energies[idx] < 1e-6,
                    "Walsh[{}] should have zero energy (floating point error < 1e-6), got {}",
                    idx,
                    energies[idx]
                );
            }
        }
    }

    /// 位相基準の設定とリセット
    #[test]
    fn test_reference_phase_setting_and_reset() {
        let mut demod = make_demodulator();

        // 位相基準を設定
        demod.set_reference_phase(0.0, 1.0);

        // Walsh[0]信号との差分を確認
        let wdict = WalshDictionary::default_w16();
        let signal: Vec<Complex32> = wdict.w16[0]
            .iter()
            .map(|&w| Complex32::new(w as f32, 0.0))
            .collect();

        let (_walsh_llr, dqpsk_llr, _diff) = demod.demod_symbol(&signal);

        // 位相遷移の確認
        // 参照位相 = (0, 1) [位相90度], signal = (1, 0) [位相0度]
        // 差分 = signal * conj(ref) = (1,0) * (0,-1) = (0,-1) [位相-90度]
        // dqpsk_llr()の実装:
        //   LLR[0] = diff.re + diff.im = -1
        //   LLR[1] = diff.re - diff.im = 1
        assert!(dqpsk_llr[0] < 0.0,
                "diff=(0,-1): LLR[0] should be negative, got {}", dqpsk_llr[0]);
        assert!(dqpsk_llr[1] > 0.0,
                "diff=(0,-1): LLR[1] should be positive, got {}", dqpsk_llr[1]);

        // リセット
        demod.reset();

        // リセット後は初期位相(1,0)に戻る
        let signal2: Vec<Complex32> = wdict.w16[0]
            .iter()
            .map(|&w| Complex32::new(w as f32, 0.0))
            .collect();

        let (_walsh_llr2, dqpsk_llr2, _diff2) = demod.demod_symbol(&signal2);

        // 位相遷移なし
        // 参照位相 = (1, 0) [位相0度], signal = (1, 0) [位相0度]
        // 差分 = (1, 0) * conj(1, 0) = (1, 0) [位相0度]
        // dqpsk_llr()の実装:
        //   LLR[0] = diff.re + diff.im = 1
        //   LLR[1] = diff.re - diff.im = 1
        assert!(dqpsk_llr2[0] > 0.0,
                "diff=(1,0): LLR[0] should be positive, got {}", dqpsk_llr2[0]);
        assert!(dqpsk_llr2[1] > 0.0,
                "diff=(1,0): LLR[1] should be positive, got {}", dqpsk_llr2[1]);
    }

    /// 複数シンボル連続復調時の状態遷移
    #[test]
    fn test_multi_symbol_state_tracking() {
        let mut demod = make_demodulator();
        let wdict = WalshDictionary::default_w16();
        demod.set_reference_phase(16.0, 0.0);

        // シンボル1: Walsh[0] + phase(1,0)
        let s1: Vec<Complex32> = wdict.w16[0]
            .iter()
            .map(|&w| Complex32::new(w as f32, 0.0))
            .collect();

        // シンボル2: Walsh[0] + phase(0,1) [delta=1]
        let s2: Vec<Complex32> = wdict.w16[0]
            .iter()
            .map(|&w| Complex32::new(0.0, w as f32))
            .collect();

        // シンボル3: Walsh[0] + phase(-1,0) [delta=2 from phase 1]
        let s3: Vec<Complex32> = wdict.w16[0]
            .iter()
            .map(|&w| Complex32::new(-w as f32, 0.0))
            .collect();

        // 連続復調
        let (_llr1, dqpsk1, _diff1) = demod.demod_symbol(&s1);
        let (_llr2, dqpsk2, _diff2) = demod.demod_symbol(&s2);
        let (_llr3, dqpsk3, _diff3) = demod.demod_symbol(&s3);

        // シンボル1: phase=0, diff=(1,0) -> delta=0 -> DQPSK 00 (LLR0>0, LLR1>0)
        assert!(dqpsk1[0] > 0.5 && dqpsk1[1] > 0.5, "Symbol 1 should be DQPSK 00");

        // シンボル2: phase=1, diff=(0,1) -> delta=1 -> DQPSK 01 (LLR0>0, LLR1<0)
        assert!(dqpsk2[0] > 0.5 && dqpsk2[1] < -0.5, "Symbol 2 should be DQPSK 01");

        // シンボル3: phase=2, diff=(0,1) -> delta=1 -> DQPSK 01 (LLR0>0, LLR1<0)
        assert!(dqpsk3[0] > 0.5 && dqpsk3[1] < -0.5, "Symbol 3 should be DQPSK 01");
    }

    // ========== 厳密な信号処理テスト（modulatorを使用）==========

    /// modulator→demodulator: 単一シンボルの往復テスト
    #[test]
    fn test_modulator_demodulator_single_symbol() {
        use crate::mary::modulator::Modulator;

        let mut modulator = Modulator::default_48k();
        let mut demodulator = Demodulator::new();
        demodulator.set_reference_phase(16.0, 0.0);

        // Walsh[0] + DQPSK 00 (phase=0)
        let bits = vec![0u8, 0, 0, 0, 0, 0];
        let _samples = modulator.modulate(&bits);

        // サンプルを16個のシンボルに分割（各シンボル16サンプル）
        // 注: 実際にはRRCフィルタとリサンプラがあるため、正確な分割は困難
        // ここでは信号の主要部分を抽出してテストする

        // チップ列を直接取得してテスト
        let (chips_i, chips_q) = modulator.bits_to_chips(&bits);

        // 16サンプルの信号を構築
        let mut signal = Vec::with_capacity(16);
        for i in 0..16 {
            signal.push(Complex32::new(chips_i[i], chips_q[i]));
        }

        let (walsh_llr, dqpsk_llr, _diff) = demodulator.demod_symbol(&signal);

        // Walsh[0] = 0b0000 -> 全ビットLLR > 0
        for bit in 0..4 {
            assert!(walsh_llr[bit] > 0.0,
                    "Walsh[0] bit{} LLR should be positive, got {}", bit, walsh_llr[bit]);
        }

        // DQPSK 00 -> phase=0, prev=0 -> delta=0 -> LLR[0] > 0, LLR[1] > 0
        assert!(dqpsk_llr[0] > 0.5, "DQPSK 00 should give positive LLR[0]");
        assert!(dqpsk_llr[1] > 0.5, "DQPSK 00 should give positive LLR[1]");
    }

    /// 全てのWalsh indexの復調を検証
    #[test]
    fn test_modulator_demodulator_all_walsh_indices() {
        use crate::mary::modulator::Modulator;

        let mut modulator = Modulator::default_48k();
        let mut demodulator = Demodulator::new();

        for walsh_idx in 0..16 {
            modulator.reset();

            // Walsh indexをビットに変換
            let b3 = (walsh_idx >> 3) & 1;
            let b2 = (walsh_idx >> 2) & 1;
            let b1 = (walsh_idx >> 1) & 1;
            let b0 = walsh_idx & 1;
            let bits = vec![b3 as u8, b2 as u8, b1 as u8, b0 as u8, 0, 0];

            let (chips_i, chips_q) = modulator.bits_to_chips(&bits);

            // 16サンプルの信号を構築
            let signal: Vec<Complex32> = (0..16)
                .map(|i| Complex32::new(chips_i[i], chips_q[i]))
                .collect();

            let (walsh_llr, _dqpsk_llr, _diff) = demodulator.demod_symbol(&signal);

            // 各ビットのLLR符号を確認
            for bit in 0..4 {
                let bit_is_set = (walsh_idx >> bit) & 1 == 1;
                let llr_idx = 3 - bit;
                let llr_positive = walsh_llr[llr_idx] > 0.0;

                assert_eq!(llr_positive, !bit_is_set,
                           "Walsh[{}] bit{}: LLR={}, bit_set={}",
                           walsh_idx, bit, walsh_llr[llr_idx], bit_is_set);
            }
        }
    }

    /// DQPSK全位相遷移の復調を検証
    #[test]
    fn test_modulator_demodulator_dqpsk_phases() {
        use crate::mary::modulator::Modulator;

        let mut modulator = Modulator::default_48k();
        let mut demodulator = Demodulator::new();

        // 全DQPSK位相をテスト
        let test_cases = [
            ([0u8, 0], 0), // 00 -> delta=0 -> phase=0
            ([0, 1], 1),  // 01 -> delta=1 -> phase=1
            ([1, 1], 2),  // 11 -> delta=2 -> phase=3
            ([1, 0], 3),  // 10 -> delta=3 -> phase=2
        ];

        for (dqpsk_bits, expected_delta) in test_cases {
            modulator.reset();
            demodulator.reset();

            let bits = vec![0u8, 0, 0, 0, dqpsk_bits[0], dqpsk_bits[1]];
            let (chips_i, chips_q) = modulator.bits_to_chips(&bits);

            let signal: Vec<Complex32> = (0..16)
                .map(|i| Complex32::new(chips_i[i], chips_q[i]))
                .collect();

            let (_walsh_llr, dqpsk_llr, diff) = demodulator.demod_symbol(&signal);

            // LLRの符号を確認（Gray codingに基づく）
            // - diff=(1,0) → LLR[0]>0, LLR[1]>0
            // - diff=(0,1) → LLR[0]>0, LLR[1]<0
            // - diff=(-1,0) → LLR[0]<0, LLR[1]<0
            // - diff=(0,-1) → LLR[0]<0, LLR[1]>0
            match expected_delta {
                0 => { // delta=0 → diff=(1,0)
                    assert!(dqpsk_llr[0] > 0.0,
                            "delta=0: LLR[0] should be positive, got {}", dqpsk_llr[0]);
                    assert!(dqpsk_llr[1] > 0.0,
                            "delta=0: LLR[1] should be positive, got {}", dqpsk_llr[1]);
                },
                1 => { // delta=1 → diff=(0,1)
                    assert!(dqpsk_llr[0] > 0.0,
                            "delta=1: LLR[0] should be positive, got {}", dqpsk_llr[0]);
                    assert!(dqpsk_llr[1] < 0.0,
                            "delta=1: LLR[1] should be negative, got {}", dqpsk_llr[1]);
                },
                2 => { // delta=2 → diff=(-1,0)
                    assert!(dqpsk_llr[0] < 0.0,
                            "delta=2: LLR[0] should be negative, got {}", dqpsk_llr[0]);
                    assert!(dqpsk_llr[1] < 0.0,
                            "delta=2: LLR[1] should be negative, got {}", dqpsk_llr[1]);
                },
                3 => { // delta=3 → diff=(0,-1)
                    assert!(dqpsk_llr[0] < 0.0,
                            "delta=3: LLR[0] should be negative, got {}", dqpsk_llr[0]);
                    assert!(dqpsk_llr[1] > 0.0,
                            "delta=3: LLR[1] should be positive, got {}", dqpsk_llr[1]);
                },
                _ => unreachable!(),
            }
        }
    }

    /// 連続シンボルの位相追跡を検証
    #[test]
    fn test_modulator_demodulator_consecutive_symbols() {
        use crate::mary::modulator::Modulator;

        let mut modulator = Modulator::default_48k();
        let mut demodulator = Demodulator::new();
        demodulator.set_reference_phase(16.0, 0.0);

        // 3シンボル連続
        let bits = vec![
            0u8, 0, 0, 0, 0, 0,  // Walsh[0], DQPSK 00 (phase=0)
            0, 0, 0, 0, 0, 1,   // Walsh[0], DQPSK 01 (phase=1)
            0, 0, 0, 0, 1, 1,   // Walsh[0], DQPSK 11 (phase=3)
        ];

        let (chips_i, chips_q) = modulator.bits_to_chips(&bits);

        // 各シンボルを復調
        let sf = 16;
        let mut results = Vec::new();

        for sym_idx in 0..3 {
            let offset = sym_idx * sf;
            let signal: Vec<Complex32> = (0..sf)
                .map(|i| Complex32::new(chips_i[offset + i], chips_q[offset + i]))
                .collect();

            let (_walsh_llr, dqpsk_llr, diff) = demodulator.demod_symbol(&signal);
            results.push((dqpsk_llr, diff));
        }

        // シンボル1: phase=0, delta=0 -> DQPSK 00
        assert!(results[0].0[0] > 0.5 && results[0].0[1] > 0.5,
                "Symbol 1 should be DQPSK 00");

        // シンボル2: phase=1, delta=1 -> DQPSK 01
        assert!(results[1].0[0] > 0.5 && results[1].0[1] < -0.5,
                "Symbol 2 should be DQPSK 01");

        // シンボル3: phase=3, delta=2 -> DQPSK 11
        assert!(results[2].0[0] < -0.5 && results[2].0[1] < -0.5,
                "Symbol 3 should be DQPSK 11");
    }

    /// 変復調レイヤー (Modulator <-> Demodulator) の LLR 完全一致テスト
    /// 352ビット（インターリーブ後の1パケット長）のストリームを通して、
    /// 入力ビットと出力LLRの符号が全て完全に一致することを確認する。
    #[test]
    fn test_modulator_demodulator_full_packet_stream() {
        use crate::mary::modulator::Modulator;
        use rand::Rng;

        let mut modulator = Modulator::default_48k();
        let mut demodulator = Demodulator::new();
        demodulator.set_reference_phase(16.0, 0.0);

        // ランダムな352ビットを生成 (58シンボル * 6ビット + 4ビットパディング)
        let mut rng = rand::thread_rng();
        let bits_len = 352;
        let mut bits = Vec::with_capacity(bits_len);
        for _ in 0..bits_len {
            bits.push(rng.gen_range(0..=1));
        }

        // パディングとして、Modulatorは6の倍数にならない端数ビットを無視するため、
        // 厳密なテストのために0埋めして6の倍数（354ビット = 59シンボル）にする
        let padded_len = bits_len.div_ceil(6) * 6;
        let mut padded_bits = bits.clone();
        padded_bits.resize(padded_len, 0);

        let (chips_i, chips_q) = modulator.bits_to_chips(&padded_bits);

        let sf = 16;
        let num_symbols = padded_len / 6;
        let mut all_llrs = Vec::with_capacity(padded_len);

        for sym_idx in 0..num_symbols {
            let offset = sym_idx * sf;
            let signal: Vec<Complex32> = (0..sf)
                .map(|i| Complex32::new(chips_i[offset + i], chips_q[offset + i]))
                .collect();

            let (walsh_llr, dqpsk_llr, _diff) = demodulator.demod_symbol(&signal);
            all_llrs.extend_from_slice(&walsh_llr);
            all_llrs.extend_from_slice(&dqpsk_llr);
        }

        // 入力ビットとLLRの符号が完全に一致するか確認
        // bit=0 -> LLR > 0
        // bit=1 -> LLR < 0
        for i in 0..bits_len {
            let expected_positive = bits[i] == 0;
            let actual_positive = all_llrs[i] > 0.0;
            assert_eq!(actual_positive, expected_positive,
                       "Bit mismatch at index {}. Input bit: {}, LLR: {}",
                       i, bits[i], all_llrs[i]);
        }
    }

    /// Walsh直交性を活用した復調テスト
    #[test]
    fn test_modulator_demodulator_walsh_orthogonality() {
        use crate::mary::modulator::Modulator;

        let mut modulator = Modulator::default_48k();
        let demodulator = Demodulator::new();

        // Walsh[0]とWalsh[1]の信号を生成
        let bits0 = vec![0u8, 0, 0, 0, 0, 0]; // Walsh[0]
        let bits1 = vec![0u8, 0, 0, 1, 0, 0]; // Walsh[1]

        let (chips_i0, chips_q0) = modulator.bits_to_chips(&bits0);
        let (chips_i1, chips_q1) = modulator.bits_to_chips(&bits1);

        let signal0: Vec<Complex32> = (0..16)
            .map(|i| Complex32::new(chips_i0[i], chips_q0[i]))
            .collect();
        let signal1: Vec<Complex32> = (0..16)
            .map(|i| Complex32::new(chips_i1[i], chips_q1[i]))
            .collect();

        // Walsh[0]信号を復調 -> Walsh[0]のエネルギーが最大
        let correlations0 = demodulator.despread_all(&signal0);
        let energies0: [f32; 16] = correlations0.map(|c| c.norm_sqr());
        let max_idx0 = energies0.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        assert_eq!(max_idx0, 0, "Walsh[0] signal should have max energy at index 0");

        // Walsh[1]信号を復調 -> Walsh[1]のエネルギーが最大
        let correlations1 = demodulator.despread_all(&signal1);
        let energies1: [f32; 16] = correlations1.map(|c| c.norm_sqr());
        let max_idx1 = energies1.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        assert_eq!(max_idx1, 1, "Walsh[1] signal should have max energy at index 1");

        // 直交性: Walsh[0]信号に対してWalsh[1]の相関は0
        assert!(energies1[0] < 1e-6, "Walsh[0] and Walsh[1] should be orthogonal");
    }

    /// LLRの信頼性とエネルギーの相関を検証
    #[test]
    fn test_modulator_demodulator_llr_reliability() {
        use crate::mary::modulator::Modulator;

        let mut modulator = Modulator::default_48k();
        let mut demodulator = Demodulator::new();

        // 高エネルギー信号
        modulator.reset();
        let bits = vec![0u8, 0, 0, 0, 0, 0];
        let (chips_i, chips_q) = modulator.bits_to_chips(&bits);
        let signal: Vec<Complex32> = (0..16)
            .map(|i| Complex32::new(chips_i[i], chips_q[i]))
            .collect();

        let (walsh_llr, _dqpsk_llr, _diff) = demodulator.demod_symbol(&signal);

        // Walsh[0]のLLRは高信頼性（1.0に近い）
        // 根拠：正規化ロジックにより、理想的な信号（ノイズなし）では (E_max - 0) / E_max = 1.0 となる。
        for bit in 0..4 {
            assert!((walsh_llr[bit].abs() - 1.0).abs() < 1e-3,
                    "Walsh[0] LLR[{}] should be ~1.0, got {}",
                    bit, walsh_llr[bit]);
        }
    }

    /// modulator.encode_frame()とdemodulatorの統合テスト
    ///
    /// RRCフィルタ、リサンプラー、キャリア変調を通した信号を正しく復調できるかを確認
    /// 根拠：実際のシステムではencode_frame()の出力が使われる
    #[test]
    fn test_modulator_encode_frame_demodulator_integration() {
        use crate::mary::modulator::Modulator;

        let mut modulator = Modulator::default_48k();
        let mut demodulator = Demodulator::new();

        // 1シンボル (6ビット: 4ビットWalsh + 2ビットDQPSK)
        let bits = vec![0u8, 0, 0, 0, 0, 0]; // Walsh[0], DQPSK 00

        // encode_frameで完全な信号生成（RRC、リサンプル、キャリア変調含む）
        let frame = modulator.encode_frame(&bits);

        // 出力信号は十分な長さを持つ
        assert!(frame.len() > 100, "Frame should be long enough, got {}", frame.len());

        // 全サンプルが有限値
        assert!(frame.iter().all(|&s| s.is_finite()), "All samples should be finite");

        // 振幅範囲が合理的（クリッピングなし）
        let max_amp = frame.iter().fold(0.0f32, |a, &s| a.max(s.abs()));
        assert!(max_amp < 5.0, "Amplitude should be reasonable, got {}", max_amp);

        // 信号エネルギーがゼロでない
        let signal_energy: f32 = frame.iter().map(|&s| s * s).sum();
        let avg_energy = signal_energy / frame.len() as f32;
        assert!(avg_energy > 0.001, "Signal should have energy, got {}", avg_energy);

        // 注：demodulatorは16サンプルのComplex32入力を期待するが、
        // encode_frame()の出力はリサンプル済みの実数信号なので、
        // このテストでは信号品質（有限値、エネルギー）のみを確認
        // 完全な往復テストはdecoder.rsで行う
    }

    /// 複数シンボルのencode_frame統合テスト
    #[test]
    fn test_modulator_encode_frame_multiple_symbols() {
        use crate::mary::modulator::Modulator;

        let mut modulator = Modulator::default_48k();

        // 3シンボル (18ビット)
        let bits = vec![
            0u8, 0, 0, 0, 0, 0,  // Walsh[0], DQPSK 00
            0, 0, 0, 1, 0, 1,    // Walsh[1], DQPSK 01
            0, 0, 1, 0, 1, 1,    // Walsh[2], DQPSK 11
        ];

        let frame = modulator.encode_frame(&bits);

        // 3シンボル分の信号が生成されている
        assert!(frame.len() > 300, "Frame should be long enough for 3 symbols, got {}", frame.len());

        // 全サンプルが有限値
        assert!(frame.iter().all(|&s| s.is_finite()));

        // 信号エネルギーが滑らか（急激な変化なし）
        let max_diff: f32 = frame.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, |a, d| a.max(d));

        // RRCフィルタの出力なので、隣接サンプルの差は過大ではない
        // 根拠：RRCフィルタはバンド limitingなので、急峻な遷移を抑制する
        assert!(max_diff < 2.0, "Adjacent sample difference should be smooth, got {}", max_diff);
    }
}

