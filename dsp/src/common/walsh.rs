//! ウォルシュ・アダマール直交系列とPNスクランブル辞書生成
//!
//! 16-ary DQPSK などのため、長さ16の完全に直交する16個の系列を生成する。
//! そのままだと直流成分等が多く音響特性や同期特性が悪いため、
//! PN系列（長さ15のM系列 + 1）を用いて全系列をスクランブルしホワイトノイズ化する。

use num_complex::Complex32;

use super::msequence::MSequence;

/// 指定サイズのウォルシュ・アダマール行列を生成する。
fn generate_hadamard_matrix(order: usize) -> Vec<Vec<i8>> {
    let mut matrix = vec![vec![1i8]];

    let mut current_size = 1;
    while current_size < order {
        let mut next_matrix = vec![vec![0i8; current_size * 2]; current_size * 2];
        for r in 0..current_size {
            for c in 0..current_size {
                let val = matrix[r][c];
                next_matrix[r][c] = val;
                next_matrix[r][c + current_size] = val;
                next_matrix[r + current_size][c] = val;
                next_matrix[r + current_size][c + current_size] = -val;
            }
        }
        matrix = next_matrix;
        current_size *= 2;
    }

    matrix
}

/// 16x16の直交系列辞書 (W_16)
///
/// H_16 の各行に対して指定されたPN系列でスクランブル（要素ごとの乗算）を行った結果。
#[derive(Clone, Debug)]
pub struct WalshDictionary {
    /// 16個の系列。各系列は長さ16。
    pub w16: Vec<Vec<i8>>,
    /// スクランブルに使用されたPN系列 (長さ16)
    pub pn: Vec<i8>,
}

impl WalshDictionary {
    /// PN系列によるスクランブル済みの16x16辞書を生成する
    ///
    /// `pn`: スクランブルに用いる長さ16のPN系列。
    /// （例: 長さ15のM系列の末尾に `1` または `-1` を追加したもの）
    pub fn new(pn: &[i8]) -> Self {
        assert_eq!(pn.len(), 16, "PN sequence must have length 16");
        let h16 = generate_hadamard_matrix(16);
        let mut w16 = vec![vec![0i8; 16]; 16];

        for r in 0..16 {
            for c in 0..16 {
                w16[r][c] = h16[r][c] * pn[c];
            }
        }

        Self {
            w16,
            pn: pn.to_vec(),
        }
    }

    /// M系列 (次数4, 長さ15) の末尾に +1 を追加したデフォルトのPN系列による辞書を生成
    pub fn default_w16() -> Self {
        let mut mseq = MSequence::new(4);
        let mut pn = mseq.generate(15);
        pn.push(1); // 16要素目を追加
        Self::new(&pn)
    }
}

/// Walsh相関器（1系列分）
///
/// 指定されたWalsh系列と受信信号の相関を計算する。
#[derive(Clone, Debug)]
pub struct WalshCorrelator {
    sequence: Vec<i8>,
}

impl WalshCorrelator {
    /// 新しいWalsh相関器を作成する
    ///
    /// `sequence`: 長さ16のWalsh系列 ({-1, +1}の値)
    pub fn new(sequence: Vec<i8>) -> Self {
        assert_eq!(sequence.len(), 16, "Walsh sequence must have length 16");
        for &val in &sequence {
            assert!(val == 1 || val == -1, "Walsh sequence values must be +/- 1");
        }
        Self { sequence }
    }

    pub fn sequence(&self) -> &[i8] {
        &self.sequence
    }

    /// 信号との相関を計算する
    ///
    /// `signal`: 長さ16の複素数信号
    ///
    /// 戻り値: 相関値（複素数）
    pub fn correlate(&self, signal: &[Complex32]) -> Complex32 {
        assert_eq!(signal.len(), 16, "Signal must have length 16");
        let mut result = Complex32::new(0.0, 0.0);
        for (&seq_val, sig_val) in self.sequence.iter().zip(signal.iter()) {
            let seq_f = seq_val as f32;
            result += sig_val * seq_f;
        }
        result
    }

    /// 信号との相関の絶対値（エネルギー）を計算する
    ///
    /// `signal`: 長さ16の複素数信号
    ///
    /// 戻り値: 相関値の絶対値
    pub fn correlate_energy(&self, signal: &[Complex32]) -> f32 {
        self.correlate(signal).norm_sqr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard_matrix_generation() {
        let h16 = generate_hadamard_matrix(16);
        assert_eq!(h16.len(), 16);
        for row in &h16 {
            assert_eq!(row.len(), 16);
            for &val in row {
                assert!(val == 1 || val == -1);
            }
        }

        // 行の直交性チェック
        for i in 0..16 {
            for j in 0..16 {
                let dot: i32 = h16[i]
                    .iter()
                    .zip(h16[j].iter())
                    .map(|(&a, &b)| a as i32 * b as i32)
                    .sum();
                if i == j {
                    assert_eq!(dot, 16);
                } else {
                    assert_eq!(dot, 0);
                }
            }
        }
    }

    #[test]
    fn test_walsh_dictionary_orthogonality() {
        let dict = WalshDictionary::default_w16();
        let w16 = &dict.w16;

        assert_eq!(w16.len(), 16);
        for row in w16 {
            assert_eq!(row.len(), 16);
        }

        // スクランブル後でも直交性が保たれていることの確認
        for i in 0..16 {
            for j in 0..16 {
                let dot: i32 = w16[i]
                    .iter()
                    .zip(w16[j].iter())
                    .map(|(&a, &b)| a as i32 * b as i32)
                    .sum();
                if i == j {
                    assert_eq!(dot, 16);
                } else {
                    assert_eq!(dot, 0);
                }
            }
        }
    }

    #[test]
    fn test_pn_contains_pm1() {
        let dict = WalshDictionary::default_w16();
        assert_eq!(dict.pn.len(), 16);
        for &val in &dict.pn {
            assert!(val == 1 || val == -1);
        }
    }

    #[test]
    fn test_pn_alignment() {
        let mut mseq = crate::common::msequence::MSequence::new(4);
        let pn1 = mseq.generate(16);
        let dict = WalshDictionary::default_w16();
        let pn2 = dict.pn.clone();

        let pn1_i8: Vec<i8> = pn1.into_iter().map(|x| x as i8).collect();
        assert_eq!(pn1_i8, pn2, "pn1={:?} pn2={:?}", pn1_i8, pn2);
    }

    // WalshCorrelatorのテスト
    #[test]
    fn test_walsh_correlator_construction() {
        let sequence = vec![1i8; 16];
        let correlator = WalshCorrelator::new(sequence.clone());
        assert_eq!(correlator.sequence, sequence);
    }

    #[test]
    fn test_walsh_correlator_invalid_length() {
        let sequence = vec![1i8; 15];
        let result = std::panic::catch_unwind(|| {
            WalshCorrelator::new(sequence);
        });
        assert!(result.is_err(), "Should panic for invalid length");
    }

    #[test]
    fn test_walsh_correlator_invalid_values() {
        let sequence = vec![0i8; 16];
        let result = std::panic::catch_unwind(|| {
            WalshCorrelator::new(sequence);
        });
        assert!(result.is_err(), "Should panic for invalid values");
    }

    #[test]
    fn test_walsh_correlate_perfect_match() {
        let sequence = vec![1i8; 16];
        let correlator = WalshCorrelator::new(sequence);
        let signal: Vec<Complex32> = (0..16).map(|_| Complex32::new(1.0, 0.0)).collect();
        let result = correlator.correlate(&signal);
        assert!((result.re - 16.0).abs() < 1e-6);
        assert!((result.im).abs() < 1e-6);
    }

    #[test]
    fn test_walsh_correlate_energy() {
        let sequence = vec![1i8; 16];
        let correlator = WalshCorrelator::new(sequence);
        let signal: Vec<Complex32> = (0..16).map(|_| Complex32::new(1.0, 0.0)).collect();
        let energy = correlator.correlate_energy(&signal);
        assert!((energy - 256.0).abs() < 1e-6);
    }

    #[test]
    fn test_walsh_correlate_complex_signal() {
        let sequence = vec![1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1];
        let correlator = WalshCorrelator::new(sequence);
        let signal: Vec<Complex32> = (0..16)
            .map(|i| {
                if i % 2 == 0 {
                    Complex32::new(1.0, 0.0)
                } else {
                    Complex32::new(-1.0, 0.0)
                }
            })
            .collect();
        let result = correlator.correlate(&signal);
        assert!((result.re - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_walsh_dictionary_correlators_orthogonality() {
        let dict = WalshDictionary::default_w16();
        let signal0: Vec<Complex32> = dict.w16[0]
            .iter()
            .map(|&v| Complex32::new(v as f32, 0.0))
            .collect();

        for i in 0..16 {
            let correlator = WalshCorrelator::new(dict.w16[i].clone());
            let correlation = correlator.correlate(&signal0);

            if i == 0 {
                // 自己相関は16
                assert!(
                    (correlation.re - 16.0).abs() < 1e-6,
                    "Self-correlation should be 16"
                );
            } else {
                // 直交系列との相関は0
                assert!(correlation.norm() < 1e-6, "Cross-correlation should be 0");
            }
        }
    }
}
