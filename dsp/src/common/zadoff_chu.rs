//! Zadoff-Chu 系列 生成器
//!
//! Zadoff-Chu (ZC) 系列は、一定の包絡線（CAZAC: Constant Amplitude Zero Autocorrelation Waveform）
//! を持ち、理想的な周期自己相関特性を示す複素数系列である。
//!
//! ## 定義
//! 系列長を N_ZC、ルートインデックスを u (0 < u < N_ZC, gcd(u, N_ZC) = 1) とすると、
//!
//! N_ZC が偶数の場合:
//!   x_u(n) = exp(-j * pi * u * n^2 / N_ZC)
//!
//! N_ZC が奇数の場合:
//!   x_u(n) = exp(-j * pi * u * n * (n + 1) / N_ZC)
//!
//! ## 特性
//! - 周期自己相関のピーク以外の値は 0 になる。
//! - マルチパスのような遅延プロファイル推定に非常に有利。
//! - プリアンブル等での利用が適している。

use std::f32::consts::PI;
use num_complex::Complex32;

pub struct ZadoffChu {
    n_zc: usize,
    u: usize,
}

impl ZadoffChu {
    /// 長さ `n_zc`、ルートインデックス `u` のZC系列生成器を作成する
    ///
    /// # 制約 (Preconditions)
    /// - `u` は `(0, n_zc)` の範囲にあること。
    /// - **[重要]** `u` と `n_zc` は互いに素（Coprime, GCDが1）でなければならない。
    ///   （互いに素でない場合、理想的な周期自己相関特性が失われます。この条件の保証は呼び出し側の責務です。）
    pub fn new(n_zc: usize, u: usize) -> Self {
        assert!(u > 0 && u < n_zc, "u must be in (0, N_ZC)");
        Self { n_zc, u }
    }

    /// 系列の `n` 番目の要素を生成する
    pub fn generate_element(&self, n: usize) -> Complex32 {
        let root = self.u as f32;
        let n_f32 = n as f32;
        let n_zc_f32 = self.n_zc as f32;

        let phase = if self.n_zc % 2 == 0 {
            // 偶数の場合: -pi * u * n^2 / N_ZC
            -PI * root * n_f32 * n_f32 / n_zc_f32
        } else {
            // 奇数の場合: -pi * u * n * (n + 1) / N_ZC
            -PI * root * n_f32 * (n_f32 + 1.0) / n_zc_f32
        };

        let (sin_v, cos_v) = phase.sin_cos();
        Complex32::new(cos_v, sin_v)
    }

    /// 1周期分の系列を生成する
    pub fn generate_sequence(&self) -> Vec<Complex32> {
        let mut seq = Vec::with_capacity(self.n_zc);
        let root = self.u as f64;
        let n_zc_f64 = self.n_zc as f64;
        let pi = std::f64::consts::PI;
        
        if self.n_zc % 2 == 0 {
            for n in 0..self.n_zc {
                let n_f64 = n as f64;
                let phase = -pi * root * n_f64 * n_f64 / n_zc_f64;
                let (sin_v, cos_v) = phase.sin_cos();
                seq.push(Complex32::new(cos_v as f32, sin_v as f32));
            }
        } else {
            for n in 0..self.n_zc {
                let n_f64 = n as f64;
                let phase = -pi * root * n_f64 * (n_f64 + 1.0) / n_zc_f64;
                // 位相を 2*PI でモジュロを取ることで f32 にダウンキャストした際も正確にする
                // f64 で計算していれば sin_cos 自体は正確だが、念のため。
                // 実際には f64::sin_cos で十分高精度。
                let (sin_v, cos_v) = phase.sin_cos();
                seq.push(Complex32::new(cos_v as f32, sin_v as f32));
            }
        }
        seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 振幅が一定 (Constant Amplitude) であることを確認
    #[test]
    fn test_constant_amplitude() {
        let zc = ZadoffChu::new(13, 1);
        let seq = zc.generate_sequence();
        for val in seq {
            let mag_sq = val.norm_sqr();
            assert!((mag_sq - 1.0).abs() < 1e-4, "マグニチュード: {}", mag_sq);
        }
    }

    /// 周期自己相関 (Periodic Autocorrelation) の確認
    #[test]
    fn test_periodic_autocorrelation() {
        let n_zc = 13;
        let zc = ZadoffChu::new(n_zc, 1);
        let seq = zc.generate_sequence();

        // R(0)
        let mut r0 = Complex32::new(0.0, 0.0);
        for i in 0..n_zc {
            r0 += seq[i] * seq[i].conj();
        }

        assert!((r0.re - n_zc as f32).abs() < 1e-3, "R(0) should be N_ZC, got {}", r0.re);
        assert!(r0.im.abs() < 1e-3);

        // R(k) for k = 1..N_ZC-1
        for k in 1..n_zc {
            let mut rk = Complex32::new(0.0, 0.0);
            for i in 0..n_zc {
                rk += seq[i] * seq[(i + k) % n_zc].conj();
            }
            
            assert!(rk.norm() < 1e-3, "R({}) should be 0, got {}", k, rk.norm());
        }
    }

    #[test]
    fn test_even_length() {
        let n_zc = 16;
        let zc = ZadoffChu::new(n_zc, 3);
        let seq = zc.generate_sequence();

        for k in 1..n_zc {
            let mut rk = Complex32::new(0.0, 0.0);
            for i in 0..n_zc {
                rk += seq[i] * seq[(i + k) % n_zc].conj();
            }
            assert!(rk.norm() < 1e-3, "R({}) should be 0 for N_ZC=16, got {}", k, rk.norm());
        }
    }

    #[test]
    fn test_sweep_lengths() {
        for sf in [13, 31, 63, 127] {
            let zc = ZadoffChu::new(sf, 1);
            let seq = zc.generate_sequence();

            let mut max_side_lobe = 0.0f32;
            for k in 1..sf {
                let mut rk = Complex32::new(0.0, 0.0);
                for i in 0..sf {
                    rk += seq[i] * seq[(i + k) % sf].conj();
                }
                let norm = rk.norm() / sf as f32;
                if norm > max_side_lobe {
                    max_side_lobe = norm;
                }
            }
            println!("SF={}: max_side_lobe={:.4}", sf, max_side_lobe);
            // 理想的なZC系列の自己相関サイドローブは0（計算誤差を考慮して非常に小さい値であるべき）
            assert!(max_side_lobe < 1e-2, "SF={} side lobe too high: {:.4}", sf, max_side_lobe);
        }
    }
}
