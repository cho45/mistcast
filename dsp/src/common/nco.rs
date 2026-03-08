use num_complex::Complex;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use std::arch::wasm32::{f32x4, f32x4_add, f32x4_mul, f32x4_sub, i32x4_shuffle, v128};

/// 複素ベースバンド変換のためのNCO。
pub struct Nco {
    osc: Complex<f32>,
    phase_inc: Complex<f32>,
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    phase_pairs: [v128; 4],
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    phase_inc8: Complex<f32>,
    renorm_counter: u32,
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
pub(crate) fn complex_mul_interleaved2_simd(input: v128, osc: v128) -> v128 {
    let osc_swapped = i32x4_shuffle::<1, 0, 3, 2>(osc, osc);

    let prod_re = f32x4_mul(input, osc);
    let prod_im = f32x4_mul(input, osc_swapped);

    let prod_re_swapped = i32x4_shuffle::<1, 0, 3, 2>(prod_re, prod_re);
    let prod_im_swapped = i32x4_shuffle::<1, 0, 3, 2>(prod_im, prod_im);

    let re = f32x4_sub(prod_re, prod_re_swapped);
    let im = f32x4_add(prod_im, prod_im_swapped);

    i32x4_shuffle::<0, 4, 2, 6>(re, im)
}

impl Nco {
    pub fn new(freq_hz: f32, sample_rate: f32) -> Self {
        let dphi = 2.0 * std::f32::consts::PI * freq_hz / sample_rate;
        let phase_inc = Complex::new(dphi.cos(), dphi.sin());
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        let (phase_pairs, phase_inc8) = {
            let mut p = Complex::new(1.0, 0.0);
            let mut powers = [Complex::new(0.0, 0.0); 8];
            for slot in &mut powers {
                *slot = p;
                p *= phase_inc;
            }
            (
                [
                    f32x4(powers[0].re, powers[0].im, powers[1].re, powers[1].im),
                    f32x4(powers[2].re, powers[2].im, powers[3].re, powers[3].im),
                    f32x4(powers[4].re, powers[4].im, powers[5].re, powers[5].im),
                    f32x4(powers[6].re, powers[6].im, powers[7].re, powers[7].im),
                ],
                p,
            )
        };

        Self {
            osc: Complex::new(1.0, 0.0),
            phase_inc,
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            phase_pairs,
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            phase_inc8,
            renorm_counter: 0,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.osc = Complex::new(1.0, 0.0);
        self.renorm_counter = 0;
    }

    /// 現在の発振値を返し、1サンプル進める。
    #[inline]
    pub fn step(&mut self) -> Complex<f32> {
        let val = self.osc;
        self.osc *= self.phase_inc;
        self.renorm_counter = self.renorm_counter.wrapping_add(1);

        if self.renorm_counter >= 1024 {
            self.renorm_counter = 0;
            let norm = self.osc.norm();
            if norm > 1e-12 {
                self.osc /= norm;
            } else {
                self.osc = Complex::new(1.0, 0.0);
            }
        }
        val
    }

    /// nサンプル分発振を進める。
    pub fn skip(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[inline]
    pub fn step8_interleaved(&mut self) -> (v128, v128, v128, v128) {
        let osc_pair = f32x4(self.osc.re, self.osc.im, self.osc.re, self.osc.im);
        let n0 = complex_mul_interleaved2_simd(osc_pair, self.phase_pairs[0]);
        let n1 = complex_mul_interleaved2_simd(osc_pair, self.phase_pairs[1]);
        let n2 = complex_mul_interleaved2_simd(osc_pair, self.phase_pairs[2]);
        let n3 = complex_mul_interleaved2_simd(osc_pair, self.phase_pairs[3]);

        self.osc *= self.phase_inc8;
        self.renorm_counter = self.renorm_counter.wrapping_add(8);
        if self.renorm_counter >= 1024 {
            self.renorm_counter = 0;
            let norm = self.osc.norm();
            if norm > 1e-12 {
                self.osc /= norm;
            } else {
                self.osc = Complex::new(1.0, 0.0);
            }
        }

        (n0, n1, n2, n3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    use std::arch::wasm32::{v128, v128_store};
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    use wasm_bindgen_test::wasm_bindgen_test;

    #[test]
    fn test_nco_frequency() {
        let sample_rate = 1000.0;
        let freq = 250.0;
        let mut nco = Nco::new(freq, sample_rate);

        let c0 = nco.step();
        assert!((c0.re - 1.0).abs() < 1e-6);
        assert!((c0.im - 0.0).abs() < 1e-6);

        let c1 = nco.step();
        assert!(c1.re.abs() < 1e-6);
        assert!((c1.im - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_nco_reset() {
        let mut nco = Nco::new(-100.0, 1000.0);
        for _ in 0..64 {
            let _ = nco.step();
        }
        nco.reset();
        let c = nco.step();
        assert!((c.re - 1.0).abs() < 1e-6);
        assert!(c.im.abs() < 1e-6);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn unpack_interleaved_pair(v: v128) -> [Complex<f32>; 2] {
        let mut lanes = [0.0f32; 4];
        unsafe {
            v128_store(lanes.as_mut_ptr() as *mut v128, v);
        }
        [
            Complex::new(lanes[0], lanes[1]),
            Complex::new(lanes[2], lanes[3]),
        ]
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn assert_complex_close(a: Complex<f32>, b: Complex<f32>, eps: f32) {
        assert!(
            (a.re - b.re).abs() <= eps && (a.im - b.im).abs() <= eps,
            "a=({:.8},{:.8}) b=({:.8},{:.8}) eps={}",
            a.re,
            a.im,
            b.re,
            b.im,
            eps
        );
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[wasm_bindgen_test]
    fn test_nco_step8_interleaved_matches_scalar_step_once() {
        let mut simd = Nco::new(-1234.5, 48_000.0);
        let mut scalar = Nco::new(-1234.5, 48_000.0);

        let (n0, n1, n2, n3) = simd.step8_interleaved();
        let simd_vals = [
            unpack_interleaved_pair(n0),
            unpack_interleaved_pair(n1),
            unpack_interleaved_pair(n2),
            unpack_interleaved_pair(n3),
        ];

        for i in 0..8 {
            let expected = scalar.step();
            let pair = simd_vals[i / 2];
            let actual = pair[i % 2];
            assert_complex_close(actual, expected, 1e-6);
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[wasm_bindgen_test]
    fn test_nco_step8_interleaved_matches_scalar_step_over_many_blocks() {
        let mut simd = Nco::new(789.25, 44_100.0);
        let mut scalar = Nco::new(789.25, 44_100.0);

        for _ in 0..300 {
            let (n0, n1, n2, n3) = simd.step8_interleaved();
            let simd_vals = [
                unpack_interleaved_pair(n0),
                unpack_interleaved_pair(n1),
                unpack_interleaved_pair(n2),
                unpack_interleaved_pair(n3),
            ];
            for i in 0..8 {
                let expected = scalar.step();
                let pair = simd_vals[i / 2];
                let actual = pair[i % 2];
                assert_complex_close(actual, expected, 2e-5);
            }
        }
    }
}
