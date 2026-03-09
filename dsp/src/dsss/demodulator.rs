//! 差動PSK + DSSS 復調器

use crate::{DifferentialModulation, DspConfig};
use num_complex::Complex32;

fn decode_symbol_bits(
    mode: DifferentialModulation,
    diff: Complex32,
    out: &mut Vec<u8>,
) -> Complex32 {
    match mode {
        DifferentialModulation::Dbpsk => {
            if diff.re >= 0.0 {
                out.push(0);
                Complex32::new(1.0, 0.0)
            } else {
                out.push(1);
                Complex32::new(-1.0, 0.0)
            }
        }
        DifferentialModulation::Dqpsk => {
            if diff.re.abs() >= diff.im.abs() {
                if diff.re >= 0.0 {
                    out.extend_from_slice(&[0, 0]);
                    Complex32::new(1.0, 0.0)
                } else {
                    out.extend_from_slice(&[1, 1]);
                    Complex32::new(-1.0, 0.0)
                }
            } else if diff.im >= 0.0 {
                out.extend_from_slice(&[0, 1]);
                Complex32::new(0.0, 1.0)
            } else {
                out.extend_from_slice(&[1, 0]);
                Complex32::new(0.0, -1.0)
            }
        }
    }
}

/// 差動PSK + DSSS 復調器
pub struct Demodulator {
    config: DspConfig,
    mode: DifferentialModulation,
    prev: Complex32,
}

impl Demodulator {
    pub fn new(config: DspConfig) -> Self {
        Self::new_with_mode(config, crate::params::MODULATION)
    }

    pub fn new_with_mode(config: DspConfig, mode: DifferentialModulation) -> Self {
        Demodulator {
            config,
            mode,
            prev: Complex32::new(1.0, 0.0),
        }
    }

    /// チップ列（拡散されたベースバンド信号）からビット列を復元する
    pub fn demodulate_chips(&mut self, chips_i: &[f32], chips_q: &[f32]) -> Vec<u8> {
        let sf = self.config.spread_factor();
        let num_symbols = chips_i.len() / sf;
        let mut bits = Vec::with_capacity(num_symbols * self.mode.bits_per_symbol());

        let mut mseq = crate::common::msequence::MSequence::new(self.config.mseq_order);
        let mut pn_i8 = Vec::with_capacity(sf);
        mseq.generate_into(sf, &mut pn_i8);
        let pn: Vec<f32> = pn_i8.into_iter().map(|c| c as f32).collect();

        for s_idx in 0..num_symbols {
            let start = s_idx * sf;
            let end = start + sf;
            let symbol_chips_i = &chips_i[start..end];
            let symbol_chips_q = &chips_q[start..end];

            let mut cur_i = 0.0f32;
            let mut cur_q = 0.0f32;
            for (k, &pn_val) in pn.iter().enumerate() {
                cur_i += symbol_chips_i[k] * pn_val;
                cur_q += symbol_chips_q[k] * pn_val;
            }
            cur_i /= sf as f32;
            cur_q /= sf as f32;
            let cur = Complex32::new(cur_i, cur_q);
            let diff = cur * self.prev.conj();
            let decided = decode_symbol_bits(self.mode, diff, &mut bits);
            let norm = cur.norm();
            if norm > 1e-4 {
                self.prev = cur / norm;
            } else {
                self.prev = decided;
            }
        }

        bits
    }

    pub fn reset(&mut self) {
        self.prev = Complex32::new(1.0, 0.0);
    }

    pub fn set_reference_phase(&mut self, i: f32, q: f32) {
        let mag = (i * i + q * q).sqrt().max(1e-6);
        self.prev = Complex32::new(i / mag, q / mag);
    }
}
