//! 逐次処理向け FIR デシメータ
//!
//! refs/radio/hackrf-dsp/src/filter.rs の実装方針を、
//! 本DSP向けに実数ストリーム専用で簡略化したもの。

fn design_lowpass_hamming_coeffs(num_taps: usize, cutoff_norm: f32) -> Vec<f32> {
    assert!(num_taps > 0, "num_taps must be > 0");
    assert!(
        cutoff_norm > 0.0 && cutoff_norm < 0.5,
        "Invalid cutoff_norm={}",
        cutoff_norm
    );

    let mut coeffs = vec![0.0; num_taps];
    let center = (num_taps - 1) as f32 / 2.0;
    let alpha = 0.54;
    let beta = 0.46;

    for (i, coeff) in coeffs.iter_mut().enumerate() {
        let n = i as f32 - center;
        let sinc = if n == 0.0 {
            2.0 * cutoff_norm
        } else {
            (2.0 * std::f32::consts::PI * cutoff_norm * n).sin() / (std::f32::consts::PI * n)
        };
        let window =
            alpha - beta * (2.0 * std::f32::consts::PI * i as f32 / (num_taps - 1) as f32).cos();
        *coeff = sinc * window;
    }

    let gain = coeffs.iter().sum::<f32>().max(1e-8);
    for c in &mut coeffs {
        *c /= gain;
    }
    coeffs
}

fn update_history(history: &mut [f32], input: &[f32]) {
    let hist_len = history.len();
    if hist_len == 0 {
        return;
    }

    if input.len() >= hist_len {
        history.copy_from_slice(&input[input.len() - hist_len..]);
    } else {
        let shift = input.len();
        history.copy_within(shift.., 0);
        history[hist_len - shift..].copy_from_slice(input);
    }
}

/// 実数ストリーム用 FIR デシメータ（ブロック境界を跨いで位相を保持）
pub struct FirDecimator {
    factor: usize,
    phase: usize,
    history: Vec<f32>,
    coeffs: Vec<f32>,
}

impl FirDecimator {
    pub fn new_lowpass_hamming(factor: usize, num_taps: usize, cutoff_norm: f32) -> Self {
        assert!(factor > 0, "factor must be > 0");
        assert!(num_taps > 0, "num_taps must be > 0");
        let coeffs = design_lowpass_hamming_coeffs(num_taps, cutoff_norm);
        Self {
            factor,
            phase: 0,
            history: vec![0.0; num_taps - 1],
            coeffs,
        }
    }

    pub fn factor(&self) -> usize {
        self.factor
    }

    pub fn reset(&mut self) {
        self.phase = 0;
        self.history.fill(0.0);
    }

    pub fn process_into(&mut self, input: &[f32], output: &mut Vec<f32>) {
        output.clear();
        if input.is_empty() {
            return;
        }

        output.reserve(input.len() / self.factor + 1);

        let mut current_idx = if self.phase == 0 {
            0
        } else {
            self.factor - self.phase
        };

        while current_idx < input.len() {
            let mut acc = 0.0f32;
            for (i, &coeff) in self.coeffs.iter().enumerate() {
                let val = if current_idx >= i {
                    input[current_idx - i]
                } else {
                    let history_back = i - current_idx;
                    self.history[self.history.len() - history_back]
                };
                acc += val * coeff;
            }
            output.push(acc);
            current_idx += self.factor;
        }

        self.phase = (self.phase + input.len()) % self.factor;
        update_history(&mut self.history, input);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fir_decimator_chunk_invariance() {
        let mut input = Vec::with_capacity(20_017);
        for i in 0..20_017 {
            let t = i as f32 / 48_000.0;
            input.push(
                0.8 * (2.0 * std::f32::consts::PI * 1_200.0 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 9_000.0 * t).sin(),
            );
        }

        let mut whole = FirDecimator::new_lowpass_hamming(3, 63, 0.15);
        let mut out_whole = Vec::new();
        whole.process_into(&input, &mut out_whole);

        let mut chunked = FirDecimator::new_lowpass_hamming(3, 63, 0.15);
        let mut out_chunks = Vec::new();
        let mut tmp = Vec::new();
        for chunk in input.chunks(257) {
            chunked.process_into(chunk, &mut tmp);
            out_chunks.extend_from_slice(&tmp);
        }

        assert_eq!(out_whole.len(), out_chunks.len());
        let max_err = out_whole
            .iter()
            .zip(out_chunks.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_err < 1e-5, "max_err={}", max_err);
    }
}
