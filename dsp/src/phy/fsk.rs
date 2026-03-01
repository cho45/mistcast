//! 比較評価用 BFSK PHY 実装
//!
//! 本実装は比較ベースライン用途のため、
//! 送受信フォーマットは最小構成 (preamble + sync + len + payload + crc16)。

use crate::common::crc;

#[derive(Clone, Debug)]
pub struct BfskConfig {
    pub sample_rate: f32,
    pub bit_rate: f32,
    pub freq0: f32,
    pub freq1: f32,
    pub amplitude: f32,
    pub preamble_bits: usize,
    pub sync_word: u16,
    pub sync_bits: usize,
}

impl BfskConfig {
    pub fn default_48k() -> Self {
        Self {
            sample_rate: 48_000.0,
            bit_rate: 1_200.0,
            freq0: 6_000.0,
            freq1: 9_000.0,
            amplitude: 0.5,
            preamble_bits: 32,
            sync_word: 0xD391,
            sync_bits: 16,
        }
    }

    pub fn samples_per_bit(&self) -> usize {
        (self.sample_rate / self.bit_rate).round().max(1.0) as usize
    }
}

#[derive(Clone, Debug)]
pub struct BfskModulator {
    cfg: BfskConfig,
}

impl BfskModulator {
    pub fn new(cfg: BfskConfig) -> Self {
        Self { cfg }
    }

    pub fn modulate_frame(&self, payload: &[u8]) -> Vec<f32> {
        assert!(
            payload.len() <= u8::MAX as usize,
            "payload too large for fsk test frame"
        );

        let mut bytes = Vec::with_capacity(1 + payload.len() + 2);
        bytes.push(payload.len() as u8);
        bytes.extend_from_slice(payload);
        let crc = crc::crc16(&bytes);
        bytes.push((crc >> 8) as u8);
        bytes.push((crc & 0xFF) as u8);

        let mut bits = self.header_bits();
        bits.extend(bytes_to_bits(&bytes));

        self.modulate_bits(&bits)
    }

    fn modulate_bits(&self, bits: &[u8]) -> Vec<f32> {
        let spb = self.cfg.samples_per_bit();
        let mut out = Vec::with_capacity(bits.len() * spb);
        let mut phase = 0.0f32;
        let two_pi = 2.0f32 * std::f32::consts::PI;

        for &b in bits {
            let freq = if b == 0 {
                self.cfg.freq0
            } else {
                self.cfg.freq1
            };
            let dphi = two_pi * freq / self.cfg.sample_rate;
            for _ in 0..spb {
                out.push(self.cfg.amplitude * phase.sin());
                phase += dphi;
                if phase > two_pi {
                    phase -= two_pi;
                }
            }
        }

        out
    }

    fn header_bits(&self) -> Vec<u8> {
        let mut bits = Vec::with_capacity(self.cfg.preamble_bits + self.cfg.sync_bits);
        for i in 0..self.cfg.preamble_bits {
            bits.push((i & 1) as u8);
        }
        for i in (0..self.cfg.sync_bits).rev() {
            bits.push(((self.cfg.sync_word >> i) & 1) as u8);
        }
        bits
    }
}

#[derive(Clone, Debug)]
pub struct BfskDemodulator {
    cfg: BfskConfig,
}

impl BfskDemodulator {
    pub fn new(cfg: BfskConfig) -> Self {
        Self { cfg }
    }

    pub fn decode_frame_aligned(&self, samples: &[f32]) -> Option<Vec<u8>> {
        self.decode_frame_at(samples, 0)
    }

    pub fn find_and_decode(&self, samples: &[f32]) -> Option<Vec<u8>> {
        let start = self.find_frame_start(samples)?;
        self.decode_frame_at(samples, start)
    }

    fn decode_frame_at(&self, samples: &[f32], start_sample: usize) -> Option<Vec<u8>> {
        let spb = self.cfg.samples_per_bit();
        let header_bits_len = self.cfg.preamble_bits + self.cfg.sync_bits;
        let payload_bits_start = start_sample + header_bits_len * spb;

        let len_bits = self.demod_bits(samples, payload_bits_start, 8)?;
        let frame_len = bits_to_bytes(&len_bits).first().copied()? as usize;

        let total_bytes = 1 + frame_len + 2;
        let total_bits = total_bytes * 8;
        let frame_bits = self.demod_bits(samples, payload_bits_start, total_bits)?;
        let frame = bits_to_bytes(&frame_bits);
        if frame.len() != total_bytes || frame.first().copied()? as usize != frame_len {
            return None;
        }

        let crc_pos = 1 + frame_len;
        let data_with_len = &frame[..crc_pos];
        let expected_crc = crc::crc16(data_with_len);
        let actual_crc = ((frame[crc_pos] as u16) << 8) | frame[crc_pos + 1] as u16;
        if expected_crc != actual_crc {
            return None;
        }

        Some(frame[1..crc_pos].to_vec())
    }

    fn find_frame_start(&self, samples: &[f32]) -> Option<usize> {
        let spb = self.cfg.samples_per_bit();
        let header_bits = self.header_bits();
        let min_required_bits = header_bits.len() + 8 + 16;
        let min_required_samples = min_required_bits * spb;
        if samples.len() < min_required_samples {
            return None;
        }

        let search_step = (spb / 4).max(1);
        let mut best_score = f32::NEG_INFINITY;
        let mut best_start = 0usize;
        let max_start = samples.len() - min_required_samples;

        let score_at = |start: usize| {
            let mut score = 0.0f32;
            for (bit_idx, &expected) in header_bits.iter().enumerate() {
                let from = start + bit_idx * spb;
                let to = from + spb;
                let (bit, conf) = self.demod_bit(&samples[from..to]);
                if bit == expected {
                    score += 1.0 + conf;
                } else {
                    score -= 1.0 + conf;
                }
            }
            score
        };

        for start in (0..=max_start).step_by(search_step) {
            let score = score_at(start);
            if score > best_score {
                best_score = score;
                best_start = start;
            }
        }

        let refine_from = best_start.saturating_sub(search_step);
        let refine_to = (best_start + search_step).min(max_start);
        for start in refine_from..=refine_to {
            let score = score_at(start);
            if score > best_score {
                best_score = score;
                best_start = start;
            }
        }

        let threshold = header_bits.len() as f32 * 0.2;
        if best_score > threshold {
            Some(best_start)
        } else {
            None
        }
    }

    fn demod_bits(
        &self,
        samples: &[f32],
        start_sample: usize,
        bit_count: usize,
    ) -> Option<Vec<u8>> {
        let spb = self.cfg.samples_per_bit();
        let end = start_sample.checked_add(bit_count.checked_mul(spb)?)?;
        if end > samples.len() {
            return None;
        }

        let mut bits = Vec::with_capacity(bit_count);
        for idx in 0..bit_count {
            let from = start_sample + idx * spb;
            let to = from + spb;
            let (bit, _) = self.demod_bit(&samples[from..to]);
            bits.push(bit);
        }
        Some(bits)
    }

    fn demod_bit(&self, window: &[f32]) -> (u8, f32) {
        let e0 = goertzel_energy(window, self.cfg.freq0, self.cfg.sample_rate);
        let e1 = goertzel_energy(window, self.cfg.freq1, self.cfg.sample_rate);
        let bit = if e1 >= e0 { 1 } else { 0 };
        let conf = (e1 - e0).abs() / (e1 + e0 + 1e-9);
        (bit, conf)
    }

    fn header_bits(&self) -> Vec<u8> {
        let mut bits = Vec::with_capacity(self.cfg.preamble_bits + self.cfg.sync_bits);
        for i in 0..self.cfg.preamble_bits {
            bits.push((i & 1) as u8);
        }
        for i in (0..self.cfg.sync_bits).rev() {
            bits.push(((self.cfg.sync_word >> i) & 1) as u8);
        }
        bits
    }
}

fn goertzel_energy(samples: &[f32], freq: f32, sample_rate: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let w = 2.0f32 * std::f32::consts::PI * freq / sample_rate;
    let coeff = 2.0f32 * w.cos();

    let mut s_prev = 0.0f32;
    let mut s_prev2 = 0.0f32;
    for &x in samples {
        let s = x + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }

    (s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2).max(0.0)
}

fn bytes_to_bits(bytes: &[u8]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(bytes.len() * 8);
    for &b in bytes {
        for i in (0..8).rev() {
            bits.push((b >> i) & 1);
        }
    }
    bits
}

fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bits.len().div_ceil(8));
    for chunk in bits.chunks(8) {
        let mut v = 0u8;
        for &b in chunk {
            v = (v << 1) | (b & 1);
        }
        if chunk.len() < 8 {
            v <<= 8 - chunk.len();
        }
        out.push(v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use rand_distr::{Distribution, Normal};
    use std::f32::consts::PI;

    fn add_awgn(samples: &mut [f32], sigma: f32, seed: u64) {
        if sigma <= 0.0 {
            return;
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let n = Normal::new(0.0, sigma).unwrap();
        for s in samples {
            *s += n.sample(&mut rng);
        }
    }

    fn seeded_payload(len: usize, seed: u64) -> Vec<u8> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..len).map(|_| rng.gen::<u8>()).collect()
    }

    fn overwrite_bit_window_with_tone(
        samples: &mut [f32],
        cfg: &BfskConfig,
        bit_index: usize,
        bit: u8,
    ) {
        let spb = cfg.samples_per_bit();
        let from = bit_index * spb;
        let to = from + spb;
        assert!(to <= samples.len());
        let freq = if bit == 0 { cfg.freq0 } else { cfg.freq1 };
        let omega = 2.0 * PI * freq / cfg.sample_rate;
        for (n, s) in samples[from..to].iter_mut().enumerate() {
            *s = cfg.amplitude * (omega * n as f32).sin();
        }
    }

    #[test]
    fn test_goertzel_energy_prefers_target_tone() {
        let cfg = BfskConfig::default_48k();
        let spb = cfg.samples_per_bit();
        let mut tone0 = vec![0.0f32; spb];
        let mut tone1 = vec![0.0f32; spb];
        let two_pi = 2.0f32 * std::f32::consts::PI;

        for (i, v) in tone0.iter_mut().enumerate() {
            let t = i as f32 / cfg.sample_rate;
            *v = (two_pi * cfg.freq0 * t).sin();
        }
        for (i, v) in tone1.iter_mut().enumerate() {
            let t = i as f32 / cfg.sample_rate;
            *v = (two_pi * cfg.freq1 * t).sin();
        }

        let e00 = goertzel_energy(&tone0, cfg.freq0, cfg.sample_rate);
        let e01 = goertzel_energy(&tone0, cfg.freq1, cfg.sample_rate);
        let e10 = goertzel_energy(&tone1, cfg.freq0, cfg.sample_rate);
        let e11 = goertzel_energy(&tone1, cfg.freq1, cfg.sample_rate);

        assert!(e00 > e01 * 4.0);
        assert!(e11 > e10 * 4.0);
    }

    #[test]
    fn test_bfsk_roundtrip_aligned_no_noise() {
        let cfg = BfskConfig::default_48k();
        let tx = BfskModulator::new(cfg.clone());
        let rx = BfskDemodulator::new(cfg);

        let payload = b"bfsk-aligned-roundtrip";
        let samples = tx.modulate_frame(payload);
        let decoded = rx
            .decode_frame_aligned(&samples)
            .expect("should decode aligned frame");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn test_bfsk_roundtrip_random_payloads_aligned() {
        let cfg = BfskConfig::default_48k();
        let tx = BfskModulator::new(cfg.clone());
        let rx = BfskDemodulator::new(cfg);
        let sizes = [0usize, 1, 2, 7, 31, 63, 127, 255];

        for (i, &len) in sizes.iter().enumerate() {
            let payload = seeded_payload(len, 0x5EED_0000 + i as u64);
            let samples = tx.modulate_frame(&payload);
            let decoded = rx
                .decode_frame_aligned(&samples)
                .unwrap_or_else(|| panic!("failed aligned decode for len={len}"));
            assert_eq!(decoded, payload, "roundtrip mismatch for len={len}");
        }
    }

    #[test]
    fn test_bfsk_find_and_decode_with_offset_and_noise() {
        let cfg = BfskConfig::default_48k();
        let tx = BfskModulator::new(cfg.clone());
        let rx = BfskDemodulator::new(cfg.clone());

        let payload = b"bfsk-offset-noisy";
        let frame = tx.modulate_frame(payload);

        let mut rng = StdRng::seed_from_u64(0xBEEF_1234);
        let lead = rng.gen_range(0..cfg.samples_per_bit());
        let mut channel = vec![0.0f32; lead];
        channel.extend(frame);
        channel.extend(std::iter::repeat_n(0.0f32, cfg.samples_per_bit() * 2));
        add_awgn(&mut channel, 0.04, 0xA11CE);

        let decoded = rx
            .find_and_decode(&channel)
            .expect("should find and decode with small offset/noise");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn test_bfsk_find_and_decode_across_sub_bit_offsets() {
        let cfg = BfskConfig::default_48k();
        let tx = BfskModulator::new(cfg.clone());
        let rx = BfskDemodulator::new(cfg.clone());
        let payload = b"offset-sweep";
        let frame = tx.modulate_frame(payload);
        let spb = cfg.samples_per_bit();

        for lead in 0..spb {
            let mut channel = vec![0.0f32; lead];
            channel.extend_from_slice(&frame);
            let decoded = rx
                .find_and_decode(&channel)
                .unwrap_or_else(|| panic!("failed at lead={lead}"));
            assert_eq!(decoded, payload, "payload mismatch at lead={lead}");
        }
    }

    #[test]
    fn test_bfsk_rejects_truncated_frame() {
        let cfg = BfskConfig::default_48k();
        let tx = BfskModulator::new(cfg.clone());
        let rx = BfskDemodulator::new(cfg.clone());
        let payload = b"truncate-check";
        let samples = tx.modulate_frame(payload);
        let truncated = &samples[..samples.len() - cfg.samples_per_bit()];
        assert!(rx.decode_frame_aligned(truncated).is_none());
    }

    #[test]
    fn test_bfsk_rejects_crc_corruption() {
        let cfg = BfskConfig::default_48k();
        let tx = BfskModulator::new(cfg.clone());
        let rx = BfskDemodulator::new(cfg.clone());
        let payload = b"crc-check";
        let mut samples = tx.modulate_frame(payload);

        let mut frame = Vec::with_capacity(1 + payload.len() + 2);
        frame.push(payload.len() as u8);
        frame.extend_from_slice(payload);
        let crc = crc::crc16(&frame);
        frame.push((crc >> 8) as u8);
        frame.push((crc & 0xFF) as u8);

        let frame_bits = bytes_to_bits(&frame);
        let header_bits_len = cfg.preamble_bits + cfg.sync_bits;
        let crc_start = header_bits_len + (1 + payload.len()) * 8;
        for i in 0..16 {
            let original = frame_bits[(1 + payload.len()) * 8 + i];
            overwrite_bit_window_with_tone(&mut samples, &cfg, crc_start + i, 1 - original);
        }

        assert!(rx.decode_frame_aligned(&samples).is_none());
    }

    #[test]
    fn test_bfsk_rejects_noise_only() {
        let cfg = BfskConfig::default_48k();
        let rx = BfskDemodulator::new(cfg.clone());

        let mut noise = vec![0.0f32; cfg.samples_per_bit() * 400];
        add_awgn(&mut noise, 0.08, 0xC0FFEE);

        assert!(rx.find_and_decode(&noise).is_none());
    }
}
