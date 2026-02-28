//! Fountain coding module backed by non-systematic RLNC over GF(256).

use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct FountainParams {
    pub k: usize,
    pub block_size: usize,
}

impl FountainParams {
    pub fn new(k: usize, block_size: usize) -> Self {
        FountainParams { k, block_size }
    }
}

#[derive(Clone, Debug)]
pub struct FountainPacket {
    pub seq: u32,
    pub coefficients: Vec<u8>,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
struct BasisRow {
    coeffs: Vec<u8>,
    data: Vec<u8>,
}

#[inline]
fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0u8;
    for _ in 0..8 {
        if b & 1 != 0 {
            p ^= a;
        }
        let carry = a & 0x80;
        a <<= 1;
        if carry != 0 {
            a ^= 0x1b;
        }
        b >>= 1;
    }
    p
}

fn gf_pow(mut base: u8, mut exp: u16) -> u8 {
    let mut acc = 1u8;
    while exp > 0 {
        if exp & 1 != 0 {
            acc = gf_mul(acc, base);
        }
        base = gf_mul(base, base);
        exp >>= 1;
    }
    acc
}

#[inline]
fn gf_inv(x: u8) -> u8 {
    debug_assert!(x != 0);
    gf_pow(x, 254)
}

#[inline]
fn row_axpy(dst: &mut [u8], src: &[u8], factor: u8) {
    if factor == 0 {
        return;
    }
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d ^= gf_mul(*s, factor);
    }
}

#[inline]
fn row_scale(dst: &mut [u8], factor: u8) {
    if factor == 1 {
        return;
    }
    for d in dst.iter_mut() {
        *d = gf_mul(*d, factor);
    }
}

pub fn reconstruct_packet_coefficients(seq: u32, k: usize) -> Vec<u8> {
    if k == 0 {
        return Vec::new();
    }

    // Non-systematic RLNC:
    // row(seq) = [1, x, x^2, ..., x^(k-1)] over GF(256), x != 0.
    // step is coprime with 255 to cover all non-zero field elements before repeating.
    let step = 73u32;
    let x = ((seq * step) % 255 + 1) as u8;

    let mut coeffs = vec![0u8; k];
    coeffs[0] = 1;
    for i in 1..k {
        coeffs[i] = gf_mul(coeffs[i - 1], x);
    }
    coeffs
}

pub struct FountainEncoder {
    params: FountainParams,
    blocks: Vec<Vec<u8>>,
    next_seq: u32,
}

impl FountainEncoder {
    pub fn new(data: &[u8], params: FountainParams) -> Self {
        let mut padded = data.to_vec();
        padded.resize(params.k * params.block_size, 0);
        let blocks = padded
            .chunks(params.block_size)
            .map(|c| c.to_vec())
            .collect();
        FountainEncoder {
            params,
            blocks,
            next_seq: 0,
        }
    }

    pub fn next_packet(&mut self) -> FountainPacket {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        let coefficients = reconstruct_packet_coefficients(seq, self.params.k);
        let mut data = vec![0u8; self.params.block_size];

        for (block_idx, &coeff) in coefficients.iter().enumerate() {
            if coeff == 0 {
                continue;
            }
            for (out, src) in data.iter_mut().zip(self.blocks[block_idx].iter()) {
                *out ^= gf_mul(*src, coeff);
            }
        }

        FountainPacket {
            seq,
            coefficients,
            data,
        }
    }
}

pub struct FountainDecoder {
    params: FountainParams,
    seen_seq: HashSet<u32>,
    basis: Vec<Option<BasisRow>>,
    current_rank: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReceiveOutcome {
    AcceptedRankUp,
    AcceptedNoRankUp,
    DuplicateSeq,
    InvalidPacket,
}

impl FountainDecoder {
    pub fn params(&self) -> &FountainParams {
        &self.params
    }

    pub fn new(params: FountainParams) -> Self {
        FountainDecoder {
            basis: vec![None; params.k],
            params,
            seen_seq: HashSet::new(),
            current_rank: 0,
        }
    }

    pub fn receive_with_outcome(&mut self, packet: FountainPacket) -> ReceiveOutcome {
        if packet.coefficients.len() != self.params.k
            || packet.data.len() != self.params.block_size
            || packet.coefficients.iter().all(|&x| x == 0)
        {
            return ReceiveOutcome::InvalidPacket;
        }

        if !self.seen_seq.insert(packet.seq) {
            return ReceiveOutcome::DuplicateSeq;
        }

        let rank_up = self.insert_row(packet.coefficients, packet.data);
        if rank_up {
            ReceiveOutcome::AcceptedRankUp
        } else {
            ReceiveOutcome::AcceptedNoRankUp
        }
    }

    pub fn receive(&mut self, packet: FountainPacket) {
        let _ = self.receive_with_outcome(packet);
    }

    fn insert_row(&mut self, mut coeffs: Vec<u8>, mut data: Vec<u8>) -> bool {
        let k = self.params.k;

        for col in 0..k {
            if coeffs[col] == 0 {
                continue;
            }
            if let Some(pivot_row) = self.basis[col].as_ref() {
                let factor = coeffs[col];
                row_axpy(&mut coeffs, &pivot_row.coeffs, factor);
                row_axpy(&mut data, &pivot_row.data, factor);
            }
        }

        let Some(pivot_col) = coeffs.iter().position(|&c| c != 0) else {
            return false;
        };

        let inv = gf_inv(coeffs[pivot_col]);
        row_scale(&mut coeffs, inv);
        row_scale(&mut data, inv);

        for other_col in 0..k {
            if other_col == pivot_col {
                continue;
            }
            if let Some(other_row) = self.basis[other_col].as_mut() {
                let factor = other_row.coeffs[pivot_col];
                if factor != 0 {
                    row_axpy(&mut other_row.coeffs, &coeffs, factor);
                    row_axpy(&mut other_row.data, &data, factor);
                }
            }
        }

        self.basis[pivot_col] = Some(BasisRow { coeffs, data });
        self.current_rank += 1;
        true
    }

    pub fn received_count(&self) -> usize {
        self.seen_seq.len()
    }

    pub fn rank(&self) -> usize {
        self.current_rank
    }

    pub fn needed_count(&self) -> usize {
        self.params.k
    }

    pub fn progress(&self) -> f32 {
        (self.current_rank as f32 / self.params.k as f32).min(1.0)
    }

    pub fn decode(&self) -> Option<Vec<u8>> {
        if self.current_rank < self.params.k {
            return None;
        }
        let mut result = Vec::with_capacity(self.params.k * self.params.block_size);
        for col in 0..self.params.k {
            let row = self.basis[col].as_ref()?;
            result.extend_from_slice(&row.data);
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data(k: usize, bs: usize) -> Vec<u8> {
        (0..(k * bs) as u8).collect()
    }

    #[test]
    fn test_non_systematic_prefix_is_not_identity_rows() {
        let k = 8;
        for seq in 0..k as u32 {
            let coeffs = reconstruct_packet_coefficients(seq, k);
            let one_count = coeffs.iter().filter(|&&v| v == 1).count();
            let nonzero_count = coeffs.iter().filter(|&&v| v != 0).count();
            assert!(nonzero_count > 1, "seq={seq} should not be systematic");
            assert!(
                !(one_count == 1 && nonzero_count == 1),
                "seq={seq} unexpectedly produced an identity row"
            );
        }
    }

    #[test]
    fn test_first_k_packets_decode() {
        let k = 10;
        let bs = 16;
        let data = sample_data(k, bs);
        let params = FountainParams::new(k, bs);
        let mut encoder = FountainEncoder::new(&data, params.clone());
        let mut decoder = FountainDecoder::new(params);

        for _ in 0..k {
            decoder.receive(encoder.next_packet());
        }

        assert_eq!(decoder.rank(), k);
        assert_eq!(decoder.decode().as_deref(), Some(data.as_slice()));
    }

    #[test]
    fn test_recover_after_dropping_prefix_packets() {
        let k = 10;
        let bs = 16;
        let data = sample_data(k, bs);
        let params = FountainParams::new(k, bs);
        let mut encoder = FountainEncoder::new(&data, params.clone());
        let mut decoder = FountainDecoder::new(params);

        for _ in 0..k {
            let _ = encoder.next_packet();
        }
        for _ in 0..k {
            decoder.receive(encoder.next_packet());
        }

        assert_eq!(decoder.rank(), k);
        assert_eq!(decoder.decode().as_deref(), Some(data.as_slice()));
    }

    #[test]
    fn test_duplicate_seq_is_detected() {
        let k = 4;
        let bs = 16;
        let data = sample_data(k, bs);
        let params = FountainParams::new(k, bs);
        let mut encoder = FountainEncoder::new(&data, params.clone());
        let mut decoder = FountainDecoder::new(params);

        let pkt = encoder.next_packet();
        assert_eq!(
            decoder.receive_with_outcome(pkt.clone()),
            ReceiveOutcome::AcceptedRankUp
        );
        assert_eq!(
            decoder.receive_with_outcome(pkt),
            ReceiveOutcome::DuplicateSeq
        );
    }

    #[test]
    fn test_linearly_dependent_packet_is_counted_without_rankup() {
        let k = 4;
        let bs = 16;
        let data = sample_data(k, bs);
        let params = FountainParams::new(k, bs);
        let mut encoder = FountainEncoder::new(&data, params.clone());
        let mut decoder = FountainDecoder::new(params);

        let pkt0 = encoder.next_packet(); // seq=0
        assert_eq!(
            decoder.receive_with_outcome(pkt0),
            ReceiveOutcome::AcceptedRankUp
        );
        assert_eq!(decoder.rank(), 1);

        // seq=255 は seq=0 と同じ係数行になる（GF(256) の非ゼロ元周期）。
        let mut pkt255 = None;
        for _ in 0..255 {
            pkt255 = Some(encoder.next_packet());
        }
        let pkt255 = pkt255.expect("packet must exist");
        assert_eq!(
            decoder.receive_with_outcome(pkt255),
            ReceiveOutcome::AcceptedNoRankUp
        );
        assert_eq!(decoder.rank(), 1);
    }

    #[test]
    fn test_invalid_packet_rejected() {
        let params = FountainParams::new(4, 16);
        let mut decoder = FountainDecoder::new(params);

        let invalid = FountainPacket {
            seq: 0,
            coefficients: vec![0; 4],
            data: vec![0; 16],
        };
        assert_eq!(
            decoder.receive_with_outcome(invalid),
            ReceiveOutcome::InvalidPacket
        );
    }

    #[test]
    fn test_full_rank_subset_decodes_original() {
        let k = 4;
        let bs = 16;
        let data = sample_data(k, bs);
        let params = FountainParams::new(k, bs);

        for a in 0..8u32 {
            for b in (a + 1)..8u32 {
                for c in (b + 1)..8u32 {
                    for d in (c + 1)..8u32 {
                        let mut encoder = FountainEncoder::new(&data, params.clone());
                        let mut packets = Vec::new();
                        for _ in 0..=d {
                            packets.push(encoder.next_packet());
                        }
                        let mut decoder = FountainDecoder::new(params.clone());
                        decoder.receive(packets[a as usize].clone());
                        decoder.receive(packets[b as usize].clone());
                        decoder.receive(packets[c as usize].clone());
                        decoder.receive(packets[d as usize].clone());
                        if decoder.rank() == k {
                            assert_eq!(decoder.decode().as_deref(), Some(data.as_slice()));
                        }
                    }
                }
            }
        }
    }
}
