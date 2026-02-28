//! LT符号 (Luby Transform Code) - Fountain Code 実装

use std::collections::{hash_map::Entry, HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct LtParams {
    pub k: usize,
    pub block_size: usize,
    pub c: f32,
    pub delta: f32,
}

impl LtParams {
    pub fn new(k: usize, block_size: usize) -> Self {
        LtParams {
            k,
            block_size,
            c: 0.03,
            delta: 0.05,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EncodedPacket {
    pub seq: u32,
    pub degree: usize,
    pub neighbors: Vec<usize>,
    pub data: Vec<u8>,
}

fn robust_soliton_degree(seq: u32, k: usize, c: f32, delta: f32) -> usize {
    let mut rng = XorShift32::new(seq ^ 0xDEAD_CAFE);
    let r = c * (k as f32 / delta).ln() * (k as f32).sqrt();
    let mut tau = vec![0.0f32; k + 1];
    for (d, t_val) in tau.iter_mut().enumerate().take(k + 1).skip(1) {
        if d == 1 {
            *t_val = r / k as f32;
        } else {
            let kdr = (k as f32 / (r * d as f32)).max(0.0);
            *t_val = if kdr > 1.0 || d == (k as f32 / r).floor() as usize {
                r / (d as f32 * k as f32)
            } else {
                0.0
            };
        }
    }
    let mut cdf = vec![0.0f32; k + 1];
    let mut sum = 0.0f32;
    for (d, c_val) in cdf.iter_mut().enumerate().take(k + 1).skip(1) {
        let rho = if d == 1 {
            1.0 / k as f32
        } else {
            1.0 / (d as f32 * (d as f32 - 1.0))
        };
        sum += rho + tau[d];
        *c_val = sum;
    }
    let z = if sum > 0.0 { sum } else { 1.0 };
    cdf.iter_mut().for_each(|v| *v /= z);
    let u = rng.next_f32();
    for (d, &c_val) in cdf.iter().enumerate().take(k + 1).skip(1) {
        if u <= c_val {
            return d;
        }
    }
    1
}

struct XorShift32 {
    state: u32,
}
impl XorShift32 {
    fn new(seed: u32) -> Self {
        XorShift32 {
            state: if seed == 0 { 1 } else { seed },
        }
    }
    fn next_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
    fn next_range(&mut self, max: usize) -> usize {
        (self.next_u32() as usize) % max
    }
}

fn select_neighbors(seq: u32, degree: usize, k: usize) -> Vec<usize> {
    let mut rng = XorShift32::new(seq ^ 0xCAFE_BABE);
    let mut neighbors = Vec::with_capacity(degree);
    let mut chosen = HashSet::new();
    while neighbors.len() < degree {
        let idx = rng.next_range(k);
        if chosen.insert(idx) {
            neighbors.push(idx);
        }
    }
    neighbors
}

pub struct LtEncoder {
    params: LtParams,
    blocks: Vec<Vec<u8>>,
    next_seq: u32,
}

impl LtEncoder {
    pub fn new(data: &[u8], params: LtParams) -> Self {
        let mut padded = data.to_vec();
        padded.resize(params.k * params.block_size, 0);
        let blocks = padded
            .chunks(params.block_size)
            .map(|c| c.to_vec())
            .collect();
        LtEncoder {
            params,
            blocks,
            next_seq: 0,
        }
    }
    pub fn next_packet(&mut self) -> EncodedPacket {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        let (degree, neighbors) =
            reconstruct_packet_metadata(seq, self.params.k, self.params.c, self.params.delta);
        let mut data = vec![0u8; self.params.block_size];
        for &idx in &neighbors {
            for (i, &b) in self.blocks[idx].iter().enumerate() {
                data[i] ^= b;
            }
        }
        EncodedPacket {
            seq,
            degree,
            neighbors,
            data,
        }
    }
}

pub struct LtDecoder {
    params: LtParams,
    received: HashMap<u32, EncodedPacket>,
    current_rank: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReceiveOutcome {
    AcceptedRankUp,
    AcceptedNoRankUp,
    DuplicateSeq,
    InvalidNeighbors,
}

impl LtDecoder {
    pub fn params(&self) -> &LtParams {
        &self.params
    }
    pub fn new(params: LtParams) -> Self {
        LtDecoder {
            params,
            received: HashMap::new(),
            current_rank: 0,
        }
    }

    pub fn receive_with_outcome(&mut self, packet: EncodedPacket) -> ReceiveOutcome {
        if !packet.neighbors.iter().all(|&idx| idx < self.params.k) {
            return ReceiveOutcome::InvalidNeighbors;
        }

        let before_rank = self.current_rank;
        match self.received.entry(packet.seq) {
            Entry::Occupied(_) => ReceiveOutcome::DuplicateSeq,
            Entry::Vacant(e) => {
                e.insert(packet);
                self.update_rank();
                if self.current_rank > before_rank {
                    ReceiveOutcome::AcceptedRankUp
                } else {
                    ReceiveOutcome::AcceptedNoRankUp
                }
            }
        }
    }

    pub fn receive(&mut self, packet: EncodedPacket) {
        let _ = self.receive_with_outcome(packet);
    }

    fn update_rank(&mut self) {
        let k = self.params.k;
        if self.received.is_empty() {
            self.current_rank = 0;
            return;
        }
        let n_packets = self.received.len();
        let row_words = k.div_ceil(8);
        let mut matrix = vec![vec![0u8; row_words]; n_packets];
        for (row, pkt) in self.received.values().enumerate() {
            for &idx in &pkt.neighbors {
                matrix[row][idx / 8] ^= 1 << (idx % 8);
            }
        }
        let mut next_row = 0;
        for col in 0..k {
            let mut found = None;
            for (row, m_row) in matrix.iter().enumerate().take(n_packets).skip(next_row) {
                if (m_row[col / 8] >> (col % 8)) & 1 == 1 {
                    found = Some(row);
                    break;
                }
            }
            if let Some(pivot_row) = found {
                matrix.swap(pivot_row, next_row);
                let pivot = next_row;
                let pivot_words = matrix[pivot].clone();
                for (row, m_row) in matrix.iter_mut().enumerate().take(n_packets) {
                    if row != pivot && (m_row[col / 8] >> (col % 8)) & 1 == 1 {
                        for (dst, src) in m_row.iter_mut().zip(pivot_words.iter()).take(row_words) {
                            *dst ^= *src;
                        }
                    }
                }
                next_row += 1;
            }
        }
        self.current_rank = next_row;
    }

    pub fn received_count(&self) -> usize {
        self.received.len()
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
        let k = self.params.k;
        let bs = self.params.block_size;
        if self.current_rank < k {
            return None;
        }
        let n_packets = self.received.len();
        let row_words = k.div_ceil(8);
        let mut matrix = Vec::with_capacity(n_packets);
        let mut rhs = Vec::with_capacity(n_packets);
        for pkt in self.received.values() {
            let mut m_row = vec![0u8; row_words];
            for &idx in &pkt.neighbors {
                m_row[idx / 8] ^= 1 << (idx % 8);
            }
            matrix.push(m_row);
            rhs.push(pkt.data.clone());
        }
        let mut next_row = 0;
        let mut pivot_row_for_col = vec![None; k];
        for col in 0..k {
            let mut found = None;
            for (row, m_row) in matrix.iter().enumerate().take(n_packets).skip(next_row) {
                if (m_row[col / 8] >> (col % 8)) & 1 == 1 {
                    found = Some(row);
                    break;
                }
            }
            if let Some(pivot_row) = found {
                matrix.swap(pivot_row, next_row);
                rhs.swap(pivot_row, next_row);
                let pivot = next_row;
                pivot_row_for_col[col] = Some(pivot);
                let pivot_words = matrix[pivot].clone();
                let pivot_rhs = rhs[pivot].clone();
                for (row, m_row) in matrix.iter_mut().enumerate().take(n_packets) {
                    if row != pivot && (m_row[col / 8] >> (col % 8)) & 1 == 1 {
                        for (dst, src) in m_row.iter_mut().zip(pivot_words.iter()).take(row_words) {
                            *dst ^= *src;
                        }
                        for (dst, src) in rhs[row].iter_mut().zip(pivot_rhs.iter()).take(bs) {
                            *dst ^= *src;
                        }
                    }
                }
                next_row += 1;
            }
        }
        let mut result = Vec::with_capacity(k * bs);
        for row_opt in pivot_row_for_col.iter().take(k) {
            let row = (*row_opt)?;
            result.extend_from_slice(&rhs[row]);
        }
        Some(result)
    }
}

pub fn reconstruct_packet_metadata(seq: u32, k: usize, c: f32, delta: f32) -> (usize, Vec<usize>) {
    if (seq as usize) < k {
        (1, vec![seq as usize])
    } else {
        let degree = robust_soliton_degree(seq, k, c, delta);
        let neighbors = select_neighbors(seq, degree, k);
        (degree, neighbors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_systematic_recovery() {
        let k = 10;
        let bs = 16;
        let data: Vec<u8> = (0..(k * bs) as u8).collect();
        let params = LtParams::new(k, bs);
        let mut encoder = LtEncoder::new(&data, params.clone());
        let mut decoder = LtDecoder::new(params);

        // Receive exactly the first 10 systematic packets
        for _ in 0..k {
            let pkt = encoder.next_packet();
            assert!(pkt.seq < k as u32);
            decoder.receive(pkt);
        }

        assert_eq!(
            decoder.current_rank, k,
            "Systematic packets 0..9 should yield rank K"
        );
        let recovered = decoder
            .decode()
            .expect("Should be able to decode systematic packets");
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_random_recovery() {
        let k = 10;
        let bs = 16;
        let data: Vec<u8> = (0..(k * bs) as u8).collect();
        let params = LtParams::new(k, bs);
        let mut encoder = LtEncoder::new(&data, params.clone());
        let mut decoder = LtDecoder::new(params);

        // Skip systematic packets, use only random fountain packets
        encoder.next_seq = 10;
        let mut count = 0;
        while decoder.current_rank < k && count < 100 {
            decoder.receive(encoder.next_packet());
            count += 1;
        }

        assert_eq!(
            decoder.current_rank, k,
            "Should eventually reach rank K with fountain packets"
        );
        let recovered = decoder
            .decode()
            .expect("Should be able to decode fountain packets");
        assert_eq!(recovered, data);
    }
}
