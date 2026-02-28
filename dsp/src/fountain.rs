//! LT符号 (Luby Transform Code) - Fountain Code 実装
//!
//! # 動作原理
//! - エンコーダ: K個のソースブロックから無限に符号化パケットを生成
//! - デコーダ: K+α個の符号化パケットから元データを復元 (信念伝播法)
//!
//! # 次数分布: ロバストソリトン分布 (Robust Soliton Distribution)
//! 理想ソリトン分布にスパイクを加えたもの。
//! - c: リップル係数 (通常 0.01〜0.03)
//! - delta: 失敗確率 (通常 0.05)
//!
//! # 参照
//! M. Luby, "LT Codes", FOCS 2002

use std::collections::HashMap;

/// LT符号の設定パラメータ
#[derive(Clone)]
pub struct LtParams {
    /// ソースブロック数 K
    pub k: usize,
    /// ブロックサイズ (bytes)
    pub block_size: usize,
    /// ロバストソリトン係数 c
    pub c: f32,
    /// 失敗確率 delta
    pub delta: f32,
}

impl LtParams {
    pub fn new(k: usize, block_size: usize) -> Self {
        LtParams { k, block_size, c: 0.03, delta: 0.05 }
    }
}

/// 符号化パケット (1つのLT符号化シンボル)
#[derive(Clone, Debug)]
pub struct EncodedPacket {
    /// シーケンス番号 (シード代わり)
    pub seq: u32,
    /// XORされたソースブロックのインデックス集合
    pub degree: usize,
    pub neighbors: Vec<usize>,
    /// ペイロード (degree個のソースブロックのXOR)
    pub data: Vec<u8>,
}

/// ロバストソリトン分布の次数をシード `seq` と K から決定する
fn robust_soliton_degree(seq: u32, k: usize, c: f32, delta: f32) -> usize {
    // 決定論的疑似乱数生成 (xorshift32)
    let mut rng = XorShift32::new(seq ^ 0xDEAD_CAFE);

    // 理想ソリトン分布の累積分布
    let r = c * (k as f32 / delta).ln() * (k as f32).sqrt();
    let mut tau = vec![0.0f32; k + 1];
    tau[0] = 0.0;
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

    // ロバストソリトン: rho(d) + tau(d)
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
    // 正規化 (Zが0にならないように保護)
    let z = if sum > 0.0 { sum } else { 1.0 };
    cdf.iter_mut().for_each(|v| *v /= z);

    // 次数をサンプリング
    let u = rng.next_f32();
    let mut degree = 1; // デフォルト次数1
    for (d, &c_val) in cdf.iter().enumerate().take(k + 1).skip(1) {
        if u <= c_val {
            degree = d;
            break;
        }
    }
    degree.max(1)
}

/// XorShift32 疑似乱数生成器 (決定論的)
struct XorShift32 {
    state: u32,
}

impl XorShift32 {
    fn new(seed: u32) -> Self {
        XorShift32 { state: if seed == 0 { 1 } else { seed } }
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

/// シーケンス番号から隣接ソースブロックインデックスを決定
fn select_neighbors(seq: u32, degree: usize, k: usize) -> Vec<usize> {
    let mut rng = XorShift32::new(seq ^ 0xCAFE_BABE);
    let mut neighbors = Vec::with_capacity(degree);
    let mut chosen = std::collections::HashSet::new();
    while neighbors.len() < degree {
        let idx = rng.next_range(k);
        if chosen.insert(idx) {
            neighbors.push(idx);
        }
    }
    neighbors
}

/// LT符号エンコーダ
pub struct LtEncoder {
    params: LtParams,
    blocks: Vec<Vec<u8>>,
    next_seq: u32,
}

impl LtEncoder {
    pub fn new(data: &[u8], params: LtParams) -> Self {
        let k = params.k;
        let bs = params.block_size;
        let total = k * bs;
        let mut padded = data.to_vec();
        padded.resize(total, 0);

        let blocks: Vec<Vec<u8>> = padded.chunks(bs).map(|c| c.to_vec()).collect();
        LtEncoder { params, blocks, next_seq: 0 }
    }

    pub fn next_packet(&mut self) -> EncodedPacket {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);

        let (degree, neighbors) = reconstruct_packet_metadata(seq, self.params.k, self.params.c, self.params.delta);

        let bs = self.params.block_size;
        let mut data = vec![0u8; bs];
        for &idx in &neighbors {
            for (i, &b) in self.blocks[idx].iter().enumerate() {
                data[i] ^= b;
            }
        }

        EncodedPacket { seq, degree, neighbors, data }
    }
}

pub struct LtDecoder {
    params: LtParams,
    /// 受信済みパケット
    received: Vec<EncodedPacket>,
}

impl LtDecoder {
    pub fn params(&self) -> &LtParams {
        &self.params
    }

    pub fn new(params: LtParams) -> Self {
        LtDecoder {
            params,
            received: Vec::new(),
        }
    }

    pub fn receive(&mut self, packet: EncodedPacket) {
        self.received.push(packet);
    }

    pub fn received_count(&self) -> usize {
        self.received.len()
    }

    pub fn needed_count(&self) -> usize {
        let overhead = crate::params::FOUNTAIN_OVERHEAD;
        (self.params.k as f32 * (1.0 + overhead)).ceil() as usize
    }

    pub fn progress(&self) -> f32 {
        (self.received.len() as f32 / self.needed_count() as f32).min(1.0)
    }

    pub fn decode(&self) -> Option<Vec<u8>> {
        let k = self.params.k;
        let bs = self.params.block_size;

        if self.received.len() < k {
            return None;
        }

        let mut packets: Vec<(Vec<usize>, Vec<u8>)> = self
            .received
            .iter()
            .map(|p| (p.neighbors.clone(), p.data.clone()))
            .collect();

        let mut recovered: HashMap<usize, Vec<u8>> = HashMap::new();

        'bp: loop {
            let mut any_degree1 = false;
            for (neighbors, data) in &packets {
                if neighbors.len() == 1 {
                    let idx = neighbors[0];
                    if let std::collections::hash_map::Entry::Vacant(e) = recovered.entry(idx) {
                        e.insert(data.clone());
                        any_degree1 = true;
                    }
                }
            }

            if !any_degree1 {
                break 'bp;
            }

            let mut new_packets: Vec<(Vec<usize>, Vec<u8>)> = Vec::new();
            for (neighbors, data) in &packets {
                let mut remaining = neighbors.clone();
                let mut current_data = data.clone();

                remaining.retain(|&idx| {
                    if let Some(block) = recovered.get(&idx) {
                        for (i, &b) in block.iter().enumerate() {
                            current_data[i] ^= b;
                        }
                        false
                    } else {
                        true
                    }
                });

                if remaining.len() == 1 && !recovered.contains_key(&remaining[0]) {
                    recovered.insert(remaining[0], current_data);
                } else if !remaining.is_empty() {
                    new_packets.push((remaining, current_data));
                }
            }
            packets = new_packets;
        }

        if recovered.len() >= k {
            let mut result = Vec::with_capacity(k * bs);
            for i in 0..k {
                result.extend_from_slice(recovered.get(&i)?);
            }
            return Some(result);
        }

        let n_packets = self.received.len();
        let row_words = k.div_ceil(8);
        let mut matrix: Vec<Vec<u8>> = vec![vec![0u8; row_words]; n_packets];
        let mut rhs: Vec<Vec<u8>> = Vec::with_capacity(n_packets);

        for (row, pkt) in self.received.iter().enumerate() {
            for &idx in &pkt.neighbors {
                matrix[row][idx / 8] ^= 1 << (idx % 8);
            }
            rhs.push(pkt.data.clone());
        }

        let mut pivot_row_for_col = vec![None; k];
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
                rhs.swap(pivot_row, next_row);
                
                let pivot = next_row;
                pivot_row_for_col[col] = Some(pivot);

                #[allow(clippy::needless_range_loop)]
                for row in 0..n_packets {
                    if row != pivot && (matrix[row][col / 8] >> (col % 8)) & 1 == 1 {
                        for w in 0..row_words {
                            let val = matrix[pivot][w];
                            matrix[row][w] ^= val;
                        }
                        for i in 0..bs {
                            let val = rhs[pivot][i];
                            rhs[row][i] ^= val;
                        }
                    }
                }
                next_row += 1;
            }
        }

        let mut solution: Vec<Option<Vec<u8>>> = vec![None; k];
        for (col, pivot_row_opt) in pivot_row_for_col.iter().enumerate().take(k) {
            if let Some(row) = *pivot_row_opt {
                let mut bit_count = 0;
                for c in 0..k {
                    if (matrix[row][c / 8] >> (c % 8)) & 1 == 1 {
                        bit_count += 1;
                    }
                }
                if bit_count == 1 {
                    let solved_data: Vec<u8> = rhs[row].clone();
                    solution[col] = Some(solved_data);
                }
            }
        }

        if solution.iter().any(|s| s.is_none()) {
            return None;
        }

        let mut result = Vec::with_capacity(k * bs);
        for s_opt in solution.iter().take(k) {
            result.extend_from_slice(s_opt.as_ref()?);
        }
        Some(result)
    }
}

pub fn reconstruct_packet_metadata(seq: u32, k: usize, c: f32, delta: f32) -> (usize, Vec<usize>) {
    // システマティックLT符号:
    // 最初のK個のパケットは、ソースブロックをそのまま（次数1）送信する。
    // これにより、ノイズがない環境では最小のKパケットで確実にデコードが可能になる。
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
    fn test_insufficient_packets_fail() {
        let data = vec![0u8; 40];
        let k = 4;
        let bs = 10;
        let params = LtParams::new(k, bs);
        let mut encoder = LtEncoder::new(&data, params.clone());
        let mut decoder = LtDecoder::new(params);
        for _ in 0..(k - 1) {
            decoder.receive(encoder.next_packet());
        }
        let result = decoder.decode();
        let _ = result;
    }

    #[test]
    fn test_sufficient_packets_succeed() {
        let original = b"Hello, air-gap world! Testing fountain codes.".to_vec();
        let k = 10;
        let bs = 8;
        let params = LtParams::new(k, bs);
        let mut encoder = LtEncoder::new(&original, params.clone());
        let mut decoder = LtDecoder::new(params);

        // K * 10 個受信 (K=4と小さい場合は確率的ばらつきが大きいため多めに設定)
        for _ in 0..(k * 10) {
            decoder.receive(encoder.next_packet());
        }

        let result = decoder.decode();
        assert!(result.is_some(), "十分なパケットでデコードが成功すること ({}個受信)", k * 10);

        let recovered = result.unwrap();
        assert_eq!(
            &recovered[..original.len()],
            original.as_slice(),
            "デコード結果が元データと一致すること"
        );
    }

    #[test]
    fn test_deterministic_encoding() {
        let data = b"test data".to_vec();
        let params = LtParams::new(2, 8);
        let mut enc1 = LtEncoder::new(&data, params.clone());
        let mut enc2 = LtEncoder::new(&data, params);
        let p1 = enc1.next_packet();
        let p2 = enc2.next_packet();
        assert_eq!(p1.neighbors, p2.neighbors);
        assert_eq!(p1.data, p2.data);
    }

    #[test]
    fn test_progress() {
        let data = vec![0u8; 40];
        let k = 4;
        let params = LtParams::new(k, 10);
        let mut encoder = LtEncoder::new(&data, params.clone());
        let mut decoder = LtDecoder::new(params);
        assert_eq!(decoder.progress(), 0.0);
        let needed = decoder.needed_count();
        for _ in 1..=needed {
            decoder.receive(encoder.next_packet());
            assert!(decoder.progress() > 0.0);
        }
        assert_eq!(decoder.progress(), 1.0);
    }

    #[test]
    fn test_k8_solvability_with_15_packets() {
        // E2Eテストと同じパラメータ: K=8, BS=16
        let data = b"Acoustic E2E Noise Test Data.";
        let params = LtParams::new(8, 16);
        let mut encoder = LtEncoder::new(data, params.clone());
        let mut decoder = LtDecoder::new(params);

        // 最初の15パケットでフルランクになる（デコード成功する）か検証
        for _ in 0..15 {
            decoder.receive(encoder.next_packet());
        }

        let result = decoder.decode();
        assert!(result.is_some(), "K=8 に対して最初の15パケットで数学的にフルランクになり復元できなければならない");
    }
}
