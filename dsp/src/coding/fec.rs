//! 畳み込み符号 (Convolutional Code) + Viterbiデコーダ
//!
//! # パラメータ
//! - 拘束長 K=7 (NASA Voyager 標準)
//! - 符号化率 R=1/2
//! - 生成多項式 G1=0o171 (121), G2=0o133 (91) (標準NASA多項式)
//!
//! # 動作
//! - エンコーダ: 1ビット入力 → 2ビット出力
//! - デコーダ: Viterbiアルゴリズム (ハード判定)
//!
//! # Viterbiアルゴリズムの状態数
//! 2^(K-1) = 2^6 = 64 状態

const CONSTRAINT_LEN: usize = 7;
const NUM_STATES: usize = 1 << (CONSTRAINT_LEN - 1); // 64
const G1: u8 = 0o171; // 0b1111001 = 121
const G2: u8 = 0o133; // 0b1011011 = 91

#[derive(Clone, Copy)]
struct TrellisPred {
    prev_state: u8,
    bit: u8,
    out_idx: usize,
}

const TRELLIS_PREDS: [[TrellisPred; 2]; NUM_STATES] = build_trellis_preds();

/// 生成多項式と状態からパリティビットを計算 (偶数パリティ)
#[inline]
fn parity(x: u8) -> u8 {
    x.count_ones() as u8 & 1
}

const fn parity_const(x: u8) -> u8 {
    x.count_ones() as u8 & 1
}

/// 現在の状態 `state` にビット `bit` を入力したときの出力シンボル (2ビット)
/// state は K-1=6 ビット幅: [s0 s1 s2 s3 s4 s5]
/// 出力は現在の bit と state の全ビットに依存する (計7ビット)
#[inline]
fn conv_output(state: u8, bit: u8) -> (u8, u8) {
    // 7ビットのレジスタ値を構成: [bit s0 s1 s2 s3 s4 s5]
    // bit を MSB (第6ビット) に配置
    let reg = (bit << 6) | (state & 0x3F);

    // G1, G2との積のパリティ
    // G1=0o171=0b1111001, G2=0o133=0b1011011
    let v1 = parity(reg & G1);
    let v2 = parity(reg & G2);
    (v1, v2)
}

const fn conv_output_const(state: u8, bit: u8) -> (u8, u8) {
    let reg = (bit << 6) | (state & 0x3F);
    let v1 = parity_const(reg & G1);
    let v2 = parity_const(reg & G2);
    (v1, v2)
}

const fn build_trellis_preds() -> [[TrellisPred; 2]; NUM_STATES] {
    let default = TrellisPred {
        prev_state: 0,
        bit: 0,
        out_idx: 0,
    };
    let mut table = [[default; 2]; NUM_STATES];
    let mut ns = 0usize;
    while ns < NUM_STATES {
        let bit = ((ns >> 5) & 1) as u8;
        let pred_base = ((ns & 0x1F) << 1) as u8;
        let mut i = 0usize;
        while i < 2 {
            let prev_state = pred_base | (i as u8);
            let (v1, v2) = conv_output_const(prev_state, bit);
            table[ns][i] = TrellisPred {
                prev_state,
                bit,
                out_idx: ((v1 as usize) << 1) | (v2 as usize),
            };
            i += 1;
        }
        ns += 1;
    }
    table
}

/// 次の状態を計算する
/// [bit s0 s1 s2 s3 s4] を新しい状態とする
#[inline]
fn next_state(state: u8, bit: u8) -> u8 {
    ((bit << 5) | (state >> 1)) & 0x3F
}

/// 畳み込み符号エンコーダ
///
/// ビット列を符号化する。出力は入力の2倍長。
/// 末尾にtailbits (K-1個のゼロ) を付加してトレリスをフラッシュする。
pub fn encode(bits: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    encode_into(bits, &mut out);
    out
}

/// 畳み込み符号エンコーダ（出力バッファ再利用版）
pub fn encode_into(bits: &[u8], out: &mut Vec<u8>) {
    let mut state: u8 = 0;
    out.clear();
    out.reserve((bits.len() + CONSTRAINT_LEN - 1) * 2);

    // データビット
    for &bit in bits {
        let (v1, v2) = conv_output(state, bit);
        out.push(v1);
        out.push(v2);
        state = next_state(state, bit);
    }

    // テールビット: K-1 個の 0 を入力してトレリスをリセット
    for _ in 0..(CONSTRAINT_LEN - 1) {
        let (v1, v2) = conv_output(state, 0);
        out.push(v1);
        out.push(v2);
        state = next_state(state, 0);
    }
}

/// Viterbiデコーダ (ハード判定)
///
/// 符号化ビット列を復号する。入力の約半分長のビット列を返す。
/// テールビットを考慮する。
pub fn decode(coded_bits: &[u8]) -> Vec<u8> {
    assert!(
        coded_bits.len().is_multiple_of(2),
        "符号化ビット列は偶数長であること"
    );
    let num_symbols = coded_bits.len() / 2;

    // トレリス: path_metric[state] = 累積ハミング距離
    const INF: u32 = u32::MAX / 2;
    let mut path_metrics = vec![INF; NUM_STATES];
    path_metrics[0] = 0;

    // サバイバーパス: survivor[time][state] = 前の状態
    let mut survivors: Vec<Vec<u8>> = Vec::with_capacity(num_symbols);

    let data_len = num_symbols.saturating_sub(CONSTRAINT_LEN - 1);

    for sym_idx in 0..num_symbols {
        let r0 = coded_bits[sym_idx * 2];
        let r1 = coded_bits[sym_idx * 2 + 1];

        let mut new_metrics = vec![INF; NUM_STATES];
        let mut survivor = vec![0u8; NUM_STATES];

        let bit_end = if sym_idx >= data_len { 1u8 } else { 2u8 };

        // 各状態への遷移を評価
        for (prev_state, &metric) in path_metrics.iter().enumerate().take(NUM_STATES) {
            if metric == INF {
                continue;
            }
            for bit in 0u8..bit_end {
                let (v1, v2) = conv_output(prev_state as u8, bit);
                // ハミング距離
                let branch_metric = (v1 ^ r0) as u32 + (v2 ^ r1) as u32;
                let ns = next_state(prev_state as u8, bit) as usize;
                let total = metric + branch_metric;
                if total < new_metrics[ns] {
                    new_metrics[ns] = total;
                    survivor[ns] = prev_state as u8;
                }
            }
        }

        path_metrics = new_metrics;
        survivors.push(survivor);
    }

    // 最良の終端状態を選択 (テールビットにより状態0に収束するはず)
    let best_end_state = 0; // テールビットにより状態0に必ず収束する

    // トレースバック
    let mut decoded = vec![0u8; data_len];
    let mut state = best_end_state as u8;

    for t in (0..num_symbols).rev() {
        let prev = survivors[t][state as usize];
        // このステップの入力ビットを復元
        // next_state(prev, bit) == state を満たすbitを見つける
        let bit = if next_state(prev, 1) == state {
            1u8
        } else {
            0u8
        };
        if t < data_len {
            decoded[t] = bit;
        }
        state = prev;
    }

    decoded
}

/// Viterbiデコーダ (ソフト判定, LLR入力)
///
/// `llrs` は coded bit ごとの対数尤度比 (LLR)。
/// 正: bit=0 を支持、負: bit=1 を支持。
pub fn decode_soft(llrs: &[f32]) -> Vec<u8> {
    let mut workspace = FecDecodeWorkspace::new();
    let mut out = Vec::new();
    workspace.decode_soft_into(llrs, &mut out);
    out
}

#[derive(Clone, Copy)]
struct ListSurvivor {
    prev_state: u8,
    prev_rank: usize,
    bit: u8,
}

impl ListSurvivor {
    #[inline]
    fn invalid() -> Self {
        Self {
            prev_state: 0,
            prev_rank: 0,
            bit: 0,
        }
    }
}

pub struct FecDecodeWorkspace {
    path_metrics: Vec<f32>,
    new_metrics: Vec<f32>,
    survivor_history: Vec<ListSurvivor>,
    survivor_valid_counts: Vec<usize>,
    ranked: Vec<(usize, f32)>,
    reusable_candidate_bits: Vec<u8>,
}

impl Default for FecDecodeWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

impl FecDecodeWorkspace {
    pub fn new() -> Self {
        Self {
            path_metrics: Vec::new(),
            new_metrics: Vec::new(),
            survivor_history: Vec::new(),
            survivor_valid_counts: Vec::new(),
            ranked: Vec::new(),
            reusable_candidate_bits: Vec::new(),
        }
    }

    fn ensure_capacity(&mut self, num_symbols: usize, list_size: usize) {
        let state_stride = NUM_STATES * list_size;
        if self.path_metrics.len() != state_stride {
            self.path_metrics.resize(state_stride, f32::NEG_INFINITY);
        }
        if self.new_metrics.len() != state_stride {
            self.new_metrics.resize(state_stride, f32::NEG_INFINITY);
        }
        let survivor_len = num_symbols.saturating_mul(state_stride);
        if self.survivor_history.len() < survivor_len {
            self.survivor_history
                .resize(survivor_len, ListSurvivor::invalid());
        }
        let valid_count_len = num_symbols.saturating_mul(NUM_STATES);
        if self.survivor_valid_counts.len() < valid_count_len {
            self.survivor_valid_counts.resize(valid_count_len, 0);
        }
    }

    /// 期待するLLR長（coded bits）とリストサイズに対して事前確保する。
    pub fn preallocate_for_llr_len(&mut self, llr_len: usize, list_size: usize) {
        assert!(llr_len.is_multiple_of(2), "LLR列は偶数長であること");
        let list_size = list_size.max(1);
        let num_symbols = llr_len / 2;
        self.ensure_capacity(num_symbols, list_size);
    }

    pub fn decode_soft_into(&mut self, llrs: &[f32], out_bits: &mut Vec<u8>) {
        let mut candidate_bits = std::mem::take(&mut self.reusable_candidate_bits);
        out_bits.clear();
        let _ = self.decode_soft_list_try(llrs, 1, &mut candidate_bits, |bits, _rank, _score| {
            out_bits.extend_from_slice(bits);
            Some(())
        });
        self.reusable_candidate_bits = candidate_bits;
    }

    pub fn decode_soft_list_into(
        &mut self,
        llrs: &[f32],
        list_size: usize,
        out_candidates: &mut Vec<Vec<u8>>,
    ) {
        out_candidates.clear();
        let mut candidate_bits = std::mem::take(&mut self.reusable_candidate_bits);
        let _ = self.decode_soft_list_try(llrs, list_size, &mut candidate_bits, |bits, _, _| {
            if !out_candidates.iter().any(|v| v == bits) {
                out_candidates.push(bits.to_vec());
            }
            None::<()>
        });
        self.reusable_candidate_bits = candidate_bits;
    }

    pub fn decode_soft_list_try<T, F>(
        &mut self,
        llrs: &[f32],
        list_size: usize,
        candidate_bits_scratch: &mut Vec<u8>,
        mut eval: F,
    ) -> Option<T>
    where
        F: FnMut(&[u8], usize, f32) -> Option<T>,
    {
        let (list_size, num_symbols, state_stride, data_len) =
            self.run_list_viterbi(llrs, list_size);
        for (rank0, score) in self.ranked.iter().copied().take(list_size) {
            if !self.traceback_candidate_into(
                num_symbols,
                state_stride,
                list_size,
                data_len,
                rank0,
                candidate_bits_scratch,
            ) {
                continue;
            }
            if let Some(accepted) = eval(candidate_bits_scratch, rank0, score) {
                return Some(accepted);
            }
        }
        None
    }

    fn run_list_viterbi(&mut self, llrs: &[f32], list_size: usize) -> (usize, usize, usize, usize) {
        assert!(llrs.len().is_multiple_of(2), "LLR列は偶数長であること");
        let list_size = list_size.max(1);
        let num_symbols = llrs.len() / 2;
        let state_stride = NUM_STATES * list_size;
        let data_len = num_symbols.saturating_sub(CONSTRAINT_LEN - 1);
        self.ensure_capacity(num_symbols, list_size);

        const NEG_INF: f32 = f32::NEG_INFINITY;
        self.path_metrics.fill(NEG_INF);
        self.path_metrics[0] = 0.0;

        for sym_idx in 0..num_symbols {
            let l0 = llrs[sym_idx * 2];
            let l1 = llrs[sym_idx * 2 + 1];
            let branch_scores = [l0 + l1, l0 - l1, -l0 + l1, -l0 - l1];
            let step_base = sym_idx * state_stride;
            let step_valid_base = sym_idx * NUM_STATES;
            self.survivor_valid_counts[step_valid_base..step_valid_base + NUM_STATES].fill(0);
            self.new_metrics.fill(NEG_INF);

            let tail_only_zero = sym_idx >= data_len;

            for state in 0..NUM_STATES {
                // テール区間では bit=0 の遷移のみ有効なため、上位半分(state>=32)は到達不能。
                if tail_only_zero && state >= (NUM_STATES / 2) {
                    continue;
                }

                let base = state * list_size;
                let pred0 = TRELLIS_PREDS[state][0];
                let pred1 = TRELLIS_PREDS[state][1];
                let pred0_base = pred0.prev_state as usize * list_size;
                let pred1_base = pred1.prev_state as usize * list_size;
                let add0 = branch_scores[pred0.out_idx];
                let add1 = branch_scores[pred1.out_idx];

                let mut i0 = 0usize;
                let mut i1 = 0usize;
                let mut rank = 0usize;
                while rank < list_size {
                    let score0 = if i0 < list_size {
                        self.path_metrics[pred0_base + i0] + add0
                    } else {
                        NEG_INF
                    };
                    let score1 = if i1 < list_size {
                        self.path_metrics[pred1_base + i1] + add1
                    } else {
                        NEG_INF
                    };

                    if !score0.is_finite() && !score1.is_finite() {
                        break;
                    }

                    if score0 >= score1 {
                        self.new_metrics[base + rank] = score0;
                        self.survivor_history[step_base + base + rank] = ListSurvivor {
                            prev_state: pred0.prev_state,
                            prev_rank: i0,
                            bit: pred0.bit,
                        };
                        i0 += 1;
                    } else {
                        self.new_metrics[base + rank] = score1;
                        self.survivor_history[step_base + base + rank] = ListSurvivor {
                            prev_state: pred1.prev_state,
                            prev_rank: i1,
                            bit: pred1.bit,
                        };
                        i1 += 1;
                    }
                    rank += 1;
                }
                self.survivor_valid_counts[step_valid_base + state] = rank;
            }

            std::mem::swap(&mut self.path_metrics, &mut self.new_metrics);
        }

        self.ranked.clear();
        for rank in 0..list_size {
            let score = self.path_metrics[rank];
            if score.is_finite() {
                self.ranked.push((rank, score));
            }
        }
        self.ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        (list_size, num_symbols, state_stride, data_len)
    }

    fn traceback_candidate_into(
        &self,
        num_symbols: usize,
        state_stride: usize,
        list_size: usize,
        data_len: usize,
        rank0: usize,
        decoded: &mut Vec<u8>,
    ) -> bool {
        decoded.clear();
        decoded.resize(data_len, 0u8);

        let mut state = 0usize;
        let mut rank = rank0;
        for t in (0..num_symbols).rev() {
            if rank >= list_size {
                decoded.clear();
                return false;
            }
            let valid_rank_count = self.survivor_valid_counts[t * NUM_STATES + state];
            if rank >= valid_rank_count {
                decoded.clear();
                return false;
            }
            let sv = self.survivor_history[t * state_stride + state * list_size + rank];
            if t < data_len {
                decoded[t] = sv.bit;
            }
            state = sv.prev_state as usize;
            rank = sv.prev_rank;
        }
        true
    }
}

pub fn decode_soft_into(llrs: &[f32], out_bits: &mut Vec<u8>, workspace: &mut FecDecodeWorkspace) {
    workspace.decode_soft_into(llrs, out_bits);
}

pub fn decode_soft_list_into(
    llrs: &[f32],
    list_size: usize,
    out_candidates: &mut Vec<Vec<u8>>,
    workspace: &mut FecDecodeWorkspace,
) {
    workspace.decode_soft_list_into(llrs, list_size, out_candidates);
}

/// Viterbiデコーダ (ソフト判定, LLR入力, List出力)
///
/// `list_size` 個までの候補系列をスコア順に返す（先頭が最尤）。
pub fn decode_soft_list(llrs: &[f32], list_size: usize) -> Vec<Vec<u8>> {
    let mut workspace = FecDecodeWorkspace::new();
    let mut out = Vec::new();
    workspace.decode_soft_list_into(llrs, list_size, &mut out);
    out
}

/// ビット列をバイト列に変換 (MSB first)
pub fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    bits_to_bytes_into(bits, &mut out);
    out
}

/// ビット列をバイト列に変換 (MSB first, 出力バッファ再利用版)
pub fn bits_to_bytes_into(bits: &[u8], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(bits.len().div_ceil(8));
    for chunk in bits.chunks(8) {
        let byte = chunk
            .iter()
            .enumerate()
            .fold(0u8, |acc, (i, &b)| acc | (b << (7 - i)));
        out.push(byte);
    }
}

/// バイト列をビット列に変換 (MSB first)
pub fn bytes_to_bits(bytes: &[u8]) -> Vec<u8> {
    let mut bits = Vec::new();
    bytes_to_bits_into(bytes, &mut bits);
    bits
}

/// バイト列をビット列に変換 (MSB first, 出力バッファ再利用版)
pub fn bytes_to_bits_into(bytes: &[u8], bits: &mut Vec<u8>) {
    bits.clear();
    bits.reserve(bytes.len() * 8);
    for &byte in bytes {
        for i in (0..8).rev() {
            bits.push((byte >> i) & 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// エンコード→デコードの往復テスト (エラーなし)
    #[test]
    fn test_encode_decode_no_error() {
        let original: Vec<u8> = (0..8).map(|_| 1u8).chain(vec![0u8; 8]).collect();
        let coded = encode(&original);
        // coded.len() = (original.len() + K-1) * 2
        assert_eq!(coded.len(), (original.len() + CONSTRAINT_LEN - 1) * 2);

        let decoded = decode(&coded);
        assert_eq!(decoded.len(), original.len());
        assert_eq!(
            decoded, original,
            "エラーなし時にデコード結果が一致すること"
        );
    }

    /// ランダムデータのエンコード→デコード
    #[test]
    fn test_encode_decode_random() {
        // 疑似乱数的なビット列 (u32で演算してからu8に変換)
        let original: Vec<u8> = (0..64u32).map(|i| ((i * 37 + 11) % 2) as u8).collect();
        let coded = encode(&original);
        let decoded = decode(&coded);
        assert_eq!(
            decoded, original,
            "ランダムビット列のデコード結果が一致すること"
        );
    }

    /// 1ビットエラーの訂正確認
    #[test]
    fn test_single_bit_error_correction() {
        let original: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0];
        let mut coded = encode(&original);
        // 中央付近の1ビットを反転
        let flip_pos = coded.len() / 2;
        coded[flip_pos] ^= 1;
        let decoded = decode(&coded);
        assert_eq!(decoded, original, "1ビットエラーが訂正されること");
    }

    /// エンコード長の確認
    #[test]
    fn test_encoded_length() {
        for input_len in [1, 8, 16, 32] {
            let bits = vec![1u8; input_len];
            let coded = encode(&bits);
            let expected_len = (input_len + CONSTRAINT_LEN - 1) * 2;
            assert_eq!(
                coded.len(),
                expected_len,
                "input_len={} の符号化後長が正しいこと",
                input_len
            );
        }
    }

    /// 3ビットエラーの訂正確認 (K=7, R=1/2 なら十分可能)
    #[test]
    fn test_multi_bit_error_correction() {
        let original: Vec<u8> = vec![1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1];
        let mut coded = encode(&original);

        // 分散した位置で3ビット反転
        coded[10] ^= 1;
        coded[20] ^= 1;
        coded[30] ^= 1;

        let decoded = decode(&coded);
        assert_eq!(decoded, original, "3ビットの分散エラーが訂正されること");
    }

    #[test]
    fn test_soft_decode_no_error_llr() {
        let original: Vec<u8> = (0..64u32).map(|i| ((i * 17 + 3) % 2) as u8).collect();
        let coded = encode(&original);
        let llrs: Vec<f32> = coded
            .iter()
            .map(|&b| if b == 0 { 4.0 } else { -4.0 })
            .collect();
        let decoded = decode_soft(&llrs);
        assert_eq!(decoded, original, "ソフト判定でも往復一致すること");
    }

    #[test]
    fn test_soft_decode_matches_hard_on_strong_llr() {
        let original: Vec<u8> = (0..48u32).map(|i| ((i * 13 + 5) % 2) as u8).collect();
        let coded = encode(&original);
        let hard_decoded = decode(&coded);
        let llrs: Vec<f32> = coded
            .iter()
            .map(|&b| if b == 0 { 6.0 } else { -6.0 })
            .collect();
        let soft_decoded = decode_soft(&llrs);
        assert_eq!(soft_decoded, hard_decoded);
        assert_eq!(soft_decoded, original);
    }

    #[test]
    #[should_panic(expected = "LLR列は偶数長であること")]
    fn test_soft_decode_panics_on_odd_llr_len() {
        let _ = decode_soft(&[1.0, -1.0, 0.5]);
    }

    #[test]
    fn test_soft_decode_invariant_to_positive_llr_scale() {
        let original: Vec<u8> = (0..96u32).map(|i| ((i * 29 + 7) % 2) as u8).collect();
        let coded = encode(&original);
        let base_llr: Vec<f32> = coded
            .iter()
            .map(|&b| if b == 0 { 1.7 } else { -1.7 })
            .collect();
        let scaled_llr: Vec<f32> = base_llr.iter().map(|v| v * 3.5).collect();

        let d0 = decode_soft(&base_llr);
        let d1 = decode_soft(&scaled_llr);
        assert_eq!(d0, d1);
        assert_eq!(d0, original);
    }

    #[test]
    fn test_soft_decode_list_k1_matches_decode_soft() {
        let original: Vec<u8> = (0..80u32).map(|i| ((i * 31 + 3) % 2) as u8).collect();
        let coded = encode(&original);
        let llrs: Vec<f32> = coded
            .iter()
            .map(|&b| if b == 0 { 1.2 } else { -1.2 })
            .collect();
        let d0 = decode_soft(&llrs);
        let list = decode_soft_list(&llrs, 1);
        assert_eq!(list.len(), 1);
        assert_eq!(list[0], d0);
    }

    #[test]
    fn test_soft_decode_list_contains_best_path() {
        let original: Vec<u8> = (0..80u32).map(|i| ((i * 19 + 11) % 2) as u8).collect();
        let coded = encode(&original);
        let mut llrs = Vec::with_capacity(coded.len());
        let mut state = 0x1234_5678_9abc_def0u64;
        for &b in &coded {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state as u32) as f32 / u32::MAX as f32; // [0,1]
            let noise = (u * 2.0 - 1.0) * 1.9; // [-1.9, 1.9]
            let sym = if b == 0 { 1.0 } else { -1.0 };
            llrs.push(sym + noise);
        }
        let best = decode_soft(&llrs);
        let list = decode_soft_list(&llrs, 4);
        assert!(!list.is_empty());
        assert_eq!(list[0], best);
    }

    #[test]
    fn test_soft_decode_list_try_k1_matches_decode_soft() {
        let original: Vec<u8> = (0..80u32).map(|i| ((i * 31 + 3) % 2) as u8).collect();
        let coded = encode(&original);
        let llrs: Vec<f32> = coded
            .iter()
            .map(|&b| if b == 0 { 1.2 } else { -1.2 })
            .collect();
        let expected = decode_soft(&llrs);

        let mut workspace = FecDecodeWorkspace::new();
        let mut candidate_bits = Vec::new();
        let mut observed_rank = None;
        let decoded = workspace
            .decode_soft_list_try(&llrs, 1, &mut candidate_bits, |bits, rank, _score| {
                observed_rank = Some(rank);
                Some(bits.to_vec())
            })
            .unwrap();

        assert_eq!(decoded, expected);
        assert_eq!(observed_rank, Some(0));
    }

    #[test]
    fn test_soft_decode_list_try_stops_when_callback_accepts() {
        let original: Vec<u8> = (0..80u32).map(|i| ((i * 23 + 17) % 2) as u8).collect();
        let coded = encode(&original);
        let llrs: Vec<f32> = coded
            .iter()
            .map(|&b| if b == 0 { 0.9 } else { -0.9 })
            .collect();
        let mut workspace = FecDecodeWorkspace::new();
        let mut candidate_bits = Vec::new();
        let mut call_count = 0usize;
        let accepted = workspace.decode_soft_list_try(
            &llrs,
            8,
            &mut candidate_bits,
            |_bits, _rank, _score| {
                call_count += 1;
                Some(())
            },
        );

        assert_eq!(accepted, Some(()));
        assert_eq!(call_count, 1, "accept後に追加候補を評価しないこと");
    }

    #[test]
    #[should_panic(expected = "LLR列は偶数長であること")]
    fn test_soft_decode_list_panics_on_odd_llr_len() {
        let _ = decode_soft_list(&[1.0, -1.0, 0.5], 4);
    }

    #[test]
    fn test_soft_decode_beats_hard_in_deterministic_noise_search() {
        let original: Vec<u8> = (0..80u32).map(|i| ((i * 19 + 11) % 2) as u8).collect();
        let coded = encode(&original);

        let mut found = false;
        let mut state = 0x9e3779b97f4a7c15u64;
        for amp in [1.2f32, 1.4, 1.6, 1.8, 2.0] {
            for _trial in 0..500 {
                let mut llrs = Vec::with_capacity(coded.len());
                for &b in &coded {
                    // xorshift64*
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let u = (state as u32) as f32 / u32::MAX as f32; // [0,1]
                    let noise = (u * 2.0 - 1.0) * amp; // [-amp, amp]
                    let sym = if b == 0 { 1.0 } else { -1.0 };
                    llrs.push(sym + noise);
                }
                let hard_bits: Vec<u8> = llrs
                    .iter()
                    .map(|&v| if v >= 0.0 { 0u8 } else { 1u8 })
                    .collect();
                let hard_decoded = decode(&hard_bits);
                let soft_decoded = decode_soft(&llrs);
                if hard_decoded != original && soft_decoded == original {
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
        }

        assert!(
            found,
            "deterministic search should find at least one case where soft Viterbi beats hard Viterbi"
        );
    }

    /// バーストエラーの訂正限界確認
    #[test]
    fn test_burst_error_correction() {
        let original: Vec<u8> = vec![0; 32];
        let mut coded = encode(&original);

        // 連続する4ビットを反転 (R=1/2, K=7 ではこのあたりが限界)
        for bit in coded.iter_mut().take(24).skip(20) {
            *bit ^= 1;
        }

        let decoded = decode(&coded);
        assert_eq!(decoded, original, "4ビットのバーストエラーが訂正されること");
    }

    /// フルパケット(約160ビット)でのエラー訂正能力確認
    #[test]
    fn test_full_packet_error_correction() {
        let mut original = vec![0u8; 160];
        for (i, bit) in original.iter_mut().enumerate().take(160) {
            *bit = (i % 2) as u8;
        }
        let mut coded = encode(&original);

        // 5% のビットエラーをランダム(分散)に注入 (160*2 * 0.05 = 16ビット)
        for i in (0..coded.len()).step_by(20) {
            coded[i] ^= 1;
        }

        let decoded = decode(&coded);
        assert_eq!(
            decoded, original,
            "フルパケットでの分散エラーが訂正されること"
        );
    }
}
