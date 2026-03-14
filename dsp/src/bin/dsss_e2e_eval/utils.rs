use dsp::coding::{fec, interleaver::BlockInterleaver, scrambler::Scrambler};
use dsp::frame::packet::{Packet, PACKET_BYTES};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

const MARY_BITS_PER_SYMBOL: usize = 6;
const MARY_WALSH_BITS_PER_SYMBOL: usize = 4;
const MARY_WALSH_LLR_CONF_WEAK_THRESH: f32 = 0.25;

pub fn count_bit_errors_bytes(tx: &[u8], rx: Option<&[u8]>) -> usize {
    let Some(rx) = rx else {
        return tx.len() * 8;
    };
    let mut errs = 0usize;
    for (idx, &b) in tx.iter().enumerate() {
        if let Some(&rb) = rx.get(idx) {
            errs += (b ^ rb).count_ones() as usize;
        } else {
            errs += 8;
        }
    }
    errs
}

pub fn bit_errors_and_runs_from_llr(
    expected: &[u8],
    llrs: &[f32],
) -> (usize, usize, usize, usize, usize) {
    let compare_len = expected.len().min(llrs.len());
    let mut errors = 0usize;
    let mut runs = 0usize;
    let mut run_bits = 0usize;
    let mut run_max = 0usize;
    let mut cur_run = 0usize;
    for j in 0..compare_len {
        let bit = expected[j];
        let llr = llrs[j];
        let is_err = (bit == 0 && llr <= 0.0) || (bit == 1 && llr >= 0.0);
        if is_err {
            errors += 1;
            cur_run += 1;
        } else if cur_run > 0 {
            runs += 1;
            run_bits += cur_run;
            run_max = run_max.max(cur_run);
            cur_run = 0;
        }
    }
    if cur_run > 0 {
        runs += 1;
        run_bits += cur_run;
        run_max = run_max.max(cur_run);
    }
    (errors, compare_len, runs, run_bits, run_max)
}

pub fn bit_errors_and_runs_from_bits(
    expected: &[u8],
    observed: &[u8],
) -> (usize, usize, usize, usize, usize) {
    let compare_len = expected.len().min(observed.len());
    let mut errors = 0usize;
    let mut runs = 0usize;
    let mut run_bits = 0usize;
    let mut run_max = 0usize;
    let mut cur_run = 0usize;
    for j in 0..compare_len {
        let is_err = expected[j] != observed[j];
        if is_err {
            errors += 1;
            cur_run += 1;
        } else if cur_run > 0 {
            runs += 1;
            run_bits += cur_run;
            run_max = run_max.max(cur_run);
            cur_run = 0;
        }
    }
    if cur_run > 0 {
        runs += 1;
        run_bits += cur_run;
        run_max = run_max.max(cur_run);
    }
    (errors, compare_len, runs, run_bits, run_max)
}

pub struct PreFecStats {
    pub bit_errors: usize,
    pub bits_compared: usize,
    pub walsh_bit_errors: usize,
    pub walsh_bits_compared: usize,
    pub dqpsk_bit_errors: usize,
    pub dqpsk_bits_compared: usize,
    pub dqpsk_walsh_weak_bit_errors: usize,
    pub dqpsk_walsh_weak_bits_compared: usize,
    pub dqpsk_walsh_strong_bit_errors: usize,
    pub dqpsk_walsh_strong_bits_compared: usize,
    pub error_runs: usize,
    pub error_run_bits: usize,
    pub error_run_max: usize,
    pub codeword_count: usize,
    pub codeword_error_sum: usize,
    pub codeword_error_max: usize,
    pub codeword_error_weights: Vec<usize>,
}

pub struct PostFecStats {
    pub bit_errors: usize,
    pub bits_compared: usize,
    pub error_runs: usize,
    pub error_run_bits: usize,
    pub error_run_max: usize,
    pub codeword_count: usize,
    pub codeword_error_sum: usize,
    pub codeword_error_max: usize,
    pub codeword_error_weights: Vec<usize>,
}

pub type LlrCallback = Box<dyn Fn(&[f32]) + Send + Sync>;

#[inline]
fn bits_msb_to_byte(bits: &[u8]) -> Option<u8> {
    if bits.len() < 8 {
        return None;
    }
    let mut out = 0u8;
    for &b in bits.iter().take(8) {
        out = (out << 1) | (b & 1);
    }
    Some(out)
}

#[inline]
fn packet_seq_from_bits(packet_bits: &[u8]) -> Option<u16> {
    if packet_bits.len() < 24 {
        return None;
    }
    let b1 = bits_msb_to_byte(&packet_bits[8..16])?;
    let b2 = bits_msb_to_byte(&packet_bits[16..24])?;
    Some(u16::from_be_bytes([b1, b2]))
}

#[inline]
fn interleave_f32_block(input: &[f32], rows: usize, cols: usize, output: &mut Vec<f32>) {
    let total = rows * cols;
    output.clear();
    output.resize(total, 0.0);
    for (k, &value) in input.iter().enumerate().take(total) {
        let row = k / cols;
        let col = k % cols;
        output[col * rows + row] = value;
    }
}

/// Mary BER集計用
pub struct BerAccumulator {
    expected_fec_bits: Arc<Mutex<HashMap<u16, Vec<u8>>>>,
    expected_interleaved_bits: Arc<Mutex<HashMap<u16, Vec<u8>>>>,
    expected_packet_bits: Arc<Mutex<HashMap<u16, Vec<u8>>>>,
    expected_packet_bytes: Arc<Mutex<HashMap<u16, Vec<u8>>>>,
    register_packet_bytes: Arc<Mutex<Vec<u8>>>,
    register_bits: Arc<Mutex<Vec<u8>>>,
    register_fec_bits: Arc<Mutex<Vec<u8>>>,
    raw_bit_errors: Arc<Mutex<usize>>,
    raw_bits_compared: Arc<Mutex<usize>>,
    raw_walsh_bit_errors: Arc<Mutex<usize>>,
    raw_walsh_bits_compared: Arc<Mutex<usize>>,
    raw_dqpsk_bit_errors: Arc<Mutex<usize>>,
    raw_dqpsk_bits_compared: Arc<Mutex<usize>>,
    raw_dqpsk_walsh_weak_bit_errors: Arc<Mutex<usize>>,
    raw_dqpsk_walsh_weak_bits_compared: Arc<Mutex<usize>>,
    raw_dqpsk_walsh_strong_bit_errors: Arc<Mutex<usize>>,
    raw_dqpsk_walsh_strong_bits_compared: Arc<Mutex<usize>>,
    raw_error_runs: Arc<Mutex<usize>>,
    raw_error_run_bits: Arc<Mutex<usize>>,
    raw_error_run_max: Arc<Mutex<usize>>,
    codeword_error_weights: Arc<Mutex<Vec<usize>>>,
    post_bit_errors: Arc<Mutex<usize>>,
    post_bits_compared: Arc<Mutex<usize>>,
    post_error_runs: Arc<Mutex<usize>>,
    post_error_run_bits: Arc<Mutex<usize>>,
    post_error_run_max: Arc<Mutex<usize>>,
    post_codeword_error_weights: Arc<Mutex<Vec<usize>>>,
    post_decode_attempts: Arc<Mutex<usize>>,
    post_decode_matched: Arc<Mutex<usize>>,
    accepted_packets: Arc<Mutex<usize>>,
    false_accepted_packets: Arc<Mutex<usize>>,
}

impl Default for BerAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl BerAccumulator {
    pub fn new() -> Self {
        Self {
            expected_fec_bits: Arc::new(Mutex::new(HashMap::new())),
            expected_interleaved_bits: Arc::new(Mutex::new(HashMap::new())),
            expected_packet_bits: Arc::new(Mutex::new(HashMap::new())),
            expected_packet_bytes: Arc::new(Mutex::new(HashMap::new())),
            register_packet_bytes: Arc::new(Mutex::new(Vec::with_capacity(PACKET_BYTES))),
            register_bits: Arc::new(Mutex::new(Vec::with_capacity(PACKET_BYTES * 8))),
            register_fec_bits: Arc::new(Mutex::new(Vec::with_capacity((PACKET_BYTES * 8 + 6) * 2))),
            raw_bit_errors: Arc::new(Mutex::new(0)),
            raw_bits_compared: Arc::new(Mutex::new(0)),
            raw_walsh_bit_errors: Arc::new(Mutex::new(0)),
            raw_walsh_bits_compared: Arc::new(Mutex::new(0)),
            raw_dqpsk_bit_errors: Arc::new(Mutex::new(0)),
            raw_dqpsk_bits_compared: Arc::new(Mutex::new(0)),
            raw_dqpsk_walsh_weak_bit_errors: Arc::new(Mutex::new(0)),
            raw_dqpsk_walsh_weak_bits_compared: Arc::new(Mutex::new(0)),
            raw_dqpsk_walsh_strong_bit_errors: Arc::new(Mutex::new(0)),
            raw_dqpsk_walsh_strong_bits_compared: Arc::new(Mutex::new(0)),
            raw_error_runs: Arc::new(Mutex::new(0)),
            raw_error_run_bits: Arc::new(Mutex::new(0)),
            raw_error_run_max: Arc::new(Mutex::new(0)),
            codeword_error_weights: Arc::new(Mutex::new(Vec::new())),
            post_bit_errors: Arc::new(Mutex::new(0)),
            post_bits_compared: Arc::new(Mutex::new(0)),
            post_error_runs: Arc::new(Mutex::new(0)),
            post_error_run_bits: Arc::new(Mutex::new(0)),
            post_error_run_max: Arc::new(Mutex::new(0)),
            post_codeword_error_weights: Arc::new(Mutex::new(Vec::new())),
            post_decode_attempts: Arc::new(Mutex::new(0)),
            post_decode_matched: Arc::new(Mutex::new(0)),
            accepted_packets: Arc::new(Mutex::new(0)),
            false_accepted_packets: Arc::new(Mutex::new(0)),
        }
    }

    /// パケットを登録（FEC符号化後のビット列を記録）
    pub fn register_packet(
        &self,
        seq: u16,
        packet: &dsp::coding::fountain::FountainPacket,
        k: usize,
    ) {
        let pkt = Packet::new(seq, k, &packet.data);
        let mut packet_bytes = self.register_packet_bytes.lock().unwrap();
        let mut bits = self.register_bits.lock().unwrap();
        let mut fec_encoded = self.register_fec_bits.lock().unwrap();
        pkt.serialize_into(&mut packet_bytes);
        fec::bytes_to_bits_into(&packet_bytes, &mut bits);
        fec::encode_into(&bits, &mut fec_encoded);
        let mut scrambled = fec_encoded.clone();
        let mut scrambler = Scrambler::default();
        scrambler.process_bits(&mut scrambled);
        let mut interleaved = vec![0u8; dsp::mary::interleaver_config::interleaved_bits()];
        let interleaver = BlockInterleaver::new(
            dsp::mary::interleaver_config::INTERLEAVER_ROWS,
            dsp::mary::interleaver_config::INTERLEAVER_COLS,
        );
        interleaver.interleave_in_place(&scrambled, &mut interleaved);

        self.expected_packet_bits
            .lock()
            .unwrap()
            .insert(seq, bits.clone());
        self.expected_packet_bytes
            .lock()
            .unwrap()
            .insert(seq, packet_bytes.clone());
        self.expected_fec_bits
            .lock()
            .unwrap()
            .insert(seq, fec_encoded.clone());
        self.expected_interleaved_bits
            .lock()
            .unwrap()
            .insert(seq, interleaved);
    }

    /// CRC 通過後に受理されたパケットの観測コールバックを生成
    pub fn accepted_packet_callback(&self) -> Box<dyn FnMut(&Packet) + Send> {
        let expected_packet_bytes = Arc::clone(&self.expected_packet_bytes);
        let accepted_packets = Arc::clone(&self.accepted_packets);
        let false_accepted_packets = Arc::clone(&self.false_accepted_packets);
        Box::new(move |packet: &Packet| {
            *accepted_packets.lock().unwrap() += 1;
            let expected = {
                expected_packet_bytes
                    .lock()
                    .unwrap()
                    .get(&packet.lt_seq)
                    .cloned()
            };
            let Some(expected_bytes) = expected else {
                return;
            };
            if packet.serialize() != expected_bytes {
                *false_accepted_packets.lock().unwrap() += 1;
            }
        })
    }

    /// LLRコールバックを生成
    pub fn llr_callback(&self) -> LlrCallback {
        let efb = Arc::clone(&self.expected_fec_bits);
        let eib = Arc::clone(&self.expected_interleaved_bits);
        let epb = Arc::clone(&self.expected_packet_bits);
        let rbe = Arc::clone(&self.raw_bit_errors);
        let rbc = Arc::clone(&self.raw_bits_compared);
        let rwbe = Arc::clone(&self.raw_walsh_bit_errors);
        let rwbc = Arc::clone(&self.raw_walsh_bits_compared);
        let rdbe = Arc::clone(&self.raw_dqpsk_bit_errors);
        let rdbc = Arc::clone(&self.raw_dqpsk_bits_compared);
        let rdwbe = Arc::clone(&self.raw_dqpsk_walsh_weak_bit_errors);
        let rdwbc = Arc::clone(&self.raw_dqpsk_walsh_weak_bits_compared);
        let rdsbe = Arc::clone(&self.raw_dqpsk_walsh_strong_bit_errors);
        let rdsbc = Arc::clone(&self.raw_dqpsk_walsh_strong_bits_compared);
        let rer = Arc::clone(&self.raw_error_runs);
        let rrb = Arc::clone(&self.raw_error_run_bits);
        let rrm = Arc::clone(&self.raw_error_run_max);
        let cew = Arc::clone(&self.codeword_error_weights);
        let pbe = Arc::clone(&self.post_bit_errors);
        let pbc = Arc::clone(&self.post_bits_compared);
        let per = Arc::clone(&self.post_error_runs);
        let prb = Arc::clone(&self.post_error_run_bits);
        let prm = Arc::clone(&self.post_error_run_max);
        let pcew = Arc::clone(&self.post_codeword_error_weights);
        let pda = Arc::clone(&self.post_decode_attempts);
        let pdm = Arc::clone(&self.post_decode_matched);
        let fec_workspace = Arc::new(Mutex::new(fec::FecDecodeWorkspace::new()));
        let decoded_bits_buf = Arc::new(Mutex::new(Vec::<u8>::new()));
        let il_rows = dsp::mary::interleaver_config::INTERLEAVER_ROWS;
        let il_cols = dsp::mary::interleaver_config::INTERLEAVER_COLS;
        let il_bits = dsp::mary::interleaver_config::interleaved_bits();

        Box::new(move |llrs: &[f32]| {
            *pda.lock().unwrap() += 1;
            let p_bits_len = PACKET_BYTES * 8;
            let mut ws_guard = fec_workspace.lock().unwrap();
            let mut decoded_bits_guard = decoded_bits_buf.lock().unwrap();
            fec::decode_soft_into(llrs, &mut decoded_bits_guard, &mut ws_guard);
            if decoded_bits_guard.len() < p_bits_len {
                return;
            }
            let decoded_packet_bits = &decoded_bits_guard[..p_bits_len];
            let Some(seq) = packet_seq_from_bits(decoded_packet_bits) else {
                return;
            };
            let expected_raw = match efb.lock().unwrap().get(&seq) {
                Some(bits) => bits.clone(),
                None => return,
            };
            let (errors, compare_len, runs, run_bits, run_max) =
                bit_errors_and_runs_from_llr(&expected_raw, llrs);
            *rbe.lock().unwrap() += errors;
            *rbc.lock().unwrap() += compare_len;
            *rer.lock().unwrap() += runs;
            *rrb.lock().unwrap() += run_bits;
            let mut max_guard = rrm.lock().unwrap();
            *max_guard = (*max_guard).max(run_max);
            cew.lock().unwrap().push(errors);

            if llrs.len() >= il_bits {
                if let Some(expected_interleaved) = eib.lock().unwrap().get(&seq).cloned() {
                    let mut scrambled_llrs = llrs[..il_bits].to_vec();
                    let mut scrambler = Scrambler::default();
                    for llr in &mut scrambled_llrs {
                        if scrambler.next_bit() == 1 {
                            *llr = -*llr;
                        }
                    }
                    let mut interleaved_llrs = Vec::with_capacity(il_bits);
                    interleave_f32_block(&scrambled_llrs, il_rows, il_cols, &mut interleaved_llrs);

                    let compare_len = il_bits.min(expected_interleaved.len());
                    let mut walsh_errors = 0usize;
                    let mut walsh_bits = 0usize;
                    let mut dqpsk_errors = 0usize;
                    let mut dqpsk_bits = 0usize;
                    let mut dqpsk_walsh_weak_errors = 0usize;
                    let mut dqpsk_walsh_weak_bits = 0usize;
                    let mut dqpsk_walsh_strong_errors = 0usize;
                    let mut dqpsk_walsh_strong_bits = 0usize;
                    let full_symbols = compare_len / MARY_BITS_PER_SYMBOL;

                    for sym in 0..full_symbols {
                        let base = sym * MARY_BITS_PER_SYMBOL;
                        let walsh_abs_mean = interleaved_llrs[base..base + MARY_WALSH_BITS_PER_SYMBOL]
                            .iter()
                            .map(|v| v.abs())
                            .sum::<f32>()
                            / MARY_WALSH_BITS_PER_SYMBOL as f32;
                        let walsh_is_weak = walsh_abs_mean < MARY_WALSH_LLR_CONF_WEAK_THRESH;

                        for off in 0..MARY_WALSH_BITS_PER_SYMBOL {
                            let j = base + off;
                            let bit = expected_interleaved[j];
                            let llr = interleaved_llrs[j];
                            let is_err = (bit == 0 && llr <= 0.0) || (bit == 1 && llr >= 0.0);
                            walsh_bits += 1;
                            if is_err {
                                walsh_errors += 1;
                            }
                        }
                        for off in MARY_WALSH_BITS_PER_SYMBOL..MARY_BITS_PER_SYMBOL {
                            let j = base + off;
                            let bit = expected_interleaved[j];
                            let llr = interleaved_llrs[j];
                            let is_err = (bit == 0 && llr <= 0.0) || (bit == 1 && llr >= 0.0);
                            dqpsk_bits += 1;
                            if is_err {
                                dqpsk_errors += 1;
                            }
                            if walsh_is_weak {
                                dqpsk_walsh_weak_bits += 1;
                                if is_err {
                                    dqpsk_walsh_weak_errors += 1;
                                }
                            } else {
                                dqpsk_walsh_strong_bits += 1;
                                if is_err {
                                    dqpsk_walsh_strong_errors += 1;
                                }
                            }
                        }
                    }

                    for j in (full_symbols * MARY_BITS_PER_SYMBOL)..compare_len {
                        let bit = expected_interleaved[j];
                        let llr = interleaved_llrs[j];
                        let is_err = (bit == 0 && llr <= 0.0) || (bit == 1 && llr >= 0.0);
                        if (j % MARY_BITS_PER_SYMBOL) < MARY_WALSH_BITS_PER_SYMBOL {
                            walsh_bits += 1;
                            if is_err {
                                walsh_errors += 1;
                            }
                        } else {
                            dqpsk_bits += 1;
                            if is_err {
                                dqpsk_errors += 1;
                            }
                        }
                    }
                    *rwbe.lock().unwrap() += walsh_errors;
                    *rwbc.lock().unwrap() += walsh_bits;
                    *rdbe.lock().unwrap() += dqpsk_errors;
                    *rdbc.lock().unwrap() += dqpsk_bits;
                    *rdwbe.lock().unwrap() += dqpsk_walsh_weak_errors;
                    *rdwbc.lock().unwrap() += dqpsk_walsh_weak_bits;
                    *rdsbe.lock().unwrap() += dqpsk_walsh_strong_errors;
                    *rdsbc.lock().unwrap() += dqpsk_walsh_strong_bits;
                }
            }

            let expected_post = match epb.lock().unwrap().get(&seq) {
                Some(bits) => bits.clone(),
                None => return,
            };
            *pdm.lock().unwrap() += 1;
            let (post_errors, post_compare_len, post_runs, post_run_bits, post_run_max) =
                bit_errors_and_runs_from_bits(&expected_post, decoded_packet_bits);
            *pbe.lock().unwrap() += post_errors;
            *pbc.lock().unwrap() += post_compare_len;
            *per.lock().unwrap() += post_runs;
            *prb.lock().unwrap() += post_run_bits;
            let mut post_max_guard = prm.lock().unwrap();
            *post_max_guard = (*post_max_guard).max(post_run_max);
            pcew.lock().unwrap().push(post_errors);
        })
    }

    /// Pre-FEC解析結果を取得
    pub fn extract_pre_fec(&self) -> PreFecStats {
        let raw_bit_errors = *self.raw_bit_errors.lock().unwrap();
        let raw_bits_compared = *self.raw_bits_compared.lock().unwrap();
        let raw_walsh_bit_errors = *self.raw_walsh_bit_errors.lock().unwrap();
        let raw_walsh_bits_compared = *self.raw_walsh_bits_compared.lock().unwrap();
        let raw_dqpsk_bit_errors = *self.raw_dqpsk_bit_errors.lock().unwrap();
        let raw_dqpsk_bits_compared = *self.raw_dqpsk_bits_compared.lock().unwrap();
        let raw_dqpsk_walsh_weak_bit_errors = *self.raw_dqpsk_walsh_weak_bit_errors.lock().unwrap();
        let raw_dqpsk_walsh_weak_bits_compared =
            *self.raw_dqpsk_walsh_weak_bits_compared.lock().unwrap();
        let raw_dqpsk_walsh_strong_bit_errors =
            *self.raw_dqpsk_walsh_strong_bit_errors.lock().unwrap();
        let raw_dqpsk_walsh_strong_bits_compared =
            *self.raw_dqpsk_walsh_strong_bits_compared.lock().unwrap();
        let raw_error_runs = *self.raw_error_runs.lock().unwrap();
        let raw_error_run_bits = *self.raw_error_run_bits.lock().unwrap();
        let raw_error_run_max = *self.raw_error_run_max.lock().unwrap();
        let codeword_error_weights = self.codeword_error_weights.lock().unwrap().clone();
        let codeword_count = codeword_error_weights.len();
        let codeword_error_sum = codeword_error_weights.iter().sum::<usize>();
        let codeword_error_max = codeword_error_weights.iter().copied().max().unwrap_or(0);
        PreFecStats {
            bit_errors: raw_bit_errors,
            bits_compared: raw_bits_compared,
            walsh_bit_errors: raw_walsh_bit_errors,
            walsh_bits_compared: raw_walsh_bits_compared,
            dqpsk_bit_errors: raw_dqpsk_bit_errors,
            dqpsk_bits_compared: raw_dqpsk_bits_compared,
            dqpsk_walsh_weak_bit_errors: raw_dqpsk_walsh_weak_bit_errors,
            dqpsk_walsh_weak_bits_compared: raw_dqpsk_walsh_weak_bits_compared,
            dqpsk_walsh_strong_bit_errors: raw_dqpsk_walsh_strong_bit_errors,
            dqpsk_walsh_strong_bits_compared: raw_dqpsk_walsh_strong_bits_compared,
            error_runs: raw_error_runs,
            error_run_bits: raw_error_run_bits,
            error_run_max: raw_error_run_max,
            codeword_count,
            codeword_error_sum,
            codeword_error_max,
            codeword_error_weights,
        }
    }

    /// Post-FEC解析結果を取得
    pub fn extract_post_fec(&self) -> PostFecStats {
        let post_bit_errors = *self.post_bit_errors.lock().unwrap();
        let post_bits_compared = *self.post_bits_compared.lock().unwrap();
        let post_error_runs = *self.post_error_runs.lock().unwrap();
        let post_error_run_bits = *self.post_error_run_bits.lock().unwrap();
        let post_error_run_max = *self.post_error_run_max.lock().unwrap();
        let post_codeword_error_weights = self.post_codeword_error_weights.lock().unwrap().clone();
        let post_codeword_count = post_codeword_error_weights.len();
        let post_codeword_error_sum = post_codeword_error_weights.iter().sum::<usize>();
        let post_codeword_error_max = post_codeword_error_weights
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        PostFecStats {
            bit_errors: post_bit_errors,
            bits_compared: post_bits_compared,
            error_runs: post_error_runs,
            error_run_bits: post_error_run_bits,
            error_run_max: post_error_run_max,
            codeword_count: post_codeword_count,
            codeword_error_sum: post_codeword_error_sum,
            codeword_error_max: post_codeword_error_max,
            codeword_error_weights: post_codeword_error_weights,
        }
    }

    /// デコード統計を取得
    pub fn extract_decode_stats(&self) -> (usize, usize) {
        let post_decode_attempts = *self.post_decode_attempts.lock().unwrap();
        let post_decode_matched = *self.post_decode_matched.lock().unwrap();
        (post_decode_attempts, post_decode_matched)
    }

    /// 受理パケットに対する false accept 統計を取得
    pub fn extract_false_accept_stats(&self) -> (usize, usize) {
        let accepted_packets = *self.accepted_packets.lock().unwrap();
        let false_accepted_packets = *self.false_accepted_packets.lock().unwrap();
        (accepted_packets, false_accepted_packets)
    }
}

pub fn parse_positive_f32(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("invalid float: {value}"))?;
    if parsed > 0.0 {
        Ok(parsed)
    } else {
        Err(format!("value must be > 0: {value}"))
    }
}

pub fn parse_nonnegative_f32(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("invalid float: {value}"))?;
    if parsed >= 0.0 {
        Ok(parsed)
    } else {
        Err(format!("value must be >= 0: {value}"))
    }
}

pub fn parse_unit_interval_f32(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("invalid float: {value}"))?;
    if (0.0..=1.0).contains(&parsed) {
        Ok(parsed)
    } else {
        Err(format!("value must be in [0,1]: {value}"))
    }
}

pub fn parse_positive_usize(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("invalid integer: {value}"))?;
    if parsed > 0 {
        Ok(parsed)
    } else {
        Err(format!("value must be > 0: {value}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp::coding::fountain::FountainPacket;
    use dsp::params::PAYLOAD_SIZE;

    #[test]
    fn test_count_bit_errors_bytes() {
        // Equal length
        assert_eq!(
            count_bit_errors_bytes(&[0b10101010], Some(&[0b10101010])),
            0
        );
        assert_eq!(
            count_bit_errors_bytes(&[0b10101010], Some(&[0b01010101])),
            8
        );
        assert_eq!(
            count_bit_errors_bytes(&[0xFF, 0x00], Some(&[0xFE, 0x01])),
            2
        );

        // Different length
        assert_eq!(count_bit_errors_bytes(&[0x00, 0x00], Some(&[0x00])), 8); // 2nd byte missing -> 8 errors
        assert_eq!(count_bit_errors_bytes(&[0x00], Some(&[0x00, 0xFF])), 0); // extra byte ignored

        // None case
        assert_eq!(count_bit_errors_bytes(&[0x00, 0x00, 0x00], None), 24);
    }

    #[test]
    fn test_bit_errors_and_runs_from_llr() {
        // expected: 0 -> LLR > 0 is correct
        // expected: 1 -> LLR <= 0 is correct
        let expected = vec![0, 1, 0, 1];
        let llrs = vec![1.0, -1.0, -0.5, 0.5];
        // index 2: exp 0, llr -0.5 (bit 1) -> error
        // index 3: exp 1, llr 0.5 (bit 0) -> error
        // 2 errors, 1 run (index 2-3)
        let (errs, comp, runs, run_bits, run_max) = bit_errors_and_runs_from_llr(&expected, &llrs);
        assert_eq!(errs, 2);
        assert_eq!(comp, 4);
        assert_eq!(runs, 1);
        assert_eq!(run_bits, 2);
        assert_eq!(run_max, 2);

        // Multiple runs
        let expected2 = vec![0, 0, 0, 0, 0];
        let llrs2 = vec![-1.0, 1.0, -1.0, -1.0, 1.0];
        // errors at: 0, 2, 3
        // runs: [0], [2, 3] -> 2 runs
        let (errs, _, runs, run_bits, run_max) = bit_errors_and_runs_from_llr(&expected2, &llrs2);
        assert_eq!(errs, 3);
        assert_eq!(runs, 2);
        assert_eq!(run_bits, 3);
        assert_eq!(run_max, 2);

        // LLR 0.0 is always an error for both expected 0 and 1
        assert_eq!(bit_errors_and_runs_from_llr(&[0], &[0.0]).0, 1);
        assert_eq!(bit_errors_and_runs_from_llr(&[1], &[0.0]).0, 1);
    }

    #[test]
    fn test_bit_errors_and_runs_from_bits() {
        let expected = vec![0, 1, 0, 1, 1, 0];
        let observed = vec![0, 0, 0, 1, 0, 1];
        // errors at index: 1, 4, 5
        // runs: [1], [4, 5] -> 2 runs
        let (errs, comp, runs, run_bits, run_max) =
            bit_errors_and_runs_from_bits(&expected, &observed);
        assert_eq!(errs, 3);
        assert_eq!(comp, 6);
        assert_eq!(runs, 2);
        assert_eq!(run_bits, 3);
        assert_eq!(run_max, 2);

        // Different length
        let (errs, comp, _, _, _) = bit_errors_and_runs_from_bits(&[0, 0, 0], &[1, 1]);
        assert_eq!(errs, 2);
        assert_eq!(comp, 2); // compares min length
    }

    #[test]
    fn test_count_bit_errors_bytes_edge_cases() {
        assert_eq!(count_bit_errors_bytes(&[], Some(&[])), 0);
        assert_eq!(count_bit_errors_bytes(&[0x00], Some(&[0x00, 0xFF])), 0);
        assert_eq!(count_bit_errors_bytes(&[0x00, 0x00], Some(&[0x00])), 8);
        assert_eq!(count_bit_errors_bytes(&[0x00, 0x00], None), 16);
    }

    #[test]
    fn test_bit_errors_and_runs_boundary_conditions() {
        let (errs, _, runs, _, _) = bit_errors_and_runs_from_llr(&[0, 0], &[-1.0, 1.0]);
        assert_eq!(errs, 1);
        assert_eq!(runs, 1);
        let (errs, _, runs, _, _) = bit_errors_and_runs_from_llr(&[0, 0], &[1.0, -1.0]);
        assert_eq!(errs, 1);
        assert_eq!(runs, 1);
        let (errs, comp, runs, run_bits, run_max) =
            bit_errors_and_runs_from_llr(&[0, 0, 0], &[-1.0, -1.0, -1.0]);
        assert_eq!(errs, 3);
        assert_eq!(comp, 3);
        assert_eq!(runs, 1);
        assert_eq!(run_bits, 3);
        assert_eq!(run_max, 3);
        let (errs, _, runs, _, _) =
            bit_errors_and_runs_from_llr(&[0, 0, 0, 0], &[-1.0, 1.0, -1.0, 1.0]);
        assert_eq!(errs, 2);
        assert_eq!(runs, 2);
        let (errs0, _, _, _, _) = bit_errors_and_runs_from_llr(&[0], &[0.0]);
        let (errs1, _, _, _, _) = bit_errors_and_runs_from_llr(&[1], &[0.0]);
        assert_eq!(errs0, 1);
        assert_eq!(errs1, 1);
    }

    #[test]
    fn test_ber_accumulator_robustness() {
        let accum = BerAccumulator::new();
        let k = 1;
        let data = vec![0u8; PAYLOAD_SIZE];
        accum.register_packet(
            10,
            &FountainPacket {
                seq: 10,
                data: data.clone(),
                coefficients: vec![],
            },
            k,
        );
        let cb = accum.llr_callback();
        cb(&[1.0; 10]);
        assert_eq!(accum.extract_decode_stats().0, 1);
        assert_eq!(accum.extract_pre_fec().codeword_count, 0);
        let pkt = Packet::new(10, k, &data);
        let bits = fec::bytes_to_bits(&pkt.serialize());
        let fec_encoded = fec::encode(&bits);
        let mut llrs: Vec<f32> = fec_encoded
            .iter()
            .map(|&b| if b == 0 { 1.0 } else { -1.0 })
            .collect();
        cb(&llrs);
        llrs[0] *= -1.0;
        cb(&llrs);
        let pre = accum.extract_pre_fec();
        assert_eq!(pre.codeword_count, 2);
        assert_eq!(pre.bit_errors, 1);
        assert_eq!(
            pre.walsh_bits_compared + pre.dqpsk_bits_compared,
            pre.bits_compared
        );
        assert_eq!(pre.walsh_bit_errors + pre.dqpsk_bit_errors, pre.bit_errors);
        assert_eq!(
            pre.dqpsk_walsh_weak_bits_compared + pre.dqpsk_walsh_strong_bits_compared,
            pre.dqpsk_bits_compared
        );
        assert_eq!(
            pre.dqpsk_walsh_weak_bit_errors + pre.dqpsk_walsh_strong_bit_errors,
            pre.dqpsk_bit_errors
        );
        assert_eq!(pre.codeword_error_weights, vec![0, 1]);
        let (att, mat) = accum.extract_decode_stats();
        assert_eq!(att, 3);
        assert_eq!(mat, 2);
        let p_wrap = Packet::new(0xFFFF, k, &data);
        let bits_wrap = fec::bytes_to_bits(&p_wrap.serialize());
        let fec_wrap = fec::encode(&bits_wrap);
        accum.register_packet(
            0xFFFF,
            &FountainPacket {
                seq: 0xFFFF,
                data: data.clone(),
                coefficients: vec![],
            },
            k,
        );
        cb(&fec_wrap
            .iter()
            .map(|&b| if b == 0 { 1.0 } else { -1.0 })
            .collect::<Vec<_>>());
        assert_eq!(accum.extract_pre_fec().codeword_count, 3);
    }

    #[test]
    fn test_accepted_packet_callback_false_accept_stats() {
        let accum = BerAccumulator::new();
        let k = 2usize;
        let seq = 77u16;
        let data = vec![0x11; PAYLOAD_SIZE];
        accum.register_packet(
            seq,
            &FountainPacket {
                seq: seq as u32,
                data: data.clone(),
                coefficients: vec![],
            },
            k,
        );

        let mut cb = accum.accepted_packet_callback();
        let ok_packet = Packet::new(seq, k, &data);
        cb(&ok_packet);

        let mut wrong_payload = data.clone();
        wrong_payload[0] ^= 0x80;
        let bad_packet = Packet::new(seq, k, &wrong_payload);
        cb(&bad_packet);

        let (accepted, false_accepted) = accum.extract_false_accept_stats();
        assert_eq!(accepted, 2);
        assert_eq!(false_accepted, 1);
    }
}
