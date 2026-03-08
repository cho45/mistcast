use dsp::coding::fec;
use dsp::frame::packet::{Packet, PACKET_BYTES};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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

/// Mary BER集計用
pub struct BerAccumulator {
    expected_fec_bits: Arc<Mutex<HashMap<u16, Vec<u8>>>>,
    expected_packet_bits: Arc<Mutex<HashMap<u16, Vec<u8>>>>,
    raw_bit_errors: Arc<Mutex<usize>>,
    raw_bits_compared: Arc<Mutex<usize>>,
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
            expected_packet_bits: Arc::new(Mutex::new(HashMap::new())),
            raw_bit_errors: Arc::new(Mutex::new(0)),
            raw_bits_compared: Arc::new(Mutex::new(0)),
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
        let bits = fec::bytes_to_bits(&pkt.serialize());
        let fec_encoded = fec::encode(&bits);
        self.expected_packet_bits.lock().unwrap().insert(seq, bits);
        self.expected_fec_bits
            .lock()
            .unwrap()
            .insert(seq, fec_encoded);
    }

    /// LLRコールバックを生成
    pub fn llr_callback(&self) -> LlrCallback {
        let efb = Arc::clone(&self.expected_fec_bits);
        let epb = Arc::clone(&self.expected_packet_bits);
        let rbe = Arc::clone(&self.raw_bit_errors);
        let rbc = Arc::clone(&self.raw_bits_compared);
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

        Box::new(move |llrs: &[f32]| {
            *pda.lock().unwrap() += 1;
            let decoded_bits = fec::decode_soft(llrs);
            let p_bits_len = PACKET_BYTES * 8;
            if decoded_bits.len() < p_bits_len {
                return;
            }
            let decoded_bytes = fec::bits_to_bytes(&decoded_bits[..p_bits_len]);
            if decoded_bytes.len() < 3 {
                return;
            }
            let seq = u16::from_be_bytes([decoded_bytes[1], decoded_bytes[2]]);
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

            let expected_post = match epb.lock().unwrap().get(&seq) {
                Some(bits) => bits.clone(),
                None => return,
            };
            *pdm.lock().unwrap() += 1;
            let (post_errors, post_compare_len, post_runs, post_run_bits, post_run_max) =
                bit_errors_and_runs_from_bits(&expected_post, &decoded_bits[..p_bits_len]);
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
        let (errs, comp, runs, run_bits, run_max) = bit_errors_and_runs_from_llr(&[0, 0, 0], &[-1.0, -1.0, -1.0]);
        assert_eq!(errs, 3);
        assert_eq!(comp, 3);
        assert_eq!(runs, 1);
        assert_eq!(run_bits, 3);
        assert_eq!(run_max, 3);
        let (errs, _, runs, _, _) = bit_errors_and_runs_from_llr(&[0, 0, 0, 0], &[-1.0, 1.0, -1.0, 1.0]);
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
        accum.register_packet(10, &FountainPacket { seq: 10, data: data.clone(), coefficients: vec![] }, k);
        let cb = accum.llr_callback();
        cb(&[1.0; 10]); 
        assert_eq!(accum.extract_decode_stats().0, 1);
        assert_eq!(accum.extract_pre_fec().codeword_count, 0);
        let pkt = Packet::new(10, k, &data);
        let bits = fec::bytes_to_bits(&pkt.serialize());
        let fec_encoded = fec::encode(&bits);
        let mut llrs: Vec<f32> = fec_encoded.iter().map(|&b| if b == 0 { 1.0 } else { -1.0 }).collect();
        cb(&llrs);
        llrs[0] *= -1.0;
        cb(&llrs);
        let pre = accum.extract_pre_fec();
        assert_eq!(pre.codeword_count, 2);
        assert_eq!(pre.bit_errors, 1);
        assert_eq!(pre.codeword_error_weights, vec![0, 1]);
        let (att, mat) = accum.extract_decode_stats();
        assert_eq!(att, 3);
        assert_eq!(mat, 2);
        let p_wrap = Packet::new(0xFFFF, k, &data);
        let bits_wrap = fec::bytes_to_bits(&p_wrap.serialize());
        let fec_wrap = fec::encode(&bits_wrap);
        accum.register_packet(0xFFFF, &FountainPacket { seq: 0xFFFF, data: data.clone(), coefficients: vec![] }, k);
        cb(&fec_wrap.iter().map(|&b| if b == 0 { 1.0 } else { -1.0 }).collect::<Vec<_>>());
        assert_eq!(accum.extract_pre_fec().codeword_count, 3);
    }
}
