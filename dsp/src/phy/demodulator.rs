//! DBPSK + DSSS 復調器

use crate::DspConfig;

/// DBPSK + DSSS 復調器
pub struct Demodulator {
    config: DspConfig,
    prev_i: f32,
    prev_q: f32,
}

impl Demodulator {
    pub fn new(config: DspConfig) -> Self {
        Demodulator {
            config,
            // 初期位相基準
            prev_i: 1.0,
            prev_q: 0.0,
        }
    }

    /// チップ列（拡散されたベースバンド信号）からビット列を復元する
    pub fn demodulate_chips(&mut self, chips_i: &[f32], chips_q: &[f32]) -> Vec<u8> {
        let lrs = self.demodulate_chips_soft(chips_i, chips_q);
        lrs.into_iter()
            .map(|llr| if llr > 0.0 { 0 } else { 1 })
            .collect()
    }

    /// ソフト判定 (LLR) を返す復調
    pub fn demodulate_chips_soft(&mut self, chips_i: &[f32], chips_q: &[f32]) -> Vec<f32> {
        let sf = self.config.spread_factor();
        let num_symbols = chips_i.len() / sf;
        let mut llrs = Vec::with_capacity(num_symbols);

        let mut mseq = crate::common::msequence::MSequence::new(self.config.mseq_order);
        let pn: Vec<f32> = mseq.generate(sf).iter().map(|&c| c as f32).collect();

        for s_idx in 0..num_symbols {
            let start = s_idx * sf;
            let end = start + sf;
            let symbol_chips_i = &chips_i[start..end];
            let symbol_chips_q = &chips_q[start..end];

            // 1. 逆拡散
            let mut cur_i = 0.0f32;
            let mut cur_q = 0.0f32;
            for (k, &pn_val) in pn.iter().enumerate() {
                cur_i += symbol_chips_i[k] * pn_val;
                cur_q += symbol_chips_q[k] * pn_val;
            }
            cur_i /= sf as f32;
            cur_q /= sf as f32;

            // 2. DBPSK 判定 (ドット積)
            let dot = cur_i * self.prev_i + cur_q * self.prev_q;
            llrs.push(dot); // 簡易的な LLR としてドット積をそのまま返す

            // 位相基準の更新 (信頼性が高い場合のみ強く更新)
            let mag = (cur_i * cur_i + cur_q * cur_q).sqrt().max(1e-6);
            if mag > 0.1 {
                // 指数移動平均的な更新（ノイズ耐性向上のため、瞬時の位相に引きずられすぎないようにする）
                let alpha = 0.8;
                self.prev_i = self.prev_i * (1.0 - alpha) + (cur_i / mag) * alpha;
                self.prev_q = self.prev_q * (1.0 - alpha) + (cur_q / mag) * alpha;
                let norm = (self.prev_i * self.prev_i + self.prev_q * self.prev_q)
                    .sqrt()
                    .max(1e-6);
                self.prev_i /= norm;
                self.prev_q /= norm;
            }
        }

        llrs
    }

    pub fn reset(&mut self) {
        self.prev_i = 1.0;
        self.prev_q = 0.0;
    }

    pub fn set_reference_phase(&mut self, i: f32, q: f32) {
        let mag = (i * i + q * q).sqrt().max(1e-6);
        self.prev_i = i / mag;
        self.prev_q = q / mag;
    }
}
