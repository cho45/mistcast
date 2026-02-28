//! スライディング相関による同期捕捉

use crate::msequence::MSequence;
use crate::rrc_filter::RrcFilter;
use crate::DspConfig;

/// ダウンコンバート: 受信信号にcos/sin を掛けてベースバンドI/Q信号に変換
///
/// I(t) = x(t) * cos(2π·fc·t) * 2
/// Q(t) = x(t) * (-sin(2π·fc·t)) * 2
pub fn downconvert(samples: &[f32], sample_offset: usize, config: &DspConfig) -> (Vec<f32>, Vec<f32>) {
    let two_pi = 2.0 * std::f32::consts::PI;
    let fs = config.sample_rate;
    let fc = config.carrier_freq;
    let mut i_ch = Vec::with_capacity(samples.len());
    let mut q_ch = Vec::with_capacity(samples.len());

    for (k, &s) in samples.iter().enumerate() {
        let t = (sample_offset + k) as f32 / fs;
        let cos_val = (two_pi * fc * t).cos();
        let sin_val = (two_pi * fc * t).sin();
        i_ch.push(s * cos_val * 2.0);
        q_ch.push(s * (-sin_val) * 2.0);
    }
    (i_ch, q_ch)
}

/// RRCマッチドフィルタを適用してデシメーション (チップレートに落とす)
pub fn matched_filter_decimate(
    i_ch: &[f32],
    q_ch: &[f32],
    config: &DspConfig,
) -> (Vec<f32>, Vec<f32>) {
    let mut rrc_i = RrcFilter::from_config(config);
    let mut rrc_q = RrcFilter::from_config(config);
    let spc = config.samples_per_chip();
    let mut filtered_i: Vec<f32> = i_ch.iter().map(|&s| rrc_i.process(s)).collect();
    let mut filtered_q: Vec<f32> = q_ch.iter().map(|&s| rrc_q.process(s)).collect();

    // マッチドフィルタによる遅延補正
    // Tx RRC (delay) + Rx RRC (delay) = 2 * (num_taps - 1) / 2 = num_taps - 1
    // これにより最初のチップのピークがインデックス0に来る
    let total_delay = config.rrc_num_taps().saturating_sub(1);

    if filtered_i.len() > total_delay {
        filtered_i = filtered_i[total_delay..].to_vec();
        filtered_q = filtered_q[total_delay..].to_vec();
    }

    let out_i: Vec<f32> = filtered_i.iter().step_by(spc).cloned().collect();
    let out_q: Vec<f32> = filtered_q.iter().step_by(spc).cloned().collect();

    (out_i, out_q)
}

/// 同期捕捉の結果
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// データ開始サンプルインデックス
    pub data_start_sample: usize,
    /// 相関ピーク値
    pub peak_value: f32,
    /// 信頼度 (ピーク/平均比)
    pub confidence: f32,
}

/// スライディング相関によるプリアンブル検出
pub struct SyncDetector {
    config: DspConfig,
    local_mseq: Vec<f32>,
}

impl SyncDetector {
    pub fn new(config: DspConfig) -> Self {
        let mut mseq = MSequence::new(config.mseq_order);
        let period = config.spread_factor();
        let chips: Vec<f32> = mseq.generate(period).iter().map(|&c| c as f32).collect();
        SyncDetector { config, local_mseq: chips }
    }

    pub fn default_48k() -> Self {
        Self::new(DspConfig::default_48k())
    }

    /// チップ列に対してスライディング相関を計算し、ピーク位置を返す
    pub fn slide_correlate(&self, chips: &[f32]) -> (usize, f32) {
        let n = self.local_mseq.len();
        if chips.len() < n {
            return (0, 0.0);
        }

        let mut best_pos = 0;
        let mut best_corr = f32::NEG_INFINITY;

        for start in 0..=(chips.len() - n) {
            let corr: f32 = chips[start..start + n]
                .iter()
                .zip(self.local_mseq.iter())
                .map(|(&r, &l)| r * l)
                .sum::<f32>()
                / n as f32;
            if corr > best_corr {
                best_corr = corr;
                best_pos = start;
            }
        }
        (best_pos, best_corr)
    }

    /// サンプル列から同期位置を検出する
    pub fn detect(&self, samples: &[f32]) -> Option<SyncResult> {
        let (i_ch, q_ch) = downconvert(samples, 0, &self.config);
        let (chips_i, chips_q) = matched_filter_decimate(&i_ch, &q_ch, &self.config);

        let period = self.local_mseq.len();
        let window = period * self.config.preamble_repeat;

        if chips_i.len() < window {
            return None;
        }

        let mut best_offset = 0;
        let mut best_energy = 0.0f32;
        let search_range = chips_i.len().saturating_sub(window);

        for offset in 0..search_range {
            let mut energy = 0.0f32;
            for rep in 0..self.config.preamble_repeat {
                let start = offset + rep * period;
                if start + period > chips_i.len() {
                    break;
                }
                let ci: f32 = chips_i[start..start + period]
                    .iter().zip(self.local_mseq.iter()).map(|(&r, &l)| r * l).sum();
                let cq: f32 = chips_q[start..start + period]
                    .iter().zip(self.local_mseq.iter()).map(|(&r, &l)| r * l).sum();
                energy += ci * ci + cq * cq;
            }
            if energy > best_energy {
                best_energy = energy;
                best_offset = offset;
            }
        }

        let mean_energy = if search_range > 0 {
            (0..search_range)
                .map(|offset| {
                    let ci: f32 = chips_i[offset..offset + period]
                        .iter().zip(self.local_mseq.iter()).map(|(&r, &l)| r * l).sum();
                    ci * ci
                })
                .sum::<f32>()
                / search_range as f32
        } else {
            1.0
        };

        let confidence = if mean_energy > 0.0 {
            best_energy / (mean_energy * self.config.preamble_repeat as f32)
        } else {
            0.0
        };

        let data_start_chip = best_offset + window;
        let data_start_sample = data_start_chip * self.config.samples_per_chip();

        Some(SyncResult { data_start_sample, peak_value: best_energy.sqrt(), confidence })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modulator::Modulator;

    fn test_config() -> DspConfig {
        DspConfig::default_48k()
    }

    #[test]
    fn test_sync_detection() {
        let config = test_config();
        let spc = config.samples_per_chip();
        let mut mod_ = Modulator::new(config.clone());
        let preamble = mod_.generate_preamble();
        let data_bits = vec![1u8, 0, 1, 0, 1, 0, 1, 0];
        let data_samples = mod_.modulate(&data_bits);

        let mut signal = preamble.clone();
        signal.extend_from_slice(&data_samples);

        let detector = SyncDetector::new(config);
        let result = detector.detect(&signal);

        assert!(result.is_some(), "同期捕捉が成功すること");
        let sync = result.unwrap();
        assert!(
            sync.data_start_sample >= preamble.len().saturating_sub(spc * 5),
            "データ開始位置が概ね正確であること: got={}, expected≈{}",
            sync.data_start_sample, preamble.len()
        );
    }

    #[test]
    fn test_downconvert_finite() {
        let config = test_config();
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let (i_ch, q_ch) = downconvert(&samples, 0, &config);
        assert!(i_ch.iter().all(|&s| s.is_finite()));
        assert!(q_ch.iter().all(|&s| s.is_finite()));
    }
}
