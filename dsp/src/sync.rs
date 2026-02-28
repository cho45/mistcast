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
///
/// 戻り値は `(Option<SyncResult>, usize)` であり、第2要素は「探索済みで安全に破棄できるサンプル数」を返す。
pub fn detect(&self, samples: &[f32]) -> (Option<SyncResult>, usize) {
    let (i_ch, q_ch) = downconvert(samples, 0, &self.config);
    let (chips_i, chips_q) = matched_filter_decimate(&i_ch, &q_ch, &self.config);
    self.detect_chips(&chips_i, &chips_q)
}

/// チップ列から同期位置を検出する
///
/// 戻り値の `SyncResult::data_start_sample` は、この関数に渡されたチップ列が
/// もし0サンプル目から生成されたと仮定した場合の元のサンプルインデックスを返す。
/// 実際にはチップ数ベースで管理する方が良いため、内部で変換している。
pub fn detect_chips(&self, chips_i: &[f32], chips_q: &[f32]) -> (Option<SyncResult>, usize) {
    let period = self.local_mseq.len();
    let window = period * self.config.preamble_repeat;

    if chips_i.len() < window {
        return (None, 0);
    }
        let search_range = chips_i.len().saturating_sub(window);
        let mut energies = Vec::with_capacity(search_range);
        let mut best_energy = 0.0f32;

        for offset in 0..search_range {
            let mut sum_ci = 0.0f32;
            let mut sum_cq = 0.0f32;
            for rep in 0..self.config.preamble_repeat {
                let start = offset + rep * period;
                if start + period > chips_i.len() {
                    break;
                }
                let ci: f32 = chips_i[start..start + period]
                    .iter().zip(self.local_mseq.iter()).map(|(&r, &l)| r * l).sum();
                let cq: f32 = chips_q[start..start + period]
                    .iter().zip(self.local_mseq.iter()).map(|(&r, &l)| r * l).sum();
                sum_ci += ci;
                sum_cq += cq;
            }
            // Coherent integration: square after summing across all repetitions.
            let energy = sum_ci * sum_ci + sum_cq * sum_cq;
            energies.push(energy);
            if energy > best_energy {
                best_energy = energy;
            }
        }

        let mean_energy = if !energies.is_empty() {
            energies.iter().sum::<f32>() / energies.len() as f32
        } else {
            1.0
        };

        let confidence = if mean_energy > 0.0 { best_energy / mean_energy } else { 0.0 };
        let mut best_local_offset = None;

        // 閾値をクリアしている場合のみ詳細なピーク判定を行う
        if best_energy > 0.0 && confidence > 10.0 {
            // 絶対最大値の90%以上を真のピーク候補とする (前置ピークは ~56% なので確実に弾かれる)
            let peak_threshold = best_energy * 0.9;
            
            // 候補を満たす「一番最初」のローカルピークを真の同期位置とする
            for (offset, &energy) in energies.iter().enumerate().take(search_range) {
                if energy >= peak_threshold {
                    // 前後2チップで極大値（ローカルピーク）であるか確認
                    let start_idx = offset.saturating_sub(2);
                    let end_idx = (offset + 2).min(search_range - 1);
                    
                    let mut is_peak = true;
                    for &e_val in energies.iter().take(end_idx + 1).skip(start_idx) {
                        if e_val > energy {
                            is_peak = false;
                            break;
                        }
                    }

                    if is_peak {
                        best_local_offset = Some((offset, energy));
                        break; // 最初の真のピークを発見したら即座に確定
                    }
                }
            }
        }

        let searched_samples = search_range * self.config.samples_per_chip();

        if let Some((offset, energy)) = best_local_offset {
            let data_start_chip = offset + window;
            let data_start_sample = data_start_chip * self.config.samples_per_chip();
            (Some(SyncResult { data_start_sample, peak_value: energy.sqrt(), confidence }), offset * self.config.samples_per_chip())
        } else {
            (None, searched_samples)
        }
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
    fn test_sync_detection_exact() {
        let config = test_config();
        let mut mod_ = Modulator::new(config.clone());
        let preamble = mod_.generate_preamble();
        
        let data_bits = vec![1u8, 0, 1, 0, 1, 0, 1, 0];
        let data_samples = mod_.modulate(&data_bits);

        let mut signal = Vec::new();
        // 前方に無音区間を入れて同期位置がずれないか確認
        // samples_per_chip = 6 の倍数にする
        let silence_len = 96;
        signal.extend(vec![0.0; silence_len]);
        signal.extend_from_slice(&preamble);
        signal.extend_from_slice(&data_samples);

        let detector = SyncDetector::new(config.clone());
        let (result_opt, _) = detector.detect(&signal);

        assert!(result_opt.is_some(), "同期捕捉が成功すること");
        let sync = result_opt.unwrap();
        
        // 理論的なデータ開始位置の計算
        let expected_start = silence_len + preamble.len();
        
        assert_eq!(
            sync.data_start_sample, expected_start,
            "同期位置が数学的に完全に一致すること。 expected={}, got={}",
            expected_start, sync.data_start_sample
        );
    }

    #[test]
    fn test_sync_no_false_positives() {
        let config = test_config();
        let mut mod_ = Modulator::new(config.clone());
        let preamble = mod_.generate_preamble();
        
        // 大量のランダムデータを作成して、データ部分に誤検出しないかテストする
        let mut data_bits = Vec::with_capacity(2000);
        let mut state = 12345u32; // 疑似乱数
        for _ in 0..2000 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            data_bits.push((state >> 16 & 1) as u8);
        }
        let data_samples = mod_.modulate(&data_bits);

        let mut signal = Vec::new();
        let silence_len = 240; // 40 chips
        signal.extend(vec![0.0; silence_len]);
        signal.extend_from_slice(&preamble);
        signal.extend_from_slice(&data_samples);

        let detector = SyncDetector::new(config.clone());
        let (i_ch, q_ch) = downconvert(&signal, 0, &detector.config);
        let (chips_i, chips_q) = matched_filter_decimate(&i_ch, &q_ch, &detector.config);
        
        let period = detector.local_mseq.len();
        let window = period * detector.config.preamble_repeat;
        let search_range = chips_i.len().saturating_sub(window);
        let mut energies = Vec::with_capacity(search_range);

        for offset in 0..search_range {
            let mut sum_ci = 0.0f32;
            let mut sum_cq = 0.0f32;
            for rep in 0..detector.config.preamble_repeat {
                let start = offset + rep * period;
                if start + period > chips_i.len() { break; }
                let ci: f32 = chips_i[start..start + period].iter().zip(detector.local_mseq.iter()).map(|(&r, &l)| r * l).sum();
                let cq: f32 = chips_q[start..start + period].iter().zip(detector.local_mseq.iter()).map(|(&r, &l)| r * l).sum();
                sum_ci += ci;
                sum_cq += cq;
            }
            energies.push(sum_ci * sum_ci + sum_cq * sum_cq);
        }
        
        let mean_energy = energies.iter().sum::<f32>() / energies.len() as f32;
        
        println!("--- Debugging Sync Peaks ---");
        for (offset, &energy) in energies.iter().enumerate().take(46) {
            let confidence = energy / mean_energy;
            if confidence > 5.0 || offset == 40 {
                println!("Offset: {}, Energy: {:.2}, Confidence: {:.2}", offset, energy, confidence);
            }
        }
        println!("---");

        let (result_opt, _) = detector.detect(&signal);

        assert!(result_opt.is_some(), "長大なデータが付加されていても同期捕捉が成功すること");
        let sync = result_opt.unwrap();
        
        let expected_start = silence_len + preamble.len();
        assert_eq!(
            sync.data_start_sample, expected_start,
            "大量のデータが存在しても、プリアンブルの正確な終了位置を1サンプルの狂いもなく指し示すこと。誤検出(False Positive)は許されない。 expected={}, got={}",
            expected_start, sync.data_start_sample
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
