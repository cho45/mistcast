//! 実用的な同期捕捉 (ノンコヒーレント相関)
//!
//! 1シンボルごとに相関電力を計算し、それらを足し合わせることで、
//! 位相回転やクロックズレに強い同期捕捉を実現する。

use crate::common::msequence::MSequence;
use crate::DspConfig;

#[derive(Debug, Clone)]
pub struct SyncResult {
    pub peak_sample_idx: usize,
    pub peak_iq: (f32, f32),
    pub score: f32,
}

pub struct SyncDetector {
    config: DspConfig,
    pn: Vec<f32>,
    sf: usize,
    spc: usize,
    sym_len: usize,
}

impl SyncDetector {
    pub fn new(config: DspConfig) -> Self {
        let mut mseq = MSequence::new(config.mseq_order);
        let sf = config.spread_factor();
        let spc = config.samples_per_chip().max(1);
        let pn: Vec<f32> = mseq.generate(sf).into_iter().map(|x| x as f32).collect();
        let sym_len = sf * spc;

        SyncDetector {
            config,
            pn,
            sf,
            spc,
            sym_len,
        }
    }

    pub fn filter_delay(&self) -> usize {
        self.config.rrc_num_taps() - 1
    }

    pub fn detect(
        &self,
        i_ch: &[f32],
        q_ch: &[f32],
        start_offset: usize,
    ) -> (Option<SyncResult>, usize) {
        let sym_len = self.sym_len;
        let repeat = self.config.preamble_repeat;
        let preamble_len = sym_len * repeat;
        let required_len = preamble_len + sym_len;
        let spc = self.spc;
        let coarse_step = spc.max(1);  // 修正: 2サンプル/チップ → 1サンプル/チップ

        // プリアンブル直後の1シンボル分まで見えてから同期確定する。
        // 部分一致による早期ロックを避けるためのガード。
        if i_ch.len() < start_offset + required_len {
            (None, start_offset)
        } else {
            let search_range_end = i_ch.len() - required_len;

            // --- 1. 粗同期 (チップ単位でのノンコヒーレント検索) ---
            let mut best_power_score = 0.0f32;
            let mut best_n = start_offset;
            for n in (start_offset..=search_range_end).step_by(coarse_step) {
                let (score, _) = self.score_candidate(i_ch, q_ch, n, repeat, sym_len);
                if score > best_power_score {
                    best_power_score = score;
                    best_n = n;
                }
            }

            // --- 2. 精密同期 (ピーク周辺を1サンプル刻みで) ---
            if best_power_score > 0.2 {
                let start = best_n.saturating_sub(coarse_step);
                let end = (best_n + coarse_step).min(search_range_end);

                let mut fine_best_score = 0.0f32;
                let mut fine_best_idx = best_n;
                let mut last_sym_iq = (0.0, 0.0);
                for n in start..=end {
                    let (score, last_iq) = self.score_candidate(i_ch, q_ch, n, repeat, sym_len);
                    if score > fine_best_score {
                        fine_best_score = score;
                        fine_best_idx = n;
                        last_sym_iq = last_iq;
                    }
                }

                if fine_best_score > 0.4 {
                    (
                        Some(SyncResult {
                            peak_sample_idx: fine_best_idx + preamble_len,
                            peak_iq: last_sym_iq, // 最後のシンボル(-M)の位相を返す
                            score: fine_best_score,
                        }),
                        fine_best_idx,
                    )
                } else {
                    // ストリーミング時の取りこぼしを防ぐため、次回探索用に
                    // 少なくとも preamble_len に加えて 2シンボル分を保持する。
                    let keep_tail = sym_len * 2 + spc;
                    (None, (search_range_end + 1).saturating_sub(keep_tail))
                }
            } else {
                let keep_tail = sym_len * 2 + spc;
                (None, (search_range_end + 1).saturating_sub(keep_tail))
            }
        }
    }

    /// 1シンボル分だけの相関を計算
    fn correlate_one_symbol(&self, i_ch: &[f32], q_ch: &[f32], offset: usize) -> (f32, f32, f32) {
        let mut sum_i = 0.0f32;
        let mut sum_q = 0.0f32;
        let mut sum_en = 0.0f32;

        let mut p = offset + (self.spc / 2);
        for &rv in &self.pn {
            debug_assert!(p < i_ch.len() && p < q_ch.len());
            let si = i_ch[p];
            let sq = q_ch[p];
            sum_i += si * rv;
            sum_q += sq * rv;
            sum_en += si * si + sq * sq;
            p += self.spc;
        }
        (sum_i, sum_q, sum_en)
    }

    /// プリアンブル構造 [M, M, ..., -M] を使って候補位置のスコアを計算する。
    /// 電力和に加え、シンボル間位相関係（最後だけ反転）を評価することで
    /// ノンコヒーレント電力相関の周期曖昧性を抑える。
    fn score_candidate(
        &self,
        i_ch: &[f32],
        q_ch: &[f32],
        n: usize,
        repeat: usize,
        sym_len: usize,
    ) -> (f32, (f32, f32)) {
        let mut total_power = 0.0f32;
        let mut min_power = f32::INFINITY;
        let mut pattern_score = 0.0f32;
        let mut prev_i = 0.0f32;
        let mut prev_q = 0.0f32;
        let mut prev_mag = 0.0f32;
        let mut has_prev = false;
        let mut last_iq = (0.0f32, 0.0f32);
        let inv_sf = 1.0f32 / self.sf as f32;

        for rep in 0..repeat {
            let (ci, cq, en) = self.correlate_one_symbol(i_ch, q_ch, n + rep * sym_len);
            let _ = en;
            let mag2 = ci * ci + cq * cq;
            let power = mag2 * inv_sf;
            total_power += power;
            min_power = min_power.min(power);
            let cur_mag = mag2.sqrt();
            if has_prev && prev_mag >= 1e-3 && cur_mag >= 1e-3 {
                let dot = prev_i * ci + prev_q * cq;
                let denom = prev_mag * cur_mag + 1e-9;
                let cos_rel = dot / denom;
                let expected = if rep == repeat - 1 { -1.0 } else { 1.0 };
                pattern_score += expected * cos_rel;
            }
            prev_i = ci;
            prev_q = cq;
            prev_mag = cur_mag;
            has_prev = true;
            last_iq = (ci, cq);
        }

        let avg_power = total_power / repeat as f32;

        let mut score = avg_power + 0.75 * pattern_score;

        // 区間の一部だけにエネルギーがある「部分一致」を抑制する。
        if min_power < avg_power * 0.2 {
            score *= 0.1;
        }
        (score, last_iq)
    }
}

pub fn downconvert(
    samples: &[f32],
    sample_offset: usize,
    config: &DspConfig,
) -> (Vec<f32>, Vec<f32>) {
    let two_pi = 2.0 * std::f32::consts::PI;
    let fs = config.sample_rate;
    let fc = config.carrier_freq;
    let mut i_ch = Vec::with_capacity(samples.len());
    let mut q_ch = Vec::with_capacity(samples.len());
    for (k, &s) in samples.iter().enumerate() {
        let t = (sample_offset + k) as f32 / fs;
        let (sin_v, cos_v) = (two_pi * fc * t).sin_cos();
        i_ch.push(s * cos_v * 2.0);
        q_ch.push(s * (-sin_v) * 2.0);
    }
    (i_ch, q_ch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::rrc_filter::RrcFilter;
    use crate::phy::modulator::Modulator;
    use rand::prelude::*;
    use rand_distr::Normal;

    fn add_noise(i: &mut [f32], q: &mut [f32], sigma: f32) {
        let mut rng = thread_rng();
        let dist = Normal::new(0.0, sigma).unwrap();
        for x in i.iter_mut() {
            *x += dist.sample(&mut rng);
        }
        for x in q.iter_mut() {
            *x += dist.sample(&mut rng);
        }
    }

    #[test]
    fn test_sync_robustness() {
        let config = DspConfig::default_48k();
        let mut modulator = Modulator::new(config.clone());
        let detector = SyncDetector::new(config.clone());

        for offset in [0, 123, 1000] {
            let mut signal = vec![0.0; offset];
            signal.extend(modulator.generate_preamble());
            signal.extend(vec![0.0; 500]);
            modulator.reset();

            let (i_raw, q_raw) = downconvert(&signal, 0, &config);
            let mut rrc_i = RrcFilter::from_config(&config);
            let mut rrc_q = RrcFilter::from_config(&config);
            let mut i_ch: Vec<f32> = i_raw.iter().map(|&s| rrc_i.process(s)).collect();
            let mut q_ch: Vec<f32> = q_raw.iter().map(|&s| rrc_q.process(s)).collect();

            add_noise(&mut i_ch, &mut q_ch, 0.1);

            let (res, _) = detector.detect(&i_ch, &q_ch, 0);
            let result = res.unwrap_or_else(|| panic!("Failed at offset {}", offset));

            let expected = offset
                + detector.filter_delay()
                + config.samples_per_symbol() * config.preamble_repeat;
            let tol = (config.samples_per_chip() as i32 / 2).max(2);
            assert!((result.peak_sample_idx as i32 - expected as i32).abs() <= tol);
        }
    }

    #[test]
    fn test_sync_streaming() {
        let config = DspConfig::default_48k();
        let mut modulator = Modulator::new(config.clone());
        let detector = SyncDetector::new(config.clone());

        let offset = 456;
        let mut signal = vec![0.0; offset];
        signal.extend(modulator.generate_preamble());
        signal.extend(vec![0.0; 1000]);

        let (i_raw, q_raw) = downconvert(&signal, 0, &config);
        let mut rrc_i = RrcFilter::from_config(&config);
        let mut rrc_q = RrcFilter::from_config(&config);
        let i_all: Vec<f32> = i_raw.iter().map(|&s| rrc_i.process(s)).collect();
        let q_all: Vec<f32> = q_raw.iter().map(|&s| rrc_q.process(s)).collect();

        let mut buf_i = Vec::new();
        let mut buf_q = Vec::new();
        let mut total_consumed = 0;
        let mut detected = false;

        for chunk in i_all.chunks(512).zip(q_all.chunks(512)) {
            buf_i.extend_from_slice(chunk.0);
            buf_q.extend_from_slice(chunk.1);

            let (res, next_off) = detector.detect(&buf_i, &buf_q, 0);
            if let Some(result) = res {
                let absolute = total_consumed + result.peak_sample_idx;
                let expected = offset
                    + detector.filter_delay()
                    + config.samples_per_symbol() * config.preamble_repeat;
                let tol = (config.samples_per_chip() as i32 / 2).max(2);
                assert!((absolute as i32 - expected as i32).abs() <= tol);
                detected = true;
                break;
            }
            buf_i.drain(0..next_off);
            buf_q.drain(0..next_off);
            total_consumed += next_off;
        }
        assert!(detected);
    }
}
