//! 実用的な同期捕捉 (ノンコヒーレント相関)
//!
//! 1シンボルごとに相関電力を計算し、それらを足し合わせることで、
//! 位相回転やクロックズレに強い同期捕捉を実現する。
//!
//! # 統計的性質としきい値の導出
//!
//! ## 電力相似度 rho_p の分布
//!
/// 各シンボルの相関出力: mag2 = |C|^2
/// 信号あり: E[mag2] = A^2 * sf^2 + σ^2 * sf
/// 信号ありの入力エネルギー: E[en] = A^2 * sf + σ^2 * sf
/// 信号ありの電力比: E[mag2 / (sf * en)] ≈ SNR / (SNR + 1)
///   - ∞ dB (高 SNR): → 1.0
///   - 0 dB: → 0.5
///   - -10 dB: → 0.09
///
/// ノイズのみ: E[mag2] = σ^2 * sf, E[en] = σ^2 * sf
/// ノイズのみの電力比: E[mag2 / (sf * en)] = 1/sf = 1/63 ≈ 0.016
///
/// 4シンボル平均の変動 (chi-sq 分布より):
///   ノイズ: 標準偏差 ≈ 0.016 / sqrt(4) = 0.008
///   ノイズ 3σ値: 0.016 + 3 * 0.008 ≈ 0.04
///
/// ## 位相一貫性 rho_phi の分布
///
/// 信号あり: 各シンボル間の位相関係はプリアンブル構造に従う
///   rho_phi ≈ 0.85-1.0 (CFO があっても安定)
/// 低 SNR 信号: rho_phi ≈ 0.7
/// ノイズのみ: 位相は一様分布、E[cos] = 0、rho_phi ≈ 0.5
///
/// 3ペア平均の変動 (中心極限定理より):
///   ノイズ: 標準偏差 ≈ 0.5 / sqrt(3) ≈ 0.29
///   ノイズ 3σ値: 0.5 + 3 * 0.29 ≈ 1.37 (クリップされるので実際は低い)
///
/// ## 総合スコアの分布
///
/// score = 0.5 * rho_p + 0.5 * rho_phi
///
/// 高 SNR 信号: 0.5 * 0.95 + 0.5 * 0.95 = 0.95
/// 0 dB SNR 信号: 0.5 * 0.5 + 0.5 * 0.7 = 0.6
/// -10 dB SNR 信号: 0.5 * 0.09 + 0.5 * 0.6 = 0.345
/// ノイズのみ: 0.5 * 0.016 + 0.5 * 0.5 = 0.258
///
/// 変動の考慮:
///   0 dB SNR 信号 2σ下限: ≈ 0.6 - 0.1 = 0.5
///   ノイズ 3σ上限: ≈ 0.258 + 0.05 = 0.31
///
/// ## しきい値の決定
///
/// 粗探索しきい値 (THRESHOLD_COARSE):
///   ノイズ 3σ上限 (0.31) 未満に設定し、偽陽性を抑制
///   設定値: 0.31
///
/// 精密探索しきい値 (THRESHOLD_FINE):
///   0 dB SNR 信号 2σ下限 (0.5) とノイズ 3σ上限 (0.31) の間
///   誤検出率と感度のバランスを考慮
///   設定値: 0.38

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
    /// 粗探索しきい値
    const THRESHOLD_COARSE: f32 = 0.27;
    /// 精密探索しきい値: バランスを考慮
    const THRESHOLD_FINE: f32 = 0.32;
    /// 位相スコアの重み: 電力スコアを重視
    const WEIGHT_PHASE: f32 = 0.3;

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
        let coarse_step = (spc * 2).max(1);

        // プリアンブル直後の1シンボル分まで見えてから同期確定する。
        if i_ch.len() < start_offset + required_len {
            (None, start_offset)
        } else {
            let search_range_end = i_ch.len() - required_len;
            let mut provisional_best: Option<(SyncResult, usize)> = None;

            // --- 1. 粗同期 & 精密同期 ---
            for n in (start_offset..=search_range_end).step_by(coarse_step) {
                let (score, _) = self.score_candidate(i_ch, q_ch, n, repeat, sym_len);

                if score > Self::THRESHOLD_COARSE || provisional_best.is_some() {
                    // --- 2. 精密同期 (ピーク周辺を1サンプル刻みで) ---
                    let f_start = n.saturating_sub(coarse_step);
                    let f_end = (n + coarse_step).min(search_range_end);

                    let mut fine_best_score = 0.0f32;
                    let mut fine_best_idx = n;
                    let mut last_sym_iq = (0.0, 0.0);
                    for fn_idx in f_start..=f_end {
                        let (f_score, f_iq) =
                            self.score_candidate(i_ch, q_ch, fn_idx, repeat, sym_len);
                        if f_score >= fine_best_score {
                            fine_best_score = f_score;
                            fine_best_idx = fn_idx;
                            last_sym_iq = f_iq;
                        }
                    }

                    // 信頼しきい値 (THRESHOLD_FINE) を超えた場合に追従を開始
                    if fine_best_score > Self::THRESHOLD_FINE {
                        if let Some((ref best, _)) = provisional_best {
                            if fine_best_score >= best.score {
                                // 登り坂: 暫定ベストを更新
                                provisional_best = Some((
                                    SyncResult {
                                        peak_sample_idx: fine_best_idx + preamble_len,
                                        peak_iq: last_sym_iq,
                                        score: fine_best_score,
                                    },
                                    fine_best_idx,
                                ));
                            } else {
                                // 下り坂: 最初の山の頂上を確定して早期リターン (First Peak Match)
                                let (res, idx) = provisional_best.unwrap();
                                return (Some(res), idx);
                            }
                        } else {
                            // 最初の THRESHOLD_FINE 超え
                            provisional_best = Some((
                                SyncResult {
                                    peak_sample_idx: fine_best_idx + preamble_len,
                                    peak_iq: last_sym_iq,
                                    score: fine_best_score,
                                },
                                fine_best_idx,
                            ));
                        }
                    } else if let Some((res, idx)) = provisional_best {
                        // 山を降りきったので確定
                        return (Some(res), idx);
                    }
                }
            }

            // ループ終了時に暫定ベストがあればそれを返す
            if let Some((res, idx)) = provisional_best {
                (Some(res), idx)
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

    fn score_candidate(
        &self,
        i_ch: &[f32],
        q_ch: &[f32],
        n: usize,
        repeat: usize,
        sym_len: usize,
    ) -> (f32, (f32, f32)) {
        let mut total_rho_p = 0.0f32;
        let mut min_power = f32::INFINITY;
        let mut max_power = 0.0f32;
        let mut phase_score = 0.0f32;
        let mut num_valid_pairs = 0;
        let mut corrs = Vec::with_capacity(repeat);

        for rep in 0..repeat {
            let (ci, cq, en) = self.correlate_one_symbol(i_ch, q_ch, n + rep * sym_len);
            let mag2 = ci * ci + cq * cq;

            // 電力相似度: 相関出力電力 / (期待される相関電力)
            // 信号の場合: mag2 ≈ sf * en、だから rho_p ≈ SNR/(SNR+1) ≈ 0.5-1.0
            // ノイズの場合: mag2 ≈ en、だから rho_p ≈ 1/sf ≈ 0.016
            let rho_p_sym = if en > 1e-9 {
                mag2 / (self.sf as f32 * en)
            } else {
                0.0
            };
            total_rho_p += rho_p_sym;

            min_power = min_power.min(mag2);
            max_power = max_power.max(mag2);

            corrs.push((ci, cq, mag2.sqrt()));
        }

        let rho_p = total_rho_p / repeat as f32;

        // 位相一貫性スコア: 各シンボル間の位相関係を評価
        // CFO があっても、シンボル間の位相差は一定（プリアンブル構造による）
        for rep in 1..repeat {
            let (p_i, p_q, p_mag) = corrs[rep - 1];
            let (c_i, c_q, c_mag) = corrs[rep];

            // 十分なエネルギーがあるペアのみを評価
            if p_mag >= 1e-3 && c_mag >= 1e-3 {
                // コサイン類似度で位相関係を評価
                let dot = p_i * c_i + p_q * c_q;
                let cos_rel = dot / (p_mag * c_mag);

                // 最後のシンボルは位相が反転するはず（プリアンブル構造 [M, M, ..., -M]）
                let expected = if rep == repeat - 1 { -1.0 } else { 1.0 };
                phase_score += expected * cos_rel;
                num_valid_pairs += 1;
            }
        }

        let rho_phi = if num_valid_pairs > 0 {
            let raw_phi = phase_score / num_valid_pairs as f32;
            // [-1, 1] -> [0, 1]
            // 信号の場合: 1.0、ノイズの場合: 0.5
            let normalized_phi = (raw_phi + 1.0) / 2.0;
            // ノイズの期待値 (0.5) を引いて、信号の場合に高い値になるように
            // 信号の場合: 0.5、ノイズの場合: 0
            let centered_phi = normalized_phi - 0.5;
            // [0, 0.5] -> [0, 1]
            centered_phi * 2.0
        } else {
            0.0
        };

        // 総合スコア: 電力相似度と位相一貫性の加重平均
        // 高 SNR 信号: ≈ 0.5*0.95 + 0.3*1.0 = 0.775
        // 0 dB SNR: ≈ 0.5*0.5 + 0.3*0.7 = 0.46
        // ノイズ: ≈ 0.5*0.016 + 0.3*0 = 0.15
        let mut score = (1.0 - Self::WEIGHT_PHASE) * rho_p + Self::WEIGHT_PHASE * rho_phi;

        // 電力の一様性チェック: 信号は各シンボルの電力が安定しているはず
        // 部分一致（1シンボルだけ高いなど）を抑制
        let avg_power = (min_power + max_power) / 2.0;
        if min_power < avg_power * 0.25 {
            score *= 0.1;
        }

        (score.max(0.0), (corrs[repeat - 1].0, corrs[repeat - 1].1))
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

    fn generate_signal(config: &DspConfig, offset: usize, amplitude: f32) -> (Vec<f32>, Vec<f32>) {
        let mut modulator = Modulator::new(config.clone());
        let mut signal = vec![0.0; offset];
        signal.extend(modulator.generate_preamble().iter().map(|&s| s * amplitude));
        // マージンを追加: SyncDetector はピークの後に 1シンボル分以上のサンプルを要求するため
        signal.extend(vec![0.0; 500]);

        let (i_raw, q_raw) = downconvert(&signal, 0, config);
        let mut rrc_i = RrcFilter::from_config(config);
        let mut rrc_q = RrcFilter::from_config(config);
        let i_ch: Vec<f32> = i_raw.iter().map(|&s| rrc_i.process(s)).collect();
        let q_ch: Vec<f32> = q_raw.iter().map(|&s| rrc_q.process(s)).collect();
        (i_ch, q_ch)
    }

    #[test]
    fn test_sync_first_match_scenarios() {
        let config = DspConfig::default_48k();
        let detector = SyncDetector::new(config.clone());
        let sym_len = config.samples_per_symbol();
        let repeat = config.preamble_repeat;

        // 地上実測値 (Ground Truth) を求める補助関数
        let find_ground_truth = |i: &[f32], q: &[f32]| {
            let mut best_score = -1.0f32;
            let mut best_idx = 0;
            let required_len = sym_len * (repeat + 1);
            if i.len() < required_len { return (0.0, 0); }
            for n in 0..=(i.len() - required_len) {
                let (score, _) = detector.score_candidate(i, q, n, repeat, sym_len);
                if score > best_score {
                    best_score = score;
                    best_idx = n;
                }
            }
            (best_score, best_idx)
        };

        // シナリオ1: 0.4 以上のピークがすぐに見つかる場合
        {
            let (i, q) = generate_signal(&config, 1000, 1.0);
            let (gt_score, gt_idx) = find_ground_truth(&i, &q);
            assert!(gt_score > SyncDetector::THRESHOLD_FINE);

            let (res, _) = detector.detect(&i, &q, 0);
            let sync = res.expect("Should find strong peak");
            let detected_idx = sync.peak_sample_idx - (sym_len * repeat);
            assert!((detected_idx as i32 - gt_idx as i32).abs() <= 2);
        }

        // シナリオ2: ランダムノイズのみの場合 (無視されるべき)
        {
            let mut rng = thread_rng();
            let dist = Normal::new(0.0, 0.1).unwrap();
            let i: Vec<f32> = (0..2000).map(|_| dist.sample(&mut rng)).collect();
            let q: Vec<f32> = (0..2000).map(|_| dist.sample(&mut rng)).collect();

            // 実装 (First Match) がノイズに対して同期を検出しないことを直接検証
            let (res, _) = detector.detect(&i, &q, 0);
            assert!(res.is_none(), "Should NOT detect sync on noise, but got {:?}", res);
        }

        // シナリオ3: 0.4 程度の弱ピークの後に、1.0 近い強ピークがある場合
        {
            // 弱ピークを 0.45 程度で作る
            let (i_weak, q_weak) = generate_signal(&config, 1000, 0.45);
            let (i_strong, q_strong) = generate_signal(&config, 2000, 1.0);

            let mut i_combined = i_weak;
            i_combined.resize(5000 + i_strong.len(), 0.0);
            let mut q_combined = q_weak;
            q_combined.resize(5000 + q_strong.len(), 0.0);

            for (idx, &s) in i_strong.iter().enumerate() {
                i_combined[5000 + idx] = s;
                q_combined[5000 + idx] = q_strong[idx];
            }

            // 前方の弱ピークの GT を特定
            let i_only_weak = &i_combined[0..4000];
            let q_only_weak = &q_combined[0..4000];
            let (_, weak_gt_idx) = find_ground_truth(i_only_weak, q_only_weak);

            let (res, _) = detector.detect(&i_combined, &q_combined, 0);
            let sync = res.expect("Should find the first valid peak");
            let detected_idx = sync.peak_sample_idx - (sym_len * repeat);

            // First Peak Match は後方の強ピークを待たず、前方の弱ピークを返すのが正しい。
            assert!((detected_idx as i32 - weak_gt_idx as i32).abs() <= 2);
        }
    }

    #[test]
    fn test_sync_boundary_conditions() {
        let config = DspConfig::default_48k();
        let detector = SyncDetector::new(config.clone());
        let sym_len = config.samples_per_symbol();
        let preamble_len = sym_len * config.preamble_repeat;

        // 境界1: バッファ長が足りない
        {
            let i = vec![0.0; preamble_len]; // 1シンボル分足りない
            let q = vec![0.0; preamble_len];
            let (res, next) = detector.detect(&i, &q, 0);
            assert!(res.is_none());
            assert_eq!(next, 0);
        }

        // 境界2: 探索範囲のちょうど末尾にピークがある
        {
            let offset = 500;
            let (i, q) = generate_signal(&config, offset, 1.0);
            let (res, _) = detector.detect(&i, &q, 0);
            assert!(res.is_some(), "Should find peak at the end of buffer");
        }

        // 境界3: 検索開始位置 (start_offset) がバッファ末尾に近い
        {
            let i = vec![0.0; 1000];
            let q = vec![0.0; 1000];
            let (res, next) = detector.detect(&i, &q, 950);
            assert!(res.is_none());
            assert_eq!(next, 950);
        }
    }
}
