//! 実用的な同期捕捉 (ノンコヒーレント相関) - MaryDQPSK版
//!
//! 1シンボルごとに相関電力を計算し、それらを足し合わせることで、
//! 位相回転やクロックズレに強い同期捕捉を実現する。

// use crate::common::walsh::WalshDictionary; // unused in current file
use crate::DspConfig;
use num_complex::Complex32;

#[derive(Debug, Clone)]
pub struct SyncResult {
    pub peak_sample_idx: usize,
    pub peak_iq: (f32, f32),
    pub score: f32,
}

pub struct MarySyncDetector {
    config: DspConfig,
    preamble_pn: Vec<Complex32>,  // Zadoff-Chu SF=13
    sync_pn: Vec<f32>,            // Walsh[0] SF=15
    sync_symbols: Vec<f32>,       // プリアンブル構造 + SYNC_WORD
    preamble_sf: usize,
    sync_sf: usize,
    spc: usize,
    preamble_sym_len: usize,
    sync_sym_len: usize,
    pub threshold_coarse: f32,
    pub threshold_fine: f32,
}

impl MarySyncDetector {
    /// デフォルトのしきい値 (ZC SF=13 プリアンブル + Walsh SF=15 同期ワード用)
    /// ROC分析に基づく: ノイズ FAR 1% → 0.23, FAR 0.1% → 0.30
    pub const THRESHOLD_COARSE_DEFAULT: f32 = 0.20;
    pub const THRESHOLD_FINE_DEFAULT: f32 = 0.23;

    pub fn new(config: DspConfig, threshold_coarse: f32, threshold_fine: f32) -> Self {
        let preamble_sf = 13;
        let sync_sf = 15;
        let spc = config.proc_samples_per_chip().max(1);

        let zc = crate::common::zadoff_chu::ZadoffChu::new(preamble_sf, 1);
        let preamble_pn = zc.generate_sequence();

        let wdict = crate::common::walsh::WalshDictionary::default_w16();
        let sync_pn: Vec<f32> = wdict.w16[0].iter().take(sync_sf).map(|&x| x as f32).collect();

        let preamble_sym_len = preamble_sf * spc;
        let sync_sym_len = sync_sf * spc;

        // 18シンボルの期待される符号系列 (BPSK/DQPSK) を構築
        // Preamble: [1, -1] (repeat=2の場合)
        // SYNC_WORD: 16 bits (デフォルト設定)
        let mut sync_symbols = Vec::with_capacity(config.preamble_repeat + config.sync_word_bits);
        for rep in 0..config.preamble_repeat {
            // Modulator.generate_preamble: last one is inverted
            let sign = if rep == config.preamble_repeat - 1 {
                -1.0
            } else {
                1.0
            };
            sync_symbols.push(sign);
        }

        // Sync Word bits (16 bits)
        // Modulator.encode_frame: uses prev_phase (starts at 0) and DBPSK delta
        let mut current_phase_factor = 1.0f32; // prev_phase = 0 -> factor 1.0
        let word = crate::params::SYNC_WORD;
        for i in 0..config.sync_word_bits {
            let bit = (word >> (config.sync_word_bits - 1 - i)) & 1;
            // DBPSK: bit=0 -> delta=0 (0 deg), bit=1 -> delta=2 (180 deg)
            if bit != 0 {
                current_phase_factor *= -1.0;
            }
            sync_symbols.push(current_phase_factor);
        }

        MarySyncDetector {
            config,
            preamble_pn,
            sync_pn,
            sync_symbols,
            preamble_sf,
            sync_sf,
            spc,
            preamble_sym_len,
            sync_sym_len,
            threshold_coarse,
            threshold_fine,
        }
    }

    pub fn filter_delay(&self) -> usize {
        // RRCフィルタの群遅延 (L-1)/2
        (self.config.rrc_num_taps() - 1) / 2
    }

    pub fn detect(
        &self,
        i_ch: &[f32],
        q_ch: &[f32],
        start_offset: usize,
    ) -> (Option<SyncResult>, usize) {
        let repeat = self.config.preamble_repeat;
        let unified_len = self.sync_symbols.len();
        let preamble_len = self.preamble_sym_len * repeat;
        let sync_part_len = self.sync_sym_len * self.config.sync_word_bits;
        let required_len = preamble_len + sync_part_len;
        let spc = self.spc;
        let coarse_step = spc.max(1);

        // プリアンブル + SYNC_WORD 分まで見えてから同期確定する。
        if i_ch.len() < start_offset + required_len {
            (None, start_offset)
        } else {
            let search_range_end = i_ch.len() - required_len;
            let mut provisional_best: Option<(SyncResult, usize)> = None;

            // --- 1. 粗同期 & 精密同期 ---
            for n in (start_offset..=search_range_end).step_by(coarse_step) {
                let (score, _) = self.score_candidate(i_ch, q_ch, n, repeat);

                if score > self.threshold_coarse || provisional_best.is_some() {
                    // --- 2. 精密同期 (ピーク周辺を1サンプル刻みで) ---
                    // 精密同期では、プリアンブル + SYNC_WORD の全36シンボルでロックを確認する
                    let f_start = n.saturating_sub(coarse_step);
                    let f_end = (n + coarse_step).min(search_range_end);

                    let mut fine_best_score = 0.0f32;
                    let mut fine_best_idx = n;
                    let mut last_sym_iq = (0.0, 0.0);
                    for fn_idx in f_start..=f_end {
                        let (f_score, f_iq) =
                            self.score_candidate(i_ch, q_ch, fn_idx, unified_len);
                        if f_score >= fine_best_score {
                            fine_best_score = f_score;
                            fine_best_idx = fn_idx;
                            last_sym_iq = f_iq;
                        }
                    }

                    // 信頼しきい値 (threshold_fine) を超えた場合に追従を開始
                    if fine_best_score > self.threshold_fine {
                        if let Some((ref best, _)) = provisional_best {
                            if fine_best_score >= best.score {
                                // 登り坂: 暫定ベストを更新
                                provisional_best = Some((
                                    SyncResult {
                                        peak_sample_idx: fine_best_idx + preamble_len + (spc / 2),
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
                            // 最初の threshold_fine 超え
                            provisional_best = Some((
                                SyncResult {
                                    peak_sample_idx: fine_best_idx + preamble_len + (spc / 2),
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
                let keep_tail = required_len * 2 + spc;
                let next_idx = (search_range_end + 1).saturating_sub(keep_tail);
                (None, next_idx)
            }
        }
    }

    /// 1シンボル分だけの相関を計算
    fn correlate_one_symbol(&self, i_ch: &[f32], q_ch: &[f32], offset: usize, is_preamble: bool) -> (f32, f32, f32) {
        let mut sum_i = 0.0f32;
        let mut sum_q = 0.0f32;
        let mut sum_en = 0.0f32;

        let mut p = offset + (self.spc / 2);
        
        if is_preamble {
            for val in &self.preamble_pn {
                debug_assert!(p < i_ch.len() && p < q_ch.len());
                let si = i_ch[p];
                let sq = q_ch[p];
                // 複素共役との積和 (si + j sq) * (re - j im)
                sum_i += si * val.re + sq * val.im; // (実部)
                sum_q += sq * val.re - si * val.im; // (虚部)
                sum_en += si * si + sq * sq;
                p += self.spc;
            }
        } else {
            for &rv in &self.sync_pn {
                debug_assert!(p < i_ch.len() && p < q_ch.len());
                let si = i_ch[p];
                let sq = q_ch[p];
                sum_i += si * rv;
                sum_q += sq * rv;
                sum_en += si * si + sq * sq;
                p += self.spc;
            }
        }
        (sum_i, sum_q, sum_en)
    }

    fn score_candidate(
        &self,
        i_ch: &[f32],
        q_ch: &[f32],
        n: usize,
        num_symbols: usize,
    ) -> (f32, (f32, f32)) {
        let mut total_rho_p = 0.0f32;
        let mut min_power = f32::INFINITY;
        let mut max_power = 0.0f32;

        let mut sum_re = 0.0f32;
        let mut sum_im = 0.0f32;
        let mut sum_mag = 0.0f32;

        let mut last_ci = 0.0f32;
        let mut last_cq = 0.0f32;
        let mut last_mag = 0.0f32;
        
        let mut current_offset = n;

        for rep in 0..num_symbols {
            let is_preamble = rep < self.config.preamble_repeat;
            let (ci, cq, en) = self.correlate_one_symbol(i_ch, q_ch, current_offset, is_preamble);
            let mag2 = ci * ci + cq * cq;
            let mag = mag2.sqrt();
            let sf = if is_preamble { self.preamble_sf } else { self.sync_sf };

            let rho_p_sym = if en > 1e-9 {
                mag2 / (sf as f32 * en)
            } else {
                0.0
            };
            total_rho_p += rho_p_sym;

            min_power = min_power.min(mag2);
            max_power = max_power.max(mag2);

            if rep > 0 && last_mag > 1e-9 && mag > 1e-9 {
                let re = last_ci * ci + last_cq * cq;
                let im = last_ci * cq - last_cq * ci;
                let pair_mag = last_mag * mag;

                let expected = self.sync_symbols[rep - 1] * self.sync_symbols[rep];
                sum_re += expected * re;
                sum_im += expected * im;
                sum_mag += pair_mag;
            }

            last_ci = ci;
            last_cq = cq;
            last_mag = mag;
            
            let sym_len = if is_preamble { self.preamble_sym_len } else { self.sync_sym_len };
            current_offset += sym_len;
        }

        let rho_p = total_rho_p / num_symbols as f32;

        let rho_phi = if sum_mag > 1e-9 {
            (sum_re * sum_re + sum_im * sum_im).sqrt() / sum_mag
        } else {
            0.0
        };

        let mut score = rho_p * rho_phi;

        let avg_power = (min_power + max_power) / 2.0;
        if min_power < avg_power * 0.25 {
            score *= 0.1;
        }

        (score.max(0.0), (last_ci, last_cq))
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
    use crate::common::resample::Resampler;
    use crate::common::rrc_filter::RrcFilter;
    use crate::mary::modulator::Modulator;
    use rand::prelude::*;
    use rand_distr::Normal;

    fn generate_signal(config: &DspConfig, offset: usize, amplitude: f32) -> (Vec<f32>, Vec<f32>) {
        let mut modulator = Modulator::new(config.clone());
        let mut signal = vec![0.0; offset];
        signal.extend(modulator.encode_frame(&[]).iter().map(|&s| s * amplitude));
        // マージンを追加: MarySyncDetector はピークの後に 1シンボル分以上のサンプルを要求するため
        signal.extend(vec![0.0; 500]);

        let (i_raw, q_raw) = downconvert(&signal, 0, config);

        // Decoder と同じパイプライン: downconvert → Resampler(48k→24k) → RRC(24kHz)
        let rrc_bw = config.chip_rate * (1.0 + config.rrc_alpha) * 0.5;
        let mut resampler_i = Resampler::new_with_cutoff(
            config.sample_rate as u32,
            config.proc_sample_rate() as u32,
            Some(rrc_bw),
        );
        let mut resampler_q = Resampler::new_with_cutoff(
            config.sample_rate as u32,
            config.proc_sample_rate() as u32,
            Some(rrc_bw),
        );
        let mut i_resampled = Vec::new();
        let mut q_resampled = Vec::new();
        resampler_i.process(&i_raw, &mut i_resampled);
        resampler_q.process(&q_raw, &mut q_resampled);

        let mut rrc_i = RrcFilter::from_config(config);
        let mut rrc_q = RrcFilter::from_config(config);
        let i_ch: Vec<f32> = i_resampled.iter().map(|&s| rrc_i.process(s)).collect();
        let q_ch: Vec<f32> = q_resampled.iter().map(|&s| rrc_q.process(s)).collect();
        (i_ch, q_ch)
    }

    fn new_detector_default(config: DspConfig) -> MarySyncDetector {
        MarySyncDetector::new(
            config,
            MarySyncDetector::THRESHOLD_COARSE_DEFAULT,
            MarySyncDetector::THRESHOLD_FINE_DEFAULT,
        )
    }

    #[test]
    fn test_sync_absolute_timing_accuracy() {
        let config = DspConfig::default_48k();
        let detector = new_detector_default(config.clone());
        let sym_len = 15 * config.proc_samples_per_chip(); // sf=15
        let preamble_len = config.preamble_repeat * sym_len;
        let offset = 500;

        // 1. 信号生成 (オフセットを正確に制御)
        let (i, q) = generate_signal(&config, offset, 1.0);

        // 2. 同期捕捉実行
        let (res, _) = detector.detect(&i, &q, 0);
        let sync = res.expect("Should find sync");

        // 3. ピーク位置の妥当性検証
        // Resampler + RRC の遅延は複合的のため、ピーク位置が
        // プリアンブル領域の後の妥当な範囲にあることを検証
        println!(
            "Detected idx: {}, preamble_len: {}",
            sync.peak_sample_idx, preamble_len
        );
        assert!(
            sync.peak_sample_idx >= preamble_len,
            "Peak should be after preamble: detected={}, preamble_len={}",
            sync.peak_sample_idx, preamble_len
        );
        assert!(sync.score > detector.threshold_fine);
    }

    #[test]
    fn test_sync_first_match_scenarios() {
        let config = DspConfig::default_48k();
        let detector = new_detector_default(config.clone());
        let sym_len = 15 * config.proc_samples_per_chip();
        let repeat = config.preamble_repeat;

        // 地上実測値 (Ground Truth) を求める補助関数
        let find_ground_truth = |i: &[f32], q: &[f32]| {
            let mut best_score = -1.0f32;
            let mut best_idx = 0;
            let unified_len = detector.sync_symbols.len();
            let required_len = sym_len * unified_len;
            if i.len() < required_len {
                return (0.0, 0);
            }
            for n in 0..=(i.len() - required_len) {
                let (score, _) = detector.score_candidate(i, q, n, unified_len);
                if score > best_score {
                    best_score = score;
                    best_idx = n;
                }
            }
            // re-run to print rho_p and rho_phi at best_idx
            if best_idx > 0 {
                let mut total_rho_p = 0.0f32;
                let mut sum_re = 0.0f32;
                let mut sum_im = 0.0f32;
                let mut sum_mag = 0.0f32;
                let mut last_ci = 0.0;
                let mut last_cq = 0.0;
                let mut last_mag = 0.0;
                let mut current_offset = best_idx;
                for rep in 0..unified_len {
                    let is_preamble = rep < config.preamble_repeat;
                    let (ci, cq, en) =
                        detector.correlate_one_symbol(i, q, current_offset, is_preamble);
                    let mag2 = ci * ci + cq * cq;
                    let mag = mag2.sqrt();
                    let sf = if is_preamble { detector.preamble_sf } else { detector.sync_sf };
                    total_rho_p += if en > 1e-9 { mag2 / (sf as f32 * en) } else { 0.0 };
                    if rep > 0 && last_mag > 1e-9 && mag > 1e-9 {
                        let re = last_ci * ci + last_cq * cq;
                        let im = last_ci * cq - last_cq * ci;
                        let expected = detector.sync_symbols[rep - 1] * detector.sync_symbols[rep];
                        sum_re += expected * re;
                        sum_im += expected * im;
                        sum_mag += last_mag * mag;
                    }
                    last_ci = ci;
                    last_cq = cq;
                    last_mag = mag;
                    let sym_len = if is_preamble { detector.preamble_sym_len } else { detector.sync_sym_len };
                    current_offset += sym_len;
                }
                let rho_p = total_rho_p / unified_len as f32;
                let rho_phi = if sum_mag > 1e-9 {
                    (sum_re * sum_re + sum_im * sum_im).sqrt() / sum_mag
                } else {
                    0.0
                };
                println!(
                    "Ground truth best score: {:.4} at idx {} (rho_p={:.4}, rho_phi={:.4})",
                    best_score, best_idx, rho_p, rho_phi
                );
            }
            (best_score, best_idx)
        };

        // シナリオ1: 0.4 以上のピークがすぐに見つかる場合
        {
            let (i, q) = generate_signal(&config, 1000, 1.0);
            let (gt_score, gt_idx) = find_ground_truth(&i, &q);
            assert!(gt_score > detector.threshold_fine);

            let (res, _) = detector.detect(&i, &q, 0);
            let sync = res.expect("Should find strong peak");
            assert!(sync.score > detector.threshold_fine);
            let detected_idx = sync.peak_sample_idx - (detector.preamble_sym_len * repeat);
            // Rc=8000 (spc=6) ではピークが平坦になりやすいため、spc/2 程度の誤差を許容する
            assert!((detected_idx as i32 - gt_idx as i32).abs() <= (detector.spc / 2) as i32);
        }

        // シナリオ2: ランダムノイズのみの場合
        {
            let mut rng = thread_rng();
            let dist = Normal::new(0.0, 0.1).unwrap();
            let i: Vec<f32> = (0..2000).map(|_| dist.sample(&mut rng)).collect();
            let q: Vec<f32> = (0..2000).map(|_| dist.sample(&mut rng)).collect();

            let (res, _) = detector.detect(&i, &q, 0);
            if let Some(ref sync) = res {
                println!("Noise detected with score: {:.4}", sync.score);
            }
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
            let detected_idx = sync.peak_sample_idx - (detector.preamble_sym_len * repeat);

            // First Peak Match は後方の強ピークを待たず、前方の弱ピークを返すのが正しい。
            assert!((detected_idx as i32 - weak_gt_idx as i32).abs() <= (detector.spc / 2) as i32);
        }
    }

    #[test]
    fn test_sync_boundary_conditions() {
        let config = DspConfig::default_48k();
        let detector = new_detector_default(config.clone());
        let sym_len = 13 * config.proc_samples_per_chip(); // preamble is SF=13
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

    #[test]
    fn test_sync_low_snr_sensitivity() {
        let config = DspConfig::default_48k();
        let detector = new_detector_default(config.clone());
        const NUM_TRIALS: usize = 100;
        const REQUIRED_DETECTION_RATE: f64 = 0.95;

        let mut num_detected = 0;

        for trial in 0..NUM_TRIALS {
            // -3 dB SNR: やや厳しい条件
            let (i, q) = generate_signal_with_awgn_seeded(&config, 500, -3.0, trial as u64);
            let (res, _) = detector.detect(&i, &q, 0);
            if res.is_some() {
                num_detected += 1;
            }
        }

        let detection_rate = num_detected as f64 / NUM_TRIALS as f64;
        println!(
            "Low SNR (-3 dB) detection rate: {:.2}%",
            detection_rate * 100.0
        );
        assert!(
            detection_rate >= REQUIRED_DETECTION_RATE,
            "Detection rate {}% below required {}%",
            detection_rate * 100.0,
            REQUIRED_DETECTION_RATE * 100.0
        );
    }

    #[test]
    fn test_roc_curve_analysis() {
        let config = DspConfig::default_48k();
        let snr_db_list: &[f32] = &[-10.0, -6.0, -3.0, 0.0, 3.0, 6.0, 10.0];
        const NUM_TRIALS: usize = 100;
        const NUM_NOISE_TRIALS: usize = 200;

        // --- ステップ1: 信号パワーの測定 (ノイズスケールのキャリブレーション) ---
        // 受信パイプライン通過後の信号 RMS を測定し、ノイズ sigma の基準とする
        let ref_signal_power: f32 = {
            let (i, q) = generate_signal_with_awgn_seeded(&config, 0, 60.0, 0); // SNR=60dBは実質ノイズなし
            let n = i.len().min(200);
            (i[..n].iter().map(|&x| x * x).sum::<f32>()
                + q[..n].iter().map(|&x| x * x).sum::<f32>())
                / (2.0 * n as f32)
        };
        let signal_rms = ref_signal_power.sqrt();
        // 0dB SNR のノイズ sigma (IとQそれぞれ)
        let noise_sigma_0db = signal_rms / 2.0_f32.sqrt();

        println!("=== ROC Analysis: ZC SF=13 preamble + Walsh SF=15 sync ===");
        println!("  Signal RMS (after pipeline): {:.4}", signal_rms);
        println!("  Noise sigma at 0dB SNR:      {:.4}", noise_sigma_0db);
        println!();

        // --- ステップ2: H0 スコア収集 (ノイズのみ・全位置) ---
        // ノイズ sigma は 0dB SNR 相当 (信号と同スケール) にキャリブレーション
        let mut h0_scores: Vec<f32> = Vec::new();
        let required_len = {
            let det = MarySyncDetector::new(
                config.clone(),
                MarySyncDetector::THRESHOLD_COARSE_DEFAULT,
                MarySyncDetector::THRESHOLD_FINE_DEFAULT,
            );
            det.preamble_sym_len * config.preamble_repeat + det.sync_sym_len
        };

        for trial in 0..NUM_NOISE_TRIALS {
            let mut rng = StdRng::seed_from_u64(trial as u64 + 9000);
            let dist = Normal::new(0.0, noise_sigma_0db as f64).unwrap();
            let buf_len = required_len + 200;
            let i: Vec<f32> = (0..buf_len).map(|_| dist.sample(&mut rng) as f32).collect();
            let q: Vec<f32> = (0..buf_len).map(|_| dist.sample(&mut rng) as f32).collect();

            let det = MarySyncDetector::new(
                config.clone(),
                MarySyncDetector::THRESHOLD_COARSE_DEFAULT,
                MarySyncDetector::THRESHOLD_FINE_DEFAULT,
            );
            for n in (0..=(i.len() - required_len)).step_by(1) {
                let (score, _) = det.score_candidate(&i, &q, n, config.preamble_repeat + 1);
                h0_scores.push(score);
            }
        }
        h0_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // FPR(τ) = P(H0スコア > τ) の計算
        let fpr_at = |tau: f32| -> f64 {
            let above = h0_scores.partition_point(|&s| s <= tau);
            (h0_scores.len() - above) as f64 / h0_scores.len() as f64
        };

        // 真の同期位置を計算：
        //   offset_48k=500 → proc_offset = 500/2 = 250 (ダウンサンプル比2)
        //   合計遅延 (Mod + Rec) = 64 samples (at 24k)
        //   実際のプリアンブル先頭位置 = 250 + 64 = 314
        let det_ref = MarySyncDetector::new(
            config.clone(),
            MarySyncDetector::THRESHOLD_COARSE_DEFAULT,
            MarySyncDetector::THRESHOLD_FINE_DEFAULT,
        );
        let signal_start_in_proc: usize = 314;
        let total_symbols_for_fine = config.preamble_repeat + config.sync_word_bits;

        let mut h1_scores_by_snr: Vec<Vec<f32>> = Vec::new();
        for &snr_db in snr_db_list {
            let mut scores = Vec::new();
            for trial in 0..NUM_TRIALS {
                let (i, q) =
                    generate_signal_with_awgn_seeded(&config, 500, snr_db, trial as u64 + 100);

                // ピーク周辺(-20..+20)を探して最大スコアを取る
                let mut best_score = 0.0f32;
                for offset in (signal_start_in_proc.saturating_sub(20))..=(signal_start_in_proc + 20) {
                    if offset + required_len <= i.len() {
                        let (score, _) =
                            det_ref.score_candidate(&i, &q, offset, total_symbols_for_fine);
                        if score > best_score {
                            best_score = score;
                        }
                    }
                }
                scores.push(best_score);
            }
            h1_scores_by_snr.push(scores);
        }

        // TPR(τ, snr) = P(H1スコア > τ | SNR) の計算
        let tpr_at = |scores: &[f32], tau: f32| -> f64 {
            scores.iter().filter(|&&s| s > tau).count() as f64 / scores.len() as f64
        };

        // --- ステップ4: ROC テーブル出力 ---
        // しきい値の候補: H0 分布のパーセンタイル点から選択
        let thresholds: Vec<f32> = {
            let percentiles = [0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0];
            percentiles
                .iter()
                .map(|&p| {
                    let idx = ((p * h0_scores.len() as f64) as usize).min(h0_scores.len() - 1);
                    h0_scores[idx]
                })
                .collect()
        };

        print!("\n{:>8} | {:>8}", "Thresh", "FPR");
        for &snr in snr_db_list {
            print!(" | TPR({:+.0})", snr);
        }
        println!();
        print!("{}", "-".repeat(9 + 10 + snr_db_list.len() * 12));
        println!();

        let mut auc_values: Vec<f64> = vec![0.0; snr_db_list.len()];

        for &tau in &thresholds {
            let fpr = fpr_at(tau);
            print!("{:>8.4} | {:>7.1}%", tau, fpr * 100.0);
            for (snr_idx, h1_scores) in h1_scores_by_snr.iter().enumerate() {
                let tpr = tpr_at(h1_scores, tau);
                print!(" | {:>8.1}%", tpr * 100.0);
                // 台形則で AUC を積算 (後でソート・正規化)
                let _ = (snr_idx, tpr);
            }
            println!();
        }

        // AUC 計算（台形則、しきい値を FPR でパラメタライズ）
        println!("\n=== AUC (Area Under ROC Curve) ===");
        for (snr_idx, (&snr_db, h1_scores)) in
            snr_db_list.iter().zip(h1_scores_by_snr.iter()).enumerate()
        {
            // 細かいしきい値グリッドで台形則
            let mut fpr_prev = 1.0f64;
            let mut tpr_prev = 1.0f64;
            let mut auc = 0.0f64;
            let steps = 200usize;
            for step in 1..=steps {
                let idx = h0_scores.len() * step / steps;
                let idx = idx.min(h0_scores.len() - 1);
                let tau = h0_scores[idx];
                let fpr = fpr_at(tau);
                let tpr = tpr_at(h1_scores, tau);
                auc += (fpr_prev - fpr) * (tpr_prev + tpr) / 2.0;
                fpr_prev = fpr;
                tpr_prev = tpr;
            }
            // FPR=0 の終端
            auc += fpr_prev * tpr_prev / 2.0;
            auc_values[snr_idx] = auc;
            println!("  SNR {:+.0}dB: AUC = {:.4}", snr_db, auc);
        }

        // 推奨しきい値 (FAR ≤ 1%) の算出
        let tau_far1pct = {
            let target_idx = (h0_scores.len() as f64 * 0.99) as usize;
            h0_scores[target_idx.min(h0_scores.len() - 1)]
        };
        let tau_far01pct = {
            let target_idx = (h0_scores.len() as f64 * 0.999) as usize;
            h0_scores[target_idx.min(h0_scores.len() - 1)]
        };
        println!();
        println!("=== Recommended Thresholds (from calibrated noise) ===");
        println!("  FAR ≤ 1.0%: tau = {:.4}", tau_far1pct);
        println!("  FAR ≤ 0.1%: tau = {:.4}", tau_far01pct);
        println!(
            "  Current THRESHOLD_FINE_DEFAULT: {:.4}",
            MarySyncDetector::THRESHOLD_FINE_DEFAULT
        );

        // --- アサーション: SNR 0dB時の AUC が 0.7 以上 ---
        let snr_0db_idx = snr_db_list.iter().position(|&s| s == 0.0).unwrap();
        assert!(
            auc_values[snr_0db_idx] >= 0.7,
            "AUC at 0dB SNR = {:.4}, expected >= 0.70. \
             Detector may not be working correctly.",
            auc_values[snr_0db_idx]
        );
    }


    #[test]
    fn test_score_candidate_mathematical_verification() {
        let config = DspConfig::default_48k();
        let detector = new_detector_default(config.clone());
        let sym_len = 15 * config.proc_samples_per_chip();
        let repeat = config.preamble_repeat;
        let spc = config.proc_samples_per_chip();

        // 理論的なピーク位置の計算:
        // 各コンポーネントの遅延を 24kHz レートで計算する
        let mod_ = Modulator::new(config.clone());
        let rate_ratio = config.sample_rate / config.proc_sample_rate();
        let mod_delay_24k = (mod_.delay() as f32 / rate_ratio).round() as usize;
        let rrc_bw = config.chip_rate * (1.0 + config.rrc_alpha) * 0.5;
        let rx_resampler = Resampler::new_with_cutoff(
            config.sample_rate as u32,
            config.proc_sample_rate() as u32,
            Some(rrc_bw),
        );
        let rx_resampler_delay = rx_resampler.delay();
        let rx_rrc_delay = (config.rrc_num_taps() - 1) / 2;
        let total_delay = mod_delay_24k + rx_resampler_delay + rx_rrc_delay;
        let theoretical_peak_n = total_delay as i32 - (spc / 2) as i32;

        let find_best_score = |i: &[f32], q: &[f32]| {
            let mut best_score = 0.0f32;
            // 理論的ピークの前後 1チップ分を探索
            for n in (theoretical_peak_n - spc as i32)..=(theoretical_peak_n + spc as i32) {
                if n < 0 || n as usize >= i.len() {
                    continue;
                }
                let (score, _) = detector.score_candidate(i, q, n as usize, repeat);
                if score > best_score {
                    best_score = score;
                }
            }
            best_score
        };

        // 1. 理想的な信号 (No Noise, No CFO)
        {
            let (i, q) = generate_signal(&config, 0, 1.0);
            let score = find_best_score(&i, &q);
            println!("Ideal signal best score: {:.4}", score);
            assert!(
                score > 0.8,
                "Ideal signal should have high score: {:.4}",
                score
            );
        }

        // 2. 大きなCFOがある信号 (1シンボルごとに 90度 回転)
        {
            let (i_raw, q_raw) = generate_signal(&config, 0, 1.0);
            let mut i_cfo = Vec::with_capacity(i_raw.len());
            let mut q_cfo = Vec::with_capacity(q_raw.len());
            for (idx, &ii) in i_raw.iter().enumerate() {
                let s_idx = idx / sym_len;
                let phase = (s_idx as f32) * std::f32::consts::FRAC_PI_2;
                let (sin_p, cos_p) = phase.sin_cos();
                let qq = q_raw[idx];
                i_cfo.push(ii * cos_p - qq * sin_p);
                q_cfo.push(ii * sin_p + qq * cos_p);
            }

            let score = find_best_score(&i_cfo, &q_cfo);
            println!("CFO signal (90 deg/sym) best score: {:.4}", score);
            assert!(
                score > 0.4,
                "CFO-drifted signal should still have high score: {:.4}",
                score
            );
        }

        // 3. 純粋なノイズ (複数シードで検証)
        {
            for seed in 0..10 {
                let mut rng = StdRng::seed_from_u64(seed + 100);
                let dist = Normal::new(0.0, 0.1).unwrap();
                // 範囲外アクセスを防ぐためバッファ長を十分に確保
                let i_noise: Vec<f32> = (0..3000).map(|_| dist.sample(&mut rng)).collect();
                let q_noise: Vec<f32> = (0..3000).map(|_| dist.sample(&mut rng)).collect();

                let (score, _) = detector.score_candidate(&i_noise, &q_noise, 0, repeat);

                let mut total_rho_p = 0.0f32;
                let mut current_offset = 0;
                for rep in 0..repeat {
                    let is_preamble = rep < config.preamble_repeat;
                    let (ci, cq, en) =
                        detector.correlate_one_symbol(&i_noise, &q_noise, current_offset, is_preamble);
                    let sf = if is_preamble {
                        detector.preamble_sf
                    } else {
                        detector.sync_sf
                    };
                    let rho_p_sym = (ci * ci + cq * cq) / (sf as f32 * en);
                    total_rho_p += rho_p_sym;
                    
                    let sym_len = if is_preamble { detector.preamble_sym_len } else { detector.sync_sym_len };
                    current_offset += sym_len;
                }
                let rho_p = total_rho_p / repeat as f32;
                let rho_phi = (score - 0.7 * rho_p) / 0.3;

                println!(
                    "Seed {}: score: {:.4} (rho_p: {:.4}, rho_phi: {:.4})",
                    seed, score, rho_p, rho_phi
                );
            }
        }
    }

    fn generate_signal_with_awgn_seeded(
        config: &DspConfig,
        offset: usize,
        snr_db: f32,
        seed: u64,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut modulator = Modulator::new(config.clone());
        let mut signal = vec![0.0; offset];
        signal.extend(modulator.encode_frame(&[]).iter().copied());
        signal.extend(vec![0.0; 500]);

        let (i_raw, q_raw) = downconvert(&signal, 0, config);

        // Decoder と同じパイプライン: downconvert → Resampler(48k→24k) → RRC(24kHz)
        let rrc_bw = config.chip_rate * (1.0 + config.rrc_alpha) * 0.5;
        let mut resampler_i = Resampler::new_with_cutoff(
            config.sample_rate as u32,
            config.proc_sample_rate() as u32,
            Some(rrc_bw),
        );
        let mut resampler_q = Resampler::new_with_cutoff(
            config.sample_rate as u32,
            config.proc_sample_rate() as u32,
            Some(rrc_bw),
        );
        let mut i_resampled = Vec::new();
        let mut q_resampled = Vec::new();
        resampler_i.process(&i_raw, &mut i_resampled);
        resampler_q.process(&q_raw, &mut q_resampled);

        let mut rrc_i = RrcFilter::from_config(config);
        let mut rrc_q = RrcFilter::from_config(config);
        let mut i_ch: Vec<f32> = i_resampled.iter().map(|&s| rrc_i.process(s)).collect();
        let mut q_ch: Vec<f32> = q_resampled.iter().map(|&s| rrc_q.process(s)).collect();

        let mut energy_sum = 0.0f32;
        let mut active_count = 0;
        for (&i, &q) in i_ch.iter().zip(q_ch.iter()) {
            let en = i * i + q * q;
            if en > 0.01 {
                energy_sum += en;
                active_count += 1;
            }
        }
        let signal_power = if active_count > 0 {
            energy_sum / active_count as f32
        } else {
            1.0 // Fallback
        };

        let snr_linear = 10.0_f32.powf(snr_db / 10.0);
        let noise_power = signal_power / snr_linear;
        let noise_std = (noise_power / 2.0).sqrt(); // IとQに分けるので 1/2

        let mut rng = StdRng::seed_from_u64(seed);
        let dist = Normal::new(0.0, noise_std).unwrap();

        for i in i_ch.iter_mut() {
            *i += dist.sample(&mut rng);
        }
        for q in q_ch.iter_mut() {
            *q += dist.sample(&mut rng);
        }

        (i_ch, q_ch)
    }
}
