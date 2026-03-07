//! 実用的な同期捕捉 (ノンコヒーレント相関) - MaryDQPSK版
//!
//! 1シンボルごとに相関電力を計算し、それらを足し合わせることで、
//! 位相回転やクロックズレに強い同期捕捉を実現する。

// use crate::common::walsh::WalshDictionary; // unused in current file
use crate::mary::params;
use crate::DspConfig;
use num_complex::Complex32;

#[derive(Debug, Clone)]
pub struct SyncResult {
    pub peak_sample_idx: usize,
    pub peak_iq: (f32, f32),
    pub score: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ChannelQualityEstimate {
    /// チップ中心サンプル上で推定した複素ノイズ分散 E[|n|^2]
    pub noise_var: f32,
    /// チップ中心サンプル上で推定した複素信号分散 E[|s|^2]
    pub signal_var: f32,
    /// 推定SNR[dB]。noise_var が極小の場合は None。
    pub snr_db: Option<f32>,
    /// プリアンブル反復間位相差から推定した CFO [rad/sample]
    pub cfo_rad_per_sample: f32,
    /// ノイズ推定に使えた差分対の数
    pub used_pairs: usize,
}

pub struct MarySyncDetector {
    config: DspConfig,
    preamble_pn: Vec<Complex32>, // Zadoff-Chu SF=13
    sync_pn: Vec<f32>,           // Walsh[0] SF=SYNC_SPREAD_FACTOR
    sync_symbols: Vec<f32>,      // プリアンブル構造 + SYNC_WORD
    preamble_sf: usize,
    sync_sf: usize,
    spc: usize,
    preamble_sym_len: usize,
    sync_sym_len: usize,
    pub threshold_coarse: f32,
    pub threshold_fine: f32,
}

impl MarySyncDetector {
    /// デフォルトのしきい値 (ZC SF=13 プリアンブル + Walsh 同期ワード用)
    /// ROC分析に基づく: ノイズ FAR 0.1% → 0.043, FAR 1% → 0.030
    /// ※ただし自己相関サイドローブ（無ノイズ時の信号自身との相関）による誤検出を防ぐため、実用的にはこれより高めに設定する
    pub const THRESHOLD_COARSE_DEFAULT: f32 = 0.10;
    pub const THRESHOLD_FINE_DEFAULT: f32 = 0.15;

    pub fn new(config: DspConfig, threshold_coarse: f32, threshold_fine: f32) -> Self {
        let preamble_sf = config.preamble_sf;
        let sync_sf = params::SYNC_SPREAD_FACTOR;
        let spc = config.proc_samples_per_chip().max(1);

        let zc = crate::common::zadoff_chu::ZadoffChu::new(preamble_sf, 1);
        let preamble_pn = zc.generate_sequence();

        let wdict = crate::common::walsh::WalshDictionary::default_w16();
        let sync_pn: Vec<f32> = wdict.w16[0]
            .iter()
            .take(sync_sf)
            .map(|&x| x as f32)
            .collect();

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
        // Modulator.encode_frame: maintains its own prev_phase (reset to 0 at start)
        let mut current_phase_factor = 1.0;
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

        // プリアンブル + SYNC_WORD 分まで見えてから同期確定する。
        if i_ch.len() < start_offset + required_len {
            (None, start_offset)
        } else {
            let search_range_end = i_ch.len() - required_len;
            let mut provisional_best: Option<(SyncResult, usize)> = None;
            let coarse_step = 1;

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
                        let (f_score, f_iq) = self.score_candidate(i_ch, q_ch, fn_idx, unified_len);
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

    /// プリアンブル1シンボル分の相関を計算（チップ全体で積分）
    fn correlate_preamble_symbol(
        &self,
        i_ch: &[f32],
        q_ch: &[f32],
        offset: usize,
    ) -> (f32, f32, f32) {
        let mut sum_i = 0.0f32;
        let mut sum_q = 0.0f32;
        let mut sum_en = 0.0f32;

        let mut p = offset;
        for val in &self.preamble_pn {
            for _ in 0..self.spc {
                if p >= i_ch.len() || p >= q_ch.len() {
                    break;
                }
                let si = i_ch[p];
                let sq = q_ch[p];
                // 複素共役との積和 (si + j sq) * (re - j im)
                sum_i += si * val.re + sq * val.im; // (実部)
                sum_q += sq * val.re - si * val.im; // (虚部)
                sum_en += si * si + sq * sq;
                p += 1;
            }
        }
        (sum_i, sum_q, sum_en)
    }

    /// 同期ワード1シンボル分の相関を計算（チップ全体で積分）
    fn correlate_sync_symbol(&self, i_ch: &[f32], q_ch: &[f32], offset: usize) -> (f32, f32, f32) {
        let mut sum_i = 0.0f32;
        let mut sum_q = 0.0f32;
        let mut sum_en = 0.0f32;

        let mut p = offset;
        for &rv in &self.sync_pn {
            for _ in 0..self.spc {
                if p >= i_ch.len() || p >= q_ch.len() {
                    break;
                }
                let si = i_ch[p];
                let sq = q_ch[p];
                sum_i += si * rv;
                sum_q += sq * rv;
                sum_en += si * si + sq * sq;
                p += 1;
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

        let mut sum_re = 0.0f32;
        let mut sum_im = 0.0f32;
        let mut sum_mag = 0.0f32;

        let mut last_ci = 0.0f32;
        let mut last_cq = 0.0f32;
        let mut last_mag = 0.0f32;

        let mut current_offset = n;

        for rep in 0..num_symbols {
            let is_preamble = rep < self.config.preamble_repeat;
            let (ci, cq, en) = if is_preamble {
                self.correlate_preamble_symbol(i_ch, q_ch, current_offset)
            } else {
                self.correlate_sync_symbol(i_ch, q_ch, current_offset)
            };
            let mag2 = ci * ci + cq * cq;
            let mag = mag2.sqrt();
            let sf = if is_preamble {
                self.preamble_sf
            } else {
                self.sync_sf
            };

            let rho_p_sym = if en > 1e-9 {
                mag2 / ((sf * self.spc) as f32 * en)
            } else {
                0.0
            };
            total_rho_p += rho_p_sym;

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

            let sym_len = if is_preamble {
                self.preamble_sym_len
            } else {
                self.sync_sym_len
            };
            current_offset += sym_len;
        }

        let rho_p = total_rho_p / num_symbols as f32;

        let rho_phi = if sum_mag > 1e-9 {
            (sum_re * sum_re + sum_im * sum_im).sqrt() / sum_mag
        } else {
            0.0
        };

        let score = rho_p * rho_phi;
        (score.max(0.0), (last_ci, last_cq))
    }

    fn estimate_cfo_rad_per_sample(&self, i_ch: &[f32], q_ch: &[f32], n: usize) -> f32 {
        let repeat = self.config.preamble_repeat.max(1);
        let sf = self.preamble_sf;
        let preamble_sym_len = self.preamble_sym_len;

        let mut cfo_rad_per_sample = 0.0f32;
        if repeat >= 2 {
            let mut sum = 0.0f32;
            let mut count = 0usize;
            for rep in 1..repeat {
                let prev_off = n + (rep - 1) * preamble_sym_len;
                let curr_off = n + rep * preamble_sym_len;
                if curr_off + (sf - 1) * self.spc >= i_ch.len()
                    || curr_off + (sf - 1) * self.spc >= q_ch.len()
                {
                    break;
                }
                let (pi, pq, _) = self.correlate_preamble_symbol(i_ch, q_ch, prev_off);
                let (ci, cq, _) = self.correlate_preamble_symbol(i_ch, q_ch, curr_off);
                let prev_sign = self.sync_symbols.get(rep - 1).copied().unwrap_or(1.0);
                let curr_sign = self.sync_symbols.get(rep).copied().unwrap_or(1.0);
                let prev_c = Complex32::new(pi, pq) * Complex32::new(prev_sign, 0.0);
                let curr_c = Complex32::new(ci, cq) * Complex32::new(curr_sign, 0.0);
                let dphi = (curr_c * prev_c.conj()).arg();
                sum += dphi / preamble_sym_len as f32;
                count += 1;
            }
            if count > 0 {
                cfo_rad_per_sample = sum / count as f32;
            }
        }
        cfo_rad_per_sample
    }

    /// プリアンブル相関から CIR とチャネル品質を同時推定する。
    /// - `cir_out`: サンプル解像度のCIRを書き込む先（空ならCIR推定をスキップ）
    /// - `quality_out`: ノイズ分散/SNR/CFOなどを上書き
    pub fn estimate_channel_quality(
        &self,
        i_ch: &[f32],
        q_ch: &[f32],
        n: usize,
        cir_out: &mut [Complex32],
        quality_out: &mut ChannelQualityEstimate,
    ) {
        let repeat = self.config.preamble_repeat.max(1);
        let sf = self.preamble_sf;
        let preamble_sym_len = self.preamble_sym_len;
        let cfo_rad_per_sample = self.estimate_cfo_rad_per_sample(i_ch, q_ch, n);

        let mut noise_acc = 0.0f32;
        let mut used_pairs = 0usize;
        let mut signal_acc = 0.0f32;
        let mut used_signal = 0usize;

        for k in 0..sf {
            let mut prev_z: Option<Complex32> = None;
            let mut sum_z = Complex32::new(0.0, 0.0);
            let mut count_z = 0usize;

            for rep in 0..repeat {
                let rep_start = n + rep * preamble_sym_len;
                let sign = self.sync_symbols.get(rep).copied().unwrap_or(1.0);
                let val = self.preamble_pn[k] * Complex32::new(sign, 0.0);

                let mut chip_z = Complex32::new(0.0, 0.0);
                let mut used_samples = 0;
                for s_idx in 0..self.spc {
                    let p = rep_start + k * self.spc + s_idx;
                    if p >= i_ch.len() || p >= q_ch.len() {
                        break;
                    }
                    let sig = Complex32::new(i_ch[p], q_ch[p]);
                    let mut z = sig * val.conj();
                    if cfo_rad_per_sample != 0.0 {
                        let t = (rep * preamble_sym_len + k * self.spc + s_idx) as f32;
                        let ang = -cfo_rad_per_sample * t;
                        let (s, c) = ang.sin_cos();
                        z *= Complex32::new(c, s);
                    }
                    chip_z += z;
                    used_samples += 1;
                }

                if used_samples == 0 {
                    break;
                }

                // Scale back by sqrt(spc) to preserve absolute power scale,
                // because adding identical signal samples coherently scales amplitude by `spc`
                // while noise amplitude scales by `sqrt(spc)`.
                // However, matching the `1.0 / sqrt(spc)` generation logic means:
                // Expected z should be sum of spc samples, each of amplitude 1/sqrt(spc).
                // Sum is spc * (1/sqrt(spc)) = sqrt(spc).
                // So to get the expected original power, we divide by sqrt(spc).
                let z = chip_z / (used_samples as f32).sqrt();

                if let Some(prev) = prev_z {
                    // 差分で信号成分を消し、複素ノイズ分散を推定する。
                    noise_acc += 0.5 * (z - prev).norm_sqr();
                    used_pairs += 1;
                }
                prev_z = Some(z);
                sum_z += z;
                count_z += 1;
            }

            if count_z > 0 {
                let mean_z = sum_z / count_z as f32;
                signal_acc += mean_z.norm_sqr();
                used_signal += 1;
            }
        }

        if !cir_out.is_empty() {
            for (d, out_val) in cir_out.iter_mut().enumerate() {
                let mut sum = Complex32::new(0.0, 0.0);
                let mut used = 0usize;
                for rep in 0..repeat {
                    let rep_start = n + rep * preamble_sym_len;
                    let sign = self.sync_symbols.get(rep).copied().unwrap_or(1.0);
                    let sign_c = Complex32::new(sign, 0.0);
                    for k in 0..sf {
                        let p = rep_start + d + k * self.spc;
                        if p >= i_ch.len() || p >= q_ch.len() {
                            break;
                        }
                        let val = self.preamble_pn[k] * sign_c;
                        let sig = Complex32::new(i_ch[p], q_ch[p]);
                        let mut corr = sig * val.conj();
                        if cfo_rad_per_sample != 0.0 {
                            let t = (rep * preamble_sym_len + k * self.spc) as f32;
                            let ang = -cfo_rad_per_sample * t;
                            let (s, c) = ang.sin_cos();
                            corr *= Complex32::new(c, s);
                        }
                        sum += corr;
                        used += 1;
                    }
                }
                *out_val = if used > 0 {
                    sum / used as f32
                } else {
                    Complex32::new(0.0, 0.0)
                };
            }
        }

        let noise_var = if used_pairs > 0 {
            noise_acc / used_pairs as f32
        } else {
            0.0
        };
        let signal_var = if used_signal > 0 {
            signal_acc / used_signal as f32
        } else {
            0.0
        };
        let snr_db = if noise_var > 1e-12 && signal_var > 0.0 {
            Some(10.0 * (signal_var / noise_var).log10())
        } else {
            None
        };

        *quality_out = ChannelQualityEstimate {
            noise_var,
            signal_var,
            snr_db,
            cfo_rad_per_sample,
            used_pairs,
        };
    }

    pub fn sync_symbols(&self) -> &[f32] {
        &self.sync_symbols
    }

    pub fn known_interval_len_samples(&self) -> usize {
        self.preamble_sym_len * self.config.preamble_repeat
            + self.sync_sym_len * self.config.sync_word_bits
    }

    pub fn sync_word_len_samples(&self) -> usize {
        self.sync_sym_len * self.config.sync_word_bits
    }

    fn sequence_len_samples(&self, symbol_start: usize, symbol_count: usize) -> usize {
        let repeat = self.config.preamble_repeat;
        let mut total = 0usize;
        for local_idx in 0..symbol_count {
            let global_idx = symbol_start + local_idx;
            total += if global_idx < repeat {
                self.preamble_sym_len
            } else {
                self.sync_sym_len
            };
        }
        total
    }

    fn sequence_mse_iq_window(
        &self,
        i_ch: &[f32],
        q_ch: &[f32],
        start_idx: usize,
        cfo_rad_per_sample: f32,
        symbol_start: usize,
        symbol_count: usize,
    ) -> Option<f32> {
        if i_ch.len() != q_ch.len() {
            return None;
        }
        let total_len = self.sequence_len_samples(symbol_start, symbol_count);
        if start_idx + total_len > i_ch.len() {
            return None;
        }

        let repeat = self.config.preamble_repeat;
        let mut sum_xy = Complex32::new(0.0, 0.0);
        let mut sum_yy = 0.0f32;
        let mut used = 0usize;
        let mut symbol_offset = start_idx;
        let mut sym_base = 0usize;

        for local_sym_idx in 0..symbol_count {
            let global_sym_idx = symbol_start + local_sym_idx;
            let is_preamble = global_sym_idx < repeat;
            let sign = self.sync_symbols[global_sym_idx];
            let sign_c = Complex32::new(sign, 0.0);
            let sf = if is_preamble {
                self.preamble_sf
            } else {
                self.sync_sf
            };

            for chip_idx in 0..sf {
                let mut chip_y = Complex32::new(0.0, 0.0);
                let mut used_samples = 0;
                for s_idx in 0..self.spc {
                    let p = symbol_offset + chip_idx * self.spc + s_idx;
                    let mut y = Complex32::new(i_ch[p], q_ch[p]);
                    if cfo_rad_per_sample != 0.0 {
                        let t = (sym_base + chip_idx * self.spc + s_idx) as f32;
                        let ang = -cfo_rad_per_sample * t;
                        let (s, c) = ang.sin_cos();
                        y *= Complex32::new(c, s);
                    }
                    chip_y += y;
                    used_samples += 1;
                }
                let y = if used_samples > 0 {
                    chip_y / (used_samples as f32).sqrt()
                } else {
                    Complex32::new(0.0, 0.0)
                };

                let code = if is_preamble {
                    self.preamble_pn[chip_idx]
                } else {
                    Complex32::new(self.sync_pn[chip_idx], 0.0)
                };
                let x = code * sign_c;
                sum_xy += x * y.conj();
                sum_yy += y.norm_sqr();
                used += 1;
            }

            let sym_len = if is_preamble {
                self.preamble_sym_len
            } else {
                self.sync_sym_len
            };
            symbol_offset += sym_len;
            sym_base += sym_len;
        }

        if used == 0 || sum_yy <= 1e-12 {
            return None;
        }

        let g = sum_xy / sum_yy;
        let mut mse_sum = 0.0f32;
        let mut symbol_offset = start_idx;
        let mut sym_base = 0usize;
        for local_sym_idx in 0..symbol_count {
            let global_sym_idx = symbol_start + local_sym_idx;
            let is_preamble = global_sym_idx < repeat;
            let sign = self.sync_symbols[global_sym_idx];
            let sign_c = Complex32::new(sign, 0.0);
            let sf = if is_preamble {
                self.preamble_sf
            } else {
                self.sync_sf
            };

            for chip_idx in 0..sf {
                let mut chip_y = Complex32::new(0.0, 0.0);
                let mut used_samples = 0;
                for s_idx in 0..self.spc {
                    let p = symbol_offset + chip_idx * self.spc + s_idx;
                    let mut y = Complex32::new(i_ch[p], q_ch[p]);
                    if cfo_rad_per_sample != 0.0 {
                        let t = (sym_base + chip_idx * self.spc + s_idx) as f32;
                        let ang = -cfo_rad_per_sample * t;
                        let (s, c) = ang.sin_cos();
                        y *= Complex32::new(c, s);
                    }
                    chip_y += y;
                    used_samples += 1;
                }
                let y = if used_samples > 0 {
                    chip_y / (used_samples as f32).sqrt()
                } else {
                    Complex32::new(0.0, 0.0)
                };

                let code = if is_preamble {
                    self.preamble_pn[chip_idx]
                } else {
                    Complex32::new(self.sync_pn[chip_idx], 0.0)
                };
                let x = code * sign_c;
                mse_sum += (g * y - x).norm_sqr();
            }

            let sym_len = if is_preamble {
                self.preamble_sym_len
            } else {
                self.sync_sym_len
            };
            symbol_offset += sym_len;
            sym_base += sym_len;
        }

        Some(mse_sum / used as f32)
    }

    fn sequence_mse_complex_window(
        &self,
        samples: &[Complex32],
        start_idx: usize,
        cfo_rad_per_sample: f32,
        symbol_start: usize,
        symbol_count: usize,
    ) -> Option<f32> {
        let total_len = self.sequence_len_samples(symbol_start, symbol_count);
        if start_idx + total_len > samples.len() {
            return None;
        }

        let repeat = self.config.preamble_repeat;
        let mut sum_xy = Complex32::new(0.0, 0.0);
        let mut sum_yy = 0.0f32;
        let mut used = 0usize;
        let mut symbol_offset = start_idx;
        let mut sym_base = 0usize;

        for local_sym_idx in 0..symbol_count {
            let global_sym_idx = symbol_start + local_sym_idx;
            let is_preamble = global_sym_idx < repeat;
            let sign = self.sync_symbols[global_sym_idx];
            let sign_c = Complex32::new(sign, 0.0);
            let sf = if is_preamble {
                self.preamble_sf
            } else {
                self.sync_sf
            };

            for chip_idx in 0..sf {
                let mut chip_y = Complex32::new(0.0, 0.0);
                let mut used_samples = 0;
                for s_idx in 0..self.spc {
                    let p = symbol_offset + chip_idx * self.spc + s_idx;
                    let mut y = samples[p];
                    if cfo_rad_per_sample != 0.0 {
                        let t = (sym_base + chip_idx * self.spc + s_idx) as f32;
                        let ang = -cfo_rad_per_sample * t;
                        let (s, c) = ang.sin_cos();
                        y *= Complex32::new(c, s);
                    }
                    chip_y += y;
                    used_samples += 1;
                }
                let y = if used_samples > 0 {
                    chip_y / (used_samples as f32).sqrt()
                } else {
                    Complex32::new(0.0, 0.0)
                };

                let code = if is_preamble {
                    self.preamble_pn[chip_idx]
                } else {
                    Complex32::new(self.sync_pn[chip_idx], 0.0)
                };
                let x = code * sign_c;
                sum_xy += x * y.conj();
                sum_yy += y.norm_sqr();
                used += 1;
            }

            let sym_len = if is_preamble {
                self.preamble_sym_len
            } else {
                self.sync_sym_len
            };
            symbol_offset += sym_len;
            sym_base += sym_len;
        }

        if used == 0 || sum_yy <= 1e-12 {
            return None;
        }

        let g = sum_xy / sum_yy;
        let mut mse_sum = 0.0f32;
        let mut symbol_offset = start_idx;
        let mut sym_base = 0usize;
        for local_sym_idx in 0..symbol_count {
            let global_sym_idx = symbol_start + local_sym_idx;
            let is_preamble = global_sym_idx < repeat;
            let sign = self.sync_symbols[global_sym_idx];
            let sign_c = Complex32::new(sign, 0.0);
            let sf = if is_preamble {
                self.preamble_sf
            } else {
                self.sync_sf
            };

            for chip_idx in 0..sf {
                let mut chip_y = Complex32::new(0.0, 0.0);
                let mut used_samples = 0;
                for s_idx in 0..self.spc {
                    let p = symbol_offset + chip_idx * self.spc + s_idx;
                    let mut y = samples[p];
                    if cfo_rad_per_sample != 0.0 {
                        let t = (sym_base + chip_idx * self.spc + s_idx) as f32;
                        let ang = -cfo_rad_per_sample * t;
                        let (s, c) = ang.sin_cos();
                        y *= Complex32::new(c, s);
                    }
                    chip_y += y;
                    used_samples += 1;
                }
                let y = if used_samples > 0 {
                    chip_y / (used_samples as f32).sqrt()
                } else {
                    Complex32::new(0.0, 0.0)
                };

                let code = if is_preamble {
                    self.preamble_pn[chip_idx]
                } else {
                    Complex32::new(self.sync_pn[chip_idx], 0.0)
                };
                let x = code * sign_c;
                mse_sum += (g * y - x).norm_sqr();
            }

            let sym_len = if is_preamble {
                self.preamble_sym_len
            } else {
                self.sync_sym_len
            };
            symbol_offset += sym_len;
            sym_base += sym_len;
        }

        Some(mse_sum / used as f32)
    }

    pub fn known_sequence_mse_iq(
        &self,
        i_ch: &[f32],
        q_ch: &[f32],
        start_idx: usize,
        cfo_rad_per_sample: f32,
    ) -> Option<f32> {
        self.sequence_mse_iq_window(
            i_ch,
            q_ch,
            start_idx,
            cfo_rad_per_sample,
            0,
            self.config.preamble_repeat + self.config.sync_word_bits,
        )
    }

    pub fn known_sequence_mse_complex(
        &self,
        samples: &[Complex32],
        start_idx: usize,
        cfo_rad_per_sample: f32,
    ) -> Option<f32> {
        self.sequence_mse_complex_window(
            samples,
            start_idx,
            cfo_rad_per_sample,
            0,
            self.config.preamble_repeat + self.config.sync_word_bits,
        )
    }

    pub fn sync_word_mse_iq(
        &self,
        i_ch: &[f32],
        q_ch: &[f32],
        start_idx: usize,
        cfo_rad_per_sample: f32,
    ) -> Option<f32> {
        self.sequence_mse_iq_window(
            i_ch,
            q_ch,
            start_idx,
            cfo_rad_per_sample,
            self.config.preamble_repeat,
            self.config.sync_word_bits,
        )
    }

    pub fn sync_word_mse_complex(
        &self,
        samples: &[Complex32],
        start_idx: usize,
        cfo_rad_per_sample: f32,
    ) -> Option<f32> {
        self.sequence_mse_complex_window(
            samples,
            start_idx,
            cfo_rad_per_sample,
            self.config.preamble_repeat,
            self.config.sync_word_bits,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::resample::Resampler;
    use crate::common::rrc_filter::RrcFilter;
    use crate::mary::modulator::Modulator;
    use rand::prelude::*;
    use rand_distr::Normal;

    pub fn downconvert(
        samples: &[f32],
        sample_offset: usize,
        config: &DspConfig,
    ) -> (Vec<f32>, Vec<f32>) {
        use crate::common::nco::Nco;
        let mut nco = Nco::new(-config.carrier_freq, config.sample_rate);
        nco.skip(sample_offset);
        let mut i_ch = Vec::with_capacity(samples.len());
        let mut q_ch = Vec::with_capacity(samples.len());
        for &s in samples {
            let lo = nco.step();
            i_ch.push(s * lo.re * 2.0);
            q_ch.push(s * lo.im * 2.0);
        }
        (i_ch, q_ch)
    }

    fn generate_signal(config: &DspConfig, offset: usize, amplitude: f32) -> (Vec<f32>, Vec<f32>) {
        let mut modulator = Modulator::new(config.clone());
        let mut signal = vec![0.0; offset];
        let mut frame = Vec::new();
        modulator.encode_frame(&[], &mut frame);
        signal.extend(frame.iter().map(|&s| s * amplitude));
        // マージンを追加: MarySyncDetector はピークの後に 1シンボル分以上のサンプルを要求するため
        // さらにCIR推定などのために余分に必要
        // ダウンサンプリング（48kHz→24kHz）で半分になるため、十分なマージンを確保する
        signal.extend(vec![0.0; 4000]);

        let (i_raw, q_raw) = downconvert(&signal, 0, config);

        // Decoder と同じパイプライン: downconvert → Resampler(48k→24k) → RRC(24kHz)
        let rrc_bw = config.chip_rate * (1.0 + config.rrc_alpha) * 0.5;
        let mut resampler_i = Resampler::new_with_cutoff(
            config.sample_rate as u32,
            config.proc_sample_rate() as u32,
            Some(rrc_bw),
            Some(config.rx_resampler_taps),
        );
        let mut resampler_q = Resampler::new_with_cutoff(
            config.sample_rate as u32,
            config.proc_sample_rate() as u32,
            Some(rrc_bw),
            Some(config.rx_resampler_taps),
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
        let sym_len = config.preamble_sf * config.proc_samples_per_chip();
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
            sync.peak_sample_idx,
            preamble_len
        );
        assert!(sync.score > detector.threshold_fine);
    }

    #[test]
    fn test_sync_first_match_scenarios() {
        let config = DspConfig::default_48k();
        let detector = new_detector_default(config.clone());
        let repeat = config.preamble_repeat;

        // 地上実測値 (Ground Truth) を求める補助関数
        let find_ground_truth = |i: &[f32], q: &[f32]| {
            let mut best_score = -1.0f32;
            let mut best_idx = 0;
            let unified_len = detector.sync_symbols.len();
            // score_candidate は最後のシンボルで以下の範囲にアクセスする：
            // 各シンボルで current_offset を更新し、correlate_sync_symbol などを呼ぶ
            // correlate_sync_symbol は current_offset から始まり、sf * spc サンプルにアクセスする
            // 最後のシンボル（sync, rep=9）の場合：
            // current_offset = n + preamble_repeat * preamble_sym_len + (sync_count - 1) * sync_sym_len
            //                  = n + 2 * 39 + 7 * 45 = n + 78 + 315 = n + 393
            // correlate_sync_symbol の最後のアクセス：
            // current_offset + sync_sf * spc - 1 が最後のアクセス位置
            // したがって required_len = 438 以上が必要
            let preamble_count = config.preamble_repeat;
            let sync_count = unified_len - preamble_count;
            let last_symbol_offset = preamble_count * detector.preamble_sym_len
                + (sync_count - 1) * detector.sync_sym_len;
            let last_symbol_access = last_symbol_offset + detector.sync_sf * detector.spc - 1;
            let required_len = last_symbol_access + 1; // +1 for 0-indexed
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
                    let (ci, cq, en) = if is_preamble {
                        detector.correlate_preamble_symbol(i, q, current_offset)
                    } else {
                        detector.correlate_sync_symbol(i, q, current_offset)
                    };
                    let mag2 = ci * ci + cq * cq;
                    let mag = mag2.sqrt();
                    let sf = if is_preamble {
                        detector.preamble_sf
                    } else {
                        detector.sync_sf
                    };
                    total_rho_p += if en > 1e-9 {
                        mag2 / ((sf * config.proc_samples_per_chip()) as f32 * en)
                    } else {
                        0.0
                    };
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
                    let sym_len = if is_preamble {
                        detector.preamble_sym_len
                    } else {
                        detector.sync_sym_len
                    };
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
            println!(
                "Found sync with score: {:.4} at detected_idx: {}",
                sync.score,
                sync.peak_sample_idx - (detector.preamble_sym_len * repeat)
            );
            assert!(sync.score > detector.threshold_fine);
            let detected_idx = sync.peak_sample_idx - (detector.preamble_sym_len * repeat);
            // Rc=8000 (spc=6) ではピークが平坦になりやすいため、spc/2 程度の誤差を許容する
            let error = (detected_idx as i32 - gt_idx as i32).abs();
            assert!(
                error <= (detector.spc / 2) as i32,
                "detected_idx={} gt_idx={} error={} spc={}",
                detected_idx,
                gt_idx,
                error,
                detector.spc
            );
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
            // SNR 0 dB: 従来の「-3dB」相当
            let (i, q) = generate_signal_with_awgn_seeded(&config, 500, 0.0, trial as u64);
            let (res, _) = detector.detect(&i, &q, 0);
            if res.is_some() {
                num_detected += 1;
            }
        }

        let detection_rate = num_detected as f64 / NUM_TRIALS as f64;
        println!(
            "Typical SNR (0 dB) detection rate: {:.2}%",
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
    #[ignore = "diagnostic ROC report; expensive"]
    fn test_roc_curve_analysis_diagnostic() {
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
                for offset in
                    (signal_start_in_proc.saturating_sub(20))..=(signal_start_in_proc + 20)
                {
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
    fn test_roc_curve_smoke() {
        let config = DspConfig::default_48k();
        const NUM_SIGNAL_TRIALS: usize = 20;
        const NUM_NOISE_TRIALS: usize = 40;

        let ref_signal_power: f32 = {
            let (i, q) = generate_signal_with_awgn_seeded(&config, 0, 60.0, 0);
            let n = i.len().min(200);
            (i[..n].iter().map(|&x| x * x).sum::<f32>()
                + q[..n].iter().map(|&x| x * x).sum::<f32>())
                / (2.0 * n as f32)
        };
        let signal_rms = ref_signal_power.sqrt();
        let noise_sigma_0db = signal_rms / 2.0_f32.sqrt();

        let detector = MarySyncDetector::new(
            config.clone(),
            MarySyncDetector::THRESHOLD_COARSE_DEFAULT,
            MarySyncDetector::THRESHOLD_FINE_DEFAULT,
        );
        let required_len =
            detector.preamble_sym_len * config.preamble_repeat + detector.sync_sym_len;

        let mut false_positives = 0usize;
        for trial in 0..NUM_NOISE_TRIALS {
            let mut rng = StdRng::seed_from_u64(trial as u64 + 9000);
            let dist = Normal::new(0.0, noise_sigma_0db as f64).unwrap();
            let buf_len = required_len + 200;
            let i: Vec<f32> = (0..buf_len).map(|_| dist.sample(&mut rng) as f32).collect();
            let q: Vec<f32> = (0..buf_len).map(|_| dist.sample(&mut rng) as f32).collect();
            let (res, _) = detector.detect(&i, &q, 0);
            if res.is_some() {
                false_positives += 1;
            }
        }

        let mut detections = 0usize;
        for trial in 0..NUM_SIGNAL_TRIALS {
            let (i, q) = generate_signal_with_awgn_seeded(&config, 500, 0.0, trial as u64 + 100);
            let (res, _) = detector.detect(&i, &q, 0);
            if res.is_some() {
                detections += 1;
            }
        }

        let false_positive_rate = false_positives as f64 / NUM_NOISE_TRIALS as f64;
        let detection_rate = detections as f64 / NUM_SIGNAL_TRIALS as f64;
        assert!(
            false_positive_rate <= 0.10,
            "Noise-only false positive rate too high: {:.1}%",
            false_positive_rate * 100.0
        );
        assert!(
            detection_rate >= 0.90,
            "0dB detection rate too low: {:.1}%",
            detection_rate * 100.0
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
            Some(config.rx_resampler_taps),
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
                    let (ci, cq, en) = if is_preamble {
                        detector.correlate_preamble_symbol(&i_noise, &q_noise, current_offset)
                    } else {
                        detector.correlate_sync_symbol(&i_noise, &q_noise, current_offset)
                    };
                    let sf = if is_preamble {
                        detector.preamble_sf
                    } else {
                        detector.sync_sf
                    };
                    let rho_p_sym = (ci * ci + cq * cq) / ((sf * detector.spc) as f32 * en);
                    total_rho_p += rho_p_sym;

                    let sym_len = if is_preamble {
                        detector.preamble_sym_len
                    } else {
                        detector.sync_sym_len
                    };
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
        let mut frame = Vec::new();
        modulator.encode_frame(&[], &mut frame);
        signal.extend(frame.iter().copied());
        signal.extend(vec![0.0; 500]);

        let (i_raw, q_raw) = downconvert(&signal, 0, config);

        // Decoder と同じパイプライン: downconvert → Resampler(48k→24k) → RRC(24kHz)
        let rrc_bw = config.chip_rate * (1.0 + config.rrc_alpha) * 0.5;
        let mut resampler_i = Resampler::new_with_cutoff(
            config.sample_rate as u32,
            config.proc_sample_rate() as u32,
            Some(rrc_bw),
            Some(config.rx_resampler_taps),
        );
        let mut resampler_q = Resampler::new_with_cutoff(
            config.sample_rate as u32,
            config.proc_sample_rate() as u32,
            Some(rrc_bw),
            Some(config.rx_resampler_taps),
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

    fn simulate_rx_frontend(samples: &[f32], config: &DspConfig) -> (Vec<f32>, Vec<f32>) {
        let (i_raw, q_raw) = downconvert(samples, 0, config);

        let rrc_bw = config.chip_rate * (1.0 + config.rrc_alpha) * 0.5;
        let mut resampler_i = Resampler::new_with_cutoff(
            config.sample_rate as u32,
            config.proc_sample_rate() as u32,
            Some(rrc_bw),
            Some(config.rx_resampler_taps),
        );
        let mut resampler_q = Resampler::new_with_cutoff(
            config.sample_rate as u32,
            config.proc_sample_rate() as u32,
            Some(rrc_bw),
            Some(config.rx_resampler_taps),
        );
        let mut i_res = Vec::new();
        let mut q_res = Vec::new();
        resampler_i.process(&i_raw, &mut i_res);
        resampler_q.process(&q_raw, &mut q_res);

        let mut rrc_i = RrcFilter::from_config(config);
        let mut rrc_q = RrcFilter::from_config(config);
        let i_ch: Vec<f32> = i_res.iter().map(|&s| rrc_i.process(s)).collect();
        let q_ch: Vec<f32> = q_res.iter().map(|&s| rrc_q.process(s)).collect();

        (i_ch, q_ch)
    }

    fn synth_preamble_observation(
        detector: &MarySyncDetector,
        taps: &[(usize, Complex32)],
        noise_std_iq: f32,
        seed: u64,
    ) -> (Vec<f32>, Vec<f32>) {
        synth_preamble_observation_with_cfo(detector, taps, noise_std_iq, 0.0, seed)
    }

    fn synth_preamble_observation_with_cfo(
        detector: &MarySyncDetector,
        taps: &[(usize, Complex32)],
        noise_std_iq: f32,
        cfo_rad_per_sample: f32,
        seed: u64,
    ) -> (Vec<f32>, Vec<f32>) {
        let repeat = detector.config.preamble_repeat.max(1);
        let sf = detector.preamble_sf;
        let spc = detector.spc;
        let preamble_sym_len = detector.preamble_sym_len;
        let max_delay = taps.iter().map(|(d, _)| *d).max().unwrap_or(0);
        let len = repeat * preamble_sym_len + max_delay + spc + 8;

        let mut x = vec![Complex32::new(0.0, 0.0); len];
        let chip_scale = 1.0 / (spc as f32).sqrt();
        for rep in 0..repeat {
            let rep_start = rep * preamble_sym_len;
            let sign = detector.sync_symbols.get(rep).copied().unwrap_or(1.0);
            for k in 0..sf {
                for s_idx in 0..spc {
                    let p = rep_start + k * spc + s_idx;
                    if p < len {
                        x[p] = detector.preamble_pn[k] * Complex32::new(sign * chip_scale, 0.0);
                    }
                }
            }
        }

        let mut y = vec![Complex32::new(0.0, 0.0); len];
        for &(delay, gain) in taps {
            for n in delay..len {
                y[n] += x[n - delay] * gain;
            }
        }

        if cfo_rad_per_sample != 0.0 {
            for (idx, v) in y.iter_mut().enumerate() {
                let ang = cfo_rad_per_sample * idx as f32;
                let (s, c) = ang.sin_cos();
                *v *= Complex32::new(c, s);
            }
        }

        if noise_std_iq > 0.0 {
            let mut rng = StdRng::seed_from_u64(seed);
            let dist = Normal::new(0.0, noise_std_iq).unwrap();
            for v in &mut y {
                v.re += dist.sample(&mut rng);
                v.im += dist.sample(&mut rng);
            }
        }

        let i = y.iter().map(|c| c.re).collect();
        let q = y.iter().map(|c| c.im).collect();
        (i, q)
    }

    fn estimate_quality_only(
        detector: &MarySyncDetector,
        i_ch: &[f32],
        q_ch: &[f32],
        n: usize,
    ) -> ChannelQualityEstimate {
        let mut est = ChannelQualityEstimate::default();
        let mut cir_dummy: [Complex32; 0] = [];
        detector.estimate_channel_quality(i_ch, q_ch, n, &mut cir_dummy, &mut est);
        est
    }

    fn synth_known_observation_with_cfo(
        detector: &MarySyncDetector,
        taps: &[(usize, Complex32)],
        noise_std_iq: f32,
        cfo_rad_per_sample: f32,
        seed: u64,
    ) -> (Vec<f32>, Vec<f32>) {
        let repeat = detector.config.preamble_repeat.max(1);
        let total_symbols = repeat + detector.config.sync_word_bits;
        let spc = detector.spc;
        let total_len = detector.known_interval_len_samples();
        let max_delay = taps.iter().map(|(d, _)| *d).max().unwrap_or(0);
        let len = total_len + max_delay + spc + 8;

        let mut x = vec![Complex32::new(0.0, 0.0); len];
        let chip_scale = 1.0 / (spc as f32).sqrt();
        let mut symbol_offset = 0usize;
        for sym_idx in 0..total_symbols {
            let is_preamble = sym_idx < repeat;
            let sign = detector.sync_symbols[sym_idx];
            let sign_c = Complex32::new(sign * chip_scale, 0.0);
            let sf = if is_preamble {
                detector.preamble_sf
            } else {
                detector.sync_sf
            };
            for chip_idx in 0..sf {
                for s_idx in 0..spc {
                    let p = symbol_offset + chip_idx * spc + s_idx;
                    if p >= len {
                        break;
                    }
                    let code = if is_preamble {
                        detector.preamble_pn[chip_idx]
                    } else {
                        Complex32::new(detector.sync_pn[chip_idx], 0.0)
                    };
                    x[p] = code * sign_c;
                }
            }
            symbol_offset += if is_preamble {
                detector.preamble_sym_len
            } else {
                detector.sync_sym_len
            };
        }

        let mut y = vec![Complex32::new(0.0, 0.0); len];
        for &(delay, gain) in taps {
            for n in delay..len {
                y[n] += x[n - delay] * gain;
            }
        }

        if cfo_rad_per_sample != 0.0 {
            for (idx, v) in y.iter_mut().enumerate() {
                let ang = cfo_rad_per_sample * idx as f32;
                let (s, c) = ang.sin_cos();
                *v *= Complex32::new(c, s);
            }
        }

        if noise_std_iq > 0.0 {
            let mut rng = StdRng::seed_from_u64(seed);
            let dist = Normal::new(0.0, noise_std_iq).unwrap();
            for v in &mut y {
                v.re += dist.sample(&mut rng);
                v.im += dist.sample(&mut rng);
            }
        }

        let i = y.iter().map(|c| c.re).collect();
        let q = y.iter().map(|c| c.im).collect();
        (i, q)
    }

    #[test]
    fn test_known_sequence_mse_zero_noise_is_small_and_iq_complex_match() {
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 127;
        config.preamble_repeat = 2;
        config.sync_word_bits = 8;
        let detector = MarySyncDetector::new(config, 0.0, 0.0);

        let (i, q) = synth_known_observation_with_cfo(
            &detector,
            &[(0usize, Complex32::new(0.8, -0.3))],
            0.0,
            0.0,
            1,
        );
        let mse_iq = detector
            .known_sequence_mse_iq(&i, &q, 0, 0.0)
            .expect("known_sequence_mse_iq");
        let samples: Vec<Complex32> = i
            .iter()
            .zip(q.iter())
            .map(|(&ii, &qq)| Complex32::new(ii, qq))
            .collect();
        let mse_c = detector
            .known_sequence_mse_complex(&samples, 0, 0.0)
            .expect("known_sequence_mse_complex");

        assert!(mse_iq < 1e-8, "mse_iq={}", mse_iq);
        assert!(
            (mse_iq - mse_c).abs() < 1e-8,
            "mse_iq={} mse_c={}",
            mse_iq,
            mse_c
        );
    }

    #[test]
    fn test_known_sequence_mse_increases_with_noise() {
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 71;
        config.preamble_repeat = 2;
        config.sync_word_bits = 8;
        let detector = MarySyncDetector::new(config, 0.0, 0.0);

        let trials = 80usize;
        let mut sum_lo = 0.0f32;
        let mut sum_hi = 0.0f32;
        for t in 0..trials {
            let (i_lo, q_lo) = synth_known_observation_with_cfo(
                &detector,
                &[(0usize, Complex32::new(1.0, 0.0))],
                0.01,
                0.0,
                t as u64 + 10,
            );
            let (i_hi, q_hi) = synth_known_observation_with_cfo(
                &detector,
                &[(0usize, Complex32::new(1.0, 0.0))],
                0.05,
                0.0,
                t as u64 + 10010,
            );
            sum_lo += detector
                .known_sequence_mse_iq(&i_lo, &q_lo, 0, 0.0)
                .unwrap();
            sum_hi += detector
                .known_sequence_mse_iq(&i_hi, &q_hi, 0, 0.0)
                .unwrap();
        }
        let avg_lo = sum_lo / trials as f32;
        let avg_hi = sum_hi / trials as f32;
        assert!(
            avg_hi > avg_lo * 4.0,
            "known-sequence MSE should grow with noise: lo={} hi={}",
            avg_lo,
            avg_hi
        );
    }

    #[test]
    fn test_known_sequence_mse_cfo_compensation_effective() {
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 127;
        config.preamble_repeat = 2;
        config.sync_word_bits = 8;
        let detector = MarySyncDetector::new(config, 0.0, 0.0);
        let injected_cfo = 0.004f32;

        let (i, q) = synth_known_observation_with_cfo(
            &detector,
            &[(0usize, Complex32::new(1.0, 0.0))],
            0.0,
            injected_cfo,
            77,
        );

        let mse_no_comp = detector.known_sequence_mse_iq(&i, &q, 0, 0.0).unwrap();
        let mse_comp = detector
            .known_sequence_mse_iq(&i, &q, 0, injected_cfo)
            .unwrap();
        assert!(
            mse_comp < mse_no_comp * 0.1,
            "CFO compensation should reduce known-sequence MSE: no_comp={} comp={}",
            mse_no_comp,
            mse_comp
        );
    }

    #[test]
    fn test_known_sequence_mse_supports_nonzero_start_offset() {
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 71;
        config.preamble_repeat = 2;
        config.sync_word_bits = 8;
        let detector = MarySyncDetector::new(config, 0.0, 0.0);
        let offset = 37usize;

        let (i_sig, q_sig) = synth_known_observation_with_cfo(
            &detector,
            &[(0usize, Complex32::new(0.9, 0.2))],
            0.0,
            0.0,
            123,
        );
        let mut i = vec![0.0f32; offset];
        let mut q = vec![0.0f32; offset];
        i.extend(i_sig);
        q.extend(q_sig);

        let mse = detector
            .known_sequence_mse_iq(&i, &q, offset, 0.0)
            .expect("known_sequence_mse_iq with offset");
        assert!(mse < 1e-8, "mse={}", mse);
    }

    #[test]
    fn test_sync_word_mse_zero_noise_is_small_and_iq_complex_match() {
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 127;
        config.preamble_repeat = 2;
        config.sync_word_bits = 8;
        let detector = MarySyncDetector::new(config, 0.0, 0.0);

        let (i, q) = synth_known_observation_with_cfo(
            &detector,
            &[(0usize, Complex32::new(0.8, -0.3))],
            0.0,
            0.0,
            11,
        );
        let sync_start = detector.known_interval_len_samples() - detector.sync_word_len_samples();
        let mse_iq = detector
            .sync_word_mse_iq(&i, &q, sync_start, 0.0)
            .expect("sync_word_mse_iq");
        let samples: Vec<Complex32> = i
            .iter()
            .zip(q.iter())
            .map(|(&ii, &qq)| Complex32::new(ii, qq))
            .collect();
        let mse_c = detector
            .sync_word_mse_complex(&samples, sync_start, 0.0)
            .expect("sync_word_mse_complex");

        assert!(mse_iq < 1e-8, "mse_iq={}", mse_iq);
        assert!(
            (mse_iq - mse_c).abs() < 1e-8,
            "mse_iq={} mse_c={}",
            mse_iq,
            mse_c
        );
    }

    #[test]
    fn test_sync_word_mse_cfo_compensation_effective() {
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 127;
        config.preamble_repeat = 2;
        config.sync_word_bits = 8;
        let detector = MarySyncDetector::new(config, 0.0, 0.0);
        let injected_cfo = 0.004f32;

        let (i, q) = synth_known_observation_with_cfo(
            &detector,
            &[(0usize, Complex32::new(1.0, 0.0))],
            0.0,
            injected_cfo,
            77,
        );
        let sync_start = detector.known_interval_len_samples() - detector.sync_word_len_samples();

        let mse_no_comp = detector.sync_word_mse_iq(&i, &q, sync_start, 0.0).unwrap();
        let mse_comp = detector
            .sync_word_mse_iq(&i, &q, sync_start, injected_cfo)
            .unwrap();
        assert!(
            mse_comp < mse_no_comp * 0.1,
            "CFO compensation should reduce sync-word MSE: no_comp={} comp={}",
            mse_no_comp,
            mse_comp
        );
    }

    #[test]
    fn test_estimate_channel_quality_zero_noise_baseline() {
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 127;
        config.preamble_repeat = 2;
        let detector = MarySyncDetector::new(config, 0.0, 0.0);

        let gain = Complex32::new(0.8, -0.3);
        let (i, q) = synth_preamble_observation(&detector, &[(0, gain)], 0.0, 1);

        let est = estimate_quality_only(&detector, &i, &q, 0);
        assert!(est.noise_var < 1e-9, "noise_var={}", est.noise_var);
        assert!(
            (est.signal_var - gain.norm_sqr()).abs() < 1e-5,
            "signal_var={}, expected={}",
            est.signal_var,
            gain.norm_sqr()
        );
        assert!(est.snr_db.is_none(), "snr_db should be None for zero-noise");
        assert!(
            est.cfo_rad_per_sample.abs() < 1e-6,
            "cfo_rad_per_sample={}",
            est.cfo_rad_per_sample
        );
        assert!(
            est.used_pairs > 0,
            "used_pairs should be positive, got {}",
            est.used_pairs
        );
    }

    #[test]
    fn test_estimate_channel_quality_noise_variance_accuracy() {
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 127;
        config.preamble_repeat = 2;
        let detector = MarySyncDetector::new(config, 0.0, 0.0);

        let gain = Complex32::new(1.0, 0.0);
        let noise_std = 0.05f32;
        let expected_noise_var = 2.0 * noise_std * noise_std;

        let trials = 200usize;
        let mut sum = 0.0f32;
        for t in 0..trials {
            let (i, q) = synth_preamble_observation(&detector, &[(0, gain)], noise_std, t as u64);
            let est = estimate_quality_only(&detector, &i, &q, 0);
            sum += est.noise_var;
        }
        let avg = sum / trials as f32;
        let rel_err = (avg - expected_noise_var).abs() / expected_noise_var.max(1e-12);
        assert!(
            rel_err < 0.18,
            "avg_noise_var={} expected={} rel_err={}",
            avg,
            expected_noise_var,
            rel_err
        );
    }

    #[test]
    fn test_estimate_channel_quality_noise_variance_scales_with_sigma() {
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 127;
        config.preamble_repeat = 2;
        let detector = MarySyncDetector::new(config, 0.0, 0.0);
        let gain = Complex32::new(1.0, 0.0);

        let sigma_lo = 0.03f32;
        let sigma_hi = 0.09f32;
        let expected_ratio = (sigma_hi / sigma_lo).powi(2);

        let trials = 120usize;
        let mut sum_lo = 0.0f32;
        let mut sum_hi = 0.0f32;
        for t in 0..trials {
            let (i_lo, q_lo) =
                synth_preamble_observation(&detector, &[(0, gain)], sigma_lo, t as u64 + 11);
            let (i_hi, q_hi) =
                synth_preamble_observation(&detector, &[(0, gain)], sigma_hi, t as u64 + 5011);
            sum_lo += estimate_quality_only(&detector, &i_lo, &q_lo, 0).noise_var;
            sum_hi += estimate_quality_only(&detector, &i_hi, &q_hi, 0).noise_var;
        }
        let avg_lo = sum_lo / trials as f32;
        let avg_hi = sum_hi / trials as f32;
        let ratio = avg_hi / avg_lo.max(1e-12);
        assert!(
            ratio > expected_ratio * 0.75 && ratio < expected_ratio * 1.25,
            "ratio={} expected_ratio={}",
            ratio,
            expected_ratio
        );
    }

    #[test]
    fn test_estimate_channel_quality_snr_db_accuracy() {
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 127;
        config.preamble_repeat = 2;
        let detector = MarySyncDetector::new(config, 0.0, 0.0);

        let gain = Complex32::new(0.9, 0.0);
        let noise_std = 0.05f32;
        let expected_noise_var = 2.0 * noise_std * noise_std;
        let expected_signal_var = gain.norm_sqr();
        let expected_snr_db = 10.0 * (expected_signal_var / expected_noise_var).log10();

        let trials = 160usize;
        let mut sum_snr = 0.0f32;
        let mut used = 0usize;
        for t in 0..trials {
            let (i, q) = synth_preamble_observation(&detector, &[(0, gain)], noise_std, t as u64);
            let est = estimate_quality_only(&detector, &i, &q, 0);
            if let Some(v) = est.snr_db {
                sum_snr += v;
                used += 1;
            }
        }
        let avg_snr = sum_snr / used.max(1) as f32;
        assert!(
            (avg_snr - expected_snr_db).abs() < 1.5,
            "avg_snr={} expected_snr_db={}",
            avg_snr,
            expected_snr_db
        );
    }

    #[test]
    fn test_estimate_channel_quality_cfo_estimation_accuracy() {
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 127;
        config.preamble_repeat = 4;
        let detector = MarySyncDetector::new(config, 0.0, 0.0);

        let gain = Complex32::new(1.0, 0.0);
        let injected_cfo = 0.0045f32;
        let trials = 80usize;
        let mut sum = 0.0f32;

        for t in 0..trials {
            let (i, q) = synth_preamble_observation_with_cfo(
                &detector,
                &[(0, gain)],
                0.01,
                injected_cfo,
                t as u64 + 2000,
            );
            let est = estimate_quality_only(&detector, &i, &q, 0);
            sum += est.cfo_rad_per_sample;
        }
        let avg = sum / trials as f32;
        assert!(
            (avg - injected_cfo).abs() < 5e-4,
            "avg_cfo={} injected_cfo={}",
            avg,
            injected_cfo
        );
    }

    #[test]
    fn test_estimate_channel_quality_noise_var_monotonic_under_multipath() {
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 127;
        config.preamble_repeat = 3;
        let detector = MarySyncDetector::new(config, 0.0, 0.0);
        let noise_std_lo = 0.02f32;
        let noise_std_hi = 0.06f32;
        let trials = 140usize;
        let taps_mp = [
            (0usize, Complex32::new(1.0, 0.0)),
            (3usize, Complex32::new(0.45, 0.2)),
            (9usize, Complex32::new(0.2, -0.15)),
        ];

        let mut sum_noise_lo = 0.0f32;
        let mut sum_noise_hi = 0.0f32;
        let mut sum_snr_lo = 0.0f32;
        let mut sum_snr_hi = 0.0f32;
        for t in 0..trials {
            let (i1, q1) =
                synth_preamble_observation(&detector, &taps_mp, noise_std_lo, t as u64 + 9007);
            let (i2, q2) =
                synth_preamble_observation(&detector, &taps_mp, noise_std_hi, t as u64 + 19007);
            let est_lo = estimate_quality_only(&detector, &i1, &q1, 0);
            let est_hi = estimate_quality_only(&detector, &i2, &q2, 0);
            sum_noise_lo += est_lo.noise_var;
            sum_noise_hi += est_hi.noise_var;
            sum_snr_lo += est_lo.snr_db.unwrap_or(-100.0);
            sum_snr_hi += est_hi.snr_db.unwrap_or(-100.0);
        }
        let avg_noise_lo = sum_noise_lo / trials as f32;
        let avg_noise_hi = sum_noise_hi / trials as f32;
        let avg_snr_lo = sum_snr_lo / trials as f32;
        let avg_snr_hi = sum_snr_hi / trials as f32;

        assert!(
            avg_noise_hi > avg_noise_lo * 2.0,
            "noise_var should increase with sigma under multipath: lo={} hi={}",
            avg_noise_lo,
            avg_noise_hi
        );
        assert!(
            avg_snr_hi + 3.0 < avg_snr_lo,
            "snr should decrease with sigma under multipath: snr_lo={} snr_hi={}",
            avg_snr_lo,
            avg_snr_hi
        );
    }

    #[test]
    fn test_estimate_channel_quality_used_pairs_scales_with_repeat() {
        for repeat in [2usize, 3, 4] {
            let mut config = DspConfig::default_48k();
            config.preamble_sf = 127;
            config.preamble_repeat = repeat;
            let detector = MarySyncDetector::new(config, 0.0, 0.0);

            let (i, q) = synth_preamble_observation(
                &detector,
                &[(0usize, Complex32::new(1.0, 0.0))],
                0.02,
                11,
            );
            let est = estimate_quality_only(&detector, &i, &q, 0);
            let expected = detector.preamble_sf * (repeat - 1);
            assert_eq!(
                est.used_pairs, expected,
                "used_pairs mismatch for repeat={}",
                repeat
            );
        }
    }

    #[test]
    fn test_estimate_channel_quality_frontend_snr_monotonicity() {
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 71;
        config.preamble_repeat = 2;
        let detector = MarySyncDetector::new(config.clone(), 0.15, 0.18);

        let spc = config.proc_samples_per_chip();
        let preamble_len = config.preamble_sf * config.preamble_repeat * spc;

        let mut snr0_vals = Vec::new();
        let mut snr10_vals = Vec::new();
        for seed in 0..40u64 {
            for (snr_db, dst) in [(-0.0f32, &mut snr0_vals), (10.0f32, &mut snr10_vals)] {
                let (i, q) = generate_signal_with_awgn_seeded(&config, 500, snr_db, 1000 + seed);
                let (res, _) = detector.detect(&i, &q, 0);
                let sync = res.expect("sync should be detected in frontend monotonicity test");
                let preamble_start = sync
                    .peak_sample_idx
                    .saturating_sub(preamble_len)
                    .saturating_sub(spc / 2);
                let est = estimate_quality_only(&detector, &i, &q, preamble_start);
                if let Some(v) = est.snr_db {
                    dst.push(v);
                }
            }
        }

        let avg0 = snr0_vals.iter().sum::<f32>() / snr0_vals.len().max(1) as f32;
        let avg10 = snr10_vals.iter().sum::<f32>() / snr10_vals.len().max(1) as f32;
        assert!(
            avg10 > avg0 + 3.0,
            "frontend SNR estimate should be monotonic: avg0={} avg10={}",
            avg0,
            avg10
        );
    }

    #[test]
    fn test_estimate_channel_quality_tiny_noise_is_finite() {
        let mut config = DspConfig::default_48k();
        config.preamble_sf = 127;
        config.preamble_repeat = 2;
        let detector = MarySyncDetector::new(config, 0.0, 0.0);

        let (i, q) =
            synth_preamble_observation(&detector, &[(0usize, Complex32::new(1.0, 0.0))], 1e-6, 99);
        let est = estimate_quality_only(&detector, &i, &q, 0);
        assert!(est.noise_var.is_finite(), "noise_var must be finite");
        assert!(est.signal_var.is_finite(), "signal_var must be finite");
        if let Some(snr) = est.snr_db {
            assert!(snr.is_finite(), "snr_db must be finite");
            assert!(
                snr > 40.0,
                "snr_db should be high under tiny noise: {}",
                snr
            );
        }
    }

    #[test]
    fn test_sync_sf_sweep() {
        for sf in [13, 31, 63, 127] {
            let mut config = DspConfig::default_48k();
            // test downsampling behavior
            config.preamble_sf = sf;

            let mut modulator = Modulator::new(config.clone());
            let mut frame = Vec::new();
            modulator.encode_frame(&[], &mut frame);
            // フィルタ遅延によって波形が後ろにズレるため、受信十分なマージン（無音）を追加する
            frame.extend(vec![0.0; 4000]);
            let (i_ch, q_ch) = simulate_rx_frontend(&frame, &config);

            let detector = MarySyncDetector::new(config.clone(), 0.15, 0.18);

            let required_len = (sf * config.preamble_repeat
                + params::SYNC_SPREAD_FACTOR * config.sync_word_bits)
                * config.proc_samples_per_chip();
            assert!(i_ch.len() > required_len, "i_ch not large enough");
            let mut sync_found = None;
            for offset in 0..(i_ch.len() - required_len) {
                let (res, _) = detector.detect(&i_ch, &q_ch, offset);
                if res.is_some() {
                    sync_found = res;
                    break;
                }
            }

            if sync_found.is_none() {
                // 最大スコアを全範囲で探してデバッグ出力
                let mut max_score = 0.0f32;
                let mut max_idx = 0;
                for offset in 0..(i_ch.len() - required_len) {
                    let (score, _) = detector.score_candidate(
                        &i_ch,
                        &q_ch,
                        offset,
                        config.preamble_repeat + config.sync_word_bits,
                    );
                    if score > max_score {
                        max_score = score;
                        max_idx = offset;
                    }
                }
                println!(
                    "  SF={} sync failed. Max manual score was {:.4} at offset {}",
                    sf, max_score, max_idx
                );
            }

            assert!(sync_found.is_some(), "Sync failed for SF={}", sf);
            let result = sync_found.unwrap();
            println!(
                "SF={}: score={:.4}, idx={}",
                sf, result.score, result.peak_sample_idx
            );
            assert!(
                result.score > 0.65,
                "Score too low for SF={}: {}",
                sf,
                result.score
            );
        }
    }

    #[test]
    fn test_estimate_cir_simple_multipath() {
        let sf = 31;
        let mut config = DspConfig::default_48k();
        config.preamble_sf = sf;

        let mut modulator = Modulator::new(config.clone());
        let mut frame = Vec::new();
        modulator.encode_frame(&[], &mut frame);
        frame.extend(vec![0.0; 8000]);

        // まずマルチパスをパスバンドでかける（本当はベースバンドでも良いが、モジュレータ出力はパスバンド）
        let mut multipath_frame = frame.clone();
        let delay_samples = 5
            * config.proc_samples_per_chip()
            * (config.sample_rate / config.proc_sample_rate()) as usize;
        // 位相がどう回るかは複雑なのでここでは単純なゲイン減衰だけにする
        let gain = 0.5f32;
        for t in delay_samples..frame.len() {
            multipath_frame[t] += frame[t - delay_samples] * gain;
        }

        let (i_ch, q_ch) = simulate_rx_frontend(&multipath_frame, &config);

        let detector = MarySyncDetector::new(config.clone(), 0.15, 0.18); // 少し低めに設定
        let (res, _) = detector.detect(&i_ch, &q_ch, 0);
        assert!(res.is_some(), "Sync failed in multipath");
        let sync_res = res.unwrap();

        let spc = config.proc_samples_per_chip();
        let sync_idx = sync_res
            .peak_sample_idx
            .saturating_sub(sf * config.preamble_repeat * spc)
            .saturating_sub(spc / 2);

        // サンプル解像度のCIRを期待して、チップ数 * spc 分のバッファを用意
        let mut est_cir = vec![Complex32::new(0.0, 0.0); sf * spc];
        let mut chq = ChannelQualityEstimate::default();
        detector.estimate_channel_quality(&i_ch, &q_ch, sync_idx, &mut est_cir, &mut chq);

        // 第0タップで正規化（位相回転も補正）
        let ref_val = est_cir[0];
        for val in est_cir.iter_mut() {
            if ref_val.norm() > 1e-9 {
                *val /= ref_val;
            }
        }

        println!("Estimated CIR (normalized by Tap 0):");
        for (i, val) in est_cir.iter().enumerate().take(20) {
            println!(
                "  [{}] {:.4} + {:.4}j (mag={:.4})",
                i,
                val.re,
                val.im,
                val.norm()
            );
        }

        assert!((est_cir[0].re - 1.0).abs() < 0.1);
        assert!((est_cir[0].im).abs() < 0.1);

        // 5チップ遅延 = 5 * spc サンプル遅延の位置にピークがあるはず
        // パスバンドでの遅延は位相回転を引き起こすため、マグニチュードでのみ検証
        let expected_sample_idx = 5 * spc;
        let tap_mag = est_cir[expected_sample_idx].norm();
        // 0.5のゲインで重畳しているが、位相回転やノイズの影響で完全には一致しない
        // ピークが存在することを確認する（周囲より大きければOK）
        let neighborhood_avg: f32 = est_cir[expected_sample_idx - 2..expected_sample_idx + 3]
            .iter()
            .map(|v| v.norm())
            .sum::<f32>()
            / 5.0;
        assert!(
            tap_mag > neighborhood_avg * 1.1,
            "Tap at {} should be a peak (mag={:.4}, neighborhood_avg={:.4})",
            expected_sample_idx,
            tap_mag,
            neighborhood_avg
        );
    }

    #[test]
    fn test_estimate_cir_pure_unit() {
        let sf = 13;
        let spc = 3;
        let mut config = DspConfig::default_48k();
        config.preamble_sf = sf;
        let detector = MarySyncDetector::new(config, 0.0, 0.0);

        // 1. ZC系列をベースに、理想的なマルチパス信号を作成
        // preamble_repeat=2回分の信号を生成する必要がある
        let preamble_sym_len = sf * spc;
        let signal_len = preamble_sym_len * detector.config.preamble_repeat + 100;
        let mut i_ch = vec![0.0f32; signal_len];
        let mut q_ch = vec![0.0f32; signal_len];

        // メインパス (idx=0) - 2回の反復を生成
        // sync_symbols の符号を考慮する必要がある
        let zc = &detector.preamble_pn;
        for rep in 0..detector.config.preamble_repeat {
            let offset = rep * preamble_sym_len;
            let sign = detector.sync_symbols[rep]; // 符号を取得 (rep=0: +1.0, rep=1: -1.0)
            for k in 0..sf {
                i_ch[offset + k * spc] = zc[k].re * sign;
                q_ch[offset + k * spc] = zc[k].im * sign;
            }
        }

        // 遅延パス (idx=1 に 0.7 倍の強度で重畳) - 2回の反復を生成
        let alpha = 0.7f32;
        for rep in 0..detector.config.preamble_repeat {
            let offset = rep * preamble_sym_len;
            let sign = detector.sync_symbols[rep]; // 符号を取得
            for k in 0..sf {
                i_ch[offset + 1 + k * spc] += zc[k].re * alpha * sign;
                q_ch[offset + 1 + k * spc] += zc[k].im * alpha * sign;
            }
        }

        // 2. CIR 推定
        let mut est_cir = vec![Complex32::new(0.0, 0.0); 5];
        let mut chq = ChannelQualityEstimate::default();
        detector.estimate_channel_quality(&i_ch, &q_ch, 0, &mut est_cir, &mut chq);

        println!("Pure Unit Test CIR:");
        for (i, val) in est_cir.iter().enumerate() {
            println!(
                "  [{}] {:.4} + {:.4}j (mag={:.4})",
                i,
                val.re,
                val.im,
                val.norm()
            );
        }

        // 3. 検証: 理想的な環境なので、誤差なく抽出できるはず
        assert!((est_cir[0].norm() - 1.0).abs() < 1e-5, "Main tap error");
        assert!((est_cir[1].norm() - 0.7).abs() < 1e-5, "Delayed tap error");
        assert!(est_cir[2].norm() < 1e-5, "Ghost tap at idx 2");
    }
}
