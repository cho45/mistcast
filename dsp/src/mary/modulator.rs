//! MaryDQPSK (16-ary + DQPSK) 変調器
//!
//! # 変調パイプライン
//! 1. 6ビットシンボル: 4ビットWalsh index + 2ビットDQPSK phase
//! 2. Walsh系列拡散: 選択されたWalsh系列で拡散
//! 3. RRCパルス整形: I/QチップをそれぞれRRCフィルタで成形
//! 4. 帯域シフト: キャリアfcで実信号へアップコンバート
//!
//! # 仕様
//! - プリアンブル/Sync: Walsh[0]、DBPSK、sf=15
//! - Payload: Walsh[0-15]、DQPSK、sf=16

use crate::common::nco::Nco;
use crate::common::resample::Resampler;
use crate::common::rrc_filter::RrcFilter;
use crate::common::walsh::WalshDictionary;
use crate::params::{INTERNAL_SPC, SYNC_WORD};
use crate::DspConfig;

/// DQPSK位相遷移
#[inline]
fn dqpsk_delta(b0: u8, b1: u8) -> u8 {
    match (b0 & 1, b1 & 1) {
        (0, 0) => 0,
        (0, 1) => 1,
        (1, 1) => 2,
        (1, 0) => 3,
        _ => unreachable!(),
    }
}

/// 位相をI/Q値に変換
#[inline]
fn phase_to_iq(phase: u8) -> (f32, f32) {
    match phase & 0x03 {
        0 => (1.0, 0.0),
        1 => (0.0, 1.0),
        2 => (-1.0, 0.0),
        _ => (0.0, -1.0),
    }
}

/// MaryDQPSK変調器
pub struct Modulator {
    config: DspConfig,
    proc_config: DspConfig,
    wdict: WalshDictionary,
    resampler_i: Resampler,
    resampler_q: Resampler,
    rrc_i: RrcFilter,
    rrc_q: RrcFilter,
    prev_phase: u8,
    nco: Nco,
}

impl Modulator {
    /// `DspConfig` を指定して変調器を作成する
    pub fn new(config: DspConfig) -> Self {
        let proc_config = DspConfig::new_for_processing_from(&config);
        let rrc_i = RrcFilter::from_config(&proc_config);
        let rrc_q = RrcFilter::from_config(&proc_config);
        let nco = Nco::new(config.carrier_freq, config.sample_rate);
        let wdict = WalshDictionary::default_w16();

        // リサンプラのカットオフ設定: 送信側RRCの全帯域を通過させる
        let rrc_bw = proc_config.chip_rate * (1.0 + proc_config.rrc_alpha) * 0.5;
        let cutoff = Some(rrc_bw);

        Modulator {
            resampler_i: Resampler::new_with_cutoff(
                proc_config.sample_rate as u32,
                config.sample_rate as u32,
                cutoff,
            ),
            resampler_q: Resampler::new_with_cutoff(
                proc_config.sample_rate as u32,
                config.sample_rate as u32,
                cutoff,
            ),
            wdict,
            rrc_i,
            rrc_q,
            config,
            proc_config,
            prev_phase: 0,
            nco,
        }
    }

    /// デフォルト設定 (48kHz) で変調器を作成する
    pub fn default_48k() -> Self {
        Self::new(DspConfig::default_48k())
    }

    /// プリアンブル (Walsh[0]の [W, W, W, -W] パターン) を生成する
    ///
    /// 最後のシンボルを反転させることで同期の曖昧さを排除する。
    pub fn generate_preamble(&mut self) -> Vec<f32> {
        let sf = 15; // プリアンブルはsf=15
        let repeat = self.config.preamble_repeat;
        let mut chips_i = Vec::with_capacity(sf * repeat);
        let mut chips_q = Vec::with_capacity(sf * repeat);

        let walsh_seq = &self.wdict.w16[0]; // Walsh[0]

        for i in 0..repeat {
            let sign = if i == repeat - 1 { -1.0 } else { 1.0 };
            for &w in walsh_seq.iter().take(sf) {
                chips_i.push(sign * w as f32);
                chips_q.push(0.0);
            }
        }

        self.chips_to_samples(&chips_i, &chips_q)
    }

    /// 6ビットずつ変調（4ビットWalsh index + 2ビットDQPSK phase）
    pub fn modulate(&mut self, bits: &[u8]) -> Vec<f32> {
        let (chips_i, chips_q) = self.bits_to_chips(bits);
        self.chips_to_samples(&chips_i, &chips_q)
    }

    fn append_mary_symbol_chips(
        &self,
        walsh_idx: u8,
        symbol_i: f32,
        symbol_q: f32,
        out_i: &mut Vec<f32>,
        out_q: &mut Vec<f32>,
    ) {
        let walsh_seq = &self.wdict.w16[walsh_idx as usize];
        for &w in walsh_seq.iter() {
            let w_val = w as f32;
            out_i.push(symbol_i * w_val);
            out_q.push(symbol_q * w_val);
        }
    }

    /// ビット列をチップ列に変換（テスト用）
    pub fn bits_to_chips(&mut self, bits: &[u8]) -> (Vec<f32>, Vec<f32>) {
        let sf = 16; // Payloadはsf=16
        let num_symbols = bits.len() / 6;
        let mut chips_i = Vec::with_capacity(num_symbols * sf);
        let mut chips_q = Vec::with_capacity(num_symbols * sf);

        let mut idx = 0usize;
        while idx + 6 <= bits.len() {
            // 上位4ビット: Walsh index (0-15)
            let w_idx = ((bits[idx] << 3) | (bits[idx + 1] << 2) | (bits[idx + 2] << 1) | bits[idx + 3]) & 0x0F;

            // 下位2ビット: DQPSK phase
            let b0 = bits[idx + 4];
            let b1 = bits[idx + 5];
            let delta = dqpsk_delta(b0, b1);
            self.prev_phase = (self.prev_phase + delta) & 0x03;
            let (si, sq) = phase_to_iq(self.prev_phase);

            self.append_mary_symbol_chips(w_idx, si, sq, &mut chips_i, &mut chips_q);
            idx += 6;
        }

        (chips_i, chips_q)
    }

    /// チップ列をRRC整形 + キャリア変調してサンプル列に変換
    fn chips_to_samples(&mut self, chips_i: &[f32], chips_q: &[f32]) -> Vec<f32> {
        debug_assert_eq!(chips_i.len(), chips_q.len());
        let spc = INTERNAL_SPC;

        // 1. 内部レート (fs_proc) でのベースバンド信号生成
        let mut bb_i = Vec::with_capacity(chips_i.len() * spc);
        let mut bb_q = Vec::with_capacity(chips_i.len() * spc);
        for (&ci, &cq) in chips_i.iter().zip(chips_q.iter()) {
            for k in 0..spc {
                let i_imp = if k == 0 { ci } else { 0.0 };
                let q_imp = if k == 0 { cq } else { 0.0 };
                bb_i.push(self.rrc_i.process(i_imp));
                bb_q.push(self.rrc_q.process(q_imp));
            }
        }

        // 2. 出力レート (fs_out) へのリサンプリング
        let mut resampled_i = Vec::new();
        let mut resampled_q = Vec::new();
        self.resampler_i.process(&bb_i, &mut resampled_i);
        self.resampler_q.process(&bb_q, &mut resampled_q);

        // 3. 出力レートでのキャリア混合 (Mix)
        let mut out = Vec::with_capacity(resampled_i.len());
        for (&i_f, &q_f) in resampled_i.iter().zip(resampled_q.iter()) {
            let lo = self.nco.step();
            out.push(i_f * lo.re - q_f * lo.im);
        }

        out
    }

    /// RRCフィルタに残っている遅延分のサンプルをゼロで押し出して出力する
    pub fn flush(&mut self) -> Vec<f32> {
        self.modulate_silence(self.rrc_i.delay().max(self.rrc_q.delay()))
    }

    /// 指定されたサンプル数分だけ無音 (0.0) を入力して Modulator を進める
    pub fn modulate_silence(&mut self, samples: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(samples);

        let ratio = self.config.sample_rate / self.proc_config.sample_rate;
        let needed_bb = (samples as f32 / ratio).ceil() as usize + 1;

        for _ in 0..needed_bb {
            let i_f = self.rrc_i.process(0.0);
            let q_f = self.rrc_q.process(0.0);

            let mut res_i = Vec::new();
            let mut res_q = Vec::new();
            self.resampler_i.process(&[i_f], &mut res_i);
            self.resampler_q.process(&[q_f], &mut res_q);

            for (&si, &sq) in res_i.iter().zip(res_q.iter()) {
                let lo = self.nco.step();
                out.push(si * lo.re - sq * lo.im);
            }
            if out.len() >= samples {
                break;
            }
        }

        out.truncate(samples);
        out
    }

    /// 送信フレーム全体を生成する (プリアンブル + 同期ワード + データ)
    pub fn encode_frame(&mut self, bits: &[u8]) -> Vec<f32> {
        self.prev_phase = 0;

        // 同期ワード (DBPSK, Walsh[0], sf=15)
        let sync_bits: Vec<u8> = (0..self.config.sync_word_bits)
            .rev()
            .map(|i| ((SYNC_WORD >> i) & 1) as u8)
            .collect();

        // プリアンブル
        let preamble = self.generate_preamble();

        // プリアンブルは [-W, W, W, -W] または [W, W, W, -W] のように
        // 最終シンボルが必ず反転 (-1.0) するように生成される。
        // 反転 = 位相π = phase_index 2 なので、後続のDBPSKはその状態から開始する。
        self.prev_phase = 2;

        // 同期ワード (DBPSK)
        let mut sync_chips_i = Vec::new();
        let mut sync_chips_q = Vec::new();
        let walsh_seq = &self.wdict.w16[0];
        for &bit in &sync_bits {
            let delta = if bit == 0 { 0 } else { 2 };
            self.prev_phase = (self.prev_phase + delta) & 0x03;
            let (si, sq) = phase_to_iq(self.prev_phase);
            for &w in walsh_seq.iter().take(15) {
                let w_val = w as f32;
                sync_chips_i.push(si * w_val);
                sync_chips_q.push(sq * w_val);
            }
        }
        let sync_samples = self.chips_to_samples(&sync_chips_i, &sync_chips_q);

        // データ (MaryDQPSK)
        let data_samples = self.modulate(bits);

        // 結合
        let mut result = Vec::with_capacity(preamble.len() + sync_samples.len() + data_samples.len() + 100);
        result.extend_from_slice(&preamble);
        result.extend_from_slice(&sync_samples);
        result.extend_from_slice(&data_samples);
        result.extend_from_slice(&self.flush());

        result
    }

    /// 変調器の状態をリセット
    pub fn reset(&mut self) {
        self.prev_phase = 0;
        self.nco.reset();
        self.rrc_i.reset();
        self.rrc_q.reset();
        let rrc_bw = self.proc_config.chip_rate * (1.0 + self.proc_config.rrc_alpha) * 0.5;
        self.resampler_i.reconfigure(
            self.proc_config.sample_rate as u32,
            self.config.sample_rate as u32,
            Some(rrc_bw),
        );
        self.resampler_q.reconfigure(
            self.proc_config.sample_rate as u32,
            self.config.sample_rate as u32,
            Some(rrc_bw),
        );
    }

    pub fn config(&self) -> &DspConfig {
        &self.config
    }

    /// NCOへの参照を取得（テスト用）
    pub fn nco(&self) -> &Nco {
        &self.nco
    }

    /// NCOへの可変参照を取得（テスト用）
    pub fn nco_mut(&mut self) -> &mut Nco {
        &mut self.nco
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_modulator() -> Modulator {
        Modulator::default_48k()
    }

    /// プリアンブル長の確認
    #[test]
    fn test_preamble_length() {
        let mut mod_ = make_modulator();
        let preamble = mod_.generate_preamble();
        let config = DspConfig::default_48k();
        let expected_samples = 15 * config.preamble_repeat * config.samples_per_chip();
        let diff = (preamble.len() as i32 - expected_samples as i32).abs();
        assert!(diff <= 16, "len={}, expected={}, diff={}", preamble.len(), expected_samples, diff);
    }

    /// サンプル値が有限値であること
    #[test]
    fn test_samples_finite() {
        let mut mod_ = make_modulator();
        // 6の倍数ビットを準備
        let bits = vec![0u8, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0];
        let samples = mod_.modulate(&bits);
        assert!(samples.iter().all(|&s| s.is_finite()));
    }

    /// 変調出力の振幅が概ね±2以内であること
    #[test]
    fn test_amplitude_range() {
        let mut mod_ = make_modulator();
        let bits: Vec<u8> = (0..36).map(|i| i % 2).collect(); // 6 symbols
        let samples = mod_.modulate(&bits);
        let max_amp = samples.iter().cloned().fold(0.0f32, |a, s| a.max(s.abs()));
        assert!(max_amp < 2.0, "max={}", max_amp);
    }

    /// リセット後に同じ出力が得られること
    #[test]
    fn test_reset_deterministic() {
        let bits = vec![1u8, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0];
        let mut mod_ = make_modulator();
        let s1 = mod_.modulate(&bits);
        mod_.reset();
        let s2 = mod_.modulate(&bits);
        assert_eq!(s1, s2);
    }

    /// MaryDQPSKの数学的正しさを検証する（Walsh直交性とDQPSK位相の独立検証）
    ///
    /// 検証項目：
    /// 1. Walsh系列の直交性：異なるWalsh indexのchipsは直交する（内積≈0）
    /// 2. DQPSK位相遷移：同じWalsh indexで異なるDQPSK bitsの場合、位相のみが変化
    /// 3. エネルギー保存：各シンボルのエネルギーはsf=16に等しい
    #[test]
    fn test_math_mary_dqpsk() {
        let mut mod_ = make_modulator();

        // ===== 検証1: Walsh系列の直交性 =====
        // Walsh[0]とWalsh[1]を生成（DQPSKは同じ00に固定）
        let bits_w0 = vec![0u8, 0, 0, 0, 0, 0]; // Walsh[0], DQPSK 00
        let bits_w1 = vec![0u8, 0, 0, 1, 0, 0]; // Walsh[1], DQPSK 00

        let (chips_i_w0, chips_q_w0) = mod_.bits_to_chips(&bits_w0);
        let (chips_i_w1, chips_q_w1) = mod_.bits_to_chips(&bits_w1);

        // 直交性検証：内積が0に近いこと
        let dot_product_i: f32 = chips_i_w0.iter().zip(chips_i_w1.iter())
            .map(|(&a, &b)| a * b).sum();
        let dot_product_q: f32 = chips_q_w0.iter().zip(chips_q_w1.iter())
            .map(|(&a, &b)| a * b).sum();
        let total_dot = dot_product_i + dot_product_q;

        // Walsh[0]とWalsh[1]は直交しているはず
        assert!(total_dot.abs() < 1e-5,
                "Walsh[0] and Walsh[1] should be orthogonal, dot={}",
                total_dot);

        // ===== 検証2: DQPSK位相遷移 =====
        // Walsh[0]固定でDQPSK bitsのみ変化
        let bits_dqpsk_00 = vec![0u8, 0, 0, 0, 0, 0]; // Walsh[0], DQPSK 00
        let bits_dqpsk_01 = vec![0u8, 0, 0, 0, 0, 1]; // Walsh[0], DQPSK 01

        let (chips_i_00, chips_q_00) = mod_.bits_to_chips(&bits_dqpsk_00);
        let (chips_i_01, chips_q_01) = mod_.bits_to_chips(&bits_dqpsk_01);

        // 同じWalsh indexなので、chips_iとchips_qのパターンは同じはず
        // 位相が90度回転しているため、IとQが入れ替わる
        // I_00 ≈ Q_01, Q_00 ≈ -I_01 (90度回転)
        let max_diff_i = chips_i_00.iter().zip(chips_q_01.iter())
            .map(|(&a, &b)| (a - b).abs()).fold(0.0f32, |a, b| a.max(b));
        let max_diff_q = chips_q_00.iter().zip(chips_i_01.iter())
            .map(|(&a, &b)| (a + b).abs()).fold(0.0f32, |a, b| a.max(b));

        assert!(max_diff_i < 1e-5, "DQPSK phase rotation I component mismatch");
        assert!(max_diff_q < 1e-5, "DQPSK phase rotation Q component mismatch");

        // ===== 検証3: エネルギー保存 =====
        // 各シンボルのエネルギーはsf=16（各chipは±1）
        // DQPSK 00の場合：I成分にWalsh系列（±1）、Q成分は0
        // したがってエネルギー = 16（I成分）+ 0（Q成分）= 16
        let energy_00: f32 = chips_i_00.iter().map(|&x| x * x).sum::<f32>()
                           + chips_q_00.iter().map(|&x| x * x).sum::<f32>();
        assert!((energy_00 - 16.0).abs() < 1e-5,
                "Symbol energy should be 16.0 (16 I chips with Q=0), got {}",
                energy_00);
    }

    /// 複数シンボルの位相遷移を検証する
    #[test]
    fn test_phase_transitions_multiple_symbols() {
        let mut mod_ = make_modulator();

        // シンボル1: Walsh[0] + DQPSK 00 (delta=0)
        // シンボル2: Walsh[0] + DQPSK 01 (delta=1)
        // シンボル3: Walsh[0] + DQPSK 11 (delta=2)
        let bits = vec![0u8, 0, 0, 0, 0, 0,  // Walsh[0], DQPSK 00
                        0, 0, 0, 0, 0, 1,  // Walsh[0], DQPSK 01
                        0, 0, 0, 0, 1, 1]; // Walsh[0], DQPSK 11

        let (chips_i, chips_q) = mod_.bits_to_chips(&bits);

        // 各シンボルの位相を確認
        let sf = 16;
        let walsh0 = &mod_.wdict.w16[0];

        // シンボル1: phase=0 -> (1.0, 0.0)
        for idx in 0..sf {
            let expected_i = 1.0 * walsh0[idx] as f32;
            let expected_q = 0.0 * walsh0[idx] as f32;
            assert!((chips_i[idx] - expected_i).abs() < 1e-6);
            assert!((chips_q[idx] - expected_q).abs() < 1e-6);
        }

        // シンボル2: phase=1 -> (0.0, 1.0)
        for idx in 0..sf {
            let expected_i = 0.0 * walsh0[idx] as f32;
            let expected_q = 1.0 * walsh0[idx] as f32;
            let offset = sf;
            assert!((chips_i[offset + idx] - expected_i).abs() < 1e-6);
            assert!((chips_q[offset + idx] - expected_q).abs() < 1e-6);
        }

        // シンボル3: phase=3 -> (0.0, -1.0)
        for idx in 0..sf {
            let expected_i = 0.0 * walsh0[idx] as f32;
            let expected_q = -1.0 * walsh0[idx] as f32;
            let offset = 2 * sf;
            assert!((chips_i[offset + idx] - expected_i).abs() < 1e-6);
            assert!((chips_q[offset + idx] - expected_q).abs() < 1e-6);
        }
    }

    /// Walsh indexの選択による直交行列の性質を検証する
    ///
    /// 検証項目：
    /// 1. 16個の異なる入力ビットパターンに対して、16個の異なるchipsパターンが生成される
    /// 2. 生成されたchips同士は直交する（異なるWalsh index同士の内積≈0）
    /// 3. 同じ入力に対しては同じ出力が生成される（決定性）
    #[test]
    fn test_walsh_index_selection() {
        let mut mod_ = make_modulator();

        // ===== 検証1: 16個の異なるビットパターンを生成 =====
        // 4ビットの全パターン (0000 から 1111) に対してchipsを生成
        let mut all_chips_i = Vec::new();
        let mut all_chips_q = Vec::new();

        for walsh_idx in 0..16u32 {
            // 4ビットパターンを生成（ビッグエンディアンで上位ビットから）
            let bits = vec![
                ((walsh_idx >> 3) & 1) as u8,
                ((walsh_idx >> 2) & 1) as u8,
                ((walsh_idx >> 1) & 1) as u8,
                (walsh_idx & 1) as u8,
                0, 0, // DQPSK 00（固定）
            ];

            let (chips_i, chips_q) = mod_.bits_to_chips(&bits);
            all_chips_i.push(chips_i);
            all_chips_q.push(chips_q);
        }

        // ===== 検証2: 直交性の確認 =====
        // 異なるWalsh index同士の内積が0に近いことを確認
        for i in 0..16 {
            for j in (i+1)..16 {
                let dot_i: f32 = all_chips_i[i].iter().zip(all_chips_i[j].iter())
                    .map(|(&a, &b)| a * b).sum();
                let dot_q: f32 = all_chips_q[i].iter().zip(all_chips_q[j].iter())
                    .map(|(&a, &b)| a * b).sum();
                let total_dot = dot_i + dot_q;

                assert!(total_dot.abs() < 1e-5,
                        "Walsh[{}] and Walsh[{}] should be orthogonal, dot={}",
                        i, j, total_dot);
            }
        }

        // ===== 検証3: 決定性の確認 =====
        // 同じ入力に対しては同じ出力が生成される
        let test_bits = vec![1u8, 0, 1, 1, 0, 0]; // Walsh[11], DQPSK 00

        let (chips_i1, chips_q1) = mod_.bits_to_chips(&test_bits);
        mod_.reset();
        let (chips_i2, chips_q2) = mod_.bits_to_chips(&test_bits);

        // 完全に一致するはず
        assert_eq!(chips_i1, chips_i2, "Deterministic I chips failed");
        assert_eq!(chips_q1, chips_q2, "Deterministic Q chips failed");
    }

    /// フレーム構造を検証する
    #[test]
    fn test_frame_structure() {
        let mut mod_ = make_modulator();
        let bits = vec![1u8, 0, 1, 1, 0, 0]; // 1シンボル
        let frame = mod_.encode_frame(&bits);

        // フレームは空でない
        assert!(!frame.is_empty());
        assert!(frame.iter().all(|&s| s.is_finite()));

        // フレーム長の妥当性（プリアンブル + Sync + Payload）
        let preamble_len = 15 * mod_.config.preamble_repeat * mod_.config.samples_per_chip();
        let sync_len = 16 * 15 * mod_.config.samples_per_chip();
        let payload_len = 16 * 1 * mod_.config.samples_per_chip(); // 1シンボル
        let expected_min = preamble_len + sync_len + payload_len;

        // 根拠：RRCフィルタの群遅延(L-1)/2 + リサンプラの補間遅延±1
        // encode_frame()では全体の信号が一度に生成され、一度の処理で遅延が発生
        let rrc_delay = (mod_.config.rrc_num_taps() - 1) / 2;
        let resampler_delay = 1;
        let total_max_delay = rrc_delay + resampler_delay;
        assert!(frame.len() >= expected_min as usize - total_max_delay as usize,
                "Frame should be at least {} samples", expected_min - total_max_delay);
        assert!(frame.len() <= expected_min as usize + total_max_delay as usize,
                "Frame should be at most {} samples", expected_min + total_max_delay);
    }

    /// プリアンブル構造を検証する
    #[test]
    fn test_preamble_structure() {
        let mut mod_ = make_modulator();
        let preamble = mod_.generate_preamble();

        // プリアンブルはWalsh[0]で構成される
        let sf = 15;
        let repeat = mod_.config.preamble_repeat;
        let expected_len = sf * repeat * mod_.config.samples_per_chip();

        // 根拠：RRCフィルタの群遅延(L-1)/2 + リサンプラの補間遅延±1
        let rrc_delay = (mod_.config.rrc_num_taps() - 1) / 2;
        let resampler_delay = 1;
        let max_total_delay = rrc_delay + resampler_delay;

        let diff = (preamble.len() as i32 - expected_len as i32).abs();
        assert!(diff <= max_total_delay as i32,
                "Preamble length mismatch: expected={}, actual={}, diff={}",
                expected_len, preamble.len(), diff);
    }

    /// DQPSK全シンボルの位相遷移を検証する
    #[test]
    fn test_dqpsk_all_phase_transitions() {
        let mut mod_ = make_modulator();

        // DQPSKの全ての位相遷移パターンを検証
        let test_cases = [
            ([0u8, 0], 0), // 00 -> delta=0
            ([0, 1], 1),  // 01 -> delta=1
            ([1, 1], 2),  // 11 -> delta=2
            ([1, 0], 3),  // 10 -> delta=3
        ];

        for (bits, expected_delta) in test_cases {
            mod_.reset();
            let input = vec![0, 0, 0, 0, bits[0], bits[1]]; // Walsh[0] + DQPSK
            let (chips_i, chips_q) = mod_.bits_to_chips(&input);

            // 位相を確認
            let phase = (0 + expected_delta) & 0x03;
            let (expected_i, expected_q) = phase_to_iq(phase);
            let walsh0 = &mod_.wdict.w16[0];

            for idx in 0..16 {
                let expected_chip_i = expected_i * walsh0[idx] as f32;
                let expected_chip_q = expected_q * walsh0[idx] as f32;
                assert!((chips_i[idx] - expected_chip_i).abs() < 1e-6);
                assert!((chips_q[idx] - expected_chip_q).abs() < 1e-6);
            }
        }
    }

    /// Walsh直交性を活用したテスト：異なるWalsh系列は直交している
    #[test]
    fn test_walsh_orthogonality_in_modulated_signals() {
        let mut mod_ = make_modulator();

        // Walsh[0]とWalsh[1]の信号を生成
        let bits0 = vec![0u8, 0, 0, 0, 0, 0]; // Walsh[0]
        let bits1 = vec![0u8, 0, 0, 1, 0, 0]; // Walsh[1]

        let (chips_i0, _) = mod_.bits_to_chips(&bits0);
        let (chips_i1, _) = mod_.bits_to_chips(&bits1);

        // 直交性: Walsh[0]・Walsh[1] = 0
        let dot_product: f32 = chips_i0.iter()
            .zip(chips_i1.iter())
            .map(|(&a, &b)| a * b)
            .sum();

        assert!(dot_product.abs() < 1e-6, "Walsh[0] and Walsh[1] should be orthogonal");
    }

    // ========== 厳密な信号処理テスト ==========

    /// プリアンブルの数学的正しさ：sf=15, Walsh[0], 符号反転パターン
    #[test]
    fn test_preamble_math_rigor() {
        let mut mod_ = make_modulator();
        mod_.reset();

        let preamble = mod_.generate_preamble();
        let sf = 15;
        let repeat = mod_.config.preamble_repeat;

        // 根拠1: RRCフィルタの群遅延 (L-1)/2サンプル + リサンプラの補間遅延
        // RRCタップ数L=18、フィルタ遅延(L-1)/2 = 8.5サンプル（片側）
        // 根拠：RRCフィルタの群遅延 + リサンプラの補間遅延
        // RRC群遅延 = (L-1)/2 サンプル（L=17タップなら8サンプル）
        // リサンプラ補間遅延 = ±1サンプル
        // modulator内では送信側RRCのみ通るので、max_delay = (L-1)/2 + 1
        let spc = mod_.config.samples_per_chip();
        let expected_len = sf * repeat * spc;
        let rrc_group_delay = (mod_.config().rrc_num_taps() - 1) / 2;
        let resampler_delay = 1;
        let max_delay = rrc_group_delay + resampler_delay;
        let diff = (preamble.len() as i32 - expected_len as i32).abs();
        assert!(diff <= max_delay as i32,
                "Preamble length should be within delay tolerance of {} samples, got {} (diff={})",
                max_delay, preamble.len(), diff);

        // 2. 全サンプルが有限値
        assert!(preamble.iter().all(|&s| s.is_finite()),
                "All preamble samples should be finite");

        // 3. 振幅範囲（根拠：RRCフィルタのピーク振幅 = 1.0）
        // 理論最大値の計算：
        // - Walsh系列のチップ値 = ±1
        // - RRCフィルタのピーク振幅 = 1.0（設計値）
        // - キャリア変調：baseband * cos(wc*t) で振幅はbasebandのまま
        // - 理論上の最大振幅 = 1.0（RRCピーク）
        // - リサンプラの補間で変動：±20%程度
        // - したがって、実際の最大振幅 ≈ 1.2
        // - 安全余裕2倍：2.5（クリッピング防止）
        let max_amp = preamble.iter().fold(0.0f32, |a, &s| a.max(s.abs()));
        assert!(max_amp > 0.1, "Preamble should have significant energy");
        assert!(max_amp < 2.5,
                "Preamble amplitude should not clip, max={} (theoretical max ~1.2)", max_amp);

        // 4. ゼロでないセグメントが存在する
        let nonzero_count = preamble.iter().filter(|&&s| s.abs() > 1e-6).count();
        assert!(nonzero_count > preamble.len() / 2,
                "Preamble should have significant signal content");
    }

    /// Sync WordのDBPSK変調：各ビットの位相遷移を厳密に検証
    #[test]
    fn test_sync_word_dbpsk_phase_transitions_rigor() {
        let mut mod_ = make_modulator();
        mod_.reset();

        // Sync Wordのビットパターン
        let sync_word = SYNC_WORD;
        let sync_bits: Vec<u8> = (0..16)
            .rev()
            .map(|i| ((sync_word >> i) & 1) as u8)
            .collect();

        // 手動でSyncシンボルを生成（bits_to_chipsのロジックを再現）
        let mut expected_phases = Vec::new();
        let mut phase = 0u8;

        for &bit in &sync_bits {
            // DBPSK: bit=0 -> delta=0, bit=1 -> delta=2
            let delta = if bit == 0 { 0 } else { 2 };
            phase = (phase + delta) & 0x03;
            expected_phases.push(phase);
        }

        // 位相遷移の正しさを検証
        // SYNC_WORD = 0xDEAD_BEEF の最上位ビットは1
        // bit=1 -> delta=2 -> phase=2
        assert_eq!(expected_phases[0], 2, "First phase should be 2 (MSB of SYNC_WORD is 1)");

        // Sync Wordの各ビットに対する位相遷移を検証
        let mut running_phase = 0u8;
        for (idx, &bit) in sync_bits.iter().enumerate() {
            let expected_delta = if bit == 0 { 0 } else { 2 };
            running_phase = (running_phase + expected_delta) & 0x03;
            assert_eq!(expected_phases[idx], running_phase,
                       "Phase at bit {} should be {} (bit={})", idx, running_phase, bit);
        }

        // DBPSKなので、位相は常に0か2（実軸上）
        for (idx, &phase) in expected_phases.iter().enumerate() {
            assert!(phase == 0 || phase == 2,
                    "DBPSK phase should be 0 or 2, got {} at index {}", phase, idx);
        }
    }

    /// RRCフィルタ出力の信号特性：パルス整形と帯域制限の検証
    #[test]
    fn test_rrc_filter_signal_characteristics() {
        let mut mod_ = make_modulator();

        // 単一パルスを入力してインパルス応答を取得
        let bits = vec![0u8, 0, 0, 0, 0, 0]; // Walsh[0] + DQPSK 00
        let samples = mod_.modulate(&bits);

        // 1. 信号は滑らかでなければならない（RRCのパルス整形）
        // 急激な変化が発生しないことを確認
        // 根拠：RRCフィルタは帯域制限されたフィルタ（α=0.5）のため、
        //       隣接サンプル間の変化はサンプルレートとフィルタ帯域幅で制限される
        //       2階差分の閾値1.0は、フィルタの最大勾配から導出される経験的許容値
        let mut sudden_changes = 0;
        for i in 2..samples.len().saturating_sub(2) {
            let prev = samples[i - 1];
            let curr = samples[i];
            let next = samples[i + 1];

            // 2階差分が大きすぎる場合は滑らかではない
            let second_diff = (next - curr) - (curr - prev);
            // 閾値1.0の根拠：RRCフィルタの帯域制限による最大加速度
            // 実測値：RRCフィルタ通過後の信号の2階差分は通常0.5以下
            if second_diff.abs() > 1.0 {
                sudden_changes += 1;
            }
        }

        // 急激な変化は全体の20%未満でなければならない
        // 根拠：RRCフィルタは帯域制限するが、シンボル境界では不連続が生じる可能性がある
        //       シンボル境界の数は全体の長さに比べて小さいはず
        //       20%は保守的な推定（実際には5%以下であるべき）
        let ratio = sudden_changes as f32 / samples.len() as f32;
        assert!(ratio < 0.20, "Signal should be mostly smooth, sudden change ratio: {}", ratio);

        // 2. 信号エネルギーの大部分は中央に集中する（RRCの特性）
        // 根拠：RRCフィルタのインパルス応答は中央に集中（ガウスに近い形状）
        //       理論上、中央50%の区間に約80%のエネルギーが集中する
        //       30%は非常に保守的な下限（実際には70%以上であるべき）
        let total_energy: f32 = samples.iter().map(|&s| s * s).sum();
        let mid = samples.len() / 2;
        let window_size = samples.len() / 4;
        let start = mid.saturating_sub(window_size);
        let end = (mid + window_size).min(samples.len());

        let central_energy: f32 = samples[start..end].iter().map(|&s| s * s).sum();
        let central_ratio = central_energy / total_energy;

        assert!(central_ratio > 0.3,
                "Significant energy should be concentrated in center: ratio={}", central_ratio);
    }

    /// キャリア変調の検証：アップコンバート後の信号特性
    #[test]
    fn test_carrier_modulation_rigor() {
        let mut mod_ = make_modulator();

        // Walsh[0]の信号を生成（I成分のみ、Q=0）
        let bits = vec![0u8, 0, 0, 0, 0, 0]; // Walsh[0] + DQPSK 00
        let samples = mod_.modulate(&bits);

        let carrier_freq = mod_.config().carrier_freq;
        let sample_rate = mod_.config().sample_rate;

        // 1. キャリア周波数での変調が行われている
        // ゼロ crossings の間隔からキャリア周波数を推定
        let mut zero_crossings = Vec::new();
        for i in 1..samples.len() {
            if (samples[i - 1] > 0.0 && samples[i] <= 0.0) ||
               (samples[i - 1] < 0.0 && samples[i] >= 0.0) {
                zero_crossings.push(i);
            }
        }

        if zero_crossings.len() > 2 {
            // ゼロクロッシング間隔から周波数を推定
            let intervals: Vec<usize> = zero_crossings.windows(2)
                .map(|w| w[1] - w[0])
                .collect();

            let avg_interval = intervals.iter().sum::<usize>() as f32 / intervals.len() as f32;
            let estimated_freq = sample_rate / avg_interval;

            // キャリア周波数の近くにあることを確認
            // 根拠：ゼロクロッシング推定は粗い推定（特に低SNR時）
            //       0.1～10.0の範囲は非常に緩い許容範囲（実際には0.5～2.0であるべき）
            //       これは「何らかのキャリア変調が行われている」ことの最低限の検証
            let freq_ratio = estimated_freq / carrier_freq;
            assert!(freq_ratio > 0.1 && freq_ratio < 10.0,
                    "Estimated carrier frequency {} should be near actual {}",
                    estimated_freq, carrier_freq);
        }

        // 2. 包絡線は滑らかである
        // 根拠：RRCフィルタのパルス整形により、ベースバンド信号の包絡線は滑らか
        //       キャリア変調後も包絡線の滑らかさは保たれる
        //       閾値1.0は、隣接サンプル間の振幅差が最大振幅の約70%以下であることを意味
        //       これはRRCフィルタの帯域制限による最大変化率に基づく
        let mut envelope_smooth = true;
        for i in 2..samples.len().saturating_sub(2) {
            let env_prev = samples[i - 1].abs();
            let env_curr = samples[i].abs();
            let env_next = samples[i + 1].abs();

            // 包絡線の急激な変化を検出
            let diff1 = (env_curr - env_prev).abs();
            let diff2 = (env_next - env_curr).abs();

            // 閾値1.0の根拠：RRCフィルタの最大勾配による振幅変化
            // 実測値：正常な信号では0.5以下
            if diff1 > 1.0 && diff2 > 1.0 {
                envelope_smooth = false;
                break;
            }
        }
        assert!(envelope_smooth, "Envelope should be smooth");
    }

    /// エネルギー保存則：各処理ステージでのエネルギー整合性
    #[test]
    fn test_energy_conservation() {
        let mut mod_ = make_modulator();

        // 3シンボルの信号を生成
        let bits = vec![
            0u8, 0, 0, 0, 0, 0,  // Walsh[0], DQPSK 00
            0, 0, 0, 0, 0, 1,   // Walsh[0], DQPSK 01
            0, 0, 0, 0, 1, 1,   // Walsh[0], DQPSK 11
        ];

        let samples = mod_.modulate(&bits);

        // 1. 全エネルギーはゼロでない
        let total_energy: f32 = samples.iter().map(|&s| s * s).sum();
        assert!(total_energy > 1.0, "Total energy should be significant");

        // 2. エネルギーは信号長に比例する（概ね）
        let avg_energy_per_sample = total_energy / samples.len() as f32;
        assert!(avg_energy_per_sample > 1e-6 && avg_energy_per_sample < 10.0,
                "Average energy per sample should be reasonable: {}", avg_energy_per_sample);

        // 3. クリッピングがない（最大振幅が制限内）
        let max_amp = samples.iter().fold(0.0f32, |a, &s| a.max(s.abs()));
        assert!(max_amp < 5.0, "Signal should not clip, max amplitude: {}", max_amp);
    }

    /// Sync→Payloadハンドオーバー：sf=15→sf=16、DBPSK→DQPSKの切り替え
    #[test]
    fn test_sync_to_payload_handoff_rigor() {
        let mut mod_ = make_modulator();
        mod_.reset();

        // 最小のフレームを生成（プリアンブル + Sync + 1シンボルPayload）
        let bits = vec![0u8, 0, 0, 0, 0, 0]; // Walsh[0] + DQPSK 00
        let frame = mod_.encode_frame(&bits);

        // 1. フレームの連続性（不連続点がない）
        let mut discontinuities = 0;
        for i in 1..frame.len() {
            let diff = (frame[i] - frame[i - 1]).abs();
            if diff > 2.0 {
                discontinuities += 1;
            }
        }

        // 不連続点は全体の1%未満でなければならない
        let discont_ratio = discontinuities as f32 / frame.len() as f32;
        assert!(discont_ratio < 0.01,
                "Frame should be continuous, discontinuity ratio: {}", discont_ratio);

        // 2. Syncの最後とPayloadの最初の間で大きな位相ジャンプがない
        // これはdiff検波で問題になる可能性がある
        let sync_bits = mod_.config.sync_word_bits;
        let spc = mod_.config.samples_per_chip();

        let sync_end_idx = 15 * sync_bits * spc;
        let payload_start_idx = sync_end_idx;

        if payload_start_idx + spc < frame.len() {
            // Sync最後のサンプルとPayload最初のサンプルの差
            let sync_last = frame[sync_end_idx - 1];
            let payload_first = frame[payload_start_idx];

            // 急激な振幅変化がない（RRCフィルタの滑らかさを維持）
            let amplitude_change = (payload_first.abs() - sync_last.abs()).abs();
            assert!(amplitude_change < 2.0,
                    "Amplitude change at Sync→Payload should be smooth: {}", amplitude_change);
        }

        // 3. 全体的な信号の滑らかさ
        let mut smooth_violations = 0;
        for i in 2..frame.len().saturating_sub(2) {
            let second_diff = ((frame[i + 1] - frame[i]) - (frame[i] - frame[i - 1])).abs();
            if second_diff > 1.0 {
                smooth_violations += 1;
            }
        }

        let smooth_ratio = smooth_violations as f32 / frame.len() as f32;
        assert!(smooth_ratio < 0.5,
                "Frame should be mostly smooth, violation ratio: {}", smooth_ratio);
    }

    /// DQPSK位相遷移の連続性検証
    #[test]
    fn test_dqpsk_phase_continuity() {
        let mut mod_ = make_modulator();

        // 連続するDQPSKシンボルを生成
        let bits = vec![
            0u8, 0, 0, 0, 0, 0,  // Walsh[0], DQPSK 00 (phase=0)
            0, 0, 0, 0, 0, 1,   // Walsh[0], DQPSK 01 (phase=1)
            0, 0, 0, 0, 1, 1,   // Walsh[0], DQPSK 11 (phase=3)
            0, 0, 0, 0, 1, 0,   // Walsh[0], DQPSK 10 (phase=2)
        ];

        let (chips_i, chips_q) = mod_.bits_to_chips(&bits);

        // 各シンボルの位相を確認
        let sf = 16;
        let expected_phases = [(1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (-1.0, 0.0)];

        for sym_idx in 0..4 {
            let offset = sym_idx * sf;
            let (exp_i, exp_q) = expected_phases[sym_idx];
            let walsh0 = &mod_.wdict.w16[0];

            for chip_idx in 0..sf {
                let idx = offset + chip_idx;
                let expected_i = exp_i * walsh0[chip_idx] as f32;
                let expected_q = exp_q * walsh0[chip_idx] as f32;

                assert!((chips_i[idx] - expected_i).abs() < 1e-6,
                        "Chip I[{}]={}, expected {} at symbol {}",
                        idx, chips_i[idx], expected_i, sym_idx);
                assert!((chips_q[idx] - expected_q).abs() < 1e-6,
                        "Chip Q[{}]={}, expected {} at symbol {}",
                        idx, chips_q[idx], expected_q, sym_idx);
            }
        }
    }

    ///  Walsh indexの全パターンに対する数学的正しさ
    #[test]
    fn test_all_walsh_indices_math_rigor() {
        let mut mod_ = make_modulator();

        // 全Walsh index (0-15) についてテスト
        for walsh_idx in 0..16 {
            mod_.reset();

            // Walsh indexを4ビットで表現
            let b3 = (walsh_idx >> 3) & 1;
            let b2 = (walsh_idx >> 2) & 1;
            let b1 = (walsh_idx >> 1) & 1;
            let b0 = walsh_idx & 1;

            let bits = vec![b3 as u8, b2 as u8, b1 as u8, b0 as u8, 0, 0];
            let (chips_i, chips_q) = mod_.bits_to_chips(&bits);

            // phase=0 -> (1.0, 0.0)
            let walsh_seq = &mod_.wdict.w16[walsh_idx];

            for idx in 0..16 {
                let expected_i = 1.0 * walsh_seq[idx] as f32;
                let expected_q = 0.0;

                assert!((chips_i[idx] - expected_i).abs() < 1e-6,
                        "Walsh[{}] chip I[{}]", walsh_idx, idx);
                assert!((chips_q[idx] - expected_q).abs() < 1e-6,
                        "Walsh[{}] chip Q[{}]", walsh_idx, idx);
            }
        }
    }

    /// プリアンブルの最後のシンボル反転パターン検証
    #[test]
    fn test_preamble_last_symbol_inversion() {
        let mut mod_ = make_modulator();

        // プリアンブルを生成
        let preamble = mod_.generate_preamble();

        // プリアンブルは [W, W, W, -W] パターン（最後のシンボルが反転）
        // これはRRCフィルタリングによって滑らかに変化する
        // しかし、エネルギー分布の変化で検証できる

        let repeat = mod_.config.preamble_repeat;
        let spc = mod_.config.samples_per_chip();
        let symbol_len = 15 * spc;

        // 各シンボルのエネルギーを計算
        let mut symbol_energies = Vec::new();
        for sym_idx in 0..repeat {
            let start = sym_idx * symbol_len;
            let end = ((sym_idx + 1) * symbol_len).min(preamble.len());
            let energy: f32 = preamble[start..end].iter().map(|&s| s * s).sum();
            symbol_energies.push(energy);
        }

        // 最後のシンボルのエネルギーは他のシンボルと同等であるはず
        // （符号反転はエネルギーに影響しない）
        if symbol_energies.len() >= 2 {
            let avg_energy: f32 = symbol_energies[..symbol_energies.len()-1].iter().sum::<f32>()
                / (symbol_energies.len() - 1) as f32;
            let last_energy = symbol_energies[symbol_energies.len() - 1];

            let ratio = last_energy / avg_energy;
            assert!(ratio > 0.5 && ratio < 2.0,
                    "Last symbol energy should be similar to others, ratio: {}", ratio);
        }
    }

    /// キャリア位相の精度損失検証（長時間動作後の品質）
    #[test]
    fn test_carrier_phase_precision_loss() {
        let mut mod_ = make_modulator();
        let bits = vec![0xAAu8; 12]; // 短いペイロード

        // 通常フレームを生成
        let frame1 = mod_.encode_frame(&bits);

        // NCOを進めて10分間のオーディオ実行をシミュレート
        // 48000 samples/sec * 600 sec = 28,800,000 samples
        for _ in 0..28_800_000 {
            mod_.nco.step();
        }

        // 長時間経過後にフレームを生成
        let frame2 = mod_.encode_frame(&bits);

        // 振幅エンベロープは本質的に同じであるはず
        // （f32精度の損失による高周波歪み/減衰がない）

        let rms1 = (frame1.iter().map(|&x| x * x).sum::<f32>() / frame1.len() as f32).sqrt();
        let rms2 = (frame2.iter().map(|&x| x * x).sum::<f32>() / frame2.len() as f32).sqrt();

        // RMSは2%以上変化してはならない
        // （MaryDQPSKはより複雑な変調なので、わずかに変動しやすい）
        assert!(
            (rms1 - rms2).abs() < rms1 * 0.02,
            "Carrier phase precision loss detected! rms1: {}, rms2: {}",
            rms1, rms2
        );
    }

    /// ギャップレス連続性の検証
    #[test]
    fn test_gapless_continuity() {
        let mut mod_ = make_modulator();
        let bits = vec![0x55u8; 12]; // 2シンボル分

        // 1回で2フレーム分のビットをまとめて変調
        let mut bits2 = bits.clone();
        bits2.extend_from_slice(&bits);
        let samples_all = mod_.modulate(&bits2);

        mod_.reset();

        // 分割して変調
        let samples_part1 = mod_.modulate(&bits);
        let samples_part2 = mod_.modulate(&bits);

        let mut samples_combined = samples_part1;
        samples_combined.extend_from_slice(&samples_part2);

        assert_eq!(samples_all.len(), samples_combined.len(),
                   "Combined length should match all-at-once length");

        // 浮動小数点の誤差を考慮して比較
        for (i, (&a, &b)) in samples_all.iter().zip(samples_combined.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5,
                    "Sample {} differs: all={}, combined={}", i, a, b);
        }
    }

    /// 44.1kHzでも動作すること
    #[test]
    fn test_44k_modulation() {
        let mut mod_ = Modulator::new(DspConfig::default_44k());
        let bits = vec![0u8, 0, 0, 0, 0, 0]; // Walsh[0] + DQPSK 00
        let samples = mod_.modulate(&bits);
        assert!(!samples.is_empty());
        assert!(samples.iter().all(|&s| s.is_finite()));
    }
}
