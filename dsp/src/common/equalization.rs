use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct MmseSettings {
    pub snr_db: f32,
    pub lambda_scale: f32,
    pub lambda_floor: f32,
    pub max_inv_gain: Option<f32>,
}

impl MmseSettings {
    pub fn new(
        snr_db: f32,
        lambda_scale: f32,
        lambda_floor: f32,
        max_inv_gain: Option<f32>,
    ) -> Self {
        Self {
            snr_db,
            lambda_scale,
            lambda_floor,
            max_inv_gain,
        }
    }

    pub fn lambda_eff(self) -> f32 {
        let lambda = 10f32.powf(-self.snr_db / 10.0);
        self.lambda_scale.max(0.0) * lambda + self.lambda_floor.max(0.0)
    }
}

impl Default for MmseSettings {
    fn default() -> Self {
        Self {
            snr_db: 15.0,
            lambda_scale: 1.0,
            lambda_floor: 0.0,
            max_inv_gain: None,
        }
    }
}

/// Minimum Mean Square Error (MMSE) 基準を用いた周波数領域等化器 (FDE)。
/// Overlap-Save法を用いて、連続するストリームデータに対して線形畳み込み（デコンボリューション）を数学的に正しく適用する。
pub struct FrequencyDomainEqualizer {
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,

    /// FFTサイズ (N)
    fft_size: usize,

    /// 周波数領域でのMMSE等化重み W(k) (サイズは `fft_size`)
    /// 既に因果的FIRフィルタとして適切にウィンドウ処理され、FFTされた状態。
    weights: Vec<Complex<f32>>,

    /// Overlap-Save法におけるオーバーラップ長 (L_eq - 1)
    overlap_len: usize,

    /// 1ブロックの処理で消費する新規サンプルの数 (N - overlap_len)
    step_size: usize,

    /// 因果的FIRフィルタ化によって生じる遅延サンプル数 (M)
    filter_delay: usize,

    /// Overlap-Save法のための内部状態バッファ。
    /// 前回のブロックの末尾サンプルと新しい入力データを保持する。
    buffer: Vec<Complex<f32>>,

    /// 出力時に先頭から破棄すべき残りサンプル数（フィルタ遅延の相殺用）
    samples_to_drop: usize,

    /// これまでに入力されたトータルサンプル数
    total_input_samples: usize,

    /// これまでに出力されたトータルサンプル数
    total_output_samples: usize,

    /// `process` メソッド内でのゼロアロケーションを実現するための作業用バッファ
    scratch: Vec<Complex<f32>>,
}

/// フレームごとに呼ばれる MSE 予測用の固定長FFTインスタンス。
/// `fft_size` ごとに使い回すことで `FftPlanner` の再生成コストを避ける。
pub struct ChannelMsePredictor {
    fft_size: usize,
    fft: Arc<dyn Fft<f32>>,
    h_spectrum: Vec<Complex<f32>>,
    w_spectrum: Vec<Complex<f32>>,
}

impl ChannelMsePredictor {
    pub fn new(fft_size: usize) -> Self {
        assert!(
            fft_size.is_multiple_of(4),
            "fft_size must be a multiple of 4"
        );
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        Self {
            fft_size,
            fft,
            h_spectrum: vec![Complex::new(0.0, 0.0); fft_size],
            w_spectrum: vec![Complex::new(0.0, 0.0); fft_size],
        }
    }

    fn load_channel_spectrum(&mut self, cir: &[Complex<f32>]) {
        assert!(
            self.fft_size >= cir.len() * 2,
            "fft_size must be at least twice the CIR length"
        );
        self.h_spectrum.fill(Complex::new(0.0, 0.0));
        self.h_spectrum[..cir.len()].copy_from_slice(cir);
        self.fft.process(&mut self.h_spectrum);
    }

    /// 線形モデル Y=H*X+N に基づく FDE 経路の予測MSEを返す。
    /// スカラー比較のため、全周波数ビンの平均値を返す。
    pub fn predict_mse_fde(
        &mut self,
        cir: &[Complex<f32>],
        signal_var: f32,
        noise_var: f32,
        mmse: MmseSettings,
    ) -> f32 {
        let sx2 = signal_var.max(0.0);
        let sn2 = noise_var.max(0.0);
        self.load_channel_spectrum(cir);
        FrequencyDomainEqualizer::mmse_weights_from_h_in_place(
            &self.h_spectrum,
            &mut self.w_spectrum,
            mmse,
        );

        let mut mse_sum = 0.0f32;
        for idx in 0..self.fft_size {
            let wh_minus_1 = self.w_spectrum[idx] * self.h_spectrum[idx] - Complex::new(1.0, 0.0);
            mse_sum += wh_minus_1.norm_sqr() * sx2 + self.w_spectrum[idx].norm_sqr() * sn2;
        }
        mse_sum / self.fft_size as f32
    }

    /// 線形モデル Y=H*X+N で、等化しない経路 (W=1) の予測MSEを返す。
    /// 受信器の位相/ゲイン追従を反映するため、周波数一定の複素1タップ補償を最適化してから誤差を評価する。
    pub fn predict_mse_raw(
        &mut self,
        cir: &[Complex<f32>],
        signal_var: f32,
        noise_var: f32,
    ) -> f32 {
        let sx2 = signal_var.max(0.0);
        let sn2 = noise_var.max(0.0);
        if sx2 <= 0.0 && sn2 <= 0.0 {
            return 0.0;
        }

        self.load_channel_spectrum(cir);

        // g = argmin E[|gY - X|^2] の閉形式。Y=H*X+N の下で
        // g = sx2 * E[H*] / (sx2 * E[|H|^2] + sn2)
        let n_inv = 1.0 / self.fft_size as f32;
        let mut mean_h = Complex::new(0.0, 0.0);
        let mut mean_h2 = 0.0f32;
        for &h in &self.h_spectrum {
            mean_h += h;
            mean_h2 += h.norm_sqr();
        }
        mean_h *= n_inv;
        mean_h2 *= n_inv;

        let denom = sx2 * mean_h2 + sn2;
        let g = if denom > 1e-12 && sx2 > 0.0 {
            mean_h.conj() * (sx2 / denom)
        } else {
            Complex::new(0.0, 0.0)
        };

        let mut mse_sum = 0.0f32;
        for &h_val in &self.h_spectrum {
            let gh_minus_1 = g * h_val - Complex::new(1.0, 0.0);
            mse_sum += gh_minus_1.norm_sqr() * sx2 + g.norm_sqr() * sn2;
        }
        mse_sum * n_inv
    }
}

impl FrequencyDomainEqualizer {
    fn mmse_weights_from_h_in_place(
        h: &[Complex<f32>],
        out: &mut [Complex<f32>],
        mmse: MmseSettings,
    ) {
        assert_eq!(h.len(), out.len(), "h and out must have same length");
        let lambda_eff = mmse.lambda_eff();
        out.fill(Complex::new(0.0, 0.0));
        for (idx, &h_val) in h.iter().enumerate() {
            let mag_sq = h_val.norm_sqr();
            let denom = mag_sq + lambda_eff;
            if denom > 1e-12 {
                let mut wi = h_val.conj() / denom;
                if let Some(limit) = mmse.max_inv_gain {
                    let n = wi.norm();
                    if n > limit && n > 1e-12 {
                        wi *= limit / n;
                    }
                }
                out[idx] = wi;
            }
        }
    }

    /// 線形モデル Y=H*X+N に基づく FDE 経路の予測MSEを返す。
    /// スカラー比較のため、全周波数ビンの平均値を返す。
    pub fn predict_mse_fde(
        cir: &[Complex<f32>],
        fft_size: usize,
        signal_var: f32,
        noise_var: f32,
        mmse: MmseSettings,
    ) -> f32 {
        let mut predictor = ChannelMsePredictor::new(fft_size);
        predictor.predict_mse_fde(cir, signal_var, noise_var, mmse)
    }

    /// 線形モデル Y=H*X+N で、等化しない経路 (W=1) の予測MSEを返す。
    /// スカラー比較のため、全周波数ビンの平均値を返す。
    pub fn predict_mse_raw(
        cir: &[Complex<f32>],
        fft_size: usize,
        signal_var: f32,
        noise_var: f32,
    ) -> f32 {
        let mut predictor = ChannelMsePredictor::new(fft_size);
        predictor.predict_mse_raw(cir, signal_var, noise_var)
    }

    /// FDEの初期化と重み W(k) の事前計算を行う。
    ///
    /// # 引数
    /// * `cir` - プリアンブル等から推定されたチャネルインパルス応答。
    /// * `fft_size` - FFTのサイズ。効率のため、`fft_size >= cir.len() * 2` を満たす4の倍数（通常は2の冪乗）を強く推奨。
    /// * `snr_db` - 推定された信号対雑音比(dB)。MMSEの正則化項として働き、ノイズの過剰増幅を防ぐ。
    ///
    /// # パニック
    /// * `fft_size` が `cir.len() * 2` 未満の場合、または4の倍数でない場合。
    pub fn new(cir: &[Complex<f32>], fft_size: usize, snr_db: f32) -> Self {
        assert!(
            fft_size >= cir.len() * 2,
            "fft_size must be at least twice the CIR length"
        );
        assert!(
            fft_size.is_multiple_of(4),
            "fft_size must be a multiple of 4"
        );

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);

        // FIR等化フィルタの長さを決定
        let l_eq = fft_size / 2;
        let filter_delay = l_eq / 2;

        let overlap_len = l_eq - 1;
        let step_size = fft_size - overlap_len;
        let buffer = vec![Complex::new(0.0, 0.0); overlap_len];
        let scratch = vec![Complex::new(0.0, 0.0); fft_size];

        let mut instance = Self {
            fft,
            ifft,
            fft_size,
            weights: vec![Complex::new(0.0, 0.0); fft_size],
            overlap_len,
            step_size,
            filter_delay,
            buffer,
            samples_to_drop: filter_delay,
            total_input_samples: 0,
            total_output_samples: 0,
            scratch,
        };

        instance.set_cir(cir, snr_db);
        instance
    }

    /// 新しいCIR（チャネルインパルス応答）とSNRを用いて等化器の重みを再計算し、状態をリセットする。
    /// インスタンス（特にFFTプラン等のヒープアロケーション）を使い回したまま、
    /// 新しいパケットの受信に備えることができる。
    ///
    /// # 引数
    /// * `cir` - 新しいチャネルインパルス応答。長さは初期化時に指定した条件 (`fft_size >= cir.len() * 2`) を満たす必要がある。
    /// * `snr_db` - 推定された信号対雑音比(dB)。
    ///
    /// # パニック
    /// * `cir.len() * 2 > fft_size` の場合。
    pub fn set_cir(&mut self, cir: &[Complex<f32>], snr_db: f32) {
        self.set_cir_with_mmse(
            cir,
            MmseSettings {
                snr_db,
                ..MmseSettings::default()
            },
        );
    }

    /// `set_cir` の拡張版。MMSE正則化と逆フィルタ利得制限を細かく制御する。
    pub fn set_cir_with_mmse(&mut self, cir: &[Complex<f32>], mmse: MmseSettings) {
        assert!(
            self.fft_size >= cir.len() * 2,
            "CIR length is too large for the configured fft_size"
        );

        let l_eq = self.fft_size / 2;

        // 1. CIRをゼロパディングしてFFT
        self.scratch.fill(Complex::new(0.0, 0.0));
        self.scratch[..cir.len()].copy_from_slice(cir);
        self.fft.process(&mut self.scratch);

        // 2. 理想的なMMSE重みの算出とIFFT (weightsバッファを一時的に利用)
        Self::mmse_weights_from_h_in_place(&self.scratch, &mut self.weights, mmse);
        self.ifft.process(&mut self.weights);

        // IFFTの正規化
        let scale = 1.0 / (self.fft_size as f32);
        for x in &mut self.weights {
            *x *= scale;
        }

        // 3. 長さ l_eq の因果的FIRフィルタの抽出
        // scratchを再度一時バッファとして利用して再配置を行う
        self.scratch.fill(Complex::new(0.0, 0.0));
        self.scratch[..self.filter_delay]
            .copy_from_slice(&self.weights[self.fft_size - self.filter_delay..]);
        self.scratch[self.filter_delay..l_eq]
            .copy_from_slice(&self.weights[..l_eq - self.filter_delay]);

        // 4. Overlap-Save用の重みとしてFFT
        self.weights.copy_from_slice(&self.scratch);
        self.fft.process(&mut self.weights);

        // 内部状態を新しいパケット向けにリセット
        self.reset();
    }

    pub fn overlap_len(&self) -> usize {
        self.overlap_len
    }

    /// 任意の長さのストリーム入力サンプルを受け取り、等化処理を行う。
    /// 内部バッファにデータが蓄積され、ブロックサイズに達するごとに出力バッファへ追記される。
    /// 返り値は、今回の呼び出しで出力バッファへ追記されたサンプルの総数。
    pub fn process(&mut self, input: &[Complex<f32>], output: &mut Vec<Complex<f32>>) -> usize {
        let initial_output_len = output.len();
        self.total_input_samples += input.len();
        self.buffer.extend_from_slice(input);

        while self.buffer.len() >= self.fft_size {
            self.scratch.copy_from_slice(&self.buffer[..self.fft_size]);

            // FDE: 周波数領域での乗算
            self.fft.process(&mut self.scratch);
            for i in 0..self.fft_size {
                self.scratch[i] *= self.weights[i];
            }
            self.ifft.process(&mut self.scratch);

            // IFFTの正規化
            let scale = 1.0 / (self.fft_size as f32);
            for x in &mut self.scratch {
                *x *= scale;
            }

            // Overlap-Save: エイリアス成分を破棄し、有効なサンプルのみを抽出
            let valid_samples = &self.scratch[self.overlap_len..self.fft_size];

            // フィルタリングによって生じた遅延分のサンプルを破棄し、時間軸を揃える
            let mut start_idx = 0;
            if self.samples_to_drop > 0 {
                let drop = self.samples_to_drop.min(valid_samples.len());
                start_idx = drop;
                self.samples_to_drop -= drop;
            }

            let to_output = &valid_samples[start_idx..];
            output.extend_from_slice(to_output);
            self.total_output_samples += to_output.len();

            // バッファを進める (step_size 分のデータを消費)
            self.buffer.drain(..self.step_size);
        }
        output.len() - initial_output_len
    }

    /// ストリームの終端に達した際、内部に滞留しているサンプルを強制的に等化して出力する。
    /// このメソッドを呼ぶことで、入力サンプル数と出力サンプル数が完全に一致する。
    pub fn flush(&mut self, output: &mut Vec<Complex<f32>>) {
        let target_total_out = self.total_input_samples;
        if self.total_output_samples >= target_total_out {
            return; // 既に出力済み
        }

        let original_total_input = self.total_input_samples;
        let mut temp_out = Vec::new();

        // 目標の出力サンプル数に到達するまでゼロをパディングして処理を進める
        let zeros = vec![Complex::new(0.0, 0.0); self.step_size];
        while self.total_output_samples < target_total_out {
            self.process(&zeros, &mut temp_out);
        }

        // 過剰に出力されたゼロパディング由来のサンプルを切り詰める
        let excess = self.total_output_samples - target_total_out;
        if excess > 0 {
            temp_out.truncate(temp_out.len() - excess);
            self.total_output_samples = target_total_out;
        }

        // processによって加算された total_input_samples を論理的な値に戻す
        self.total_input_samples = original_total_input;

        output.extend_from_slice(&temp_out);
    }

    /// フィルタリングによって生じる内部遅延サンプル数を返す。
    pub fn filter_delay(&self) -> usize {
        self.filter_delay
    }

    /// 等化器の内部状態をリセットし、新しいパケットの受信に備える。
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.buffer.resize(self.overlap_len, Complex::new(0.0, 0.0));
        self.samples_to_drop = self.filter_delay;
        self.total_input_samples = 0;
        self.total_output_samples = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32, tol: f32, label: &str) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tol,
            "{}: actual={}, expected={}, diff={}, tol={}",
            label,
            actual,
            expected,
            diff,
            tol
        );
    }

    fn convolve(input: &[Complex<f32>], h: &[Complex<f32>]) -> Vec<Complex<f32>> {
        let mut y = vec![Complex::new(0.0, 0.0); input.len()];
        for i in 0..input.len() {
            for j in 0..h.len() {
                if i + j < y.len() {
                    y[i + j] += input[i] * h[j];
                }
            }
        }
        y
    }

    fn mse(a: &[Complex<f32>], b: &[Complex<f32>]) -> f32 {
        let n = a.len().min(b.len()).max(1);
        let mut acc = 0.0f32;
        for i in 0..n {
            acc += (a[i] - b[i]).norm_sqr();
        }
        acc / n as f32
    }

    fn best_scalar_compensation(
        reference: &[Complex<f32>],
        observed: &[Complex<f32>],
    ) -> Complex<f32> {
        let n = reference.len().min(observed.len());
        let mut num = Complex::new(0.0, 0.0);
        let mut den = 0.0f32;
        for i in 0..n {
            num += reference[i] * observed[i].conj();
            den += observed[i].norm_sqr();
        }
        if den > 1e-12 {
            num / den
        } else {
            Complex::new(0.0, 0.0)
        }
    }

    fn apply_scalar(input: &[Complex<f32>], g: Complex<f32>) -> Vec<Complex<f32>> {
        input.iter().map(|&x| g * x).collect()
    }

    #[test]
    fn test_predict_mse_identity_matches_closed_form() {
        let cir = vec![Complex::new(1.0, 0.0)];
        let fft_size = 64;
        let signal_var = 1.3f32;
        let noise_var = 0.2f32;
        let mmse = MmseSettings::new(0.0, 0.0, 0.25, None); // lambda_eff = 0.25
        let lambda_eff = mmse.lambda_eff();

        let mse_fde =
            FrequencyDomainEqualizer::predict_mse_fde(&cir, fft_size, signal_var, noise_var, mmse);
        let mse_raw =
            FrequencyDomainEqualizer::predict_mse_raw(&cir, fft_size, signal_var, noise_var);

        let expected_fde =
            (lambda_eff * lambda_eff * signal_var + noise_var) / (1.0 + lambda_eff).powi(2);
        let expected_raw = signal_var * noise_var / (signal_var + noise_var);

        assert_close(mse_fde, expected_fde, 1e-6, "mse_fde(identity)");
        assert_close(mse_raw, expected_raw, 1e-6, "mse_raw(identity)");
    }

    #[test]
    fn test_predict_mse_raw_matches_closed_form_flat_channel() {
        let cir = vec![Complex::new(0.5, 0.0)];
        let fft_size = 64;
        let signal_var = 2.0f32;
        let noise_var = 0.1f32;

        let mse_raw =
            FrequencyDomainEqualizer::predict_mse_raw(&cir, fft_size, signal_var, noise_var);
        let h = cir[0];
        let g = h.conj() * (signal_var / (signal_var * h.norm_sqr() + noise_var));
        let expected =
            (g * h - Complex::new(1.0, 0.0)).norm_sqr() * signal_var + g.norm_sqr() * noise_var;
        assert_close(mse_raw, expected, 1e-6, "mse_raw(flat)");
    }

    #[test]
    fn test_predict_mse_fde_noiseless_zero_lambda_recovers_channel() {
        let cir = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.3, 0.0),
            Complex::new(-0.1, 0.0),
        ];
        let fft_size = 128;
        let signal_var = 1.0f32;
        let noise_var = 0.0f32;
        let mmse = MmseSettings::new(30.0, 0.0, 0.0, None); // lambda_eff = 0

        let mse_fde =
            FrequencyDomainEqualizer::predict_mse_fde(&cir, fft_size, signal_var, noise_var, mmse);
        let mse_raw =
            FrequencyDomainEqualizer::predict_mse_raw(&cir, fft_size, signal_var, noise_var);
        assert!(
            mse_fde < 1e-4,
            "noiseless inversion should be near-perfect: {}",
            mse_fde
        );
        assert!(
            mse_fde < mse_raw,
            "FDE should beat raw in noiseless ISI channel: fde={} raw={}",
            mse_fde,
            mse_raw
        );
    }

    #[test]
    fn test_predict_mse_fde_is_not_worse_than_raw_with_matched_lambda() {
        let cir = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.4, 0.2),
            Complex::new(-0.2, 0.1),
        ];
        let fft_size = 128;
        let signal_var = 1.0f32;
        let noise_var = 0.1f32;
        // lambda_eff = noise/signal = 0.1
        let mmse = MmseSettings::new(10.0, 1.0, 0.0, None);

        let mse_fde =
            FrequencyDomainEqualizer::predict_mse_fde(&cir, fft_size, signal_var, noise_var, mmse);
        let mse_raw =
            FrequencyDomainEqualizer::predict_mse_raw(&cir, fft_size, signal_var, noise_var);
        assert!(
            mse_fde <= mse_raw + 1e-6,
            "MMSE should not underperform raw in this model: fde={} raw={}",
            mse_fde,
            mse_raw
        );
    }

    #[test]
    fn test_predict_mse_monotonic_with_noise_var() {
        let cir = vec![Complex::new(1.0, 0.0), Complex::new(0.35, -0.1)];
        let fft_size = 128;
        let signal_var = 1.0f32;
        let mmse = MmseSettings::new(10.0, 1.0, 0.0, None);

        let mse_fde_lo =
            FrequencyDomainEqualizer::predict_mse_fde(&cir, fft_size, signal_var, 0.01, mmse);
        let mse_fde_hi =
            FrequencyDomainEqualizer::predict_mse_fde(&cir, fft_size, signal_var, 0.1, mmse);
        let mse_raw_lo =
            FrequencyDomainEqualizer::predict_mse_raw(&cir, fft_size, signal_var, 0.01);
        let mse_raw_hi = FrequencyDomainEqualizer::predict_mse_raw(&cir, fft_size, signal_var, 0.1);

        assert!(
            mse_fde_hi > mse_fde_lo,
            "FDE MSE must increase with noise variance"
        );
        assert!(
            mse_raw_hi > mse_raw_lo,
            "RAW MSE must increase with noise variance"
        );
    }

    #[test]
    fn test_predict_mse_finite_with_deep_notch_channel() {
        let cir = vec![Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0)]; // bin0に深いノッチ
        let fft_size = 64;
        let signal_var = 1.0f32;
        let noise_var = 0.05f32;
        let mmse = MmseSettings::new(20.0, 1.0, 1e-4, Some(2.0));

        let mse_fde =
            FrequencyDomainEqualizer::predict_mse_fde(&cir, fft_size, signal_var, noise_var, mmse);
        let mse_raw =
            FrequencyDomainEqualizer::predict_mse_raw(&cir, fft_size, signal_var, noise_var);
        assert!(
            mse_fde.is_finite() && mse_fde >= 0.0,
            "mse_fde must be finite/non-negative"
        );
        assert!(
            mse_raw.is_finite() && mse_raw >= 0.0,
            "mse_raw must be finite/non-negative"
        );
    }

    #[test]
    fn test_predict_mse_rank_matches_measured_mse_for_block_model() {
        let cir = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.35, 0.15),
            Complex::new(-0.1, 0.05),
        ];
        let fft_size = 256;
        let signal_var = 1.0f32;
        let noise_var = 0.03f32;
        // lambda_eff ≈ noise/signal
        let mmse = MmseSettings::new(15.0, 1.0, 0.0, None);

        let pred_fde =
            FrequencyDomainEqualizer::predict_mse_fde(&cir, fft_size, signal_var, noise_var, mmse);
        let pred_raw =
            FrequencyDomainEqualizer::predict_mse_raw(&cir, fft_size, signal_var, noise_var);
        assert!(
            pred_fde < pred_raw,
            "precondition failed: pred_fde={} pred_raw={}",
            pred_fde,
            pred_raw
        );

        let tx: Vec<Complex<f32>> = (0..fft_size)
            .map(|i| {
                let x = (i as f32 * 0.07).sin();
                let y = (i as f32 * 0.11).cos();
                Complex::new(x, y)
            })
            .collect();
        let mut rx = convolve(&tx, &cir);
        // deterministic pseudo-noise
        for (i, r) in rx.iter_mut().enumerate() {
            let n1 = ((i as f32 * 1.37).sin()) * (noise_var / 2.0).sqrt();
            let n2 = ((i as f32 * 0.91).cos()) * (noise_var / 2.0).sqrt();
            r.re += n1;
            r.im += n2;
        }

        let mut fde = FrequencyDomainEqualizer::new(&cir, fft_size, 15.0);
        let mut fde_out = Vec::new();
        fde.process(&rx, &mut fde_out);
        fde.flush(&mut fde_out);
        fde_out.truncate(tx.len());

        let g_raw = best_scalar_compensation(&tx, &rx);
        let raw_out = apply_scalar(&rx, g_raw);
        let mse_fde_meas = mse(&tx, &fde_out);
        let mse_raw_meas = mse(&tx, &raw_out);
        assert!(
            mse_fde_meas <= mse_raw_meas * 1.15,
            "measured rank mismatch: mse_fde_meas={} mse_raw_meas={}",
            mse_fde_meas,
            mse_raw_meas
        );
    }

    #[test]
    fn test_predict_mse_raw_is_phase_invariant_for_unit_magnitude_channel() {
        let fft_size = 64;
        let signal_var = 1.0f32;
        let noise_var = 0.01f32;

        let cir_ref = vec![Complex::new(1.0, 0.0)];
        let cir_rot = vec![Complex::new(0.0, 1.0)];
        let mse_ref =
            FrequencyDomainEqualizer::predict_mse_raw(&cir_ref, fft_size, signal_var, noise_var);
        let mse_rot =
            FrequencyDomainEqualizer::predict_mse_raw(&cir_rot, fft_size, signal_var, noise_var);
        assert_close(mse_ref, mse_rot, 1e-6, "raw phase invariance");
    }

    #[test]
    fn test_predict_mse_raw_rank_matches_measured_scalar_compensated_mse() {
        let fft_size = 256;
        let signal_var = 1.0f32;
        let noise_var = 0.04f32;

        let cir_good = vec![Complex::new(1.0, 0.0)];
        let cir_bad = vec![Complex::new(0.35, 0.35)];
        let pred_good =
            FrequencyDomainEqualizer::predict_mse_raw(&cir_good, fft_size, signal_var, noise_var);
        let pred_bad =
            FrequencyDomainEqualizer::predict_mse_raw(&cir_bad, fft_size, signal_var, noise_var);
        assert!(
            pred_good < pred_bad,
            "prediction precondition failed: pred_good={} pred_bad={}",
            pred_good,
            pred_bad
        );

        let tx: Vec<Complex<f32>> = (0..fft_size)
            .map(|i| {
                let x = (i as f32 * 0.07).sin();
                let y = (i as f32 * 0.13).cos();
                Complex::new(x, y)
            })
            .collect();

        let mut rx_good = convolve(&tx, &cir_good);
        let mut rx_bad = convolve(&tx, &cir_bad);
        for i in 0..fft_size {
            let n1 = ((i as f32 * 1.23).sin()) * (noise_var / 2.0).sqrt();
            let n2 = ((i as f32 * 0.87).cos()) * (noise_var / 2.0).sqrt();
            let n = Complex::new(n1, n2);
            rx_good[i] += n;
            rx_bad[i] += n;
        }

        let g_good = best_scalar_compensation(&tx, &rx_good);
        let g_bad = best_scalar_compensation(&tx, &rx_bad);
        let meas_good = mse(&tx, &apply_scalar(&rx_good, g_good));
        let meas_bad = mse(&tx, &apply_scalar(&rx_bad, g_bad));
        assert!(
            meas_good < meas_bad,
            "measured rank mismatch: meas_good={} meas_bad={}",
            meas_good,
            meas_bad
        );
    }

    #[test]
    fn test_predictor_reuse_matches_static_predict_functions() {
        let fft_size = 128;
        let signal_var = 1.0f32;
        let noise_var = 0.05f32;
        let mmse = MmseSettings::new(13.0, 1.0, 1e-4, Some(2.0));

        let cir_a = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.3, 0.1),
            Complex::new(-0.1, 0.05),
        ];
        let cir_b = vec![Complex::new(0.8, -0.2), Complex::new(0.1, 0.05)];

        let mut predictor = ChannelMsePredictor::new(fft_size);

        let fde_a_inst = predictor.predict_mse_fde(&cir_a, signal_var, noise_var, mmse);
        let raw_a_inst = predictor.predict_mse_raw(&cir_a, signal_var, noise_var);
        let fde_a_static = FrequencyDomainEqualizer::predict_mse_fde(
            &cir_a, fft_size, signal_var, noise_var, mmse,
        );
        let raw_a_static =
            FrequencyDomainEqualizer::predict_mse_raw(&cir_a, fft_size, signal_var, noise_var);
        assert_close(fde_a_inst, fde_a_static, 1e-6, "fde reuse/static (A)");
        assert_close(raw_a_inst, raw_a_static, 1e-6, "raw reuse/static (A)");

        let fde_b_inst = predictor.predict_mse_fde(&cir_b, signal_var, noise_var, mmse);
        let raw_b_inst = predictor.predict_mse_raw(&cir_b, signal_var, noise_var);
        let fde_b_static = FrequencyDomainEqualizer::predict_mse_fde(
            &cir_b, fft_size, signal_var, noise_var, mmse,
        );
        let raw_b_static =
            FrequencyDomainEqualizer::predict_mse_raw(&cir_b, fft_size, signal_var, noise_var);
        assert_close(fde_b_inst, fde_b_static, 1e-6, "fde reuse/static (B)");
        assert_close(raw_b_inst, raw_b_static, 1e-6, "raw reuse/static (B)");
    }

    #[test]
    fn test_predict_mse_fde_max_inv_gain_trades_noise_and_bias_in_notch_channel() {
        let cir = vec![Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0)];
        let fft_size = 64;
        let signal_var = 1.0f32;
        let noise_var = 1e-4f32;
        let base = MmseSettings::new(20.0, 1.0, 1e-4, None);

        let mse_unlimited =
            FrequencyDomainEqualizer::predict_mse_fde(&cir, fft_size, signal_var, noise_var, base);
        let mse_limited = FrequencyDomainEqualizer::predict_mse_fde(
            &cir,
            fft_size,
            signal_var,
            noise_var,
            MmseSettings {
                max_inv_gain: Some(0.5),
                ..base
            },
        );
        assert!(
            mse_limited >= mse_unlimited - 1e-6,
            "low-noise notch should prefer unlimited inverse gain: limited={} unlimited={}",
            mse_limited,
            mse_unlimited
        );
    }

    #[test]
    fn test_predict_mse_boundary_signal_and_noise_zero() {
        let cir = vec![Complex::new(0.7, -0.2), Complex::new(0.1, 0.05)];
        let fft_size = 64;
        let mmse = MmseSettings::new(10.0, 1.0, 0.0, None);

        let m00 = FrequencyDomainEqualizer::predict_mse_fde(&cir, fft_size, 0.0, 0.0, mmse);
        let m10 = FrequencyDomainEqualizer::predict_mse_fde(&cir, fft_size, 1.0, 0.0, mmse);
        let m01 = FrequencyDomainEqualizer::predict_mse_fde(&cir, fft_size, 0.0, 0.1, mmse);
        assert!(m00.abs() < 1e-8, "mse(0,0) must be 0, got {}", m00);
        assert!(
            m10.is_finite() && m10 >= 0.0,
            "mse(1,0) must be finite/non-negative"
        );
        assert!(
            m01.is_finite() && m01 >= 0.0,
            "mse(0,0.1) must be finite/non-negative"
        );
    }

    #[test]
    fn test_predict_mse_is_reasonably_stable_across_fft_size() {
        let cir = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.33, 0.18),
            Complex::new(-0.12, 0.05),
        ];
        let signal_var = 1.0f32;
        let noise_var = 0.05f32;
        let mmse = MmseSettings::new(13.0, 1.0, 0.0, None);

        let m64 = FrequencyDomainEqualizer::predict_mse_fde(&cir, 64, signal_var, noise_var, mmse);
        let m128 =
            FrequencyDomainEqualizer::predict_mse_fde(&cir, 128, signal_var, noise_var, mmse);
        let m256 =
            FrequencyDomainEqualizer::predict_mse_fde(&cir, 256, signal_var, noise_var, mmse);

        let max_m = m64.max(m128).max(m256);
        let min_m = m64.min(m128).min(m256);
        let rel_span = (max_m - min_m) / max_m.max(1e-12);
        assert!(
            rel_span < 0.2,
            "predict_mse_fde varies too much across fft_size: m64={} m128={} m256={} rel_span={}",
            m64,
            m128,
            m256,
            rel_span
        );
    }

    #[test]
    fn test_identity_channel() {
        // CIRが [1.0] つまり無遅延・無歪みの場合
        let cir = vec![Complex::new(1.0, 0.0)];
        let fft_size = 64;
        let mut fde = FrequencyDomainEqualizer::new(&cir, fft_size, 60.0); // 高SNR

        let input: Vec<Complex<f32>> = (0..100)
            .map(|i| Complex::new((i as f32) / 100.0, -(i as f32) / 100.0))
            .collect();

        let mut output = Vec::new();
        fde.process(&input, &mut output);
        fde.flush(&mut output);

        assert_eq!(
            input.len(),
            output.len(),
            "Input and output lengths must match exactly"
        );

        // FDEを通した結果は元の入力とほぼ一致するはず
        for i in 0..input.len() {
            let diff = (input[i] - output[i]).norm();
            assert!(
                diff < 1e-4,
                "Mismatch at index {}: expected {}, got {}",
                i,
                input[i],
                output[i]
            );
        }
    }

    #[test]
    fn test_delayed_channel() {
        // CIRが [0.0, 0.0, 1.0] のように遅延している場合
        // FDEは遅延を打ち消すように働くため、出力は時間軸が補正されて入力と一致するはず。
        let cir = vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ];
        let fft_size = 64;
        let mut fde = FrequencyDomainEqualizer::new(&cir, fft_size, 60.0);

        let mut input = vec![Complex::new(0.0, 0.0); 100];
        // インパルスを入力
        input[10] = Complex::new(1.0, 0.0);

        // 受信信号はチャネルを通って遅延しているとする
        let mut rx_signal = vec![Complex::new(0.0, 0.0); 100];
        rx_signal[12] = Complex::new(1.0, 0.0); // 10 + 2 = 12

        let mut output = Vec::new();
        fde.process(&rx_signal, &mut output);
        fde.flush(&mut output);

        assert_eq!(rx_signal.len(), output.len());

        // 出力は元のインパルス位置 (index 10) に戻っているはず
        let diff = (output[10] - Complex::new(1.0, 0.0)).norm();
        assert!(
            diff < 1e-4,
            "Peak not restored properly, expected 1.0, got {}",
            output[10]
        );

        // それ以外の部分は0に近い
        let diff_other = output[12].norm();
        assert!(
            diff_other < 1e-4,
            "Delayed peak should be removed, expected 0.0, got {}",
            output[12]
        );
    }

    #[test]
    fn test_multipath_channel() {
        // マルチパス環境: 直接波と反射波
        let cir = vec![Complex::new(1.0, 0.0), Complex::new(0.5, 0.0)];
        let fft_size = 64;
        let mut fde = FrequencyDomainEqualizer::new(&cir, fft_size, 40.0);

        let mut tx_signal = vec![Complex::new(0.0, 0.0); 100];
        tx_signal[5] = Complex::new(1.0, 0.0);
        tx_signal[15] = Complex::new(-1.0, 0.0);

        // チャネルとの線形畳み込み（受信信号の生成）
        let mut rx_signal = vec![Complex::new(0.0, 0.0); 100];
        for i in 0..tx_signal.len() {
            for j in 0..cir.len() {
                if i + j < rx_signal.len() {
                    rx_signal[i + j] += tx_signal[i] * cir[j];
                }
            }
        }

        let mut output = Vec::new();
        // 小さなチャンクで入力してみる（ストリーム処理のテスト）
        for chunk in rx_signal.chunks(7) {
            fde.process(chunk, &mut output);
        }
        fde.flush(&mut output);

        assert_eq!(tx_signal.len(), output.len());

        // 等化後はマルチパスが除去され、元の送信信号が復元される
        for i in 0..tx_signal.len() {
            // FDEのFIRフィルタ近似とエッジ効果のため、パケットの極端な末尾などは誤差が出やすいが
            // 中央部分は非常に精度良く一致するはず。
            if i < 80 {
                let diff = (tx_signal[i] - output[i]).norm();
                assert!(
                    diff < 1e-2,
                    "Mismatch at index {}: expected {}, got {}",
                    i,
                    tx_signal[i],
                    output[i]
                );
            }
        }
    }

    #[test]
    fn test_stream_continuity_and_chunking() {
        let cir = vec![
            Complex::new(1.0, 0.0),
            Complex::new(-0.8, 0.0),
            Complex::new(0.4, 0.0),
        ];
        let fft_size = 128;
        let mut fde_bulk = FrequencyDomainEqualizer::new(&cir, fft_size, 30.0);
        let mut fde_chunked = FrequencyDomainEqualizer::new(&cir, fft_size, 30.0);

        let tx_signal: Vec<Complex<f32>> = (0..500)
            .map(|i| Complex::new(f32::sin(i as f32 * 0.1), f32::cos(i as f32 * 0.1)))
            .collect();

        // 畳み込みで受信信号生成
        let mut rx_signal = vec![Complex::new(0.0, 0.0); tx_signal.len() + cir.len() - 1];
        for i in 0..tx_signal.len() {
            for j in 0..cir.len() {
                rx_signal[i + j] += tx_signal[i] * cir[j];
            }
        }
        rx_signal.truncate(tx_signal.len()); // 簡単のため長さを揃える

        // 1. 一括処理
        let mut bulk_output = Vec::new();
        fde_bulk.process(&rx_signal, &mut bulk_output);
        fde_bulk.flush(&mut bulk_output);

        // 2. 素数サイズの細切れチャンク処理
        let mut chunked_output = Vec::new();
        let chunk_sizes = [3, 7, 13, 29, 101];
        let mut offset = 0;
        let mut chunk_idx = 0;
        while offset < rx_signal.len() {
            let size = chunk_sizes[chunk_idx % chunk_sizes.len()].min(rx_signal.len() - offset);
            fde_chunked.process(&rx_signal[offset..offset + size], &mut chunked_output);
            offset += size;
            chunk_idx += 1;
        }
        fde_chunked.flush(&mut chunked_output);

        assert_eq!(
            bulk_output.len(),
            chunked_output.len(),
            "Lengths must match"
        );

        // チャンク処理と一括処理の結果が数学的に完全に一致することを証明
        for i in 0..bulk_output.len() {
            let diff = (bulk_output[i] - chunked_output[i]).norm();
            assert!(
                diff < 1e-5,
                "Mismatch between bulk and chunked at {}: {} vs {}",
                i,
                bulk_output[i],
                chunked_output[i]
            );
        }
    }

    #[test]
    fn test_boundary_crossing_echo() {
        // FFTサイズ64に対し、遅延が非常に大きいエコー（15サンプル遅延）
        let mut cir = vec![Complex::new(0.0, 0.0); 16];
        cir[0] = Complex::new(1.0, 0.0);
        cir[15] = Complex::new(0.8, 0.0); // 強い遅延波

        let fft_size = 64;
        let mut fde = FrequencyDomainEqualizer::new(&cir, fft_size, 40.0);

        // 入力信号: インパルスを配置
        let mut tx_signal = vec![Complex::new(0.0, 0.0); 100];
        tx_signal[20] = Complex::new(1.0, 0.0);

        let mut rx_signal = vec![Complex::new(0.0, 0.0); 100];
        rx_signal[20] = Complex::new(1.0, 0.0);
        rx_signal[35] = Complex::new(0.8, 0.0); // エコー（20 + 15 = 35）

        // Step size は fft_size(64) - overlap_len(31) = 33
        // index 35 のエコーは、最初の処理ブロック(0~63)に含まれるが、
        // 次のインパルスを index 40 に配置した場合、そのエコーはブロック境界を跨ぐ。
        tx_signal[40] = Complex::new(-1.0, 0.0);
        rx_signal[40] = Complex::new(-1.0, 0.0);
        rx_signal[55] = Complex::new(-0.8, 0.0); // エコー (40 + 15 = 55)

        let mut output = Vec::new();
        fde.process(&rx_signal, &mut output);
        fde.flush(&mut output);

        // エコーが綺麗に消去され、元のインパルスだけが残るか確認
        assert!((output[20] - Complex::new(1.0, 0.0)).norm() < 1e-2);
        assert!(
            (output[35] - Complex::new(0.0, 0.0)).norm() < 1e-2,
            "Echo at 35 not cancelled"
        );

        assert!((output[40] - Complex::new(-1.0, 0.0)).norm() < 1e-2);
        assert!(
            (output[55] - Complex::new(0.0, 0.0)).norm() < 1e-2,
            "Echo at 55 not cancelled"
        );
    }

    #[test]
    fn test_flush_exactness_with_single_sample() {
        let cir = vec![Complex::new(1.0, 0.0)];
        let fft_size = 64;
        let mut fde = FrequencyDomainEqualizer::new(&cir, fft_size, 50.0);

        // たった1サンプルだけ入力してフラッシュする過酷なエッジケース
        let input = vec![Complex::new(0.5, -0.5)];
        let mut output = Vec::new();
        fde.process(&input, &mut output);
        fde.flush(&mut output);

        assert_eq!(
            output.len(),
            1,
            "Output length must exactly match input length even for 1 sample"
        );
        assert!(
            (output[0] - input[0]).norm() < 1e-4,
            "The single sample must be correctly preserved"
        );
    }

    #[test]
    fn test_scale_invariance() {
        let cir = vec![Complex::new(0.5, 0.0), Complex::new(0.2, 0.0)];
        let fft_size = 64;
        let mut fde_normal = FrequencyDomainEqualizer::new(&cir, fft_size, 40.0);
        let mut fde_scaled = FrequencyDomainEqualizer::new(&cir, fft_size, 40.0);

        let input_normal = vec![
            Complex::new(1.0, -1.0),
            Complex::new(-0.5, 0.5),
            Complex::new(0.0, 1.0),
        ];

        // 信号の振幅を100倍にする
        let scale_factor = 100.0;
        let input_scaled: Vec<Complex<f32>> =
            input_normal.iter().map(|x| *x * scale_factor).collect();

        let mut out_normal = Vec::new();
        fde_normal.process(&input_normal, &mut out_normal);
        fde_normal.flush(&mut out_normal);

        let mut out_scaled = Vec::new();
        fde_scaled.process(&input_scaled, &mut out_scaled);
        fde_scaled.flush(&mut out_scaled);

        assert_eq!(out_normal.len(), out_scaled.len());

        // スケールされた入力に対する出力は、通常の出力の正確に100倍になるべき
        for i in 0..out_normal.len() {
            let expected = out_normal[i] * scale_factor;
            let diff = (expected - out_scaled[i]).norm();

            // F32の計算誤差を考慮して相対誤差で評価する
            let relative_error = diff / expected.norm().max(1e-10);
            assert!(
                relative_error < 1e-4,
                "Scale invariance failed at index {}: expected {}, got {} (relative error: {})",
                i,
                expected,
                out_scaled[i],
                relative_error
            );
        }
    }

    #[test]
    fn test_process_returns_added_count() {
        let cir = vec![Complex::new(1.0, 0.0)];
        let fft_size = 64;
        let mut fde = FrequencyDomainEqualizer::new(&cir, fft_size, 60.0);
        let overlap = fde.overlap_len();

        let mut output = Vec::new();

        // 最初の投入: fft_size に満たない場合は 0 が返るはず
        let count1 = fde.process(&[Complex::new(1.0, 0.0); 10], &mut output);
        assert_eq!(count1, 0);
        assert_eq!(output.len(), 0);

        // fft_size を超えるように投入: 最初のブロックが処理される
        // ただし、初回は filter_delay 分が samples_to_drop で削られる
        let needed = fft_size - 10;
        let count2 = fde.process(&vec![Complex::new(1.0, 0.0); needed], &mut output);

        let expected_first_block = (fft_size - overlap).saturating_sub(fde.filter_delay());
        assert_eq!(count2, expected_first_block);
        assert_eq!(output.len(), expected_first_block);
    }
}
