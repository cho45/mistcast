use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

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

impl FrequencyDomainEqualizer {
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
        assert!(fft_size >= cir.len() * 2, "fft_size must be at least twice the CIR length");
        assert!(fft_size % 4 == 0, "fft_size must be a multiple of 4");

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
        assert!(self.fft_size >= cir.len() * 2, "CIR length is too large for the configured fft_size");

        let l_eq = self.fft_size / 2;

        // 1. CIRをゼロパディングしてFFT
        self.scratch.fill(Complex::new(0.0, 0.0));
        self.scratch[..cir.len()].copy_from_slice(cir);
        self.fft.process(&mut self.scratch);

        // 2. 理想的なMMSE重みの算出とIFFT (weightsバッファを一時的に利用)
        let sigma_sq = 10f32.powf(-snr_db / 10.0);
        for i in 0..self.fft_size {
            let h_val = self.scratch[i];
            let mag_sq = h_val.norm_sqr();
            let denom = mag_sq + sigma_sq;
            if denom > 1e-12 {
                self.weights[i] = h_val.conj() / denom;
            } else {
                self.weights[i] = Complex::new(0.0, 0.0);
            }
        }
        self.ifft.process(&mut self.weights);
        
        // IFFTの正規化
        let scale = 1.0 / (self.fft_size as f32);
        for x in &mut self.weights {
            *x *= scale;
        }

        // 3. 長さ l_eq の因果的FIRフィルタの抽出
        // scratchを再度一時バッファとして利用して再配置を行う
        self.scratch.fill(Complex::new(0.0, 0.0));
        self.scratch[..self.filter_delay].copy_from_slice(&self.weights[self.fft_size - self.filter_delay..]);
        self.scratch[self.filter_delay..l_eq].copy_from_slice(&self.weights[..l_eq - self.filter_delay]);

        // 4. Overlap-Save用の重みとしてFFT
        self.weights.copy_from_slice(&self.scratch);
        self.fft.process(&mut self.weights);

        // 内部状態を新しいパケット向けにリセット
        self.reset();
    }

    /// 任意の長さのストリーム入力サンプルを受け取り、等化処理を行う。
    /// 内部バッファにデータが蓄積され、ブロックサイズに達するごとに出力バッファへ追記される。
    pub fn process(&mut self, input: &[Complex<f32>], output: &mut Vec<Complex<f32>>) {
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

        assert_eq!(input.len(), output.len(), "Input and output lengths must match exactly");

        // FDEを通した結果は元の入力とほぼ一致するはず
        for i in 0..input.len() {
            let diff = (input[i] - output[i]).norm();
            assert!(diff < 1e-4, "Mismatch at index {}: expected {}, got {}", i, input[i], output[i]);
        }
    }

    #[test]
    fn test_delayed_channel() {
        // CIRが [0.0, 0.0, 1.0] のように遅延している場合
        // FDEは遅延を打ち消すように働くため、出力は時間軸が補正されて入力と一致するはず。
        let cir = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];
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
        assert!(diff < 1e-4, "Peak not restored properly, expected 1.0, got {}", output[10]);
        
        // それ以外の部分は0に近い
        let diff_other = output[12].norm();
        assert!(diff_other < 1e-4, "Delayed peak should be removed, expected 0.0, got {}", output[12]);
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
                assert!(diff < 1e-2, "Mismatch at index {}: expected {}, got {}", i, tx_signal[i], output[i]);
            }
        }
    }

    #[test]
    fn test_stream_continuity_and_chunking() {
        let cir = vec![Complex::new(1.0, 0.0), Complex::new(-0.8, 0.0), Complex::new(0.4, 0.0)];
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

        assert_eq!(bulk_output.len(), chunked_output.len(), "Lengths must match");
        
        // チャンク処理と一括処理の結果が数学的に完全に一致することを証明
        for i in 0..bulk_output.len() {
            let diff = (bulk_output[i] - chunked_output[i]).norm();
            assert!(diff < 1e-5, "Mismatch between bulk and chunked at {}: {} vs {}", i, bulk_output[i], chunked_output[i]);
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
        assert!((output[35] - Complex::new(0.0, 0.0)).norm() < 1e-2, "Echo at 35 not cancelled");
        
        assert!((output[40] - Complex::new(-1.0, 0.0)).norm() < 1e-2);
        assert!((output[55] - Complex::new(0.0, 0.0)).norm() < 1e-2, "Echo at 55 not cancelled");
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

        assert_eq!(output.len(), 1, "Output length must exactly match input length even for 1 sample");
        assert!((output[0] - input[0]).norm() < 1e-4, "The single sample must be correctly preserved");
    }

    #[test]
    fn test_scale_invariance() {
        let cir = vec![Complex::new(0.5, 0.0), Complex::new(0.2, 0.0)];
        let fft_size = 64;
        let mut fde_normal = FrequencyDomainEqualizer::new(&cir, fft_size, 40.0);
        let mut fde_scaled = FrequencyDomainEqualizer::new(&cir, fft_size, 40.0);

        let input_normal = vec![Complex::new(1.0, -1.0), Complex::new(-0.5, 0.5), Complex::new(0.0, 1.0)];
        
        // 信号の振幅を100倍にする
        let scale_factor = 100.0;
        let input_scaled: Vec<Complex<f32>> = input_normal.iter()
            .map(|x| *x * scale_factor)
            .collect();

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
            assert!(relative_error < 1e-4, 
                "Scale invariance failed at index {}: expected {}, got {} (relative error: {})", 
                i, expected, out_scaled[i], relative_error);
        }
    }
}
