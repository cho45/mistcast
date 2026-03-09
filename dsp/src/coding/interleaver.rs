//! ブロックインターリーバー
//!
//! バーストエラー (連続したビット誤り) を分散させ、
//! FECが独立したランダムエラーを処理できるようにする。

/// ブロックインターリーバー (rows × cols 行列)
pub struct BlockInterleaver {
    rows: usize,
    cols: usize,
}

impl BlockInterleaver {
    /// `rows` 行 `cols` 列のインターリーバーを作成する
    pub fn new(rows: usize, cols: usize) -> Self {
        BlockInterleaver { rows, cols }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// インターリーブ処理 (送信側)
    pub fn interleave(&self, bits: &[u8]) -> Vec<u8> {
        let total = self.rows * self.cols;
        let mut matrix = vec![0u8; total];
        for (i, &bit) in bits.iter().enumerate().take(total) {
            matrix[i] = bit;
        }
        let mut out = Vec::with_capacity(total);
        for col in 0..self.cols {
            for row in 0..self.rows {
                out.push(matrix[row * self.cols + col]);
            }
        }
        out
    }

    /// インターリーブ処理 (送信側, インプレース版)
    ///
    /// # 引数
    /// - `input`: 入力ビット列
    /// - `output`: 出力バッファ（十分なサイズが必要）
    ///
    /// # パニック
    /// output.len() < input.len() の場合パニックします
    pub fn interleave_in_place(&self, input: &[u8], output: &mut [u8]) {
        let total = self.rows * self.cols;
        assert!(output.len() >= input.len(), "output buffer too small");

        // 行列に書き込み（列優先で読み出すため、まず行優先で格納）
        let mut matrix = vec![0u8; total];
        for (i, &bit) in input.iter().enumerate().take(total) {
            matrix[i] = bit;
        }

        // 列優先で読み出し
        let mut k = 0;
        for col in 0..self.cols {
            for row in 0..self.rows {
                output[k] = matrix[row * self.cols + col];
                k += 1;
            }
        }
    }

    /// デインターリーブ処理 (受信側)
    pub fn deinterleave(&self, bits: &[u8]) -> Vec<u8> {
        let total = self.rows * self.cols;
        let mut matrix = vec![0u8; total];
        for (k, &bit) in bits.iter().enumerate().take(total) {
            let col = k / self.rows;
            let row = k % self.rows;
            matrix[row * self.cols + col] = bit;
        }
        matrix
    }

    /// デインターリーブ処理 (受信側, インプレース版)
    ///
    /// # 引数
    /// - `input`: 入力ビット列
    /// - `output`: 出力バッファ（十分なサイズが必要）
    ///
    /// # パニック
    /// output.len() < input.len() の場合パニックします
    pub fn deinterleave_in_place(&self, input: &[u8], output: &mut [u8]) {
        let total = self.rows * self.cols;
        assert!(output.len() >= input.len(), "output buffer too small");

        // 一時的な行列バッファ
        let mut matrix = vec![0u8; total];
        for (k, &bit) in input.iter().enumerate().take(total) {
            let col = k / self.rows;
            let row = k % self.rows;
            matrix[row * self.cols + col] = bit;
        }

        // 行列を出力にコピー
        output[..total].copy_from_slice(&matrix);
    }

    /// デインターリーブ処理 (受信側, f32値)
    pub fn deinterleave_f32(&self, values: &[f32]) -> Vec<f32> {
        let total = self.rows * self.cols;
        let mut matrix = vec![0.0f32; total];
        for (k, &value) in values.iter().enumerate().take(total) {
            let col = k / self.rows;
            let row = k % self.rows;
            matrix[row * self.cols + col] = value;
        }
        matrix
    }

    /// デインターリーブ処理 (受信側, f32値, インプレース版)
    ///
    /// # 引数
    /// - `input`: 入力f32値列
    /// - `output`: 出力バッファ（十分なサイズが必要）
    ///
    /// # パニック
    /// output.len() < input.len() の場合パニックします
    pub fn deinterleave_f32_in_place(&self, input: &[f32], output: &mut [f32]) {
        let total = self.rows * self.cols;
        assert!(output.len() >= input.len(), "output buffer too small");

        // 一時的な行列バッファ
        let mut matrix = vec![0.0f32; total];
        for (k, &value) in input.iter().enumerate().take(total) {
            let col = k / self.rows;
            let row = k % self.rows;
            matrix[row * self.cols + col] = value;
        }

        // 行列を出力にコピー
        output[..total].copy_from_slice(&matrix);
    }

    pub fn reset(&mut self) {}

    pub fn block_size(&self) -> usize {
        self.rows * self.cols
    }
}

/// 代数インターリーバー (Algebraic / Prime Interleaver)
///
/// `i_out = (i_in * q) % N` の数式を用いてマッピングを行う。
/// N と q は互いに素である必要がある。
pub struct AlgebraicInterleaver {
    n: usize,
    q: usize,
    q_inv: usize,
}

impl AlgebraicInterleaver {
    /// サイズ `n`、ステップ `q` の代数インターリーバを作成する
    ///
    /// # パニック
    /// `n` と `q` が互いに素でない場合、または `n == 0` の場合にパニックする
    pub fn new(n: usize, q: usize) -> Self {
        assert!(n > 0, "size must be greater than 0");

        // 互いに素であることの確認と、モジュラ逆元の計算 (拡張ユークリッドの互除法)
        let mut t_new = 1isize;
        let mut t_old = 0isize;
        let mut r_new = q as isize;
        let mut r_old = n as isize;

        while r_new != 0 {
            let quotient = r_old / r_new;

            let t_temp = t_old - quotient * t_new;
            t_old = t_new;
            t_new = t_temp;

            let r_temp = r_old - quotient * r_new;
            r_old = r_new;
            r_new = r_temp;
        }

        assert!(r_old == 1, "n and q must be coprime");

        let mut q_inv = t_old;
        if q_inv < 0 {
            q_inv += n as isize;
        }

        AlgebraicInterleaver {
            n,
            q,
            q_inv: q_inv as usize,
        }
    }

    pub fn size(&self) -> usize {
        self.n
    }

    pub fn q(&self) -> usize {
        self.q
    }

    /// インターリーブ処理 (送信側)
    pub fn interleave(&self, bits: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; self.n];
        for (i, dst) in out.iter_mut().enumerate().take(self.n) {
            let src_idx = (i * self.q) % self.n;
            if src_idx < bits.len() {
                *dst = bits[src_idx];
            }
        }
        out
    }

    /// インターリーブ処理 (送信側, インプレース版)
    pub fn interleave_in_place(&self, input: &[u8], output: &mut [u8]) {
        assert!(output.len() >= self.n, "output buffer too small");
        for (i, dst) in output.iter_mut().enumerate().take(self.n) {
            let src_idx = (i * self.q) % self.n;
            if src_idx < input.len() {
                *dst = input[src_idx];
            } else {
                *dst = 0;
            }
        }
    }

    /// デインターリーブ処理 (受信側)
    pub fn deinterleave(&self, bits: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; self.n];
        for (i, dst) in out.iter_mut().enumerate().take(self.n) {
            let src_idx = (i * self.q_inv) % self.n;
            if src_idx < bits.len() {
                *dst = bits[src_idx];
            }
        }
        out
    }

    /// デインターリーブ処理 (受信側, インプレース版)
    pub fn deinterleave_in_place(&self, input: &[u8], output: &mut [u8]) {
        assert!(output.len() >= self.n, "output buffer too small");
        for (i, dst) in output.iter_mut().enumerate().take(self.n) {
            let src_idx = (i * self.q_inv) % self.n;
            if src_idx < input.len() {
                *dst = input[src_idx];
            } else {
                *dst = 0;
            }
        }
    }

    /// デインターリーブ処理 (受信側, f32値)
    pub fn deinterleave_f32(&self, values: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n];
        for (i, dst) in out.iter_mut().enumerate().take(self.n) {
            let src_idx = (i * self.q_inv) % self.n;
            if src_idx < values.len() {
                *dst = values[src_idx];
            }
        }
        out
    }

    /// デインターリーブ処理 (受信側, f32値, インプレース版)
    pub fn deinterleave_f32_in_place(&self, input: &[f32], output: &mut [f32]) {
        assert!(output.len() >= self.n, "output buffer too small");
        for (i, dst) in output.iter_mut().enumerate().take(self.n) {
            let src_idx = (i * self.q_inv) % self.n;
            if src_idx < input.len() {
                *dst = input[src_idx];
            } else {
                *dst = 0.0;
            }
        }
    }

    pub fn reset(&mut self) {}

    pub fn block_size(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algebraic_roundtrip() {
        let il = AlgebraicInterleaver::new(348, 55);
        let input: Vec<u8> = (0..255).chain(0..93).collect();
        let interleaved = il.interleave(&input);
        let recovered = il.deinterleave(&interleaved);
        assert_eq!(recovered, input);
    }

    #[test]
    fn test_algebraic_f32_roundtrip() {
        let il = AlgebraicInterleaver::new(348, 55);
        let input: Vec<f32> = (0..348).map(|i| i as f32 * 0.1).collect();

        // Manual interleave for test setup
        let mut interleaved = vec![0.0f32; 348];
        for i in 0..348 {
            interleaved[i] = input[(i * 55) % 348];
        }

        let recovered = il.deinterleave_f32(&interleaved);
        assert_eq!(recovered, input);
    }

    #[test]
    fn test_algebraic_interleave_in_place() {
        let il = AlgebraicInterleaver::new(348, 55);
        let input: Vec<u8> = (0..255).chain(0..93).collect();

        let expected = il.interleave(&input);

        let mut output = vec![0u8; 348];
        il.interleave_in_place(&input, &mut output);

        assert_eq!(output, expected);
    }

    #[test]
    fn test_algebraic_deinterleave_in_place() {
        let il = AlgebraicInterleaver::new(348, 55);
        let input: Vec<u8> = (0..255).chain(0..93).collect();

        let interleaved = il.interleave(&input);
        let expected = il.deinterleave(&interleaved);

        let mut output = vec![0u8; 348];
        il.deinterleave_in_place(&interleaved, &mut output);

        assert_eq!(output, expected);
    }

    #[test]
    #[should_panic(expected = "n and q must be coprime")]
    fn test_algebraic_not_coprime() {
        // 348 and 12 are not coprime (gcd = 12)
        AlgebraicInterleaver::new(348, 12);
    }

    #[test]
    fn test_roundtrip() {
        let il = BlockInterleaver::new(4, 8);
        let input: Vec<u8> = (0..32u8).collect();
        let interleaved = il.interleave(&input);
        let recovered = il.deinterleave(&interleaved);
        assert_eq!(recovered, input);
    }

    #[test]
    fn test_deinterleave_f32_roundtrip() {
        let il = BlockInterleaver::new(4, 8);
        let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.25 - 3.0).collect();
        let mut interleaved = Vec::with_capacity(input.len());
        for col in 0..il.cols() {
            for row in 0..il.rows() {
                interleaved.push(input[row * il.cols() + col]);
            }
        }
        let recovered = il.deinterleave_f32(&interleaved);
        assert_eq!(recovered, input);
    }

    /// インプレースAPIのテスト: interleave_in_place
    #[test]
    fn test_interleave_in_place() {
        let il = BlockInterleaver::new(4, 8);
        let input: Vec<u8> = (0..32u8).collect();

        // 通常の処理
        let expected = il.interleave(&input);

        // インプレース処理
        let mut output = vec![0u8; expected.len()];
        il.interleave_in_place(&input, &mut output);

        assert_eq!(output, expected);
    }

    /// インプレースAPIのテスト: deinterleave_in_place
    #[test]
    fn test_deinterleave_in_place() {
        let il = BlockInterleaver::new(4, 8);
        let input: Vec<u8> = (0..32u8).collect();

        // インターリーブ
        let interleaved = il.interleave(&input);

        // 通常のデインターリーブ
        let expected = il.deinterleave(&interleaved);

        // インプレースデインターリーブ
        let mut output = vec![0u8; expected.len()];
        il.deinterleave_in_place(&interleaved, &mut output);

        assert_eq!(output, expected);
    }

    /// インプレースAPIのテスト: deinterleave_f32_in_place
    #[test]
    fn test_deinterleave_f32_in_place() {
        let il = BlockInterleaver::new(4, 8);
        let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.25 - 3.0).collect();

        // インターリーブ
        let mut interleaved = Vec::with_capacity(input.len());
        for col in 0..il.cols() {
            for row in 0..il.rows() {
                interleaved.push(input[row * il.cols() + col]);
            }
        }

        // 通常のデインターリーブ
        let expected = il.deinterleave_f32(&interleaved);

        // インプレースデインターリーブ
        let mut output = vec![0.0f32; expected.len()];
        il.deinterleave_f32_in_place(&interleaved, &mut output);

        assert_eq!(output, expected);
    }

    /// インプレースAPIの往復テスト
    #[test]
    fn test_roundtrip_in_place() {
        let il = BlockInterleaver::new(4, 8);
        let input: Vec<u8> = (0..32u8).collect();

        let mut temp1 = vec![0u8; input.len()];
        let mut temp2 = vec![0u8; input.len()];

        // インターリーブ -> デインターリーブ
        il.interleave_in_place(&input, &mut temp1);
        il.deinterleave_in_place(&temp1, &mut temp2);

        assert_eq!(&temp2[..input.len()], &input[..]);
    }

    /// インプレースAPIのf32往復テスト
    #[test]
    fn test_f32_roundtrip_in_place() {
        let il = BlockInterleaver::new(4, 8);
        let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.25 - 3.0).collect();

        let mut interleaved = vec![0.0f32; input.len()];
        let mut output = vec![0.0f32; input.len()];

        // インターリーブ（手動で、インターリーバにはf32版のinterleaveがないため）
        for col in 0..il.cols() {
            for row in 0..il.rows() {
                interleaved[col * il.rows() + row] = input[row * il.cols() + col];
            }
        }

        // デインターリーブ
        il.deinterleave_f32_in_place(&interleaved, &mut output);

        assert_eq!(&output[..input.len()], &input[..]);
    }
}
