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

    pub fn reset(&mut self) {}

    pub fn block_size(&self) -> usize {
        self.rows * self.cols
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
