//! ブロックインターリーバー
//!
//! バーストエラー (連続したビット誤り) を分散させ、
//! FECが独立したランダムエラーを処理できるようにする。
//!
//! 動作原理:
//!   エンコード: 行単位で書き込み → 列単位で読み出し
//!   デコード:   列単位で書き込み → 行単位で読み出し
//!
//! これにより、バースト長 L のエラーが L/rows 間隔のエラーに分散する。

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

    /// インターリーブ処理 (送信側)
    ///
    /// 入力ビット列を rows×cols 行列に行単位で書き込み、列単位で読み出す。
    /// 入力長が rows×cols に満たない場合はゼロパディング。
    pub fn interleave(&self, bits: &[u8]) -> Vec<u8> {
        let total = self.rows * self.cols;
        let mut matrix = vec![0u8; total];

        // 行単位で書き込み
        for (i, &bit) in bits.iter().enumerate().take(total) {
            matrix[i] = bit;
        }

        // 列単位で読み出し
        let mut out = Vec::with_capacity(total);
        for col in 0..self.cols {
            for row in 0..self.rows {
                out.push(matrix[row * self.cols + col]);
            }
        }
        out
    }

    /// デインターリーブ処理 (受信側)
    ///
    /// interleave の逆操作。
    pub fn deinterleave(&self, bits: &[u8]) -> Vec<u8> {
        let total = self.rows * self.cols;
        let mut matrix = vec![0u8; total];

        // 列単位で書き込み (逆操作)
        for (k, &bit) in bits.iter().enumerate().take(total) {
            let col = k / self.rows;
            let row = k % self.rows;
            matrix[row * self.cols + col] = bit;
        }

        matrix
    }

    pub fn block_size(&self) -> usize {
        self.rows * self.cols
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 往復テスト: interleave → deinterleave で元データに戻ること
    #[test]
    fn test_roundtrip() {
        let il = BlockInterleaver::new(4, 8); // 4行×8列
        let input: Vec<u8> = (0..32u8).collect();
        let interleaved = il.interleave(&input);
        let recovered = il.deinterleave(&interleaved);
        assert_eq!(recovered, input, "デインターリーブで元データに復元されること");
    }

    /// インターリーブで隣接ビットが分散されることを確認
    #[test]
    fn test_dispersion() {
        let rows = 4;
        let cols = 8;
        let il = BlockInterleaver::new(rows, cols);
        let input: Vec<u8> = (0..32u8).collect();
        let interleaved = il.interleave(&input);

        // インターリーブ前後で順序が変わっていること
        assert_ne!(interleaved, input, "インターリーブで並び替えが行われること");

        // 連続する入力ビット0,1,2,3がインターリーブ後に間隔 rows を持つこと
        // 入力[0]→matrix[0,0]→出力位置0
        // 入力[1]→matrix[0,1]→出力位置rows
        assert_eq!(interleaved[0], input[0]);
        assert_eq!(interleaved[rows], input[1]);
    }

    /// バーストエラー分散の効果確認
    /// 連続4ビットのバーストエラーがデインターリーブ後に分散されること
    #[test]
    fn test_burst_error_dispersion() {
        let rows = 4;
        let cols = 8;
        let il = BlockInterleaver::new(rows, cols);
        let input: Vec<u8> = vec![1u8; rows * cols];
        let mut interleaved = il.interleave(&input);

        // 連続4ビットのバーストエラー (位置0〜3を反転)
        for i in 0..rows {
            interleaved[i] = 0;
        }

        let recovered = il.deinterleave(&interleaved);

        // エラーがcols列に分散されていること (各列1ビットずつ)
        let error_count = recovered.iter().filter(|&&b| b == 0).count();
        assert_eq!(error_count, rows, "バーストエラーが分散されること");

        // エラービットの間隔がcols以上であること
        let error_positions: Vec<usize> = recovered
            .iter()
            .enumerate()
            .filter(|(_, &b)| b == 0)
            .map(|(i, _)| i)
            .collect();
        for w in error_positions.windows(2) {
            assert_eq!(
                w[1] - w[0],
                cols,
                "エラービットの間隔が cols={} であること",
                cols
            );
        }
    }
}
