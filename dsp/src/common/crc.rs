//! CRC-16 (Koopman最適多項式, <=241bitでHD=5)
//!
//! 多項式: x^16 + x^14 + x^12 + x^11 + x^8 + x^5 + x^4 + x^2 + 1 (0x5935)
//! 初期値: 0xFFFF
//!
//! パケットの完全性確認に使用する。

const POLY: u16 = 0x5935;
const INIT: u16 = 0xFFFF;

/// CRC-16テーブル (コンパイル時生成)
const TABLE: [u16; 256] = {
    let mut table = [0u16; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = (i as u16) << 8;
        let mut j = 0;
        while j < 8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ POLY;
            } else {
                crc <<= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// データ列のCRC-16を計算する
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc = INIT;
    for &byte in data {
        let idx = ((crc >> 8) ^ byte as u16) as usize;
        crc = (crc << 8) ^ TABLE[idx];
    }
    crc
}

/// CRCを含むデータ列が正しいかを検証する
///
/// 最後の2バイトがCRC-16 (big-endian) であると仮定する
pub fn verify(data_with_crc: &[u8]) -> bool {
    if data_with_crc.len() < 2 {
        return false;
    }
    let (data, crc_bytes) = data_with_crc.split_at(data_with_crc.len() - 2);
    let expected = ((crc_bytes[0] as u16) << 8) | crc_bytes[1] as u16;
    crc16(data) == expected
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 空データのCRC
    #[test]
    fn test_empty() {
        let crc = crc16(&[]);
        assert_eq!(crc, INIT, "空データのCRCは初期値(0xFFFF)であること");
    }

    /// 既知ベクタによる確認
    /// 実装条件 (poly=0x5935, init=0xFFFF, xorout=0, non-reflected) で
    /// "123456789" の CRC は 0x772B
    #[test]
    fn test_known_vector() {
        let data = b"123456789";
        let crc = crc16(data);
        assert_eq!(crc, 0x772B, "既知ベクタとCRCが一致すること");
    }

    /// CRC追加→検証の往復確認
    #[test]
    fn test_roundtrip() {
        let data = b"Hello, acoustic world!";
        let crc = crc16(data);
        let mut packet = data.to_vec();
        packet.push((crc >> 8) as u8);
        packet.push((crc & 0xFF) as u8);
        assert!(verify(&packet), "CRC付きパケットの検証が成功すること");
    }

    /// 1ビット反転でCRC検証が失敗することを確認
    #[test]
    fn test_detect_corruption() {
        let data = b"test data";
        let crc = crc16(data);
        let mut packet = data.to_vec();
        packet.push((crc >> 8) as u8);
        packet.push((crc & 0xFF) as u8);
        // 先頭バイトを反転
        packet[0] ^= 0xFF;
        assert!(!verify(&packet), "破損データのCRC検証が失敗すること");
    }
}
