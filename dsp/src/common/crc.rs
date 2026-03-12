//! CRC 実装（CRC-16 / CRC-24）
//!
//! - CRC-16: poly=0x5935, init=0xFFFF
//! - CRC-24: poly=0x1101DCD, init=0xB704CE
//!
//! CRC-24 多項式は、Mistcast の固定パケット長（保護対象 27B=216bit）を主眼に
//! 短尺領域での誤り検出距離を優先して選定している。

const POLY: u16 = 0x5935;
const INIT: u16 = 0xFFFF;
// CRC-24 polynomial (full form: 0x1101DCD)
// 27B(216bit)前後の固定長パケットでの検出性能を重視して採用。
const POLY24: u32 = 0x101DCD;
const INIT24: u32 = 0xB704CE;

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

/// CRC-24テーブル (コンパイル時生成)
const TABLE24: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = (i as u32) << 16;
        let mut j = 0;
        while j < 8 {
            if crc & 0x80_0000 != 0 {
                crc = (crc << 1) ^ POLY24;
            } else {
                crc <<= 1;
            }
            crc &= 0xFF_FFFF;
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

/// データ列のCRC-24を計算する (poly=0x1101DCD, init=0xB704CE)
pub fn crc24(data: &[u8]) -> u32 {
    let mut crc = INIT24;
    for &byte in data {
        let idx = ((crc >> 16) ^ byte as u32) as usize & 0xFF;
        crc = ((crc << 8) & 0xFF_FFFF) ^ TABLE24[idx];
    }
    crc & 0xFF_FFFF
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

    #[test]
    fn test_crc24_known_vector() {
        let data = b"123456789";
        assert_eq!(crc24(data), 0xEAE922);
    }
}
