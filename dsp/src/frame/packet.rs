//! パケット構造の定義
//!
//! # 簡略化パケットフォーマット (20 bytes)
//! ```text
//! [lt_seq: u16 2byte] [payload: 16bytes] [crc-16: 2bytes]
//! ```
//!
//! プリアンブルと同期ワードは変調レイヤーで処理される。

use crate::common::crc;
use crate::params::PAYLOAD_SIZE;

/// ヘッダサイズ (lt_seq u16 = 2 bytes)
pub const HEADER_SIZE: usize = 2;
/// CRCサイズ
pub const CRC_SIZE: usize = 2;
/// 1パケットの合計バイト数 (ヘッダ + ペイロード + CRC = 20 bytes)
pub const PACKET_BYTES: usize = HEADER_SIZE + PAYLOAD_SIZE + CRC_SIZE;

/// 音響通信パケット
#[derive(Clone, Debug, PartialEq)]
pub struct Packet {
    /// LT符号のシーケンス番号 (Fountain Codeパケット番号)
    pub lt_seq: u16,
    /// ペイロードデータ (PAYLOAD_SIZE bytes)
    pub payload: [u8; PAYLOAD_SIZE],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PacketParseError {
    InvalidLength { actual: usize },
    CrcMismatch { expected: u16, actual: u16 },
}

impl Packet {
    /// ペイロードを指定してパケットを作成する
    pub fn new(lt_seq: u16, payload: &[u8]) -> Self {
        assert!(
            payload.len() <= PAYLOAD_SIZE,
            "ペイロードが最大サイズを超えている"
        );
        let mut p = [0u8; PAYLOAD_SIZE];
        p[..payload.len()].copy_from_slice(payload);
        Packet { lt_seq, payload: p }
    }

    /// パケットをバイト列にシリアライズする (CRCを末尾に付加)
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(PACKET_BYTES);
        buf.extend_from_slice(&self.lt_seq.to_be_bytes());
        buf.extend_from_slice(&self.payload);
        let crc = crc::crc16(&buf);
        buf.push((crc >> 8) as u8);
        buf.push((crc & 0xFF) as u8);
        buf
    }

    /// バイト列からパケットをデシリアライズする
    pub fn deserialize(data: &[u8]) -> Result<Self, PacketParseError> {
        if data.len() != PACKET_BYTES {
            return Err(PacketParseError::InvalidLength { actual: data.len() });
        }

        // CRC検証
        let (payload_data, crc_bytes) = data.split_at(PACKET_BYTES - CRC_SIZE);
        let expected_crc = ((crc_bytes[0] as u16) << 8) | crc_bytes[1] as u16;
        let actual_crc = crc::crc16(payload_data);
        if actual_crc != expected_crc {
            return Err(PacketParseError::CrcMismatch {
                expected: expected_crc,
                actual: actual_crc,
            });
        }

        let lt_seq = u16::from_be_bytes(
            data[0..2]
                .try_into()
                .map_err(|_| PacketParseError::InvalidLength { actual: data.len() })?,
        );
        let mut payload = [0u8; PAYLOAD_SIZE];
        payload.copy_from_slice(&data[2..2 + PAYLOAD_SIZE]);

        Ok(Packet { lt_seq, payload })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// シリアライズ→デシリアライズの往復テスト
    #[test]
    fn test_serialize_deserialize() {
        let payload = b"Hello world 1234"; // 16 bytes
        let pkt = Packet::new(42, payload);
        let bytes = pkt.serialize();
        assert_eq!(bytes.len(), PACKET_BYTES);

        let recovered = Packet::deserialize(&bytes).expect("デシリアライズ成功すること");
        assert_eq!(recovered, pkt);
    }

    /// CRC破損検出
    #[test]
    fn test_crc_corruption_detection() {
        let pkt = Packet::new(1, b"test data!!!!!  ");
        let mut bytes = pkt.serialize();
        bytes[0] ^= 0xFF; // 先頭バイトを破損
        assert!(
            matches!(
                Packet::deserialize(&bytes),
                Err(PacketParseError::CrcMismatch { .. })
            ),
            "破損パケットのデシリアライズが失敗すること"
        );
    }

    /// パケットサイズの確認
    #[test]
    fn test_packet_size() {
        let pkt = Packet::new(0, &[0u8; PAYLOAD_SIZE]);
        let bytes = pkt.serialize();
        assert_eq!(bytes.len(), 20);
        assert_eq!(PACKET_BYTES, 20);
    }
}
