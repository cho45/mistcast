//! パケット構造の定義
//!
//! # 簡略化パケットフォーマット (30 bytes)
//! ```text
//! [lt_meta: 3byte] [payload: 24bytes] [crc-24: 3bytes]
//!
//! lt_meta のビット割当:
//! - 上位8bit: LT k - 1 (1..=255 を表現)
//! - 下位16bit: LT seq (0..=65535)
//! ```
//!
//! プリアンブルと同期ワードは変調レイヤーで処理される。

use crate::common::crc;
use crate::params::PAYLOAD_SIZE;

/// ヘッダサイズ (lt_k + lt_seq = 3 bytes)
pub const HEADER_SIZE: usize = 3;
pub const LT_K_BITS: usize = 8;
pub const LT_SEQ_BITS: usize = 16;
pub const LT_K_MAX: usize = ((1u16 << LT_K_BITS) - 1) as usize; // 255
pub const LT_SEQ_MAX: u16 = u16::MAX; // 65535
/// CRCサイズ
pub const CRC_SIZE: usize = 3;
/// 1パケットの合計バイト数 (ヘッダ + ペイロード + CRC = 30 bytes)
pub const PACKET_BYTES: usize = HEADER_SIZE + PAYLOAD_SIZE + CRC_SIZE;

/// 音響通信パケット
#[derive(Clone, Debug, PartialEq)]
pub struct Packet {
    /// LT符号のシーケンス番号 (Fountain Codeパケット番号)
    pub lt_seq: u16,
    /// LT復号に必要なK (1..=255)
    pub lt_k: u8,
    /// ペイロードデータ (PAYLOAD_SIZE bytes)
    pub payload: [u8; PAYLOAD_SIZE],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PacketParseError {
    InvalidLength { actual: usize },
    CrcMismatch { expected: u32, actual: u32 },
}

impl Packet {
    #[inline]
    fn encode_lt_meta(lt_seq: u16, lt_k: usize) -> [u8; HEADER_SIZE] {
        assert!(
            (1..=LT_K_MAX).contains(&lt_k),
            "lt_k must be in 1..={LT_K_MAX}"
        );
        let seq_bytes = lt_seq.to_be_bytes();
        [(lt_k - 1) as u8, seq_bytes[0], seq_bytes[1]]
    }

    #[inline]
    fn decode_lt_meta(lt_meta: &[u8]) -> (u16, usize) {
        debug_assert_eq!(lt_meta.len(), HEADER_SIZE);
        let lt_k = (lt_meta[0] as usize) + 1;
        let lt_seq = u16::from_be_bytes([lt_meta[1], lt_meta[2]]);
        (lt_seq, lt_k)
    }

    /// ペイロードを指定してパケットを作成する
    pub fn new(lt_seq: u16, lt_k: usize, payload: &[u8]) -> Self {
        assert!(
            payload.len() <= PAYLOAD_SIZE,
            "ペイロードが最大サイズを超えている"
        );
        assert!(
            (1..=LT_K_MAX).contains(&lt_k),
            "lt_k must be in 1..={LT_K_MAX}"
        );
        let mut p = [0u8; PAYLOAD_SIZE];
        p[..payload.len()].copy_from_slice(payload);
        Packet {
            lt_seq,
            lt_k: lt_k as u8,
            payload: p,
        }
    }

    /// パケットをバイト列にシリアライズする (CRCを末尾に付加)
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.serialize_into(&mut buf);
        buf
    }

    /// パケットをバイト列にシリアライズする (CRCを末尾に付加, 出力バッファ再利用版)
    pub fn serialize_into(&self, out: &mut Vec<u8>) {
        out.clear();
        out.reserve(PACKET_BYTES);
        let lt_meta = Self::encode_lt_meta(self.lt_seq, self.lt_k as usize);
        out.extend_from_slice(&lt_meta);
        out.extend_from_slice(&self.payload);
        let crc = crc::crc24(out);
        out.push((crc >> 16) as u8);
        out.push((crc >> 8) as u8);
        out.push((crc & 0xFF) as u8);
    }

    /// バイト列からパケットをデシリアライズする
    pub fn deserialize(data: &[u8]) -> Result<Self, PacketParseError> {
        if data.len() != PACKET_BYTES {
            return Err(PacketParseError::InvalidLength { actual: data.len() });
        }

        // CRC検証
        let (payload_data, crc_bytes) = data.split_at(PACKET_BYTES - CRC_SIZE);
        let expected_crc =
            ((crc_bytes[0] as u32) << 16) | ((crc_bytes[1] as u32) << 8) | crc_bytes[2] as u32;
        let actual_crc = crc::crc24(payload_data);
        if actual_crc != expected_crc {
            return Err(PacketParseError::CrcMismatch {
                expected: expected_crc,
                actual: actual_crc,
            });
        }

        let (lt_seq, lt_k) = Self::decode_lt_meta(&data[0..HEADER_SIZE]);
        let mut payload = [0u8; PAYLOAD_SIZE];
        payload.copy_from_slice(&data[HEADER_SIZE..HEADER_SIZE + PAYLOAD_SIZE]);

        Ok(Packet {
            lt_seq,
            lt_k: lt_k as u8,
            payload,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// シリアライズ→デシリアライズの往復テスト
    #[test]
    fn test_serialize_deserialize() {
        let payload = b"Hello world 1234"; // 16 bytes
        let pkt = Packet::new(42, 10, payload);
        let bytes = pkt.serialize();
        assert_eq!(bytes.len(), PACKET_BYTES);

        let recovered = Packet::deserialize(&bytes).expect("デシリアライズ成功すること");
        assert_eq!(recovered, pkt);
    }

    /// CRC破損検出
    #[test]
    fn test_crc_corruption_detection() {
        let pkt = Packet::new(1, 10, b"test data!!!!!  ");
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
        let pkt = Packet::new(0, 10, &[0u8; PAYLOAD_SIZE]);
        let bytes = pkt.serialize();
        assert_eq!(bytes.len(), 30);
        assert_eq!(PACKET_BYTES, 30);
    }

    #[test]
    fn test_lt_meta_pack_unpack() {
        let pkt = Packet::new(65535, 255, &[0u8; PAYLOAD_SIZE]);
        let raw = pkt.serialize();
        let parsed = Packet::deserialize(&raw).unwrap();
        assert_eq!(parsed.lt_seq, 65535);
        assert_eq!(parsed.lt_k, 255);
    }
}
