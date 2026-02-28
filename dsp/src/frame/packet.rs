//! パケット構造の定義
//!
//! # パケットフォーマット
//! ```text
//! [プリアンブル: M系列×4周期] [同期ワード: 0xDEADBEEF 32bit]
//! [シーケンス番号: u32 4byte] [LTシーケンス番号: u32 4byte]
//! [ペイロード長: u16 2byte] [ペイロード: PAYLOAD_SIZE bytes]
//! [CRC-16: u16 2byte]
//! ```
//!
//! プリアンブルと同期ワードは変調レイヤーで処理される。
//! パケット構造体はペイロード以降を表す。

use crate::params::{PAYLOAD_SIZE, SYNC_WORD};
use crate::common::crc;

/// ヘッダサイズ (seq u32 + lt_seq u32 + payload_len u16 = 10 bytes)
pub const HEADER_SIZE: usize = 10;
/// CRCサイズ
pub const CRC_SIZE: usize = 2;
/// 1パケットの合計バイト数 (ヘッダ + ペイロード + CRC)
pub const PACKET_BYTES: usize = HEADER_SIZE + PAYLOAD_SIZE + CRC_SIZE;

/// 音響通信パケット
#[derive(Clone, Debug, PartialEq)]
pub struct Packet {
    /// パケットシーケンス番号
    pub seq: u32,
    /// LT符号のシーケンス番号 (Fountain Codeパケット番号)
    pub lt_seq: u32,
    /// ペイロードの有効バイト数
    pub payload_len: u16,
    /// ペイロードデータ (PAYLOAD_SIZE bytes, 余剰はゼロパディング)
    pub payload: [u8; PAYLOAD_SIZE],
}

impl Packet {
    /// ペイロードを指定してパケットを作成する
    pub fn new(seq: u32, lt_seq: u32, payload: &[u8]) -> Self {
        assert!(payload.len() <= PAYLOAD_SIZE, "ペイロードが最大サイズを超えている");
        let mut p = [0u8; PAYLOAD_SIZE];
        p[..payload.len()].copy_from_slice(payload);
        Packet {
            seq,
            lt_seq,
            payload_len: payload.len() as u16,
            payload: p,
        }
    }

    /// パケットをバイト列にシリアライズする (CRCを末尾に付加)
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(PACKET_BYTES);
        buf.extend_from_slice(&self.seq.to_be_bytes());
        buf.extend_from_slice(&self.lt_seq.to_be_bytes());
        buf.extend_from_slice(&self.payload_len.to_be_bytes());
        buf.extend_from_slice(&self.payload);
        let crc = crc::crc16(&buf);
        buf.push((crc >> 8) as u8);
        buf.push((crc & 0xFF) as u8);
        buf
    }

    /// バイト列からパケットをデシリアライズする
    ///
    /// CRCが不正な場合は `None` を返す
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() != PACKET_BYTES {
            return None;
        }

        // CRC検証
        let (payload_data, crc_bytes) = data.split_at(PACKET_BYTES - CRC_SIZE);
        let expected_crc = ((crc_bytes[0] as u16) << 8) | crc_bytes[1] as u16;
        if crc::crc16(payload_data) != expected_crc {
            return None;
        }

        let seq = u32::from_be_bytes(data[0..4].try_into().ok()?);
        let lt_seq = u32::from_be_bytes(data[4..8].try_into().ok()?);
        let payload_len = u16::from_be_bytes(data[8..10].try_into().ok()?);
        let mut payload = [0u8; PAYLOAD_SIZE];
        payload.copy_from_slice(&data[10..10 + PAYLOAD_SIZE]);

        Some(Packet { seq, lt_seq, payload_len, payload })
    }

    /// ペイロードの有効データを返す
    pub fn payload_data(&self) -> &[u8] {
        &self.payload[..self.payload_len as usize]
    }

    /// 同期ワードのバイト表現 (big-endian)
    pub fn sync_word_bytes() -> [u8; 4] {
        SYNC_WORD.to_be_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// シリアライズ→デシリアライズの往復テスト
    #[test]
    fn test_serialize_deserialize() {
        let payload = b"Hello, world!  !";
        let pkt = Packet::new(42, 7, payload);
        let bytes = pkt.serialize();
        assert_eq!(bytes.len(), PACKET_BYTES);

        let recovered = Packet::deserialize(&bytes).expect("デシリアライズ成功すること");
        assert_eq!(recovered, pkt);
        assert_eq!(recovered.payload_data(), payload);
    }

    /// CRC破損検出
    #[test]
    fn test_crc_corruption_detection() {
        let pkt = Packet::new(1, 0, b"test data!!!!!");  // 14バイト (≤16)
        let mut bytes = pkt.serialize();
        bytes[0] ^= 0xFF; // 先頭バイトを破損
        assert!(
            Packet::deserialize(&bytes).is_none(),
            "破損パケットのデシリアライズが失敗すること"
        );
    }

    /// パケットサイズの確認
    #[test]
    fn test_packet_size() {
        let pkt = Packet::new(0, 0, &[0u8; PAYLOAD_SIZE]);
        let bytes = pkt.serialize();
        assert_eq!(bytes.len(), PACKET_BYTES);
        assert_eq!(PACKET_BYTES, HEADER_SIZE + PAYLOAD_SIZE + CRC_SIZE);
    }

    /// ペイロード長が0のパケット
    #[test]
    fn test_empty_payload() {
        let pkt = Packet::new(0, 0, &[]);
        let bytes = pkt.serialize();
        let recovered = Packet::deserialize(&bytes).unwrap();
        assert_eq!(recovered.payload_data().len(), 0);
    }
}
