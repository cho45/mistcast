//! Fountain受理モジュール
//!
//! 復元済み Packet を FountainDecoder へ渡し、
//! 受理結果に応じた統計更新と復号完了判定を担当する。

use super::decoder_stats::DecoderStats;
use crate::coding::fountain::{
    reconstruct_packet_coefficients, FountainDecoder, FountainPacket, ReceiveOutcome,
};
use crate::frame::packet::Packet;

pub(crate) struct PacketReceiveResult {
    pub recovered_data: Option<Vec<u8>>,
}

pub(crate) fn receive_packet(
    fountain_decoder: &mut FountainDecoder,
    stats: &mut DecoderStats,
    packet: Packet,
) -> PacketReceiveResult {
    stats.last_packet_seq = Some(packet.lt_seq as u32);

    let fountain_packet = FountainPacket {
        seq: packet.lt_seq as u32,
        coefficients: reconstruct_packet_coefficients(
            packet.lt_seq as u32,
            fountain_decoder.params().k,
        ),
        data: packet.payload.to_vec(),
    };

    let outcome = fountain_decoder.receive_with_outcome(fountain_packet);
    match outcome {
        ReceiveOutcome::AcceptedRankUp => {
            stats.received_packets += 1;
            stats.last_rank_up_seq = Some(packet.lt_seq as u32);
        }
        ReceiveOutcome::AcceptedNoRankUp => {
            stats.received_packets += 1;
            stats.stalled_packets += 1;
            stats.dependent_packets += 1;
        }
        ReceiveOutcome::DuplicateSeq => {
            stats.duplicate_packets += 1;
        }
        ReceiveOutcome::InvalidPacket => {
            stats.parse_error_packets += 1;
        }
    }

    PacketReceiveResult {
        recovered_data: fountain_decoder.decode(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coding::fountain::{FountainEncoder, FountainParams};
    use crate::params::PAYLOAD_SIZE;

    #[test]
    fn test_receive_packet_updates_stats_and_recovers_data() {
        let data = vec![0x5a; PAYLOAD_SIZE];
        let params = FountainParams::new(1, PAYLOAD_SIZE);
        let mut encoder = FountainEncoder::new(&data, params.clone());
        let mut decoder = FountainDecoder::new(params);
        let mut stats = DecoderStats::new();

        let fountain_packet = encoder.next_packet();
        let packet = Packet::new(fountain_packet.seq as u16, 1, &fountain_packet.data);
        let result = receive_packet(&mut decoder, &mut stats, packet);

        assert_eq!(stats.received_packets, 1);
        assert_eq!(stats.last_packet_seq, Some(0));
        assert_eq!(stats.last_rank_up_seq, Some(0));
        assert_eq!(result.recovered_data.as_deref(), Some(data.as_slice()));
    }

    #[test]
    fn test_receive_packet_marks_duplicate_seq_without_incrementing_received() {
        let data = vec![0x5a; PAYLOAD_SIZE];
        let params = FountainParams::new(2, PAYLOAD_SIZE);
        let mut encoder = FountainEncoder::new(&data, params.clone());
        let mut decoder = FountainDecoder::new(params);
        let mut stats = DecoderStats::new();

        let fountain_packet = encoder.next_packet();
        let packet = Packet::new(fountain_packet.seq as u16, 2, &fountain_packet.data);

        let first = receive_packet(&mut decoder, &mut stats, packet.clone());
        let second = receive_packet(&mut decoder, &mut stats, packet);

        assert!(first.recovered_data.is_none());
        assert!(second.recovered_data.is_none());
        assert_eq!(stats.received_packets, 1);
        assert_eq!(stats.duplicate_packets, 1);
        assert_eq!(stats.last_packet_seq, Some(0));
        assert_eq!(stats.last_rank_up_seq, Some(0));
    }

    #[test]
    fn test_receive_packet_marks_dependent_packet_as_stalled() {
        let data = vec![0x33; PAYLOAD_SIZE * 2];
        let params = FountainParams::new(1, PAYLOAD_SIZE);
        let mut encoder = FountainEncoder::new(&data, params.clone());
        let mut decoder = FountainDecoder::new(params);
        let mut stats = DecoderStats::new();

        let packet0 = encoder.next_packet();
        let packet1 = encoder.next_packet();

        let packet0 = Packet::new(packet0.seq as u16, 1, &packet0.data);
        let packet1 = Packet::new(packet1.seq as u16, 1, &packet1.data);

        let first = receive_packet(&mut decoder, &mut stats, packet0);
        let second = receive_packet(&mut decoder, &mut stats, packet1);

        assert!(first.recovered_data.is_some());
        assert!(second.recovered_data.is_some());
        assert_eq!(stats.received_packets, 2);
        assert_eq!(stats.stalled_packets, 1);
        assert_eq!(stats.dependent_packets, 1);
        assert_eq!(stats.last_packet_seq, Some(1));
        assert_eq!(stats.last_rank_up_seq, Some(0));
    }
}
