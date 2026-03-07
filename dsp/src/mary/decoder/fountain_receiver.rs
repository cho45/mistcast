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
}
