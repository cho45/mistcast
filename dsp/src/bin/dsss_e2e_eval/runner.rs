use crate::channel::{apply_channel, ChannelImpairment};
use crate::config::{Cli, MaryFdeMode};
use crate::engine::{
    process_samples_in_chunks, run_flush, run_tail, run_warmup, signal_energy, ControlFlow,
    SimulationConfig,
};
use crate::metrics::{Metrics, PhaseStats};
use crate::utils::{count_bit_errors_bytes, BerAccumulator};
use dsp::dsss::encoder::{Encoder as DsssEncoder, EncoderConfig as DsssEncoderConfig};
use dsp::mary::decoder::Decoder as MaryDecoder;
use dsp::mary::encoder::Encoder as MaryEncoder;
use dsp::params::PAYLOAD_SIZE;
use dsp::{dsss::decoder::Decoder as DsssDecoder, DspConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub fn make_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.gen::<u8>()).collect()
}

pub fn apply_mary_fde_mode(decoder: &mut MaryDecoder, mode: MaryFdeMode) {
    match mode {
        MaryFdeMode::On => {
            decoder.set_fde_enabled(true);
            decoder.set_fde_auto_path_select(false);
        }
        MaryFdeMode::Off => {
            decoder.set_fde_enabled(false);
            decoder.set_fde_auto_path_select(false);
        }
        MaryFdeMode::Auto => {
            decoder.set_fde_enabled(true);
            decoder.set_fde_auto_path_select(true);
        }
    }
}

pub fn run_trial_dsss_e2e(imp: &ChannelImpairment, cli: &Cli, seed: u64) -> Metrics {
    let mut tx_cfg = DspConfig::new(cli.sample_rate);
    tx_cfg.chip_rate = cli.chip_rate;
    tx_cfg.carrier_freq = cli.carrier_freq;
    tx_cfg.mseq_order = cli.mseq_order;
    tx_cfg.rrc_alpha = cli.rrc_alpha;
    tx_cfg.sync_word_bits = cli.sync_word_bits;
    tx_cfg.preamble_repeat = cli.preamble_repeat;
    tx_cfg.packets_per_burst = cli.packets_per_frame;
    tx_cfg.preamble_sf = cli.preamble_sf;

    let mut rx_cfg = tx_cfg.clone();
    rx_cfg.carrier_freq += imp.cfo_hz;

    let payload = make_bytes(cli.payload_bytes, seed ^ 0x1234_5678);
    let k = payload.len().div_ceil(PAYLOAD_SIZE).max(1);
    let mut enc_cfg = DsssEncoderConfig::new(tx_cfg.clone());
    enc_cfg.fountain_k = k;
    enc_cfg.packets_per_sync_burst = cli.packets_per_frame;
    let mut encoder = DsssEncoder::new(enc_cfg);
    let mut decoder = DsssDecoder::new(payload.len(), k, rx_cfg);
    decoder.config.packets_per_burst = cli.packets_per_frame;

    let mut rng = StdRng::seed_from_u64(seed ^ 0xD55A_0001);
    let mut m = Metrics::new(cli.packets_per_frame);

    let chunk = cli.chunk_samples.max(1);

    let mut sim_cfg = SimulationConfig {
        sample_rate: tx_cfg.sample_rate,
        imp,
        rng: &mut rng,
        chunk_size: chunk,
    };

    let mut last_reset_sec = 0.0f32;

    run_warmup(&mut encoder, &mut decoder, &mut m, &mut sim_cfg);
    {
        let mut stream = encoder.encode_stream(&payload);
        loop {
            if m.total_sim_sec >= cli.total_sim_sec {
                break;
            }
            let Some(frame) = stream.next() else {
                break;
            };
            m.total_frame_attempts += 1;
            m.total_tx_signal_energy += signal_energy(&frame);
            m.total_tx_signal_samples += frame.len();

            let drop_frame = sim_cfg.rng.gen::<f32>() < imp.frame_loss;
            if drop_frame {
                m.dropped_frames += 1;
            }
            let rx_frame = apply_channel(&frame, imp, sim_cfg.rng, drop_frame);
            m.total_sim_sec += rx_frame.len() as f32 / tx_cfg.sample_rate;

            process_samples_in_chunks(
                &rx_frame,
                chunk,
                &mut decoder,
                &mut m,
                |decoder: &mut DsssDecoder, progress, m: &mut Metrics| {
                    if progress.complete {
                        let recovered = decoder.recovered_data();
                        let errs = count_bit_errors_bytes(&payload, recovered);
                        m.add_frame_event(
                            progress.synced_frames,
                            progress.received_packets,
                            progress.crc_error_packets,
                        );
                        m.add_recovery_event(
                            m.total_sim_sec - last_reset_sec,
                            errs,
                            payload.len() * 8,
                        );

                        decoder.reset_fountain_decoder();
                        last_reset_sec = m.total_sim_sec;
                    }
                    ControlFlow::Continue
                },
            );
        }
    }

    run_flush(&mut encoder, &mut decoder, &mut m, &mut sim_cfg);
    run_tail(&mut encoder, &mut decoder, &mut m, &mut sim_cfg);

    let final_progress = decoder.process_samples(&[]);
    m.add_frame_event(
        final_progress
            .synced_frames,
        final_progress
            .received_packets,
        final_progress
            .crc_error_packets
    );

    let recovered = decoder.recovered_data();
    if recovered.is_some() {
        let errs = count_bit_errors_bytes(&payload, recovered);
        m.add_recovery_event(m.total_sim_sec - last_reset_sec, errs, payload.len() * 8);
    }

    m
}

pub fn run_trial_mary_e2e(imp: &ChannelImpairment, cli: &Cli, seed: u64) -> Metrics {
    let mut tx_cfg = dsp::mary::params::dsp_config(cli.sample_rate);
    tx_cfg.chip_rate = cli.chip_rate;
    tx_cfg.carrier_freq = cli.carrier_freq;
    tx_cfg.mseq_order = cli.mseq_order;
    tx_cfg.rrc_alpha = cli.rrc_alpha;
    tx_cfg.sync_word_bits = cli.sync_word_bits;
    tx_cfg.preamble_repeat = cli.preamble_repeat;
    tx_cfg.packets_per_burst = cli.packets_per_frame;
    tx_cfg.preamble_sf = cli.preamble_sf;

    let mut rx_cfg = tx_cfg.clone();
    rx_cfg.carrier_freq += imp.cfo_hz;

    let payload = make_bytes(cli.payload_bytes, seed ^ 0x1234_5678);
    let k = payload.len().div_ceil(PAYLOAD_SIZE).max(1);

    let mut encoder = MaryEncoder::new(tx_cfg.clone());
    encoder.set_data(&payload);

    let mut decoder = MaryDecoder::new(payload.len(), k, rx_cfg);
    decoder.config.packets_per_burst = cli.packets_per_frame;
    apply_mary_fde_mode(&mut decoder, cli.mary_fde_mode);
    decoder.set_fde_mmse_settings(
        cli.mary_fde_snr_db,
        cli.mary_fde_lambda_scale,
        cli.mary_fde_lambda_floor,
        cli.mary_fde_max_inv_gain,
    );
    decoder.set_cir_postprocess(cli.mary_cir_norm.into(), cli.mary_cir_tap_alpha);
    decoder.set_viterbi_list_size(cli.mary_viterbi_list);
    decoder.set_llr_erasure_second_pass(
        cli.mary_llr_erasure_second_pass,
        cli.mary_llr_erasure_q,
        cli.mary_llr_erasure_list,
    );

    let mut rng = StdRng::seed_from_u64(seed ^ 0xD55A_0001);
    let mut m = Metrics::new(cli.packets_per_frame);

    let chunk = cli.chunk_samples.max(1);

    let mut sim_cfg = SimulationConfig {
        sample_rate: tx_cfg.sample_rate,
        imp,
        rng: &mut rng,
        chunk_size: chunk,
    };

    let fountain_params = dsp::coding::fountain::FountainParams::new(k, PAYLOAD_SIZE);
    let mut fountain_encoder =
        dsp::coding::fountain::FountainEncoder::new(&payload, fountain_params);

    let ber_accum = BerAccumulator::new();
    decoder.llr_callback = Some(ber_accum.llr_callback());

    let mut last_reset_sec = 0.0f32;
    let mut last_total_llr_attempts = 0usize;
    let mut last_total_llr_rescued = 0usize;
    let mut last_total_post_attempts = 0usize;
    let mut last_total_post_matched = 0usize;

    run_warmup(&mut encoder, &mut decoder, &mut m, &mut sim_cfg);

    loop {
        if m.total_sim_sec >= cli.total_sim_sec {
            break;
        }

        let packet_count_per_frame = cli.packets_per_frame.max(1);
        let mut packets = Vec::with_capacity(packet_count_per_frame);
        for _ in 0..packet_count_per_frame {
            let fp = fountain_encoder.next_packet();
            ber_accum.register_packet((fp.seq % (u32::from(u16::MAX) + 1)) as u16, &fp, k);
            packets.push(fp);
        }
        let frame = encoder.encode_burst(&packets);

        m.total_frame_attempts += 1;
        m.total_tx_signal_energy += signal_energy(&frame);
        m.total_tx_signal_samples += frame.len();

        let drop_frame = sim_cfg.rng.gen::<f32>() < imp.frame_loss;
        if drop_frame {
            m.dropped_frames += 1;
        }
        let rx_frame = apply_channel(&frame, imp, sim_cfg.rng, drop_frame);
        m.total_sim_sec += rx_frame.len() as f32 / tx_cfg.sample_rate;

        process_samples_in_chunks(
            &rx_frame,
            chunk,
            &mut decoder,
            &mut m,
            |decoder: &mut MaryDecoder, progress, m: &mut Metrics| {
                m.add_mary_phase(PhaseStats {
                    last_est_snr_db: progress.last_est_snr_db,
                    phase_gate_on_symbols: progress.phase_gate_on_symbols,
                    phase_gate_off_symbols: progress.phase_gate_off_symbols,
                    phase_innovation_reject_symbols: progress.phase_innovation_reject_symbols,
                    phase_err_abs_sum_rad: progress.phase_err_abs_sum_rad,
                    phase_err_abs_count: progress.phase_err_abs_count,
                    phase_err_abs_ge_0p5_symbols: progress.phase_err_abs_ge_0p5_symbols,
                    phase_err_abs_ge_1p0_symbols: progress.phase_err_abs_ge_1p0_symbols,
                });

                m.add_mary_llr(
                    progress
                        .llr_second_pass_attempts
                        .saturating_sub(last_total_llr_attempts),
                    progress
                        .llr_second_pass_rescued
                        .saturating_sub(last_total_llr_rescued),
                );
                last_total_llr_attempts = progress.llr_second_pass_attempts;
                last_total_llr_rescued = progress.llr_second_pass_rescued;

                let (post_attempts, post_matched) = ber_accum.extract_decode_stats();
                m.add_mary_decode_stats(
                    post_attempts.saturating_sub(last_total_post_attempts),
                    post_matched.saturating_sub(last_total_post_matched),
                );
                last_total_post_attempts = post_attempts;
                last_total_post_matched = post_matched;

                if progress.complete {
                    let recovered = decoder.recovered_data();
                    let errs = count_bit_errors_bytes(&payload, recovered);
                    m.add_frame_event(
                        progress.synced_frames,
                        progress.received_packets,
                        progress.crc_error_packets,
                    );
                    m.add_recovery_event(m.total_sim_sec - last_reset_sec, errs, payload.len() * 8);

                    decoder.reset_fountain_decoder();
                    last_reset_sec = m.total_sim_sec;
                    last_total_llr_attempts = 0;
                    last_total_llr_rescued = 0;
                    // Note: ber_accum は累積し続ける
                }
                ControlFlow::Continue
            },
        );
    }

    run_flush(&mut encoder, &mut decoder, &mut m, &mut sim_cfg);
    run_tail(&mut encoder, &mut decoder, &mut m, &mut sim_cfg);

    let final_progress = decoder.process_samples(&[]);
    m.add_frame_event(
        final_progress
             .synced_frames,
         final_progress
             .received_packets,
         final_progress
             .crc_error_packets
    );

    let recovered = decoder.recovered_data();
    if recovered.is_some() {
        let errs = count_bit_errors_bytes(&payload, recovered);
        m.add_recovery_event(m.total_sim_sec - last_reset_sec, errs, payload.len() * 8);
    }

    m.set_mary_raw_ber(ber_accum.extract_pre_fec());
    m.set_mary_post_ber(ber_accum.extract_post_fec());

    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::MultipathProfile;
    use crate::config::{CirNormArg, EvalMode, OutputFormat, Phy};

    #[test]
    fn test_make_bytes_determinism() {
        let b1 = make_bytes(32, 123);
        let b2 = make_bytes(32, 123);
        let b3 = make_bytes(32, 456);
        assert_eq!(b1, b2);
        assert_ne!(b1, b3);
        assert_eq!(b1.len(), 32);
    }

    #[test]
    fn test_apply_mary_fde_mode_smoke() {
        let mut decoder = MaryDecoder::new(16, 1, dsp::mary::params::dsp_config(48000.0));
        apply_mary_fde_mode(&mut decoder, MaryFdeMode::On);
        apply_mary_fde_mode(&mut decoder, MaryFdeMode::Off);
        apply_mary_fde_mode(&mut decoder, MaryFdeMode::Auto);
    }

    fn dummy_cli(phy: Phy) -> Cli {
        Cli {
            phy,
            mode: EvalMode::Point,
            total_sim_sec: 3.0,
            payload_bytes: 50,
            chunk_samples: 1024,
            seed: 789,
            target_p_complete: 0.95,
            sigma: 0.0,
            cfo_hz: 0.0,
            ppm: 0.0,
            frame_loss: 0.0,
            fading_depth: 0.0,
            multipath: MultipathProfile::none(),
            sweep_awgn: vec![],
            sweep_ppm: vec![],
            sweep_loss: vec![],
            sweep_fading: vec![],
            sweep_chip_rate: vec![],
            sweep_carrier_freq: vec![],
            sample_rate: 48000.0,
            chip_rate: 8000.0,
            carrier_freq: 15000.0,
            mseq_order: 4,
            rrc_alpha: 0.3,
            sync_word_bits: 16,
            preamble_repeat: 2,
            packets_per_frame: 1,
            preamble_sf: 127,
            mary_fde_mode: MaryFdeMode::On,
            mary_fde_snr_db: 15.0,
            mary_fde_lambda_scale: 1.0,
            mary_fde_lambda_floor: 0.0,
            mary_fde_max_inv_gain: None,
            mary_cir_norm: CirNormArg::None,
            mary_cir_tap_alpha: 0.0,
            mary_viterbi_list: 1,
            mary_llr_erasure_second_pass: false,
            mary_llr_erasure_q: 0.2,
            mary_llr_erasure_list: 8,
            columns: None,
            output: OutputFormat::Csv,
            show_metrics_desc: false,
        }
    }

    #[test]
    fn test_run_trial_dsss_e2e_full_metrics() {
        let cli = dummy_cli(Phy::Dsss);
        let imp = cli.base_impairment(); // AWGN(0)
        let m = run_trial_dsss_e2e(&imp, &cli, cli.seed);

        // --- DSSS メトリクスの厳格な検証 ---
        // 1. 基本統計
        assert!(m.total_sim_sec >= 1.0);
        assert!(m.total_frame_attempts > 0);
        assert_eq!(m.dropped_frames, 0);

        // 2. 到達率 (理想条件)
        // DSSS の synced_frames は 1以上であることを許容する
        assert!(
            m.total_synced_frames >= 1,
            "DSSS synced_frames must be >= 1"
        );
        // accepted_packets は送信されたフレーム数に近い値になるはず
        assert!(m.total_accepted_packets > 0);
        assert_eq!(m.total_crc_error_packets, 0);
        assert_eq!(m.crc_pass_ratio(), 1.0);
        assert!(m.p_complete() > 0.9);

        // 3. エラー統計 (理想条件)
        assert!(m.total_successes > 0);
        assert_eq!(
            m.total_bit_errors, 0,
            "DSSS must have 0 bit errors in AWGN(0)"
        );
        assert_eq!(m.total_bits_compared, m.total_successes * cli.payload_bytes * 8);
        assert_eq!(m.ber(), 0.0);

        // 4. 物理・タイミング
        assert!(m.total_tx_signal_energy > 0.0);
        assert!(m.total_tx_signal_samples > 0);
        assert!(m.total_process_time_ns > 0);
        assert!(m.mean_completion_sec().is_some());
        assert!(m.goodput_effective_bps(16 * 8) > 0.0);
    }

    #[test]
    fn test_run_trial_mary_e2e_full_metrics() {
        let cli = dummy_cli(Phy::Mary);
        let imp = cli.base_impairment(); // AWGN(0)
        let m = run_trial_mary_e2e(&imp, &cli, cli.seed);

        // --- Mary メトリクスの厳格な検証 ---
        // 1. 基本統計
        assert!(m.total_sim_sec >= 1.0);
        assert!(m.total_frame_attempts > 0);
        assert_eq!(m.dropped_frames, 0);

        // 2. 到達率
        // Mary は synced_frames == accepted_packets (packets_per_frame=1)
        assert!(m.total_accepted_packets > 0);
        assert_eq!(m.total_crc_error_packets, 0);
        assert!(m.p_complete() > 0.95);
        assert_eq!(
            m.total_synced_frames, m.total_accepted_packets,
            "Mary synced vs accepted mismatch"
        );

        // 3. エラー統計
        assert!(m.total_successes > 0);
        assert_eq!(
            m.total_bit_errors, 0,
            "Mary must have 0 bit errors in AWGN(0)"
        );
        assert_eq!(m.ber(), 0.0);

        // Mary 特有: 生/適用後 BER
        assert_eq!(m.total_raw_bit_errors, 0);
        assert_eq!(m.raw_ber(), 0.0);
        assert_eq!(m.total_post_bit_errors, 0);
        assert_eq!(m.post_ber(), 0.0);

        // Viterbi / LLR
        assert!(m.total_post_decode_attempts > 0);
        assert_eq!(m.total_post_decode_matched, m.total_post_decode_attempts);
        assert_eq!(m.total_llr_second_pass_attempts, 0);

        // 4. 位相統計 (理想条件)
        assert!(m.total_phase_gate_on_symbols > 0);
        assert_eq!(m.total_phase_gate_off_symbols, 0);
        assert_eq!(m.phase_gate_on_ratio(), 1.0);
        assert!(m.phase_err_abs_mean_rad().unwrap() < 0.1, "Phase error mean should be near 0 in ideal conditions {}", m.phase_err_abs_mean_rad().unwrap());
        assert_eq!(m.phase_err_abs_ge_0p5_ratio(), 0.0);

        // SNR
        assert!(m.avg_last_est_snr_db().unwrap() > 20.0);
    }
}
