use crate::channel::{apply_channel, ChannelImpairment};
use crate::config::{Cli, MaryFdeMode};
use crate::engine::{
    process_samples_in_chunks, run_flush, run_gap_with_completion, run_tail, run_warmup,
    signal_energy, ControlFlow, SimulationConfig,
};
use crate::metrics::{TrialResult, TrialResultBuilder, TrialState, PhaseStats};
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

pub fn run_trial_dsss_e2e(imp: &ChannelImpairment, cli: &Cli, seed: u64) -> TrialResult {
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
    let mut state = TrialState {
        elapsed_sec: 0.0f32,
        frame_attempts: 0,
        dropped_frames: 0,
        tx_signal_energy_sum: 0.0f64,
        tx_signal_samples: 0,
        total_process_ns: 0,
    };

    let chunk = cli.chunk_samples.max(1);
    let gap = cli.gap_samples;

    let mut sim_cfg = SimulationConfig {
        sample_rate: tx_cfg.sample_rate,
        imp,
        rng: &mut rng,
        chunk_size: chunk,
    };

    let mut total_synced_frames = 0usize;
    let mut total_accepted_packets = 0usize;
    let mut total_crc_error_packets = 0usize;
    let mut completion_secs = Vec::new();
    let mut total_bit_errors = 0usize;
    let mut total_bits_compared = 0usize;
    let mut last_reset_sec = 0.0f32;

    run_warmup(&mut encoder, &mut decoder, &mut state, &mut sim_cfg);
    {
        let mut stream = encoder.encode_stream(&payload);
        loop {
            if state.elapsed_sec >= cli.total_sim_sec {
                break;
            }
            let Some(frame) = stream.next() else {
                break;
            };
            state.frame_attempts += 1;
            state.tx_signal_energy_sum += signal_energy(&frame);
            state.tx_signal_samples += frame.len();

            let drop_frame = sim_cfg.rng.gen::<f32>() < imp.frame_loss;
            if drop_frame {
                state.dropped_frames += 1;
            }
            let rx_frame = apply_channel(&frame, imp, sim_cfg.rng, drop_frame);
            state.elapsed_sec += rx_frame.len() as f32 / tx_cfg.sample_rate;
            
            process_samples_in_chunks(
                &rx_frame,
                chunk,
                &mut decoder,
                &mut state,
                |decoder, progress, state| {
                    if progress.complete {
                        total_synced_frames += progress.synced_frames;
                        total_accepted_packets += progress.received_packets;
                        total_crc_error_packets += progress.crc_error_packets;

                        let recovered = decoder.recovered_data();
                        let errs = count_bit_errors_bytes(&payload, recovered);
                        total_bit_errors += errs;
                        total_bits_compared += payload.len() * 8;
                        completion_secs.push(state.elapsed_sec - last_reset_sec);
                        
                        decoder.reset();
                        last_reset_sec = state.elapsed_sec;
                    }
                    ControlFlow::Continue
                },
            );

            run_gap_with_completion(
                &mut decoder,
                &mut state,
                &mut sim_cfg,
                gap,
                |decoder, progress, state| {
                    if progress.complete {
                        total_synced_frames += progress.synced_frames;
                        total_accepted_packets += progress.received_packets;
                        total_crc_error_packets += progress.crc_error_packets;

                        let recovered = decoder.recovered_data();
                        let errs = count_bit_errors_bytes(&payload, recovered);
                        total_bit_errors += errs;
                        total_bits_compared += payload.len() * 8;
                        completion_secs.push(state.elapsed_sec - last_reset_sec);
                        
                        decoder.reset();
                        last_reset_sec = state.elapsed_sec;
                    }
                    false
                },
            );
        }
    }

    run_flush(&mut encoder, &mut decoder, &mut state, &mut sim_cfg);
    run_tail(&mut encoder, &mut decoder, &mut state, &mut sim_cfg);

    let final_progress = decoder.process_samples(&[]);
    total_synced_frames += final_progress.synced_frames;
    total_accepted_packets += final_progress.received_packets;
    total_crc_error_packets += final_progress.crc_error_packets;
    let recovered = decoder.recovered_data();
    if recovered.is_some() {
        let errs = count_bit_errors_bytes(&payload, recovered);
        total_bit_errors += errs;
        total_bits_compared += payload.len() * 8;
        completion_secs.push(state.elapsed_sec - last_reset_sec);
    }

    let avg_completion = if completion_secs.is_empty() {
        None
    } else {
        Some(completion_secs.iter().sum::<f32>() / completion_secs.len() as f32)
    };

    TrialResultBuilder::new(&state, cli.packets_per_frame)
        .bit_errors(total_bit_errors, total_bits_compared)
        .completion_sec(avg_completion)
        .packet_stats(
            total_synced_frames,
            total_accepted_packets,
            total_crc_error_packets,
        )
        .build()
}

pub fn run_trial_mary_e2e(imp: &ChannelImpairment, cli: &Cli, seed: u64) -> TrialResult {
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
    let mut state = TrialState {
        elapsed_sec: 0.0f32,
        frame_attempts: 0,
        dropped_frames: 0,
        tx_signal_energy_sum: 0.0f64,
        tx_signal_samples: 0,
        total_process_ns: 0,
    };

    let chunk = cli.chunk_samples.max(1);
    let gap = cli.gap_samples;

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

    let mut total_synced_frames = 0usize;
    let mut total_accepted_packets = 0usize;
    let mut total_crc_error_packets = 0usize;
    let mut completion_secs = Vec::new();
    let mut total_bit_errors = 0usize;
    let mut total_bits_compared = 0usize;
    let mut last_phase_stats = None;
    let mut last_llr_attempts = 0usize;
    let mut last_llr_rescued = 0usize;
    let mut last_reset_sec = 0.0f32;

    run_warmup(&mut encoder, &mut decoder, &mut state, &mut sim_cfg);

    loop {
        if state.elapsed_sec >= cli.total_sim_sec {
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

        state.frame_attempts += 1;
        state.tx_signal_energy_sum += signal_energy(&frame);
        state.tx_signal_samples += frame.len();

        let drop_frame = sim_cfg.rng.gen::<f32>() < imp.frame_loss;
        if drop_frame {
            state.dropped_frames += 1;
        }
        let rx_frame = apply_channel(&frame, imp, sim_cfg.rng, drop_frame);
        state.elapsed_sec += rx_frame.len() as f32 / tx_cfg.sample_rate;

        process_samples_in_chunks(
            &rx_frame,
            chunk,
            &mut decoder,
            &mut state,
            |decoder, progress, state| {
                if progress.complete {
                    total_synced_frames += progress.fde_selected_frames + progress.raw_selected_frames;
                    total_accepted_packets += progress.received_packets;
                    total_crc_error_packets += progress.crc_error_packets;
                    
                    let recovered = decoder.recovered_data();
                    let errs = count_bit_errors_bytes(&payload, recovered);
                    total_bit_errors += errs;
                    total_bits_compared += payload.len() * 8;
                    completion_secs.push(state.elapsed_sec - last_reset_sec);

                    last_phase_stats = Some(PhaseStats {
                        last_est_snr_db: progress.last_est_snr_db,
                        phase_gate_on_symbols: progress.phase_gate_on_symbols,
                        phase_gate_off_symbols: progress.phase_gate_off_symbols,
                        phase_innovation_reject_symbols: progress.phase_innovation_reject_symbols,
                        phase_err_abs_sum_rad: progress.phase_err_abs_sum_rad,
                        phase_err_abs_count: progress.phase_err_abs_count,
                        phase_err_abs_ge_0p5_symbols: progress.phase_err_abs_ge_0p5_symbols,
                        phase_err_abs_ge_1p0_symbols: progress.phase_err_abs_ge_1p0_symbols,
                    });
                    last_llr_attempts = progress.llr_second_pass_attempts;
                    last_llr_rescued = progress.llr_second_pass_rescued;

                    decoder.reset();
                    last_reset_sec = state.elapsed_sec;
                }
                ControlFlow::Continue
            },
        );

        if gap > 0 {
            run_gap_with_completion(
                &mut decoder,
                &mut state,
                &mut sim_cfg,
                gap,
                |decoder, progress, state| {
                    if progress.complete {
                        total_synced_frames += progress.fde_selected_frames + progress.raw_selected_frames;
                        total_accepted_packets += progress.received_packets;
                        total_crc_error_packets += progress.crc_error_packets;

                        let recovered = decoder.recovered_data();
                        let errs = count_bit_errors_bytes(&payload, recovered);
                        total_bit_errors += errs;
                        total_bits_compared += payload.len() * 8;
                        completion_secs.push(state.elapsed_sec - last_reset_sec);

                        last_phase_stats = Some(PhaseStats {
                            last_est_snr_db: progress.last_est_snr_db,
                            phase_gate_on_symbols: progress.phase_gate_on_symbols,
                            phase_gate_off_symbols: progress.phase_gate_off_symbols,
                            phase_innovation_reject_symbols: progress.phase_innovation_reject_symbols,
                            phase_err_abs_sum_rad: progress.phase_err_abs_sum_rad,
                            phase_err_abs_count: progress.phase_err_abs_count,
                            phase_err_abs_ge_0p5_symbols: progress.phase_err_abs_ge_0p5_symbols,
                            phase_err_abs_ge_1p0_symbols: progress.phase_err_abs_ge_1p0_symbols,
                        });
                        last_llr_attempts = progress.llr_second_pass_attempts;
                        last_llr_rescued = progress.llr_second_pass_rescued;

                        decoder.reset();
                        last_reset_sec = state.elapsed_sec;
                    }
                    false
                },
            );
        }
    }

    run_flush(&mut encoder, &mut decoder, &mut state, &mut sim_cfg);
    run_tail(&mut encoder, &mut decoder, &mut state, &mut sim_cfg);

    let final_progress = decoder.process_samples(&[]);
    total_synced_frames += final_progress.fde_selected_frames + final_progress.raw_selected_frames;
    total_accepted_packets += final_progress.received_packets;
    total_crc_error_packets += final_progress.crc_error_packets;
    let recovered = decoder.recovered_data();
    if recovered.is_some() {
        let errs = count_bit_errors_bytes(&payload, recovered);
        total_bit_errors += errs;
        total_bits_compared += payload.len() * 8;
        completion_secs.push(state.elapsed_sec - last_reset_sec);
    }

    let avg_completion = if completion_secs.is_empty() {
        None
    } else {
        Some(completion_secs.iter().sum::<f32>() / completion_secs.len() as f32)
    };

    let pre_fec = ber_accum.extract_pre_fec();
    let post_fec = ber_accum.extract_post_fec();
    let (post_decode_attempts, post_decode_matched) = ber_accum.extract_decode_stats();

    let mut builder = TrialResultBuilder::new(&state, cli.packets_per_frame)
        .bit_errors(total_bit_errors, total_bits_compared)
        .completion_sec(avg_completion)
        .packet_stats(
            total_synced_frames,
            total_accepted_packets,
            total_crc_error_packets,
        )
        .mary_raw_ber(pre_fec)
        .mary_post_ber(post_fec)
        .mary_llr(
            last_llr_attempts,
            last_llr_rescued,
        )
        .mary_decode_stats(post_decode_attempts, post_decode_matched);

    if let Some(ps) = last_phase_stats {
        builder = builder.mary_phase(ps);
    }

    builder.build()
}
