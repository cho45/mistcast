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
    tx_cfg.packets_per_burst = cli.packets_per_burst;
    tx_cfg.preamble_sf = cli.preamble_sf;

    let mut rx_cfg = tx_cfg.clone();
    rx_cfg.carrier_freq += imp.cfo_hz;

    let payload = make_bytes(cli.payload_bytes, seed ^ 0x1234_5678);
    let k = payload.len().div_ceil(PAYLOAD_SIZE).max(1);
    let mut enc_cfg = DsssEncoderConfig::new(tx_cfg.clone());
    enc_cfg.fountain_k = k;
    enc_cfg.packets_per_sync_burst = cli.packets_per_burst;
    let mut encoder = DsssEncoder::new(enc_cfg);
    let mut decoder = DsssDecoder::new(payload.len(), k, rx_cfg);
    decoder.config.packets_per_burst = cli.packets_per_burst;

    let mut rng = StdRng::seed_from_u64(seed ^ 0xD55A_0001);
    let mut state = TrialState {
        elapsed_sec: 0.0f32,
        attempts: 0,
        dropped_attempts: 0,
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

    run_warmup(&mut encoder, &mut decoder, &mut state, &mut sim_cfg);
    {
        let mut stream = encoder.encode_stream(&payload);
        loop {
            if state.elapsed_sec >= cli.max_sec {
                break;
            }
            let Some(frame) = stream.next() else {
                break;
            };
            state.attempts += 1;
            state.tx_signal_energy_sum += signal_energy(&frame);
            state.tx_signal_samples += frame.len();

            let drop_burst = sim_cfg.rng.gen::<f32>() < imp.burst_loss;
            if drop_burst {
                state.dropped_attempts += 1;
            }
            let rx_frame = apply_channel(&frame, imp, sim_cfg.rng, drop_burst);
            state.elapsed_sec += rx_frame.len() as f32 / tx_cfg.sample_rate;
            let complete = process_samples_in_chunks(
                &rx_frame,
                chunk,
                &mut decoder,
                &mut state,
                |progress, _state| {
                    if progress.complete {
                        ControlFlow::Complete
                    } else {
                        ControlFlow::Continue
                    }
                },
            );
            if complete {
                let progress = decoder.process_samples(&[]);
                let recovered = decoder.recovered_data();
                let errs = count_bit_errors_bytes(&payload, recovered);
                let bits_compared = payload.len() * 8;
                return TrialResultBuilder::new(&state)
                    .success(errs == 0, errs, bits_compared)
                    .frame_stats(
                        progress.synced_frames,
                        progress.received_packets,
                        progress.crc_error_packets,
                    )
                    .build();
            }

            let mut complete = false;
            run_gap_with_completion(
                &mut decoder,
                &mut state,
                &mut sim_cfg,
                gap,
                |progress, _state| {
                    if progress.complete {
                        complete = true;
                        true
                    } else {
                        false
                    }
                },
            );
            if complete {
                let progress = decoder.process_samples(&[]);
                let recovered = decoder.recovered_data();
                let errs = count_bit_errors_bytes(&payload, recovered);
                let bits_compared = payload.len() * 8;
                return TrialResultBuilder::new(&state)
                    .success(errs == 0, errs, bits_compared)
                    .frame_stats(
                        progress.synced_frames,
                        progress.received_packets,
                        progress.crc_error_packets,
                    )
                    .build();
            }
        }
    }

    run_flush(&mut encoder, &mut decoder, &mut state, &mut sim_cfg);
    run_tail(&mut encoder, &mut decoder, &mut state, &mut sim_cfg);

    let bits_compared = payload.len() * 8;
    let bit_errors = count_bit_errors_bytes(&payload, decoder.recovered_data());
    let final_progress = decoder.process_samples(&[]);
    TrialResultBuilder::new(&state)
        .success(false, bit_errors, bits_compared)
        .frame_stats(
            final_progress.synced_frames,
            final_progress.received_packets,
            final_progress.crc_error_packets,
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
    tx_cfg.packets_per_burst = cli.packets_per_burst;
    tx_cfg.preamble_sf = cli.preamble_sf;

    let mut rx_cfg = tx_cfg.clone();
    rx_cfg.carrier_freq += imp.cfo_hz;

    let payload = make_bytes(cli.payload_bytes, seed ^ 0x1234_5678);
    let k = payload.len().div_ceil(PAYLOAD_SIZE).max(1);

    let mut encoder = MaryEncoder::new(tx_cfg.clone());
    encoder.set_data(&payload);

    let mut decoder = MaryDecoder::new(payload.len(), k, rx_cfg);
    decoder.config.packets_per_burst = cli.packets_per_burst;
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
        attempts: 0,
        dropped_attempts: 0,
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

    // FEC前/後のBER計測用: 送信時の期待ビット列を seq -> bits で記録し、
    // 受信側LLRから復元した seq と照合して到着順ズレの影響を避けて集計する。
    let ber_accum = BerAccumulator::new();
    decoder.llr_callback = Some(ber_accum.llr_callback());

    run_warmup(&mut encoder, &mut decoder, &mut state, &mut sim_cfg);

    loop {
        if state.elapsed_sec >= cli.max_sec {
            break;
        }

        let burst_count = cli.packets_per_burst.max(1);
        let mut packets = Vec::with_capacity(burst_count);
        for _ in 0..burst_count {
            let fp = fountain_encoder.next_packet();
            ber_accum.register_packet((fp.seq % (u32::from(u16::MAX) + 1)) as u16, &fp, k);
            packets.push(fp);
        }
        let frame = encoder.encode_burst(&packets);

        state.attempts += 1;
        state.tx_signal_energy_sum += signal_energy(&frame);
        state.tx_signal_samples += frame.len();

        let drop_burst = sim_cfg.rng.gen::<f32>() < imp.burst_loss;
        if drop_burst {
            state.dropped_attempts += 1;
        }
        let rx_frame = apply_channel(&frame, imp, sim_cfg.rng, drop_burst);
        state.elapsed_sec += rx_frame.len() as f32 / tx_cfg.sample_rate;

        let complete = process_samples_in_chunks(
            &rx_frame,
            chunk,
            &mut decoder,
            &mut state,
            |progress, _state| {
                if progress.complete {
                    ControlFlow::Complete
                } else {
                    ControlFlow::Continue
                }
            },
        );
        if complete {
            let progress = decoder.process_samples(&[]);
            let recovered = decoder.recovered_data();
            let errs = count_bit_errors_bytes(&payload, recovered);
            let bits_compared = payload.len() * 8;
            let pre_fec = ber_accum.extract_pre_fec();
            let post_fec = ber_accum.extract_post_fec();
            let (post_decode_attempts, post_decode_matched) = ber_accum.extract_decode_stats();
            return TrialResultBuilder::new(&state)
                .success(errs == 0, errs, bits_compared)
                .frame_stats(
                    progress.fde_selected_frames + progress.raw_selected_frames,
                    progress.received_packets,
                    progress.crc_error_packets,
                )
                .mary_raw_ber(pre_fec)
                .mary_post_ber(post_fec)
                .mary_phase(PhaseStats {
                    last_est_snr_db: progress.last_est_snr_db,
                    phase_gate_on_symbols: progress.phase_gate_on_symbols,
                    phase_gate_off_symbols: progress.phase_gate_off_symbols,
                    phase_innovation_reject_symbols: progress.phase_innovation_reject_symbols,
                    phase_err_abs_sum_rad: progress.phase_err_abs_sum_rad,
                    phase_err_abs_count: progress.phase_err_abs_count,
                    phase_err_abs_ge_0p5_symbols: progress.phase_err_abs_ge_0p5_symbols,
                    phase_err_abs_ge_1p0_symbols: progress.phase_err_abs_ge_1p0_symbols,
                })
                .mary_llr(
                    progress.llr_second_pass_attempts,
                    progress.llr_second_pass_rescued,
                )
                .mary_decode_stats(post_decode_attempts, post_decode_matched)
                .build();
        }

        if gap > 0 {
            run_gap_with_completion(
                &mut decoder,
                &mut state,
                &mut sim_cfg,
                gap,
                |progress, _state| {
                    progress.complete
                },
            );
            // Re-check completion after gap
            if decoder.process_samples(&[]).complete {
                let progress = decoder.process_samples(&[]);
                let recovered = decoder.recovered_data();
                let errs = count_bit_errors_bytes(&payload, recovered);
                let bits_compared = payload.len() * 8;
                let pre_fec = ber_accum.extract_pre_fec();
                let post_fec = ber_accum.extract_post_fec();
                let (post_decode_attempts, post_decode_matched) = ber_accum.extract_decode_stats();
                return TrialResultBuilder::new(&state)
                    .success(errs == 0, errs, bits_compared)
                    .frame_stats(
                        progress.fde_selected_frames + progress.raw_selected_frames,
                        progress.received_packets,
                        progress.crc_error_packets,
                    )
                    .mary_raw_ber(pre_fec)
                    .mary_post_ber(post_fec)
                    .mary_phase(PhaseStats {
                        last_est_snr_db: progress.last_est_snr_db,
                        phase_gate_on_symbols: progress.phase_gate_on_symbols,
                        phase_gate_off_symbols: progress.phase_gate_off_symbols,
                        phase_innovation_reject_symbols: progress.phase_innovation_reject_symbols,
                        phase_err_abs_sum_rad: progress.phase_err_abs_sum_rad,
                        phase_err_abs_count: progress.phase_err_abs_count,
                        phase_err_abs_ge_0p5_symbols: progress.phase_err_abs_ge_0p5_symbols,
                        phase_err_abs_ge_1p0_symbols: progress.phase_err_abs_ge_1p0_symbols,
                    })
                    .mary_llr(
                        progress.llr_second_pass_attempts,
                        progress.llr_second_pass_rescued,
                    )
                    .mary_decode_stats(post_decode_attempts, post_decode_matched)
                    .build();
            }
        }
    }

    run_flush(&mut encoder, &mut decoder, &mut state, &mut sim_cfg);
    run_tail(&mut encoder, &mut decoder, &mut state, &mut sim_cfg);

    let bits_compared = payload.len() * 8;
    let bit_errors = count_bit_errors_bytes(&payload, decoder.recovered_data());
    let pre_fec = ber_accum.extract_pre_fec();
    let post_fec = ber_accum.extract_post_fec();
    let (post_decode_attempts, post_decode_matched) = ber_accum.extract_decode_stats();
    let final_progress = decoder.process_samples(&[]);
    TrialResultBuilder::new(&state)
        .success(false, bit_errors, bits_compared)
        .frame_stats(
            final_progress.synced_frames,
            final_progress.received_packets,
            final_progress.crc_error_packets,
        )
        .mary_raw_ber(pre_fec)
        .mary_post_ber(post_fec)
        .mary_phase(PhaseStats {
            last_est_snr_db: final_progress.last_est_snr_db,
            phase_gate_on_symbols: final_progress.phase_gate_on_symbols,
            phase_gate_off_symbols: final_progress.phase_gate_off_symbols,
            phase_innovation_reject_symbols: final_progress.phase_innovation_reject_symbols,
            phase_err_abs_sum_rad: final_progress.phase_err_abs_sum_rad,
            phase_err_abs_count: final_progress.phase_err_abs_count,
            phase_err_abs_ge_0p5_symbols: final_progress.phase_err_abs_ge_0p5_symbols,
            phase_err_abs_ge_1p0_symbols: final_progress.phase_err_abs_ge_1p0_symbols,
        })
        .mary_llr(
            final_progress.llr_second_pass_attempts,
            final_progress.llr_second_pass_rescued,
        )
        .mary_decode_stats(post_decode_attempts, post_decode_matched)
        .build()
}
