use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use dsp::coding::fec::{self, FecDecodeWorkspace};
use dsp::common::resample::Resampler;
use dsp::common::rrc_filter::RrcFilter;
use dsp::frame::packet::Packet;
use dsp::params::PAYLOAD_SIZE;
use dsp::DspConfig;
use std::f32::consts::PI;

fn make_signal(len: usize, sample_rate: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32 / sample_rate;
            0.8 * (2.0 * PI * 1000.0 * t).sin()
                + 0.2 * (2.0 * PI * 4300.0 * t).cos()
                + 0.1 * (2.0 * PI * 8500.0 * t).sin()
        })
        .collect()
}

fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn make_packet_fec_llrs(noise_amp: f32) -> Vec<f32> {
    let mut payload = [0u8; PAYLOAD_SIZE];
    for (i, byte) in payload.iter_mut().enumerate() {
        *byte = ((i as u8).wrapping_mul(37)).wrapping_add(11);
    }

    let packet = Packet::new(123, 10, &payload);
    let bits = fec::bytes_to_bits(&packet.serialize());
    let coded = fec::encode(&bits);

    let mut rng_state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut llrs = Vec::with_capacity(coded.len());
    for bit in coded {
        let u = (xorshift64(&mut rng_state) as u32) as f32 / u32::MAX as f32;
        let noise = (u * 2.0 - 1.0) * noise_amp;
        let symbol = if bit == 0 { 1.0 } else { -1.0 };
        llrs.push(symbol + noise);
    }
    llrs
}

fn bench_rrc_filter(c: &mut Criterion) {
    let config = DspConfig::default_48k();
    let input = make_signal(16_384, config.proc_sample_rate());

    c.bench_function("rrc_filter/process_block_in_place/16k", |b| {
        b.iter_batched(
            || {
                let filter = RrcFilter::from_config(&config);
                let buf = input.clone();
                (filter, buf)
            },
            |(mut filter, mut buf)| {
                filter.process_block_in_place(black_box(&mut buf));
                black_box(buf);
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_resampler(c: &mut Criterion) {
    let source_rate = 48_000u32;
    let target_rate = 44_100u32;
    let input = make_signal(20_000, source_rate as f32);

    c.bench_function("resampler/process+flush/20k_48k_to_44k1", |b| {
        b.iter_batched(
            || {
                let rs = Resampler::new_with_cutoff(source_rate, target_rate, None, None);
                let out = Vec::new();
                (rs, out)
            },
            |(mut rs, mut out)| {
                rs.process(black_box(&input), &mut out);
                rs.flush(&mut out);
                black_box(out);
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_fec_list_viterbi(c: &mut Criterion) {
    let llrs = make_packet_fec_llrs(1.7);
    let llr_len = llrs.len();
    let bits_per_packet = llr_len / 2;

    let mut group = c.benchmark_group("fec/decode_soft_list_into");
    group.throughput(Throughput::Elements(bits_per_packet as u64));

    for &list_size in &[1usize, 2, 4, 8, 16, 32] {
        let mut workspace = FecDecodeWorkspace::new();
        workspace.preallocate_for_llr_len(llr_len, list_size);
        let mut out_candidates = Vec::new();
        workspace.decode_soft_list_into(&llrs, list_size, &mut out_candidates);

        group.bench_function(format!("packet_llr348/k={list_size}"), |b| {
            b.iter(|| {
                workspace.decode_soft_list_into(
                    black_box(&llrs),
                    black_box(list_size),
                    &mut out_candidates,
                );
                black_box(out_candidates.len());
            });
        });
    }

    group.finish();
}

fn bench_fec_list_viterbi_try(c: &mut Criterion) {
    let llrs = make_packet_fec_llrs(1.7);
    let llr_len = llrs.len();
    let bits_per_packet = llr_len / 2;

    let mut group = c.benchmark_group("fec/decode_soft_list_try");
    group.throughput(Throughput::Elements(bits_per_packet as u64));

    for &list_size in &[1usize, 2, 4, 8, 16, 32] {
        let mut workspace = FecDecodeWorkspace::new();
        workspace.preallocate_for_llr_len(llr_len, list_size);
        let mut candidate_bits = Vec::with_capacity(bits_per_packet);

        for &(label, accept_after) in &[
            ("reject_all", usize::MAX),
            ("accept_after_1", 1usize),
            ("accept_after_4", 4usize),
        ] {
            group.bench_function(format!("packet_llr348/k={list_size}/{label}"), |b| {
                b.iter(|| {
                    let mut seen = 0usize;
                    let accepted = workspace.decode_soft_list_try(
                        black_box(&llrs),
                        black_box(list_size),
                        &mut candidate_bits,
                        |bits, _rank, _score| {
                            seen += 1;
                            black_box(bits);
                            if seen >= accept_after {
                                Some(())
                            } else {
                                None
                            }
                        },
                    );
                    black_box(accepted.is_some());
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    native_dsp_benches,
    bench_rrc_filter,
    bench_resampler,
    bench_fec_list_viterbi,
    bench_fec_list_viterbi_try
);
criterion_main!(native_dsp_benches);
