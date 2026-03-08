use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use dsp::common::resample::Resampler;
use dsp::common::rrc_filter::RrcFilter;
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

criterion_group!(native_dsp_benches, bench_rrc_filter, bench_resampler);
criterion_main!(native_dsp_benches);
