use crate::channel::{apply_channel, ChannelImpairment};
use crate::metrics::Metrics;
use dsp::dsss::decoder::Decoder as DsssDecoder;
use dsp::mary::decoder::Decoder as MaryDecoder;
use rand::rngs::StdRng;

pub const TX_WARMUP_SAMPLES: usize = 4096;
pub const TX_TAIL_SAMPLES: usize = 4096;

/// プロセスを継続するか、完了したか
pub enum ControlFlow {
    Continue,
}

/// デコーダが持つべきメソッドを示すマーカートレイト
pub trait ProcessSamples {
    type Progress;
    fn process_samples(&mut self, samples: &[f32]) -> Self::Progress;
}

// 既存の型にトレイトを実装
impl ProcessSamples for DsssDecoder {
    type Progress = dsp::dsss::decoder::DecodeProgress;
    fn process_samples(&mut self, samples: &[f32]) -> Self::Progress {
        self.process_samples(samples)
    }
}

impl ProcessSamples for MaryDecoder {
    type Progress = dsp::mary::decoder::DecodeProgress;
    fn process_samples(&mut self, samples: &[f32]) -> Self::Progress {
        self.process_samples(samples)
    }
}

/// チャンク単位でサンプルを処理し、処理時間を計測する
pub fn process_samples_in_chunks<D, F>(
    samples: &[f32],
    chunk_size: usize,
    decoder: &mut D,
    state: &mut Metrics,
    mut on_progress: F,
) where
    D: ProcessSamples,
    F: FnMut(&mut D, &D::Progress, &mut Metrics) -> ControlFlow,
{
    for piece in samples.chunks(chunk_size) {
        let start_time = std::time::Instant::now();
        let progress = decoder.process_samples(piece);
        state.total_process_time_ns += start_time.elapsed().as_nanos() as u64;

        on_progress(decoder, &progress, state);
    }
}

pub struct SimulationConfig<'a> {
    pub sample_rate: f32,
    pub imp: &'a ChannelImpairment,
    pub rng: &'a mut StdRng,
    pub chunk_size: usize,
}

/// ウォームアップ処理を実行する
pub fn run_warmup<E, D>(
    encoder: &mut E,
    decoder: &mut D,
    state: &mut Metrics,
    cfg: &mut SimulationConfig,
) where
    E: WarmupSignal,
    D: ProcessSamples,
{
    let warmup = encoder.modulate_silence(TX_WARMUP_SAMPLES);
    state.total_tx_signal_energy += signal_energy(&warmup);
    state.total_tx_signal_samples += warmup.len();
    let warmup_rx = apply_channel(&warmup, cfg.imp, cfg.rng, false);
    state.total_sim_sec += warmup_rx.len() as f32 / cfg.sample_rate;
    process_samples_in_chunks(&warmup_rx, cfg.chunk_size, decoder, state, |_, _, _| {
        ControlFlow::Continue
    });
}

/// フラッシュ処理を実行する
pub fn run_flush<E, D>(
    encoder: &mut E,
    decoder: &mut D,
    state: &mut Metrics,
    cfg: &mut SimulationConfig,
) where
    E: FlushSignal,
    D: ProcessSamples,
{
    let flush = encoder.flush();
    if flush.is_empty() {
        return;
    }
    state.total_tx_signal_energy += signal_energy(&flush);
    state.total_tx_signal_samples += flush.len();
    let flush_rx = apply_channel(&flush, cfg.imp, cfg.rng, false);
    state.total_sim_sec += flush_rx.len() as f32 / cfg.sample_rate;
    process_samples_in_chunks(&flush_rx, cfg.chunk_size, decoder, state, |_, _, _| {
        ControlFlow::Continue
    });
}

/// テール処理を実行する
pub fn run_tail<E, D>(
    encoder: &mut E,
    decoder: &mut D,
    state: &mut Metrics,
    cfg: &mut SimulationConfig,
) where
    E: WarmupSignal,
    D: ProcessSamples,
{
    let tail = encoder.modulate_silence(TX_TAIL_SAMPLES);
    if tail.is_empty() {
        return;
    }
    state.total_tx_signal_energy += signal_energy(&tail);
    state.total_tx_signal_samples += tail.len();
    let tail_rx = apply_channel(&tail, cfg.imp, cfg.rng, false);
    state.total_sim_sec += tail_rx.len() as f32 / cfg.sample_rate;
    process_samples_in_chunks(&tail_rx, cfg.chunk_size, decoder, state, |_, _, _| {
        ControlFlow::Continue
    });
}

pub trait WarmupSignal {
    fn modulate_silence(&mut self, samples: usize) -> Vec<f32>;
}

pub trait FlushSignal {
    fn flush(&mut self) -> Vec<f32>;
}

impl WarmupSignal for dsp::dsss::encoder::Encoder {
    fn modulate_silence(&mut self, samples: usize) -> Vec<f32> {
        dsp::dsss::encoder::Encoder::modulate_silence(self, samples)
    }
}

impl FlushSignal for dsp::dsss::encoder::Encoder {
    fn flush(&mut self) -> Vec<f32> {
        dsp::dsss::encoder::Encoder::flush(self)
    }
}

impl WarmupSignal for dsp::mary::encoder::Encoder {
    fn modulate_silence(&mut self, samples: usize) -> Vec<f32> {
        dsp::mary::encoder::Encoder::modulate_silence(self, samples)
    }
}

impl FlushSignal for dsp::mary::encoder::Encoder {
    fn flush(&mut self) -> Vec<f32> {
        dsp::mary::encoder::Encoder::flush(self)
    }
}

pub fn signal_energy(samples: &[f32]) -> f64 {
    samples
        .iter()
        .map(|&x| {
            let v = x as f64;
            v * v
        })
        .sum()
}
