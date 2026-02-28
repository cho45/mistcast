//! encode_wav: 入力テキストを変調してWAVに保存する例

use dsp::encoder::Encoder;
use dsp::encoder::EncoderConfig;
use dsp::DspConfig;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut input = "Hello, acoustic air-gap world!".to_string();
    let mut output = "out.wav".to_string();
    let mut sample_rate = 48000.0f32;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" if i + 1 < args.len() => { input = args[i + 1].clone(); i += 2; }
            "--output" if i + 1 < args.len() => { output = args[i + 1].clone(); i += 2; }
            "--sample-rate" if i + 1 < args.len() => {
                sample_rate = args[i + 1].parse().unwrap_or(48000.0);
                i += 2;
            }
            _ => i += 1,
        }
    }

    println!("入力データ: {:?} ({} bytes)", input, input.len());
    println!("サンプリングレート: {} Hz", sample_rate);

    let data = input.as_bytes();
    let dsp_config = DspConfig::new(sample_rate);
    let mut encoder = Encoder::new(EncoderConfig::new(dsp_config));
    let mut stream = encoder.encode_stream(data);

    let num_frames = 3;
    let mut all_samples: Vec<f32> = Vec::new();
    for frame_idx in 0..num_frames {
        let frame = stream.next().unwrap();
        println!("フレーム {}: {} サンプル ({:.2} sec)",
            frame_idx, frame.len(), frame.len() as f32 / sample_rate);
        all_samples.extend(frame);
    }

    println!("合計 {} サンプル ({:.2} sec)",
        all_samples.len(), all_samples.len() as f32 / sample_rate);

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(&output, spec).expect("WAVファイルの作成に失敗");
    for &sample in &all_samples {
        writer.write_sample(sample.clamp(-1.0, 1.0)).expect("サンプル書き込みに失敗");
    }
    writer.finalize().expect("WAVファイルのファイナライズに失敗");
    println!("出力: {}", output);
}
