use std::io::Write;

use anyhow::Result;
use clap::{Parser, ValueEnum};

use dsp::coding::fountain::FountainEncoder;
use dsp::dsss::encoder::Encoder as DsssEncoder;
use dsp::mary::encoder::Encoder as MaryEncoder;
use dsp::params;
use dsp::DspConfig;

/// MaryDQPSK/DSSSエンコーダの出力をWAVファイルとして生成する
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// 変調方式
    #[arg(short, long, default_value = "mary")]
    mode: Mode,

    /// 出力WAVファイルパス
    #[arg(short, long, default_value = "/tmp/output.wav")]
    output: String,

    /// 生成する音声の長さ（秒、目安）
    #[arg(short, long, default_value_t = 10.0)]
    duration: f32,

    /// サンプルレート (Hz)
    #[arg(short, long, default_value_t = 48000)]
    sample_rate: u32,

    /// エンコードするテキスト（UTF-8）
    #[arg(short, long)]
    text: Option<String>,
}

#[derive(Clone, Debug, ValueEnum)]
enum Mode {
    Mary,
    Dsss,
}

fn write_wav_file(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    let num_channels = 1u16;
    let bits_per_sample = 16u16;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let data_size: u32 = (samples.len() * 2) as u32;
    let file_size: u32 = 36 + data_size;

    let mut file = std::fs::File::create(path)?;

    // RIFFヘッダ
    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;

    // fmt chunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(b"\x01\x00")?; // PCM format
    file.write_all(b"\x01\x00")?; // mono
    file.write_all(&(sample_rate.to_le_bytes()))?;
    file.write_all(&(byte_rate.to_le_bytes()))?;
    file.write_all(&(block_align.to_le_bytes()))?;
    file.write_all(&(bits_per_sample.to_le_bytes()))?;

    // data chunk
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;

    // サンプルデータ（f32をi16にクリッピングして書き出し）
    for &sample in samples {
        let scaled = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        file.write_all(&scaled.to_le_bytes())?;
    }

    println!("WAVファイルを生成しました: {}", path);
    println!(
        "  サンプルレート: {} Hz, チャンネル: mono, ビット深度: 16bit",
        sample_rate
    );
    println!(
        "  長さ: {:.2} 秒, サンプル数: {}",
        samples.len() as f32 / sample_rate as f32,
        samples.len()
    );

    Ok(())
}

fn run_mary_encoder(dsp_config: DspConfig, data: &[u8], target_duration_sec: f32) -> Vec<f32> {
    let sample_rate = dsp_config.sample_rate as usize;
    let target_total = (sample_rate as f32 * target_duration_sec) as usize;

    let mut encoder = MaryEncoder::new(dsp_config.clone());
    encoder.set_data(data);

    let mut samples = Vec::new();
    let mut frame_count = 0;

    // フレーム単位で終わるようにする
    loop {
        if let Some(frame) = encoder.encode_frame() {
            samples.extend_from_slice(&frame);
            frame_count += 1;

            // 目標時間を超えていたら、フレームの境界で終了
            if samples.len() >= target_total {
                break;
            }
        } else {
            break;
        }
    }

    println!("フレーム数: {}", frame_count);
    samples
}

fn run_dsss_encoder(dsp_config: DspConfig, data: &[u8], target_duration_sec: f32) -> Vec<f32> {
    let sample_rate = dsp_config.sample_rate as usize;
    let target_total = (sample_rate as f32 * target_duration_sec) as usize;

    let dsss_config = dsp::dsss::params::dsp_config(dsp_config.sample_rate);
    let config = dsp::dsss::encoder::EncoderConfig::new(dsss_config);
    let mut encoder = DsssEncoder::new(config);

    // Fountain encoder を設定
    let needed_k = data.len().div_ceil(params::PAYLOAD_SIZE).max(1);
    encoder.set_fountain_k(needed_k);
    let fountain_params =
        dsp::coding::fountain::FountainParams::new(needed_k, params::PAYLOAD_SIZE);
    let mut fountain_encoder = FountainEncoder::new(data, fountain_params);

    let mut samples = Vec::new();
    let mut frame_count = 0;

    // フレーム単位で終わるようにする
    loop {
        // バーストサイズ分のパケットを生成
        let burst_size = encoder.config().packets_per_sync_burst;
        let mut packets = Vec::with_capacity(burst_size);
        for _ in 0..burst_size {
            packets.push(fountain_encoder.next_packet());
        }

        let frame = encoder.encode_burst(&packets);
        samples.extend_from_slice(&frame);
        frame_count += 1;

        // 目標時間を超えていたら、フレームの境界で終了
        if samples.len() >= target_total {
            break;
        }
    }

    println!("フレーム数: {}", frame_count);
    samples
}

fn main() -> Result<()> {
    let args = Args::parse();

    // テキストをバイト列に変換
    let (data, display_text): (Vec<u8>, String) = if let Some(ref text) = args.text {
        (text.clone().into_bytes(), text.clone())
    } else {
        // デフォルトのテストデータ
        (b"Hello, World!".to_vec(), "Hello, World!".to_string())
    };

    if data.is_empty() {
        anyhow::bail!("データが空です。--text でテキストを指定してください");
    }

    // JS版と同じように、元のデータの先頭にサイズ情報（2バイト、ビッグエンディアン）を埋め込む
    let original_size = data.len();
    let mut data_with_size = Vec::with_capacity(2 + data.len());
    data_with_size.extend_from_slice(&(original_size as u16).to_be_bytes());
    data_with_size.extend_from_slice(&data);

    println!("=== Waveform Generator ===");
    println!("モード: {:?}", args.mode);
    println!(
        "データ: {:?} ({} バイト, ヘッダ込 {} バイト)",
        display_text,
        original_size,
        data_with_size.len()
    );
    println!("サンプルレート: {} Hz", args.sample_rate);
    println!("目安の長さ: {:.1} 秒", args.duration);

    let dsp_config = DspConfig::new(args.sample_rate as f32);

    let samples = match args.mode {
        Mode::Mary => run_mary_encoder(dsp_config, &data_with_size, args.duration),
        Mode::Dsss => run_dsss_encoder(dsp_config, &data_with_size, args.duration),
    };

    println!("サンプル数: {}", samples.len());
    println!(
        "実際の長さ: {:.2} 秒",
        samples.len() as f32 / args.sample_rate as f32
    );

    // WAVファイルに書き出し
    write_wav_file(&args.output, &samples, args.sample_rate)?;

    Ok(())
}
