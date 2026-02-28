//! 音響通信DSPシステム
//!
//! エアギャップ環境での音響通信を実現するDSPコアライブラリ。
//! DSSS + DBPSK 変調、RRCフィルタ、FEC、LT符号(Fountain Code)を実装する。

pub mod common;
pub mod phy;
pub mod coding;
pub mod frame;

pub mod decoder;
pub mod encoder;

/// システムデフォルト定数
///
/// これらはデフォルト値であり、実行時に `DspConfig` で上書き可能。
pub mod params {
    /// デフォルトのサンプリングレート (Hz)
    /// 実際のサンプリングレートは DspConfig で設定すること。
    pub const DEFAULT_SAMPLE_RATE: f32 = 48000.0;

    /// キャリア周波数 (Hz) - 帯域の中心
    pub const CARRIER_FREQ: f32 = 8000.0;

    /// M系列の次数 (拡散符号長 = 2^ORDER - 1 = 31)
    pub const MSEQ_ORDER: usize = 5;

    /// 拡散符号長 N = 31
    pub const SPREAD_FACTOR: usize = 31;

    /// RRCフィルタのロールオフ係数
    pub const RRC_ALPHA: f32 = 0.35;

    /// チップレート (chips/sec)
    /// 48kHz / 8000 = 6 サンプル/チップ (整数比)
    /// 占有帯域: 8000 * (1 + 0.35) = 10800 Hz (24kHz以下でOK)
    pub const CHIP_RATE: f32 = 8000.0;

    /// プリアンブルのM系列繰り返し数
    pub const PREAMBLE_REPEAT: usize = 4;

    /// 同期ワード (固定パターン)
    pub const SYNC_WORD: u32 = 0xDEAD_BEEF;

    /// パケットペイロードサイズ (bytes)
    pub const PAYLOAD_SIZE: usize = 16;

    /// LT符号のオーバーヘッド係数 (K+αのα)
    pub const FOUNTAIN_OVERHEAD: f32 = 0.1;
}

/// DSP動作設定
///
/// サンプリングレートはWebAudio環境によって 48kHz / 44.1kHz などが変わるため、
/// ランタイムで設定する。
#[derive(Clone, Debug)]
pub struct DspConfig {
    /// サンプリングレート (Hz)。48000.0 または 44100.0 など。
    pub sample_rate: f32,
    /// キャリア周波数 (Hz)
    pub carrier_freq: f32,
    /// M系列次数
    pub mseq_order: usize,
    /// チップレート (chips/sec)
    pub chip_rate: f32,
    /// RRCロールオフ係数
    pub rrc_alpha: f32,
    /// RRCフィルタのタップ数 (シンボル当たり)
    pub rrc_taps_per_symbol: usize,
    /// プリアンブルのM系列繰り返し数
    pub preamble_repeat: usize,
}

impl DspConfig {
    /// 指定サンプリングレートで標準設定を作成する
    pub fn new(sample_rate: f32) -> Self {
        DspConfig {
            sample_rate,
            carrier_freq: params::CARRIER_FREQ,
            mseq_order: params::MSEQ_ORDER,
            chip_rate: params::CHIP_RATE,
            rrc_alpha: params::RRC_ALPHA,
            rrc_taps_per_symbol: 16, // 16シンボル分のタップ
            preamble_repeat: params::PREAMBLE_REPEAT,
        }
    }

    /// 48kHz のデフォルト設定
    pub fn default_48k() -> Self {
        Self::new(params::DEFAULT_SAMPLE_RATE)
    }

    /// 44.1kHz の設定
    pub fn default_44k() -> Self {
        Self::new(44100.0)
    }

    /// 1チップあたりのサンプル数 (= floor(Fs / Rc))
    #[inline]
    pub fn samples_per_chip(&self) -> usize {
        (self.sample_rate / self.chip_rate) as usize
    }

    /// M系列の1周期長 (= 2^order - 1)
    #[inline]
    pub fn spread_factor(&self) -> usize {
        (1 << self.mseq_order) - 1
    }

    /// 1シンボルあたりのサンプル数
    #[inline]
    pub fn samples_per_symbol(&self) -> usize {
        self.samples_per_chip() * self.spread_factor()
    }

    /// RRCフィルタのタップ数
    #[inline]
    pub fn rrc_num_taps(&self) -> usize {
        self.rrc_taps_per_symbol * self.samples_per_chip() + 1
    }
}

impl Default for DspConfig {
    fn default() -> Self {
        Self::default_48k()
    }
}
