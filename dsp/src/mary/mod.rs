//! MaryDQPSK (16-ary + DQPSK) 変復調モジュール
//!
//! 完全に独立したMaryDQPSK実装。既存のDBPSK/DQPSKコードとは
//! 一切依存関係がない。

pub mod modulator;
pub mod demodulator;
pub mod encoder;
pub mod decoder;
pub mod sync;

#[cfg(test)]
mod integration_tests;
