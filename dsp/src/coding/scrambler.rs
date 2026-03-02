//! スクランブラ (PN9)

/// PN9 スクランブラ/デスクランブラ (G(x) = x^9 + x^5 + 1)
pub struct Scrambler {
    state: u16,
}

impl Scrambler {
    pub const DEFAULT_SEED: u16 = 0x1FF;

    pub fn new(seed: u16) -> Self {
        Scrambler {
            state: if seed == 0 { 0x1FF } else { seed & 0x1FF },
        }
    }

    pub fn default() -> Self {
        Self::new(Self::DEFAULT_SEED)
    }

    /// 状態をリセット
    pub fn reset(&mut self) {
        self.state = Self::DEFAULT_SEED;
    }

    /// 1ビットずつスクランブル
    pub fn next_bit(&mut self) -> u8 {
        // bit 9 XOR bit 5 (0-indexed: 8 XOR 4)
        let bit = ((self.state >> 8) ^ (self.state >> 4)) & 1;
        self.state = ((self.state << 1) | bit) & 0x1FF;
        bit as u8
    }

    /// ビット列をインプレースでスクランブル
    pub fn process_bits(&mut self, bits: &mut [u8]) {
        for bit in bits.iter_mut() {
            *bit ^= self.next_bit();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scrambler_reversibility() {
        let original = vec![0u8, 1, 0, 1, 1, 0, 0, 1];
        let mut bits = original.clone();
        
        let mut s1 = Scrambler::default();
        s1.process_bits(&mut bits);
        
        assert_ne!(original, bits, "Scrambled bits must be different");
        
        let mut s2 = Scrambler::default();
        s2.process_bits(&mut bits);
        
        assert_eq!(original, bits, "Double scramble must return original bits");
    }

    #[test]
    fn test_scrambler_randomness() {
        // 0ばかりのデータがランダムに見えるか確認
        let mut data = vec![0u8; 1000];
        let mut s = Scrambler::default();
        s.process_bits(&mut data);
        
        let ones = data.iter().filter(|&&b| b == 1).count() as u32;
        let total = data.len() as u32;
        
        // 1の出現率が50%に近いことを確認
        let ratio = ones as f32 / total as f32;
        assert!(ratio > 0.4 && ratio < 0.6, "Scrambler should produce roughly 50% ones: ratio={}", ratio);
    }
}
