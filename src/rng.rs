//! A small seedable random number generator for the sampling heuristics (`no_std`, no
//! dependency).

/// The `SplitMix64` generator: small, seedable and good enough for sampling.
#[derive(Clone, Debug)]
pub(crate) struct SplitMix64(pub(crate) u64);

impl SplitMix64 {
    /// The next 64 random bits.
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..n` (`n > 0`), by multiply-shift (bias below `n / 2^64`).
    pub(crate) fn below(&mut self, n: usize) -> usize {
        let wide = u128::from(self.next()) * u128::try_from(n).expect("usize fits u128");
        usize::try_from(wide >> 64).expect("below n")
    }
}
