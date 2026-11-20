//! Float functions `core` lacks under `no_std`.

/// `ln 2`.
const LN_2: f64 = core::f64::consts::LN_2;

/// The natural logarithm of `x`: `-∞` at 0, NaN below 0, accurate to a few ulps.
///
/// Splits `x = m · 2^e` with `m` in `[√½, √2)` and sums `ln m = 2 atanh((m−1)/(m+1))`, whose
/// series converges fast there (`|z| < 0.172`).
pub(crate) fn ln(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }

    if x == 0.0 {
        return f64::NEG_INFINITY;
    }

    if x.is_infinite() {
        return f64::INFINITY;
    }

    if x < f64::MIN_POSITIVE {
        // Subnormal: scale into the normal range first.
        return ln(x * 18_014_398_509_481_984.0) - 54.0 * LN_2;
    }

    let bits = x.to_bits();
    let mut exponent = i32::try_from((bits >> 52) & 0x7ff).expect("11 bits") - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | (1023 << 52));
    if m > core::f64::consts::SQRT_2 {
        m /= 2.0;
        exponent += 1;
    }

    let z = (m - 1.0) / (m + 1.0);
    let z2 = z * z;
    let mut term = z;
    let mut sum: f64 = 0.0;
    let mut k = 1.0;
    while term.abs() > 1e-18 * sum.abs().max(1e-300) {
        sum += term / k;
        term *= z2;
        k += 2.0;
    }

    f64::from(exponent) * LN_2 + 2.0 * sum
}

#[cfg(test)]
mod tests {
    //! `ln` against `std`.

    extern crate std;

    use super::ln;

    /// Agrees with `std` across magnitudes, and handles the special values.
    #[test]
    fn matches_std() {
        for &x in &[
            1.0, 0.5, 0.75, 1e-300, 4.9e-324, 2.0, 3.0, 10.0, 1e300, 0.999_999, 1.000_001,
        ] {
            let (ours, theirs) = (ln(x), x.ln());
            assert!(
                (ours - theirs).abs() <= 1e-14 * theirs.abs().max(1.0),
                "{x}: {ours} vs {theirs}"
            );
        }

        assert_eq!(ln(0.0), f64::NEG_INFINITY);
        assert!(ln(-1.0).is_nan());
    }
}
