//! The lifting map `p ↦ (p, ℓ(p))`, made exact.
//!
//! A regular triangulation is the projection of the lower hull of the lifted
//! points (`CGAL.md` section 2.2). The project's *height* is the lift itself;
//! a *weight* `w` gives `ℓ = |p|² − w`; Delaunay is `ℓ = |p|²`.
//!
//! Every predicate on lifted points is a determinant that is linear in the
//! column of lifts, so multiplying every lift by one positive constant changes
//! no sign. The lifts are therefore brought to a common denominator and
//! carried as integers ([`IBig`]): an `f64` is an exact dyadic rational, an
//! integer is itself, and a rational contributes its denominator to the
//! common one (`CGAL.md` section 5.3, `PLAN.md` section 5.1).
//!
//! This is also where bug B3 is fixed: 0.3 computed `|p|²` in 32-bit `int`
//! arithmetic, which overflows for coordinates of `46341` and beyond. Here it
//! is exact at any `i64` coordinate.
//!
//! `|p|²` is always taken from the *original* coordinates, never from the
//! hull-reduced ones, which do not preserve distances (`PLAN.md` section 13,
//! "Phase 1 decisions").

use dashu_int::ops::{BitTest, Gcd, UnsignedAbs};
use dashu_int::{IBig, UBig};

use crate::config::PointConfig;
use crate::error::{Error, Result};

/// Lifts wider than this many bits after scaling to a common denominator are
/// rejected, rather than silently making every predicate slower.
///
/// For `f64` input the width is roughly the exponent spread of the values plus
/// 53 bits, plus about 130 bits for `|p|²` at the extremes of `i64`. The cap
/// admits any spread up to about a thousand binary orders of magnitude (e.g.
/// heights of `1e-150` next to `1e150`), which no meaningful input reaches.
pub const MAX_LIFT_BITS: usize = 1280;

/// An exact real number: a weight or a height.
///
/// `f64` values are taken at their exact binary value, so `0.1` means
/// `3602879701896397 / 2^55`, not one tenth; pass a [`Rational`] for one
/// tenth. Integers and rationals keep the whole computation exact with no
/// scaling at all, and are the recommended input for lattice-polytope work.
///
/// [`Rational`]: Self::Rational
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Scalar {
    /// A finite, non-subnormal `f64`.
    Float(f64),
    /// An integer.
    Integer(IBig),
    /// The rational `numerator / denominator`, with a non-zero denominator.
    Rational {
        /// The numerator.
        numerator: IBig,
        /// The denominator; non-zero, of either sign.
        denominator: IBig,
    },
}

impl From<f64> for Scalar {
    fn from(x: f64) -> Self {
        Self::Float(x)
    }
}

impl From<i64> for Scalar {
    fn from(x: i64) -> Self {
        Self::Integer(IBig::from(x))
    }
}

impl From<i32> for Scalar {
    fn from(x: i32) -> Self {
        Self::Integer(IBig::from(x))
    }
}

impl From<IBig> for Scalar {
    fn from(x: IBig) -> Self {
        Self::Integer(x)
    }
}

/// `num / den` with `den > 0`.
#[derive(Debug, Clone, PartialEq)]
struct Ratio {
    num: IBig,
    den: UBig,
}

fn invalid(index: usize, reason: &'static str) -> Error {
    Error::InvalidParameter { index, reason }
}

impl Scalar {
    /// The exact value, or an error naming parameter `index`.
    fn to_ratio(&self, index: usize) -> Result<Ratio> {
        match self {
            Self::Float(f) => {
                if !f.is_finite() {
                    return Err(invalid(index, "not finite"));
                }
                if f.is_subnormal() {
                    return Err(invalid(index, "subnormal floats are not accepted"));
                }
                if *f == 0.0 {
                    return Ok(Ratio {
                        num: IBig::ZERO,
                        den: UBig::ONE,
                    });
                }
                let bits = f.to_bits();
                // Normal: (-1)^s * 1.m * 2^(e - 1023) = (2^52 + m) * 2^(e - 1075).
                let mut exponent = ((bits >> 52) & 0x7ff) as i64 - 1075;
                let mut mantissa = (bits & ((1 << 52) - 1)) | (1 << 52);
                // Lowest terms, so that the common denominator stays small.
                if exponent < 0 {
                    let shift = i64::from(mantissa.trailing_zeros()).min(-exponent);
                    mantissa >>= shift;
                    exponent += shift;
                }
                let mantissa = IBig::from(mantissa);
                let mantissa = if *f < 0.0 { -mantissa } else { mantissa };
                Ok(if exponent >= 0 {
                    Ratio {
                        num: mantissa << exponent as usize,
                        den: UBig::ONE,
                    }
                } else {
                    Ratio {
                        num: mantissa,
                        den: UBig::ONE << (-exponent) as usize,
                    }
                })
            }
            Self::Integer(n) => Ok(Ratio {
                num: n.clone(),
                den: UBig::ONE,
            }),
            Self::Rational {
                numerator,
                denominator,
            } => {
                if denominator.is_zero() {
                    return Err(invalid(index, "zero denominator"));
                }
                let negative = *denominator < IBig::ZERO;
                let num = if negative {
                    -numerator.clone()
                } else {
                    numerator.clone()
                };
                Ok(Ratio {
                    num,
                    den: denominator.unsigned_abs(),
                })
            }
        }
    }
}

/// `|p|²` of point `i`, in the original coordinates.
fn norm_squared(config: &PointConfig, i: usize) -> IBig {
    config
        .coords(i)
        .unwrap_or(&[])
        .iter()
        .map(|&x| {
            let x = IBig::from(x);
            &x * &x
        })
        .sum()
}

/// Bring exact lifts to a common denominator, as integers.
fn scale(ratios: Vec<Ratio>) -> Result<Vec<IBig>> {
    let mut common = UBig::ONE;
    for r in &ratios {
        let g = (&common).gcd(&r.den);
        common = &common / &g * &r.den;
    }
    let lifts: Vec<IBig> = ratios
        .into_iter()
        .map(|r| r.num * IBig::from(&common / &r.den))
        .collect();
    for (index, l) in lifts.iter().enumerate() {
        if l.unsigned_abs().bit_len() > MAX_LIFT_BITS {
            return Err(invalid(
                index,
                "the lifts span too many binary orders of magnitude to be compared exactly \
                 (see triangulumancer::regular::lift::MAX_LIFT_BITS)",
            ));
        }
    }
    Ok(lifts)
}

fn check_count(config: &PointConfig, found: usize) -> Result<()> {
    if found != config.len() {
        return Err(Error::ParameterCountMismatch {
            expected: config.len(),
            found,
        });
    }
    Ok(())
}

/// The scaled integer lifts for heights: `ℓᵢ = hᵢ`.
///
/// # Errors
///
/// [`Error::ParameterCountMismatch`] for the wrong number of heights, and
/// [`Error::InvalidParameter`] for a non-finite or subnormal `f64`, a zero
/// denominator, or lifts beyond [`MAX_LIFT_BITS`].
pub fn from_heights(config: &PointConfig, heights: &[Scalar]) -> Result<Vec<IBig>> {
    check_count(config, heights.len())?;
    let ratios = heights
        .iter()
        .enumerate()
        .map(|(i, h)| h.to_ratio(i))
        .collect::<Result<_>>()?;
    scale(ratios)
}

/// The scaled integer lifts for weights: `ℓᵢ = |pᵢ|² − wᵢ`.
///
/// # Errors
///
/// As [`from_heights`].
pub fn from_weights(config: &PointConfig, weights: &[Scalar]) -> Result<Vec<IBig>> {
    check_count(config, weights.len())?;
    let ratios = weights
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let w = w.to_ratio(i)?;
            let norm = norm_squared(config, i);
            Ok(Ratio {
                num: norm * IBig::from(w.den.clone()) - w.num,
                den: w.den,
            })
        })
        .collect::<Result<_>>()?;
    scale(ratios)
}

/// The lifts of the Delaunay triangulation: `ℓᵢ = |pᵢ|²`.
pub fn delaunay(config: &PointConfig) -> Vec<IBig> {
    (0..config.len()).map(|i| norm_squared(config, i)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line() -> PointConfig {
        PointConfig::from_row_major(&[0, 3, -100_000], 3, 1).unwrap()
    }

    #[test]
    fn floats_are_exact_dyadic_rationals() {
        let r = Scalar::Float(0.75).to_ratio(0).unwrap();
        assert_eq!((r.num, r.den), (IBig::from(3), UBig::from(4u8)));
        let r = Scalar::Float(-6.0).to_ratio(0).unwrap();
        assert_eq!(r.num / IBig::from(r.den), IBig::from(-6));
        let r = Scalar::Float(0.1).to_ratio(0).unwrap();
        assert_eq!(r.num, IBig::from(0xc_cccc_cccc_cccd_u64));
        assert_eq!(r.den, UBig::ONE << 55);
        for bad in [f64::NAN, f64::INFINITY, f64::MIN_POSITIVE / 2.0] {
            assert!(matches!(
                Scalar::Float(bad).to_ratio(3),
                Err(Error::InvalidParameter { index: 3, .. })
            ));
        }
    }

    #[test]
    fn common_denominator() {
        let c = line();
        let h = [
            Scalar::Float(0.5),
            Scalar::from(2),
            Scalar::Rational {
                numerator: IBig::from(1),
                denominator: IBig::from(-3),
            },
        ];
        // 1/2, 2, -1/3 over 6: 3, 12, -2.
        let l = from_heights(&c, &h).unwrap();
        assert_eq!(l, [IBig::from(3), IBig::from(12), IBig::from(-2)]);
    }

    #[test]
    fn weights_use_original_coordinates_without_overflow() {
        let c = line();
        let l = from_weights(&c, &[1.into(), 0.into(), 0.5.into()]).unwrap();
        // |p|^2 - w over 2: -2, 18, 2 * 10^10 - 1.
        assert_eq!(
            l,
            [
                IBig::from(-2),
                IBig::from(18),
                IBig::from(20_000_000_000i64 - 1)
            ]
        );
        let big = PointConfig::from_row_major(&[i64::MIN, i64::MAX], 1, 2).unwrap();
        let expected = IBig::from(i64::MIN) * IBig::from(i64::MIN)
            + IBig::from(i64::MAX) * IBig::from(i64::MAX);
        assert_eq!(delaunay(&big), [expected]);
    }

    #[test]
    fn bad_parameters() {
        let c = line();
        assert_eq!(
            from_heights(&c, &[1.into()]),
            Err(Error::ParameterCountMismatch {
                expected: 3,
                found: 1
            })
        );
        let zero = Scalar::Rational {
            numerator: IBig::ONE,
            denominator: IBig::ZERO,
        };
        assert!(from_weights(&c, &[1.into(), zero, 1.into()]).is_err());
        let spread = [Scalar::Float(1e-200), Scalar::Float(1e200), 0.into()];
        assert!(matches!(
            from_heights(&c, &spread),
            Err(Error::InvalidParameter { .. })
        ));
        let fine = [Scalar::Float(1e-100), Scalar::Float(1e100), 0.into()];
        assert!(from_heights(&c, &fine).is_ok());
    }
}
