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

fn too_wide(index: usize) -> Error {
    invalid(
        index,
        "the lifts span too many binary orders of magnitude to be compared exactly \
         (see triangulumancer::regular::lift::MAX_LIFT_BITS)",
    )
}

/// `a - b`, in lowest terms.
fn difference(a: &Ratio, b: &Ratio) -> Ratio {
    let num = &a.num * IBig::from(b.den.clone()) - &b.num * IBig::from(a.den.clone());
    if num.is_zero() {
        return Ratio {
            num,
            den: UBig::ONE,
        };
    }
    let den = &a.den * &b.den;
    let g = (&num).unsigned_abs().gcd(&den);
    Ratio {
        num: num / IBig::from(g.clone()),
        den: den / g,
    }
}

/// Bring exact lifts to integers, normalised so that the smallest is zero and
/// they have no common factor.
///
/// Subtracting one constant from every lift, or dividing every lift by one
/// positive constant, changes no predicate sign (the lifted points move by an
/// affine map of the lifting direction), so neither changes the regular
/// triangulation, and normalising makes the [`MAX_LIFT_BITS`] check
/// invariant under both: heights that are all `2^1300` are the same as heights
/// that are all zero. The symbolic perturbation that breaks ties depends only
/// on the order of the points, not on the lifts, so it is unaffected too.
///
/// The common denominator is checked as it grows, so an input of many large,
/// pairwise coprime denominators fails early instead of building a huge
/// number first; the early failure is exactly the final check's verdict (see
/// the comment in the loop).
fn scale(ratios: Vec<Ratio>) -> Result<Vec<IBig>> {
    let Some(min) = ratios
        .iter()
        .min_by(|a, b| {
            (&a.num * IBig::from(b.den.clone())).cmp(&(&b.num * IBig::from(a.den.clone())))
        })
        .cloned()
    else {
        return Ok(Vec::new());
    };
    let mut diffs: Vec<Ratio> = ratios.iter().map(|r| difference(r, &min)).collect();
    // Divide out the numerators' common factor. Every fraction is in lowest
    // terms, so that factor is coprime to every denominator and they stay so.
    let g = diffs
        .iter()
        .filter(|r| !r.num.is_zero())
        .fold(UBig::ZERO, |g, r| {
            // dashu's gcd panics on a zero argument.
            if g.is_zero() {
                (&r.num).unsigned_abs()
            } else {
                g.gcd((&r.num).unsigned_abs())
            }
        });
    if g > UBig::ONE {
        let g = IBig::from(g);
        for r in &mut diffs {
            r.num = &r.num / &g;
        }
    }
    let max_den_bits = diffs.iter().map(|r| r.den.bit_len()).max().unwrap_or(0);
    let mut common = UBig::ONE;
    for (index, r) in diffs.iter().enumerate() {
        let g = (&common).gcd(&r.den);
        common = &common / &g * &r.den;
        // Each non-zero lift `num * common / den` (zero lifts have `den = 1`)
        // is at least `common / den >= common / 2^max_den_bits` in absolute
        // value, and `common` only grows; so past this bound every non-zero
        // lift is already too wide, and there is at least one, since the
        // denominator would otherwise still be one.
        if common.bit_len() > MAX_LIFT_BITS + max_den_bits {
            return Err(too_wide(index));
        }
    }
    let lifts: Vec<IBig> = diffs
        .into_iter()
        .map(|r| r.num * IBig::from(&common / &r.den))
        .collect();
    for (index, l) in lifts.iter().enumerate() {
        if l.unsigned_abs().bit_len() > MAX_LIFT_BITS {
            return Err(too_wide(index));
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
        // 1/2, 2, -1/3 shifted by 1/3 and over 6: 5, 14, 0.
        let l = from_heights(&c, &h).unwrap();
        assert_eq!(l, [IBig::from(5), IBig::from(14), IBig::from(0)]);
    }

    #[test]
    fn weights_use_original_coordinates_without_overflow() {
        let c = line();
        let l = from_weights(&c, &[1.into(), 0.into(), 0.5.into()]).unwrap();
        // |p|^2 - w = -1, 9, 10^10 - 1/2; shifted by 1 and over 2.
        assert_eq!(
            l,
            [
                IBig::from(0),
                IBig::from(20),
                IBig::from(20_000_000_000i64 + 1)
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

    fn int(x: IBig) -> Scalar {
        Scalar::Integer(x)
    }

    #[test]
    fn the_cap_is_invariant_under_shift_and_scale() {
        let c = line();
        let huge = IBig::ONE << 1300;
        // All equal: the same as all zero.
        let same = vec![int(huge.clone()); 3];
        assert_eq!(from_heights(&c, &same).unwrap(), [IBig::ZERO; 3]);
        assert_eq!(
            from_heights(&c, &vec![Scalar::Float(1e300); 3]).unwrap(),
            [IBig::ZERO; 3]
        );
        // A huge common offset, or a huge common factor, is normalised away.
        let shifted = [
            int(&huge + IBig::from(2)),
            int(huge.clone()),
            int(&huge + IBig::from(7)),
        ];
        let expected = [IBig::from(2), IBig::from(0), IBig::from(7)];
        assert_eq!(from_heights(&c, &shifted).unwrap(), expected);
        let scaled = [
            int(&huge * IBig::from(2)),
            0.into(),
            int(&huge * IBig::from(7)),
        ];
        assert_eq!(from_heights(&c, &scaled).unwrap(), expected);
        let frac = |n: i64, d: IBig| Scalar::Rational {
            numerator: IBig::from(n),
            denominator: d,
        };
        let tiny = [frac(2, huge.clone()), 0.into(), frac(7, huge.clone())];
        assert_eq!(from_heights(&c, &tiny).unwrap(), expected);
        // Weights too: |p|^2 - w with w all equal to 2^1300.
        let w = vec![int(huge.clone()); 3];
        assert_eq!(
            from_weights(&c, &w).unwrap(),
            from_weights(&c, &[0.into(), 0.into(), 0.into()]).unwrap()
        );
        // A genuine spread is still too wide, shifted or not.
        let spread = [int(huge.clone()), 0.into(), 1.into()];
        assert!(matches!(
            from_heights(&c, &spread),
            Err(Error::InvalidParameter { .. })
        ));
    }

    #[test]
    fn many_coprime_denominators_fail_early() {
        // 2000 distinct primes near 2^64 as denominators: their lcm would be
        // ~128 000 bits, built by thousands of ever larger multiplications.
        // The growing common denominator is checked as it goes.
        let n = 2000;
        let c = PointConfig::from_row_major(&vec![0; n], n, 1).unwrap();
        let mut primes = Vec::new();
        let mut p: u64 = u64::MAX;
        while primes.len() < n {
            p -= 2;
            if is_prime(p) {
                primes.push(p);
            }
        }
        let h: Vec<Scalar> = primes
            .iter()
            .map(|&p| Scalar::Rational {
                numerator: IBig::ONE,
                denominator: IBig::from(p),
            })
            .collect();
        match from_heights(&c, &h) {
            Err(Error::InvalidParameter { index, .. }) => assert!(index < 40, "{index}"),
            other => panic!("{other:?}"),
        }
    }

    fn is_prime(n: u64) -> bool {
        // Deterministic Miller-Rabin for 64-bit integers.
        let mul = |a: u64, b: u64| ((a as u128 * b as u128) % n as u128) as u64;
        let pow = |mut b: u64, mut e: u64| {
            let mut r = 1;
            while e > 0 {
                if e & 1 == 1 {
                    r = mul(r, b);
                }
                b = mul(b, b);
                e >>= 1;
            }
            r
        };
        let (mut d, mut s) = (n - 1, 0);
        while d % 2 == 0 {
            d /= 2;
            s += 1;
        }
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
            .iter()
            .all(|&a| {
                let mut x = pow(a, d);
                if x == 1 || x == n - 1 {
                    return true;
                }
                for _ in 1..s {
                    x = mul(x, x);
                    if x == n - 1 {
                        return true;
                    }
                }
                false
            })
    }
}
