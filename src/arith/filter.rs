//! Tier 1: a floating-point filter for determinant signs.
//!
//! CGAL's filter evaluates predicates in `Interval_nt_advanced` with the FPU
//! rounding mode set towards `+inf`. Stable Rust has no control over the
//! rounding mode, so this filter instead rounds to nearest — the only mode
//! there is — and widens every result outward by one ulp. That is sound: the
//! exact value of an operation on two floats lies between the two
//! representable neighbours of the rounded-to-nearest result, in every finite
//! case including underflow.
//!
//! The determinant is computed by Gaussian elimination with partial pivoting
//! over these intervals. If every pivot interval excludes zero, then the exact
//! elimination with the same row exchanges has non-zero pivots lying inside
//! those intervals (inclusion isotonicity), so the sign of the determinant is
//! the parity of the row exchanges times the product of the pivot signs. If
//! some pivot interval contains zero — always the case for a singular matrix —
//! the filter gives up and the exact tiers decide.

use super::{Sign, SquareMatrix};

/// The next representable `f64` above `x`, for finite `x`.
///
/// `f64::next_up` is stable only from Rust 1.86, past this crate's MSRV.
#[inline]
fn next_up(x: f64) -> f64 {
    if x == 0.0 {
        // Covers -0.0 as well: the smallest positive subnormal.
        return f64::from_bits(1);
    }
    let bits = x.to_bits();
    f64::from_bits(if x > 0.0 { bits + 1 } else { bits - 1 })
}

/// The next representable `f64` below `x`, for finite `x`.
#[inline]
fn next_down(x: f64) -> f64 {
    -next_up(-x)
}

/// A closed interval `[lo, hi]` of reals, with finite endpoints.
#[derive(Debug, Clone, Copy)]
struct Interval {
    lo: f64,
    hi: f64,
}

/// Integers up to this magnitude convert to `f64` exactly.
const EXACT_LIMIT: u64 = 1 << 53;

impl Interval {
    #[inline]
    fn from_i64(x: i64) -> Self {
        let f = x as f64;
        if x.unsigned_abs() <= EXACT_LIMIT {
            Self { lo: f, hi: f }
        } else {
            // Rounded to nearest, so the integer is within one ulp of `f`.
            Self {
                lo: next_down(f),
                hi: next_up(f),
            }
        }
    }

    /// Whether zero is excluded, i.e. the sign of every member is known.
    #[inline]
    fn sign(self) -> Option<Sign> {
        if self.lo > 0.0 {
            Some(Sign::Positive)
        } else if self.hi < 0.0 {
            Some(Sign::Negative)
        } else {
            None
        }
    }

    /// The smallest magnitude of any member; the pivoting criterion.
    #[inline]
    fn mignitude(self) -> f64 {
        if self.lo > 0.0 {
            self.lo
        } else if self.hi < 0.0 {
            -self.hi
        } else {
            0.0
        }
    }

    /// `self - f * other`, or `None` if anything became non-finite.
    #[inline]
    fn sub_mul(self, f: Self, other: Self) -> Option<Self> {
        let p = f.mul(other)?;
        let lo = next_down(self.lo - p.hi);
        let hi = next_up(self.hi - p.lo);
        (lo.is_finite() && hi.is_finite()).then_some(Self { lo, hi })
    }

    #[inline]
    fn mul(self, other: Self) -> Option<Self> {
        let a = self.lo * other.lo;
        let b = self.lo * other.hi;
        let c = self.hi * other.lo;
        let d = self.hi * other.hi;
        // NaN is impossible from finite factors, but infinity is not; and
        // `f64::min` would silently drop a NaN, so check before combining.
        if !(a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite()) {
            return None;
        }
        Some(Self {
            lo: next_down(a.min(b).min(c.min(d))),
            hi: next_up(a.max(b).max(c.max(d))),
        })
    }

    /// `self / other`, for `other` excluding zero.
    #[inline]
    fn div(self, other: Self) -> Option<Self> {
        let a = self.lo / other.lo;
        let b = self.lo / other.hi;
        let c = self.hi / other.lo;
        let d = self.hi / other.hi;
        if !(a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite()) {
            return None;
        }
        Some(Self {
            lo: next_down(a.min(b).min(c.min(d))),
            hi: next_up(a.max(b).max(c.max(d))),
        })
    }
}

/// Inline storage for the matrices the chirotope sees (rank up to 8).
const INLINE: usize = 64;

/// Tier 1: the determinant sign by interval Gaussian elimination, or `None`
/// if the filter cannot decide.
///
/// A `Some` answer is always exact. `None` is returned for every singular
/// matrix, and for nearly singular or badly conditioned ones.
pub fn sign_of_det(m: SquareMatrix<'_>) -> Option<Sign> {
    let n = m.n();
    let src = m.data();
    let zero = Interval { lo: 0.0, hi: 0.0 };
    if src.len() <= INLINE {
        let mut buf = [zero; INLINE];
        for (dst, &x) in buf.iter_mut().zip(src) {
            *dst = Interval::from_i64(x);
        }
        eliminate(&mut buf[..src.len()], n)
    } else {
        let mut buf: Vec<Interval> = src.iter().map(|&x| Interval::from_i64(x)).collect();
        eliminate(&mut buf, n)
    }
}

fn eliminate(m: &mut [Interval], n: usize) -> Option<Sign> {
    let mut sign = Sign::Positive;
    for k in 0..n {
        // Partial pivoting on the mignitude: the row whose entry is furthest
        // from containing zero.
        let (p, best) = (k..n)
            .map(|i| (i, m[i * n + k].mignitude()))
            .fold((k, -1.0), |acc, x| if x.1 > acc.1 { x } else { acc });
        if best <= 0.0 {
            return None;
        }
        if p != k {
            for c in k..n {
                m.swap(k * n + c, p * n + c);
            }
            sign = -sign;
        }
        let (top, bottom) = m.split_at_mut((k + 1) * n);
        let pivot_row = &top[k * n..];
        let pivot = pivot_row[k];
        sign = sign * pivot.sign()?;
        for row in bottom.chunks_exact_mut(n) {
            let f = row[k].div(pivot)?;
            for j in k + 1..n {
                row[j] = row[j].sub_mul(f, pivot_row[j])?;
            }
        }
    }
    Some(sign)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbours() {
        assert!(next_up(1.0) > 1.0);
        assert!(next_down(1.0) < 1.0);
        assert!(next_up(-1.0) > -1.0);
        assert!(next_down(-1.0) < -1.0);
        assert!(next_up(0.0) > 0.0);
        assert!(next_down(0.0) < 0.0);
        assert!(next_up(-0.0) > 0.0);
        assert_eq!(next_down(next_up(1.0)), 1.0);
    }

    #[test]
    fn inexact_conversion_is_bracketed() {
        let x = (1i64 << 53) + 1;
        let i = Interval::from_i64(x);
        assert!(i.lo <= 9007199254740992.0 && i.hi >= 9007199254740994.0);
        let i = Interval::from_i64(i64::MIN);
        assert!(i.lo < -9.223372036854775e18 && i.hi > -9.223372036854777e18);
    }

    #[test]
    fn decides_regular_and_declines_singular() {
        let m = [2, -1, 0, -1, 2, -1, 0, -1, 2];
        assert_eq!(
            sign_of_det(SquareMatrix::new(&m, 3).unwrap()),
            Some(Sign::Positive)
        );
        let m = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(sign_of_det(SquareMatrix::new(&m, 3).unwrap()), None);
        let m = [0, 1, 1, 0];
        assert_eq!(
            sign_of_det(SquareMatrix::new(&m, 2).unwrap()),
            Some(Sign::Negative)
        );
    }
}
