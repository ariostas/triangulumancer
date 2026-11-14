//! Exact determinant signs: the shared foundation of the whole crate.
//!
//! Both upstream libraries reduce to one primitive. TOPCOM's chirotope is the
//! sign of an `r x r` determinant of configuration columns, and CGAL's
//! `orientation` and `power_side_of_power_sphere` predicates are determinant
//! signs too — the latter being the former one dimension up, on lifted points.
//! A single wrong sign silently changes the oriented matroid, so this module is
//! exact, and is differentially tested against a cofactor-expansion oracle in
//! `tests/sign_of_det.rs`.
//!
//! [`sign_of_det`] runs a three-tier stack (`PLAN.md` section 5.1), each tier
//! only reached when the cheaper one cannot decide:
//!
//! 1. [`filter`] — Gaussian elimination in `f64` interval arithmetic. Sound
//!    without control over the rounding mode, which stable Rust does not have:
//!    every operation is rounded to nearest and then widened outward by one
//!    ulp. Decides any matrix that is not (nearly) singular.
//! 2. [`bareiss`] in checked `i128`. Fraction-free, so every intermediate is a
//!    minor of the input and the Hadamard bound applies to all of them.
//!    Decides exact zeros, and anything whose minors fit in 127 bits.
//! 3. [`bareiss`] over arbitrary-precision integers, on `i128` overflow.
//!
//! Orientation does not matter: a matrix and its transpose have the same
//! determinant, so callers may pass rows or columns, whichever they have.

pub mod bareiss;
pub mod filter;

use std::ops::{Mul, Neg};

use crate::error::{Error, Result};

/// The sign of an exact quantity.
///
/// Marked `#[non_exhaustive]` only to follow the crate convention for public
/// enums; a sign has three values and always will.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Sign {
    /// Strictly negative.
    Negative,
    /// Exactly zero.
    Zero,
    /// Strictly positive.
    Positive,
}

impl Sign {
    /// The sign of an integer.
    pub fn of<T: Into<i128>>(x: T) -> Self {
        match x.into() {
            0 => Self::Zero,
            v if v > 0 => Self::Positive,
            _ => Self::Negative,
        }
    }

    /// `-1`, `0` or `1`.
    pub fn as_i8(self) -> i8 {
        match self {
            Self::Negative => -1,
            Self::Zero => 0,
            Self::Positive => 1,
        }
    }

    /// Whether this is [`Sign::Zero`].
    pub fn is_zero(self) -> bool {
        self == Self::Zero
    }
}

impl Neg for Sign {
    type Output = Self;

    fn neg(self) -> Self {
        match self {
            Self::Negative => Self::Positive,
            Self::Zero => Self::Zero,
            Self::Positive => Self::Negative,
        }
    }
}

impl Mul for Sign {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Zero, _) | (_, Self::Zero) => Self::Zero,
            (a, b) if a == b => Self::Positive,
            _ => Self::Negative,
        }
    }
}

/// A borrowed `n x n` integer matrix whose shape has been checked.
///
/// Row-major or column-major are equally valid readings: the determinant of a
/// matrix and of its transpose agree. The tier functions in [`filter`] and
/// [`bareiss`] take this type so that they cannot be handed a malformed slice.
#[derive(Debug, Clone, Copy)]
pub struct SquareMatrix<'a> {
    data: &'a [i64],
    n: usize,
}

impl<'a> SquareMatrix<'a> {
    /// View `data` as an `n x n` matrix.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if `data.len() != n * n`.
    pub fn new(data: &'a [i64], n: usize) -> Result<Self> {
        let expected = n.checked_mul(n).ok_or(Error::Overflow("matrix size"))?;
        if data.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                found: data.len(),
            });
        }
        Ok(Self { data, n })
    }

    /// The side length.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The entries, `n * n` of them.
    pub fn data(&self) -> &'a [i64] {
        self.data
    }
}

/// The exact sign of the determinant of the `n x n` matrix `m`.
///
/// The empty (`0 x 0`) matrix has determinant `1`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] if `m.len() != n * n`. The computation itself
/// cannot fail.
pub fn sign_of_det(m: &[i64], n: usize) -> Result<Sign> {
    Ok(sign_of_det_square(SquareMatrix::new(m, n)?))
}

/// [`sign_of_det`] for an already validated matrix.
pub fn sign_of_det_square(m: SquareMatrix<'_>) -> Sign {
    let d = m.data;
    match m.n {
        0 => Sign::Positive,
        1 => Sign::of(d[0]),
        // Exact in i128: each product is at most 2^126 in magnitude, and the
        // difference at most 2^127 - 2^63.
        2 => Sign::of(d[0] as i128 * d[3] as i128 - d[1] as i128 * d[2] as i128),
        _ => filter::sign_of_det(m)
            .or_else(|| bareiss::sign_of_det_i128(m))
            .unwrap_or_else(|| bareiss::sign_of_det_bigint(m)),
    }
}

/// The exact rank of the `rows x cols` matrix `m`, read row-major.
///
/// As with determinants, reading the data column-major gives the transpose,
/// which has the same rank.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] if `m.len() != rows * cols`.
pub fn rank(m: &[i64], rows: usize, cols: usize) -> Result<usize> {
    let expected = rows
        .checked_mul(cols)
        .ok_or(Error::Overflow("matrix size"))?;
    if m.len() != expected {
        return Err(Error::DimensionMismatch {
            expected,
            found: m.len(),
        });
    }
    Ok(bareiss::rank_i128(m, rows, cols).unwrap_or_else(|| bareiss::rank_bigint(m, rows, cols)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_algebra() {
        use Sign::*;
        assert_eq!(-Positive, Negative);
        assert_eq!(-Zero, Zero);
        assert_eq!(Negative * Negative, Positive);
        assert_eq!(Negative * Positive, Negative);
        assert_eq!(Zero * Negative, Zero);
        assert_eq!(Sign::of(-5i64), Negative);
        assert_eq!(Sign::of(0i64), Zero);
        assert_eq!(Positive.as_i8(), 1);
    }

    #[test]
    fn small_determinants() {
        assert_eq!(sign_of_det(&[], 0), Ok(Sign::Positive));
        assert_eq!(sign_of_det(&[-3], 1), Ok(Sign::Negative));
        assert_eq!(sign_of_det(&[1, 2, 3, 4], 2), Ok(Sign::Negative));
        assert_eq!(sign_of_det(&[1, 2, 2, 4], 2), Ok(Sign::Zero));
        let extreme = [i64::MIN, i64::MIN, i64::MIN, i64::MAX];
        // MIN * MAX - MIN * MIN < 0
        assert_eq!(sign_of_det(&extreme, 2), Ok(Sign::Negative));
        assert!(sign_of_det(&[1, 2, 3], 2).is_err());
    }

    #[test]
    fn ranks() {
        assert_eq!(rank(&[], 0, 0), Ok(0));
        assert_eq!(rank(&[0, 0, 0, 0], 2, 2), Ok(0));
        assert_eq!(rank(&[1, 2, 3, 2, 4, 6], 2, 3), Ok(1));
        assert_eq!(rank(&[1, 2, 3, 2, 4, 7], 2, 3), Ok(2));
        assert_eq!(rank(&[0, 1, 0, 0, 0, 1, 0, 0, 0], 3, 3), Ok(2));
        let big = [i64::MAX, i64::MIN, i64::MIN, i64::MAX, 1, 1];
        assert_eq!(rank(&big, 3, 2), Ok(2));
        assert!(rank(&[1, 2, 3], 2, 2).is_err());
    }
}
