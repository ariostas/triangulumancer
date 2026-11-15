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
//! only reached when the cheaper one cannot decide. Matrices up to `4 x 4`
//! first try a cofactor expansion in checked `i128` (exact, and cheaper than
//! all three when nothing overflows):
//!
//! 1. [`filter`] — Gaussian elimination in `f64` with a running error bound
//!    per entry (midpoint-radius arithmetic). Sound without control over the
//!    rounding mode, which stable Rust does not have: the radii are inflated
//!    by constants that provably absorb their own rounding errors. Decides any
//!    matrix that is not (nearly) singular.
//! 2. [`bareiss`] in checked `i128`. Fraction-free, so every intermediate is a
//!    minor of the input and the Hadamard bound applies to all of them.
//!    Decides exact zeros, and anything whose minors fit in 127 bits.
//! 3. [`bareiss`] over arbitrary-precision integers, on `i128` overflow.
//!
//! Orientation does not matter: a matrix and its transpose have the same
//! determinant, so callers may pass rows or columns, whichever they have.
//!
//! Entries are `i64` for the chirotope. The regular-triangulation predicates
//! need wider ones — coordinate differences take 65 bits, and lifted heights
//! `|p|^2 - w` with exactly scaled `f64` weights need arbitrary precision — so
//! [`sign_of_det_of`] runs the same three tiers over any [`Entry`] type:
//! `i64`, `i128` or [`IBig`].

pub mod bareiss;
pub mod filter;

use std::ops::{Mul, Neg};

use dashu_int::IBig;

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
        3 | 4 => {
            let mut buf = [0i128; 16];
            for (b, &x) in buf.iter_mut().zip(d) {
                *b = i128::from(x);
            }
            det_small(&buf[..d.len()], m.n).map_or_else(|| sign_of_det_tiers(m), Sign::of)
        }
        _ => sign_of_det_tiers(m),
    }
}

/// The three tiers, for `n >= 2`.
fn sign_of_det_tiers(m: SquareMatrix<'_>) -> Sign {
    filter::sign_of_det(m)
        .or_else(|| bareiss::sign_of_det_i128(m))
        .unwrap_or_else(|| bareiss::sign_of_det_bigint(m))
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for i64 {}
    impl Sealed for i128 {}
    impl Sealed for dashu_int::IBig {}
}

/// An exact integer matrix entry: `i64`, `i128` or [`IBig`].
///
/// Sealed; the three implementers are the only ones. The methods are what the
/// tiers need from an entry.
pub trait Entry: sealed::Sealed + Clone {
    /// A midpoint and radius `(m, r)` of finite `f64`s with `|value - m| <= r`,
    /// or `None` if the value is beyond the `f64` range.
    fn approx(&self) -> Option<(f64, f64)>;
    /// The value as an `i128`, if it fits.
    fn to_i128(&self) -> Option<i128>;
    /// The value as an arbitrary-precision integer.
    fn to_ibig(&self) -> IBig;
    /// The sign of the value.
    fn sign(&self) -> Sign;
}

impl Entry for i64 {
    fn approx(&self) -> Option<(f64, f64)> {
        Some(filter::approx_i128(i128::from(*self)))
    }
    fn to_i128(&self) -> Option<i128> {
        Some(i128::from(*self))
    }
    fn to_ibig(&self) -> IBig {
        IBig::from(*self)
    }
    fn sign(&self) -> Sign {
        Sign::of(*self)
    }
}

impl Entry for i128 {
    fn approx(&self) -> Option<(f64, f64)> {
        Some(filter::approx_i128(*self))
    }
    fn to_i128(&self) -> Option<i128> {
        Some(*self)
    }
    fn to_ibig(&self) -> IBig {
        IBig::from(*self)
    }
    fn sign(&self) -> Sign {
        Sign::of(*self)
    }
}

impl Entry for IBig {
    fn approx(&self) -> Option<(f64, f64)> {
        match i128::try_from(self) {
            Ok(v) => Some(filter::approx_i128(v)),
            // Beyond i128, so beyond 2^53: rounded to nearest by dashu.
            Err(_) => filter::approx_rounded(self.to_f64().value()),
        }
    }
    fn to_i128(&self) -> Option<i128> {
        i128::try_from(self).ok()
    }
    fn to_ibig(&self) -> IBig {
        self.clone()
    }
    fn sign(&self) -> Sign {
        if self.is_zero() {
            Sign::Zero
        } else if *self > IBig::ZERO {
            Sign::Positive
        } else {
            Sign::Negative
        }
    }
}

/// The exact sign of the determinant of the `n x n` matrix `m`, for any
/// [`Entry`] type.
///
/// The same three tiers as [`sign_of_det`]: the `f64` filter, then
/// Bareiss in checked `i128` (skipped when an entry does not fit), then Bareiss
/// over [`IBig`]. For `i64` entries the answer is the same as
/// [`sign_of_det`]'s.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] if `m.len() != n * n`.
pub fn sign_of_det_of<T: Entry>(m: &[T], n: usize) -> Result<Sign> {
    let expected = n.checked_mul(n).ok_or(Error::Overflow("matrix size"))?;
    if m.len() != expected {
        return Err(Error::DimensionMismatch {
            expected,
            found: m.len(),
        });
    }
    if (2..=4).contains(&n) {
        let mut buf = [0i128; 16];
        let mut fits = true;
        for (b, x) in buf.iter_mut().zip(m) {
            match x.to_i128() {
                Some(v) => *b = v,
                None => {
                    fits = false;
                    break;
                }
            }
        }
        if let Some(det) = fits.then(|| det_small(&buf[..n * n], n)).flatten() {
            return Ok(Sign::of(det));
        }
    }
    Ok(match n {
        0 => Sign::Positive,
        1 => m[0].sign(),
        _ => filter::sign_of_det_of(m, n)
            .or_else(|| bareiss::sign_of_det_i128_of(m, n))
            .unwrap_or_else(|| bareiss::sign_of_det_bigint_of(m, n)),
    })
}

/// Tiers 2 and 3 of [`sign_of_det_of`] only, for a caller whose own tier-1
/// attempt has already declined. The caller guarantees `m.len() == n * n`.
pub(crate) fn sign_of_det_exact_of<T: Entry>(m: &[T], n: usize) -> Sign {
    match n {
        0 => Sign::Positive,
        1 => m[0].sign(),
        _ => bareiss::sign_of_det_i128_of(m, n)
            .unwrap_or_else(|| bareiss::sign_of_det_bigint_of(m, n)),
    }
}

/// The determinant of a `2 x 2`, `3 x 3` or `4 x 4` matrix by cofactor
/// expansion in checked `i128`, or `None` on overflow.
///
/// Tier 0 of the stack: for the small matrices of the low-dimensional
/// predicates and chirotopes, with entries of moderate size (below about
/// `2^31` for `4 x 4`, `2^41` for `3 x 3`), a dozen exact multiplications are
/// cheaper than either the `f64` filter or Bareiss, and decide exact zeros
/// too. Anything that overflows falls through to the three tiers.
fn det_small(m: &[i128], n: usize) -> Option<i128> {
    // Every term of the expansion takes one entry from each column, so if
    // column `j`'s entries are below `2^b_j` in magnitude, every product of
    // entries from distinct columns — and so every `2 x 2` minor and every
    // product of minors below — is below `2^(Σ b_j + 2)`. With every entry in
    // `i64` and `Σ b_j <= 120`, nothing can overflow, and the products are
    // plain multiplications rather than checked ones.
    let mut total = 0;
    for j in 0..n {
        // The bit length of the largest magnitude in column `j`.
        let or = (0..n).fold(0u128, |acc, i| acc | m[i * n + j].unsigned_abs());
        total += 128 - or.leading_zeros();
    }
    // Past this, some term of the expansion almost certainly overflows, and
    // trying costs as much as the filter that decides it next. Purely a
    // shortcut: every answer below is exact or `None`.
    if total > 140 {
        return None;
    }
    let fits_i64 = m.iter().all(|&x| x.unsigned_abs() < 1 << 63);
    if fits_i64 && total <= 120 {
        let wide = |a: usize, b: usize| i128::from(m[a] as i64) * i128::from(m[b] as i64);
        let minor = |a: usize, b: usize, c: usize, d: usize| wide(a, d) - wide(b, c);
        return Some(match n {
            2 => minor(0, 1, 2, 3),
            3 => m[0] * minor(4, 5, 7, 8) - m[1] * minor(3, 5, 6, 8) + m[2] * minor(3, 4, 6, 7),
            4 => {
                let top = |i: usize, j: usize| minor(i, j, 4 + i, 4 + j);
                let bottom = |i: usize, j: usize| minor(8 + i, 8 + j, 12 + i, 12 + j);
                top(0, 1) * bottom(2, 3) - top(0, 2) * bottom(1, 3)
                    + top(0, 3) * bottom(1, 2)
                    + top(1, 2) * bottom(0, 3)
                    - top(1, 3) * bottom(0, 2)
                    + top(2, 3) * bottom(0, 1)
            }
            _ => return None,
        });
    }
    let minor =
        |a: i128, b: i128, c: i128, d: i128| a.checked_mul(d)?.checked_sub(b.checked_mul(c)?);
    match n {
        2 => minor(m[0], m[1], m[2], m[3]),
        3 => {
            let a = m[0].checked_mul(minor(m[4], m[5], m[7], m[8])?)?;
            let b = m[1].checked_mul(minor(m[3], m[5], m[6], m[8])?)?;
            let c = m[2].checked_mul(minor(m[3], m[4], m[6], m[7])?)?;
            a.checked_sub(b)?.checked_add(c)
        }
        4 => {
            // Laplace expansion along the first two rows: each 2 x 2 minor
            // of rows 0-1 times the complementary minor of rows 2-3.
            let top = |i: usize, j: usize| minor(m[i], m[j], m[4 + i], m[4 + j]);
            let bottom = |i: usize, j: usize| minor(m[8 + i], m[8 + j], m[12 + i], m[12 + j]);
            let term =
                |i: usize, j: usize, k: usize, l: usize| top(i, j)?.checked_mul(bottom(k, l)?);
            term(0, 1, 2, 3)?
                .checked_sub(term(0, 2, 1, 3)?)?
                .checked_add(term(0, 3, 1, 2)?)?
                .checked_add(term(1, 2, 0, 3)?)?
                .checked_sub(term(1, 3, 0, 2)?)?
                .checked_add(term(2, 3, 0, 1)?)
        }
        _ => None,
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

    /// Tier 0 against bigint Bareiss, on matrices whose columns have very
    /// different widths — the case its overflow-free fast path is about —
    /// and on singular ones, at and around the width limits.
    #[test]
    fn small_determinants_agree_with_bareiss() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for trial in 0..20_000 {
            let n = 2 + trial % 3;
            let widths: Vec<u32> = (0..n).map(|_| 1 + (next() % 70) as u32).collect();
            let mut m: Vec<i128> = (0..n * n)
                .map(|k| {
                    let w = widths[k % n];
                    let x = i128::from(next()) << 64 | i128::from(next());
                    x >> (128 - w)
                })
                .collect();
            if trial % 4 == 0 {
                // Singular: the last row a copy of the first.
                let first = m[..n].to_vec();
                m[n * (n - 1)..].copy_from_slice(&first);
            }
            let expected = bareiss::sign_of_det_bigint_of(&m, n);
            if let Some(det) = det_small(&m, n) {
                assert_eq!(Sign::of(det), expected, "{m:?}");
            }
            assert_eq!(sign_of_det_of(&m, n), Ok(expected), "{m:?}");
        }
    }

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
