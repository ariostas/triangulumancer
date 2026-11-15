//! Fraction-free (Bareiss) elimination over the integers.
//!
//! Plain Gaussian elimination over the rationals inflates intermediates far
//! beyond the size of the determinant — this is what TOPCOM's `StairCaseMatrix`
//! does over `mpq_class`. Bareiss' update
//!
//! ```text
//! a[i][j] <- (a[i][j] * a[k][k] - a[i][k] * a[k][j]) / prev
//! ```
//!
//! divides exactly, and by Sylvester's identity every intermediate is a minor
//! of the input. So the Hadamard bound on the input bounds every intermediate,
//! which is what makes a fixed-width tier worth having at all.
//!
//! Two instantiations share one generic routine: checked `i128`, which reports
//! overflow as `None`, and arbitrary-precision [`IBig`], which cannot fail.

use dashu_int::IBig;

use super::{Entry, Sign, SquareMatrix};

/// The integer arithmetic Bareiss elimination needs.
trait Exact: Clone {
    fn zero() -> Self;
    fn one() -> Self;
    fn is_zero(&self) -> bool;
    fn sign(&self) -> Sign;
    /// `(a * b - c * d) / e`, where the division is known to be exact.
    /// `None` on overflow.
    fn bareiss(a: &Self, b: &Self, c: &Self, d: &Self, e: &Self) -> Option<Self>;
}

impl Exact for i128 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn is_zero(&self) -> bool {
        *self == 0
    }
    fn sign(&self) -> Sign {
        Sign::of(*self)
    }
    #[inline]
    fn bareiss(a: &Self, b: &Self, c: &Self, d: &Self, e: &Self) -> Option<Self> {
        let num = a.checked_mul(*b)?.checked_sub(c.checked_mul(*d)?)?;
        // `e` is a previous pivot and never zero; `checked_div` additionally
        // guards `i128::MIN / -1`.
        num.checked_div(*e)
    }
}

impl Exact for IBig {
    fn zero() -> Self {
        IBig::ZERO
    }
    fn one() -> Self {
        IBig::ONE
    }
    fn is_zero(&self) -> bool {
        IBig::is_zero(self)
    }
    fn sign(&self) -> Sign {
        if IBig::is_zero(self) {
            Sign::Zero
        } else if *self > IBig::ZERO {
            Sign::Positive
        } else {
            Sign::Negative
        }
    }
    fn bareiss(a: &Self, b: &Self, c: &Self, d: &Self, e: &Self) -> Option<Self> {
        Some((a * b - c * d) / e)
    }
}

/// Inline storage for the matrices the chirotope sees (rank up to 8), so that
/// the `i128` tier does not allocate.
const INLINE: usize = 64;

/// The sign of the determinant of an `n x n` row-major matrix, destroying it.
/// `None` if the arithmetic overflowed.
fn det_sign<T: Exact>(m: &mut [T], n: usize) -> Option<Sign> {
    if n == 0 {
        return Some(Sign::Positive);
    }
    let mut sign = Sign::Positive;
    let mut prev = T::one();
    for k in 0..n - 1 {
        if m[k * n + k].is_zero() {
            // Pivot: any row below with a non-zero entry in this column. A row
            // swap keeps every intermediate a minor (of the permuted matrix)
            // and negates the determinant.
            let Some(p) = (k + 1..n).find(|&i| !m[i * n + k].is_zero()) else {
                return Some(Sign::Zero);
            };
            for c in k..n {
                m.swap(k * n + c, p * n + c);
            }
            sign = -sign;
        }
        let (top, bottom) = m.split_at_mut((k + 1) * n);
        let pivot_row = &top[k * n..];
        let pivot = &pivot_row[k];
        for row in bottom.chunks_exact_mut(n) {
            let lead = row[k].clone();
            for j in k + 1..n {
                row[j] = T::bareiss(&row[j], pivot, &lead, &pivot_row[j], &prev)?;
            }
        }
        prev = pivot.clone();
    }
    Some(sign * m[n * n - 1].sign())
}

/// The rank of a `rows x cols` row-major matrix, destroying it. `None` if the
/// arithmetic overflowed.
///
/// Bareiss elimination with column skipping: a column with no pivot is left
/// alone, and Sylvester's identity still makes every update an exact division,
/// since the entries remain minors on the pivot columns chosen so far.
fn rank_of<T: Exact>(m: &mut [T], rows: usize, cols: usize) -> Option<usize> {
    let mut prev = T::one();
    let mut r = 0;
    for c in 0..cols {
        if r == rows {
            break;
        }
        let Some(p) = (r..rows).find(|&i| !m[i * cols + c].is_zero()) else {
            continue;
        };
        if p != r {
            for j in c..cols {
                m.swap(r * cols + j, p * cols + j);
            }
        }
        let (top, bottom) = m.split_at_mut((r + 1) * cols);
        let pivot_row = &top[r * cols..];
        let pivot = &pivot_row[c];
        for row in bottom.chunks_exact_mut(cols) {
            let lead = row[c].clone();
            for j in c + 1..cols {
                row[j] = T::bareiss(&row[j], pivot, &lead, &pivot_row[j], &prev)?;
            }
            row[c] = T::zero();
        }
        prev = pivot.clone();
        r += 1;
    }
    Some(r)
}

/// Tier 2: the determinant sign by Bareiss elimination in checked `i128`.
///
/// Returns `None` if an intermediate overflows, which for inputs with entries
/// below about `2^60 / n` cannot happen for `n <= 4`, and is rare for realistic
/// inputs at any size.
pub fn sign_of_det_i128(m: SquareMatrix<'_>) -> Option<Sign> {
    let n = m.n();
    let src = m.data();
    if src.len() <= INLINE {
        let mut buf = [0i128; INLINE];
        for (dst, &x) in buf.iter_mut().zip(src) {
            *dst = x as i128;
        }
        det_sign(&mut buf[..src.len()], n)
    } else {
        let mut buf: Vec<i128> = src.iter().map(|&x| x as i128).collect();
        det_sign(&mut buf, n)
    }
}

/// Tier 3: the determinant sign by Bareiss elimination over arbitrary-precision
/// integers. Always succeeds.
pub fn sign_of_det_bigint(m: SquareMatrix<'_>) -> Sign {
    let mut buf: Vec<IBig> = m.data().iter().map(|&x| IBig::from(x)).collect();
    // Infallible: IBig arithmetic does not overflow.
    det_sign(&mut buf, m.n()).unwrap_or(Sign::Zero)
}

/// Tier 2 for any [`Entry`] type: Bareiss in checked `i128`. `None` if an
/// entry does not fit in `i128` or an intermediate overflows.
///
/// # Panics
///
/// If `m.len() != n * n`; [`arith::sign_of_det_of`](crate::arith::sign_of_det_of)
/// returns an error instead.
pub fn sign_of_det_i128_of<T: Entry>(m: &[T], n: usize) -> Option<Sign> {
    assert!(
        n.checked_mul(n) == Some(m.len()),
        "sign_of_det_i128_of: {} entries do not make a {n} x {n} matrix",
        m.len()
    );
    if m.len() <= INLINE {
        let mut buf = [0i128; INLINE];
        for (dst, x) in buf.iter_mut().zip(m) {
            *dst = x.to_i128()?;
        }
        det_sign(&mut buf[..m.len()], n)
    } else {
        let mut buf = m.iter().map(Entry::to_i128).collect::<Option<Vec<_>>>()?;
        det_sign(&mut buf, n)
    }
}

/// Tier 3 for any [`Entry`] type: Bareiss over arbitrary-precision integers.
/// Always succeeds.
///
/// # Panics
///
/// If `m.len() != n * n`; [`arith::sign_of_det_of`](crate::arith::sign_of_det_of)
/// returns an error instead.
pub fn sign_of_det_bigint_of<T: Entry>(m: &[T], n: usize) -> Sign {
    assert!(
        n.checked_mul(n) == Some(m.len()),
        "sign_of_det_bigint_of: {} entries do not make a {n} x {n} matrix",
        m.len()
    );
    let mut buf: Vec<IBig> = m.iter().map(Entry::to_ibig).collect();
    // Infallible: IBig arithmetic does not overflow.
    det_sign(&mut buf, n).unwrap_or(Sign::Zero)
}

/// The rank of a `rows x cols` row-major matrix in checked `i128`, or `None`
/// on overflow. The caller guarantees `m.len() == rows * cols`.
pub(crate) fn rank_i128(m: &[i64], rows: usize, cols: usize) -> Option<usize> {
    let mut buf: Vec<i128> = m.iter().map(|&x| x as i128).collect();
    rank_of(&mut buf, rows, cols)
}

/// The rank of a `rows x cols` row-major matrix over arbitrary-precision
/// integers. The caller guarantees `m.len() == rows * cols`.
pub(crate) fn rank_bigint(m: &[i64], rows: usize, cols: usize) -> usize {
    let mut buf: Vec<IBig> = m.iter().map(|&x| IBig::from(x)).collect();
    // Infallible: IBig arithmetic does not overflow.
    rank_of(&mut buf, rows, cols).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(m: &[i64], n: usize) -> SquareMatrix<'_> {
        SquareMatrix::new(m, n).unwrap()
    }

    #[test]
    fn tiers_agree_on_small_cases() {
        let cases: &[(&[i64], usize, Sign)] = &[
            (&[2, -1, 0, -1, 2, -1, 0, -1, 2], 3, Sign::Positive),
            (&[0, 1, 0, 1, 0, 0, 0, 0, 1], 3, Sign::Negative),
            (&[1, 2, 3, 4, 5, 6, 7, 8, 9], 3, Sign::Zero),
            (&[0, 0, 1, 0, 1, 0, 1, 0, 0], 3, Sign::Negative),
        ];
        for &(m, n, s) in cases {
            assert_eq!(sign_of_det_i128(sq(m, n)), Some(s), "{m:?}");
            assert_eq!(sign_of_det_bigint(sq(m, n)), s, "{m:?}");
        }
    }

    #[test]
    fn i128_overflow_falls_back() {
        let m = [
            i64::MAX,
            i64::MIN,
            i64::MAX,
            i64::MIN,
            i64::MIN,
            i64::MAX,
            1,
            i64::MAX,
            i64::MIN,
        ];
        assert_eq!(sign_of_det_i128(sq(&m, 3)), None);
        // The determinant is about 2^128, past i128 (expanded independently).
        assert_eq!(sign_of_det_bigint(sq(&m, 3)), Sign::Positive);
    }

    #[test]
    fn large_matrices_allocate() {
        // 9 x 9 identity: past the inline buffer.
        let mut m = vec![0i64; 81];
        for i in 0..9 {
            m[i * 9 + i] = 1;
        }
        assert_eq!(sign_of_det_i128(sq(&m, 9)), Some(Sign::Positive));
    }
}
