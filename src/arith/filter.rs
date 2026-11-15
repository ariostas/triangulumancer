//! Tier 1: a floating-point filter for determinant signs.
//!
//! CGAL's filter evaluates predicates in `Interval_nt_advanced` with the FPU
//! rounding mode set towards `+inf`. Stable Rust has no control over the
//! rounding mode, so this filter works in the only mode there is, round to
//! nearest, and carries a *running error bound* instead: Gaussian elimination
//! with partial pivoting on `f64` midpoints, with a radius beside every entry
//! that bounds the distance from the midpoint to the value the same
//! elimination would have produced in exact arithmetic (midpoint-radius
//! interval arithmetic in the sense of Rump). The radii are computed in round
//! to nearest too, and inflated by a constant factor and a tiny absolute term
//! that provably absorb every rounding error made while computing them.
//!
//! Per updated entry that is two `f64` operations for the midpoint and about a
//! dozen branch-free ones for the radius, which vectorise; the `[lo, hi]`
//! interval elimination this replaced needed four products, six comparisons
//! and four `next_up`/`next_down` adjustments, and was about half the run time
//! of a Delaunay triangulation in dimension 5 and up.
//!
//! If every pivot is finite and its radius below its magnitude, then the exact
//! elimination with the same row exchanges has non-zero pivots of the same
//! signs, so the sign of the determinant is the parity of the row exchanges
//! times the product of the pivot signs. Otherwise — always the case for a
//! singular matrix — the filter gives up and the exact tiers decide.
//!
//! # Soundness
//!
//! Notation: `u = 2⁻⁵³` (the unit roundoff), `η = 2⁻¹⁰⁷⁵` (half the smallest
//! subnormal), `fl(·)` an operation rounded to nearest. A pair `(m, r)`
//! *encloses* a real `x` if `|x − m| <= r`. Facts about round to nearest used
//! below, for finite results:
//!
//! * **(F1)** `|a ∘ b − fl(a ∘ b)| <= u·|fl(a ∘ b)| + η` for `∘ ∈ {×, ÷}`, and
//!   `|a ± b − fl(a ± b)| <= u·|fl(a ± b)|`: in the normal range the error is at
//!   most half an ulp of the result, which is at most `u` times its magnitude;
//!   below it, products and quotients err by at most `η` and sums are exact.
//!   Multiplying by `2` is exact.
//! * **(F2)** For non-negative `a`, `b`: `a + b <= (1 + u)·fl(a + b)`, and
//!   `a ∘ b <= (1 + u)·fl(a ∘ b) + η` for `∘ ∈ {×, ÷}` — (F1) rearranged.
//! * **(F3)** `fl` is monotone: `x <= y` implies `fl(x) <= fl(y)`.
//!
//! **Inputs.** An integer `x` is entered as `m = fl(x)`, with `r = 0` if
//! `|x| <= 2⁵³` (exact) and `r = 2⁻⁵²·|m|` otherwise, which covers any faithful
//! rounding and is itself computed exactly (`|m| >= 2⁵³`, a power-of-two
//! scaling). A value beyond the `f64` range makes the filter decline.
//!
//! **Pivot.** At step `k` the row with the largest `|m|` in column `k` becomes
//! the pivot row, so every multiplier below satisfies `|m_a| <= |m_p|` and hence
//! `|m_f| = |fl(m_a / m_p)| <= 1` (F3). The pivot `(m_p, r_p)` is accepted only
//! if `|m_p|` is finite and `ρ̂ = fl(r_p / |m_p|) <= 1/4`; then the exact
//! `ρ = r_p / |m_p| <= (1 + u)/4 + η < 1/2` and the exact pivot `p` is non-zero
//! with the sign of `m_p`. (For the last pivot, which divides nothing,
//! `r_p < |m_p|` suffices.)
//!
//! **Multiplier.** Row `i` has `(m_a, r_a)` in column `k`, enclosing the exact
//! `a`; the exact multiplier is `f = a / p`, and `m_f = fl(m_a / m_p)`. Then
//!
//! ```text
//! |a/p − m_a/m_p| = |(a − m_a)·m_p − m_a·(p − m_p)| / (|p|·|m_p|)
//!                <= (r_a/|m_p| + |m_a/m_p|·ρ) / (1 − ρ)
//!                <= (r_a/|m_p| + |m_a/m_p|·ρ)·(1 + 2ρ)          (ρ <= 1/2)
//! ```
//!
//! and with `|m_a/m_p| <= |m_f|·(1 + u) + η` (F1) and `ρ·(1 + 2ρ) <= 1`, a valid
//! radius for the multiplier is
//!
//! ```text
//! R_f = (r_a/|m_p| + |m_f|·ρ)·(1 + 2ρ) + 2u·|m_f| + 2η.
//! ```
//!
//! **Update.** For `j > k`, the entry `(m_x, r_x)` of row `i` becomes
//! `m_z = fl(m_x − P)` with `P = fl(m_f·m_y)`, `(m_y, r_y)` the pivot row's
//! entry. Writing the exact `z = x − f·y`,
//!
//! ```text
//! z − m_z = (x − m_x) − (f·y − m_f·m_y) − (m_f·m_y − P) + (m_x − P − m_z),
//! |f·y − m_f·m_y| <= |f|·r_y + r_f·|m_y| <= |m_f|·r_y + r_f·(|m_y| + r_y),
//! ```
//!
//! so by (F1) a valid radius is
//!
//! ```text
//! R_z = r_x + |m_f|·r_y + r_f·(|m_y| + r_y) + u·(|P| + |m_z|) + η.
//! ```
//!
//! **Computing the radii.** `R_f` and `R_z` are evaluated in round to nearest
//! as written (with `ρ̂` for `ρ`), giving `ŝ`, and stored as
//! `fl(fl(ŝ + E)·C)` with `E = 2⁻¹⁰⁰⁰` and `C = 1 + 2⁻⁴⁰`. Applying (F2) term
//! by term — every operand is non-negative; `|m_f| <= 1` and `1 + 2ρ̂ <= 2`
//! bound what an underflow error `η` is later multiplied by; the absolute
//! error `2η` in `1 + 2ρ` is below `u` relative to it, as it is at least one —
//! gives `R_z <= (1 + u)⁵·ŝ + 7η` and `R_f <= (1 + u)⁸·ŝ + 14η`. Since
//! `ŝ + E >= 2⁻¹⁰⁰⁰` is normal, (F2) applies to the two final operations
//! without the `η` term, and the stored radius is at least
//! `(ŝ + E)·C / (1 + u)²`. That exceeds both bounds, because
//! `C >= (1 + u)¹⁰` and `E·C / (1 + u)² > 14η`. So the stored pair encloses
//! the exact value, by induction over the steps.
//!
//! **Overflow.** Every operation above returns a non-finite value when an
//! operand is non-finite (the division by `m_p` is the exception, and `m_p` is
//! checked first; a radius is never zero once computed, so `0 × ∞` cannot hide
//! one either). A non-finite multiplier spoils every later entry of its row,
//! and a non-finite pivot-row entry in column `j` spoils column `j` of every
//! row below, one of which is the pivot of step `j`. Each row is a pivot row
//! by the end, so every non-finite value reaches some pivot, and the filter
//! declines. Hence when the filter answers, every value it computed is finite
//! and the bounds above apply. Comparisons are written so that `NaN` fails
//! them.
//!
//! The constants are generous by design: `E` makes the filter decline only
//! for pivots below about `2⁻⁹⁹⁰`, which integer matrices do not produce
//! before their entries leave the `f64` range, and `C` costs a relative
//! `2⁻⁴⁰` per step, far below the rounding errors it covers. The differential
//! harness `tests/sign_of_det.rs` checks the filter against an exact oracle,
//! including an adversarial family swept across the filter's decision
//! threshold.

use super::{Entry, Sign, SquareMatrix};

/// The unit roundoff, `2⁻⁵³`.
const U: f64 = f64::EPSILON / 2.0;

/// The absolute slack added to every radius, `2⁻¹⁰⁰⁰`: far above any
/// underflow error, far below any pivot of an integer matrix.
const E: f64 = f64::from_bits((1023 - 1000) << 52);

/// The relative inflation of every radius, `1 + 2⁻⁴⁰`, at least `(1 + u)¹⁰`.
const C: f64 = 1.0 + 1.0 / (1u64 << 40) as f64;

// `C >= (1 + u)^10`, since `(1 + u)^10 <= exp(10u) < 1 + 11u`, with a wide
// margin.
const _: () = assert!(C > 1.0 + 11.0 * U);

/// Integers up to this magnitude convert to `f64` exactly.
const EXACT_LIMIT: u128 = 1 << 53;

/// `2⁻⁵²`: at least one ulp of a normal `f64`, relative to its magnitude.
const ULP: f64 = f64::EPSILON;

/// A midpoint and radius enclosing the integer `x`; see the module
/// documentation.
#[inline]
pub(crate) fn approx_i128(x: i128) -> (f64, f64) {
    // Through `i64` when it fits: a hardware conversion rather than the
    // software routine for `i128`, which the optimiser would otherwise call
    // unconditionally. Both round to nearest.
    if let Ok(small) = i64::try_from(x) {
        let m = small as f64;
        if small.unsigned_abs() <= EXACT_LIMIT as u64 {
            (m, 0.0)
        } else {
            (m, m.abs() * ULP)
        }
    } else {
        approx_wide(x)
    }
}

/// [`approx_i128`] beyond `i64`.
#[cold]
#[inline(never)]
fn approx_wide(x: i128) -> (f64, f64) {
    let m = x as f64;
    (m, m.abs() * ULP)
}

/// A midpoint and radius enclosing an integer of magnitude above `2⁵³` whose
/// value, rounded faithfully to an `f64`, is `m`; `None` if that is not
/// finite.
pub(crate) fn approx_rounded(m: f64) -> Option<(f64, f64)> {
    m.is_finite().then(|| (m, m.abs() * ULP))
}

/// Matrices up to this many entries are eliminated on the stack in a buffer
/// of this size; `6 x 6` covers the power test up to dimension 5.
const SMALL: usize = 36;

/// The same for up to `12 x 12`, dimension 11.
const MEDIUM: usize = 144;

/// Tier 1 for any [`Entry`] type: the determinant sign of the row-major
/// `n x n` matrix `m` by Gaussian elimination with a running error bound, or
/// `None` if the filter cannot decide (or an entry is beyond the `f64`
/// range).
///
/// The caller guarantees `m.len() == n * n`.
pub fn sign_of_det_of<T: Entry>(m: &[T], n: usize) -> Option<Sign> {
    with_buffers(m.len(), |mid, rad| {
        for ((dm, dr), x) in mid.iter_mut().zip(rad.iter_mut()).zip(m) {
            (*dm, *dr) = x.approx()?;
        }
        eliminate(mid, rad, n)
    })
}

/// Tier 1: the determinant sign by Gaussian elimination with a running error
/// bound, or `None` if the filter cannot decide.
///
/// A `Some` answer is always exact. `None` is returned for every singular
/// matrix, and for nearly singular or badly conditioned ones.
pub fn sign_of_det(m: SquareMatrix<'_>) -> Option<Sign> {
    let src = m.data();
    with_buffers(src.len(), |mid, rad| {
        for ((dm, dr), &x) in mid.iter_mut().zip(rad.iter_mut()).zip(src) {
            (*dm, *dr) = approx_i128(i128::from(x));
        }
        eliminate(mid, rad, m.n())
    })
}

/// Tier 1 on an `n x n` matrix of integers that are exactly representable
/// as `f64`: `fill` writes the row-major entries into a zeroed buffer, and
/// every radius is zero. This skips the conversion from an integer type for
/// callers that can produce the entries in `f64` directly.
///
/// Soundness depends on the caller: each entry written must *be* the matrix
/// entry, not an approximation of it.
pub(crate) fn sign_of_det_integral(n: usize, fill: impl FnOnce(&mut [f64])) -> Option<Sign> {
    with_buffers(n * n, |mid, rad| {
        fill(mid);
        eliminate(mid, rad, n)
    })
}

/// Run `f` on two scratch buffers of `len` entries each, on the stack when
/// they are small.
#[inline]
fn with_buffers<R>(len: usize, f: impl FnOnce(&mut [f64], &mut [f64]) -> R) -> R {
    if len <= SMALL {
        let (mut mid, mut rad) = ([0.0; SMALL], [0.0; SMALL]);
        f(&mut mid[..len], &mut rad[..len])
    } else if len <= MEDIUM {
        let (mut mid, mut rad) = ([0.0; MEDIUM], [0.0; MEDIUM]);
        f(&mut mid[..len], &mut rad[..len])
    } else {
        f(&mut vec![0.0; len], &mut vec![0.0; len])
    }
}

/// The elimination of the module documentation, on row-major midpoints and
/// radii.
fn eliminate(mid: &mut [f64], rad: &mut [f64], n: usize) -> Option<Sign> {
    // A copy per size of the regular-triangulation predicates up to
    // dimension 10, so that the loop bounds are constants.
    match n {
        5 => eliminate_n(mid, rad, 5),
        6 => eliminate_n(mid, rad, 6),
        7 => eliminate_n(mid, rad, 7),
        8 => eliminate_n(mid, rad, 8),
        9 => eliminate_n(mid, rad, 9),
        10 => eliminate_n(mid, rad, 10),
        11 => eliminate_n(mid, rad, 11),
        _ => eliminate_n(mid, rad, n),
    }
}

#[inline(always)]
fn eliminate_n(mid: &mut [f64], rad: &mut [f64], n: usize) -> Option<Sign> {
    debug_assert!(mid.len() == n * n && rad.len() == n * n);
    let mut sign = Sign::Positive;
    for k in 0..n {
        // Partial pivoting on the midpoint's magnitude. NaN never wins a
        // comparison, so it is never chosen while a finite candidate exists.
        let mut p = k;
        let mut best = mid[k * n + k].abs();
        for i in k + 1..n {
            let a = mid[i * n + k].abs();
            if a > best {
                best = a;
                p = i;
            }
        }
        if p != k {
            for c in k..n {
                mid.swap(k * n + c, p * n + c);
                rad.swap(k * n + c, p * n + c);
            }
            sign = -sign;
        }
        let (mp, rp) = (mid[k * n + k], rad[k * n + k]);
        let ap = mp.abs();
        if k + 1 == n {
            // Divides nothing: only the sign is needed.
            if !(ap.is_finite() && rp < ap) {
                return None;
            }
        } else {
            let rho = rp / ap;
            if !(ap.is_finite() && rho <= 0.25) {
                return None;
            }
            let grow = 1.0 + 2.0 * rho;
            let (top, bottom) = (mid.split_at_mut((k + 1) * n), rad.split_at_mut((k + 1) * n));
            let (my_row, ry_row) = (&top.0[k * n + k + 1..], &bottom.0[k * n + k + 1..]);
            for (mrow, rrow) in top.1.chunks_exact_mut(n).zip(bottom.1.chunks_exact_mut(n)) {
                let f = mrow[k] / mp;
                let af = f.abs();
                let rf = ((rrow[k] / ap + af * rho) * grow + 2.0 * U * af + E) * C;
                let (mrow, rrow) = (&mut mrow[k + 1..], &mut rrow[k + 1..]);
                for (((mx, rx), &my), &ry) in
                    mrow.iter_mut().zip(rrow.iter_mut()).zip(my_row).zip(ry_row)
                {
                    let prod = f * my;
                    let z = *mx - prod;
                    *rx =
                        (*rx + af * ry + rf * (my.abs() + ry) + U * (prod.abs() + z.abs()) + E) * C;
                    *mx = z;
                }
            }
        }
        if mp < 0.0 {
            sign = -sign;
        }
    }
    Some(sign)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants() {
        assert_eq!(U, 2f64.powi(-53));
        assert_eq!(E, 2f64.powi(-1000));
        assert_eq!(C - 1.0, 2f64.powi(-40));
    }

    #[test]
    fn inexact_conversion_is_enclosed() {
        let x = (1i128 << 53) + 1;
        let (m, r) = approx_i128(x);
        assert!(m - r <= 9007199254740993.0 && m + r >= 9007199254740993.0);
        assert!(r > 0.0);
        let (m, r) = approx_i128(i128::from(i64::MIN));
        assert_eq!((m, r), (-(2f64.powi(63)), 2f64.powi(11)));
        assert_eq!(approx_i128(-(1 << 53)), (-(2f64.powi(53)), 0.0));
        assert_eq!(approx_i128(i128::MAX).0, 2f64.powi(127));
        assert_eq!(approx_rounded(f64::INFINITY), None);
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
        assert_eq!(
            sign_of_det(SquareMatrix::new(&[], 0).unwrap()),
            Some(Sign::Positive)
        );
        assert_eq!(
            sign_of_det(SquareMatrix::new(&[-3], 1).unwrap()),
            Some(Sign::Negative)
        );
        assert_eq!(sign_of_det(SquareMatrix::new(&[0], 1).unwrap()), None);
    }

    #[test]
    fn overflow_declines() {
        // Entries near the top of the f64 range: the elimination overflows.
        let big = f64::MAX;
        let mut mid = [big, -big, big, big];
        let mut rad = [0.0; 4];
        assert_eq!(eliminate(&mut mid, &mut rad, 2), None);
        let mut mid = [f64::NAN, 1.0, 1.0, 1.0];
        let mut rad = [0.0; 4];
        assert_eq!(eliminate(&mut mid, &mut rad, 2), None);
    }
}
