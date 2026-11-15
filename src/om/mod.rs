//! The oriented-matroid layer: the seam between coordinates and combinatorics.
//!
//! Almost all of TOPCOM is a function of the *chirotope* alone — the sign of
//! every basis determinant — and not of the coordinates it came from. Its tools
//! come in `points2X` / `chiro2X` pairs for exactly that reason. This module
//! makes the same boundary explicit: [`OrientedMatroid`] is the only thing the
//! combinatorial algorithms in [`crate::triang`] accept, and a [`Chirotope`]
//! built from a [`Config`](crate::config::Config) is the one implementer that
//! touches coordinates. See `PLAN.md` section 6.1.
//!
//! Anything else that can answer [`OrientedMatroid::sign`] — a chirotope read
//! from TOPCOM's text format, a dual, a uniform matroid generated on the fly —
//! runs through every algorithm unchanged.

mod chirotope;
mod circuit;
mod cocircuit;

pub use chirotope::Chirotope;
pub use circuit::{Circuit, circuits};
pub use cocircuit::{Cocircuit, Facets, Incidence, Incidences, cocircuits};

use crate::arith::Sign;
use crate::error::{Error, Result};
use crate::simplex::{Simplex, combinations};

/// A chirotope: an alternating sign map on the ordered bases of a ground set
/// `{0, ..., n - 1}` of some rank `r`.
///
/// Only [`sign`](Self::sign) — the value on a basis given as a *set*, read in
/// increasing label order, TOPCOM's `χ(B)` — must be implemented; the other
/// methods derive from it and may be overridden for speed.
///
/// Implementers should be cheap to query repeatedly; the algorithms ask for
/// the same basis many times. They must not panic on a malformed query: a
/// basis of the wrong size or with a label out of range is an error.
pub trait OrientedMatroid {
    /// The number of elements of the ground set.
    fn n(&self) -> usize;

    /// The rank: the cardinality of every basis.
    fn rank(&self) -> usize;

    /// `χ(basis)`, the orientation of `basis` with its labels in increasing
    /// order.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] unless `basis` has exactly
    /// [`rank`](Self::rank) labels, and [`Error::IndexOutOfRange`] for a label
    /// not below [`n`](Self::n).
    fn sign(&self, basis: &Simplex) -> Result<Sign>;

    /// The orientation of the ordered basis `(facet in increasing order, x)`:
    /// `χ(facet ∪ {x})` times the sign of the permutation that sorts it. Zero
    /// if `x` is in `facet`.
    ///
    /// This is TOPCOM's `Permutation perm(facet); perm.push_back(x);
    /// perm.sign() * chiro(perm)` idiom, the side of the hyperplane spanned by
    /// `facet` that `x` lies on.
    ///
    /// # Errors
    ///
    /// As [`sign`](Self::sign), for `facet` of the wrong size or labels out of
    /// range.
    fn sign_appended(&self, facet: &Simplex, x: usize) -> Result<Sign> {
        if facet.contains(x) {
            return Ok(Sign::Zero);
        }
        let mut basis = facet.clone();
        basis.insert(x);
        let chi = self.sign(&basis)?;
        // Moving `x` from the end into its sorted position passes every label
        // of `facet` larger than it.
        let larger = facet.iter().filter(|&f| f > x).count();
        Ok(if larger % 2 == 0 { chi } else { -chi })
    }

    /// The lexicographically first basis with a non-zero sign, or `None` if
    /// there is none (which a chirotope of a full-rank configuration always
    /// has). This is TOPCOM's `find_non_deg_basis`, the seed of the placing
    /// triangulation.
    ///
    /// # Errors
    ///
    /// Whatever [`sign`](Self::sign) returns.
    fn first_basis(&self) -> Result<Option<Simplex>> {
        for basis in combinations(self.n(), self.rank()) {
            if !self.sign(&basis)?.is_zero() {
                return Ok(Some(basis));
            }
        }
        Ok(None)
    }

    /// The orientation of `(prebasis in increasing order, p)`, where `p` is the
    /// lexicographic extension `[order[0]⁺, order[1]⁺, ...]`: a new element
    /// placed at `ε₀·order[0] + ε₁·order[1] + ...` with `ε₀ ≫ ε₁ ≫ ... > 0`.
    ///
    /// Equivalently, [`sign_appended`](Self::sign_appended) for the first
    /// label of `order` that is not in `prebasis` and gives a non-zero sign;
    /// zero if there is none. TOPCOM's `chiro(prebasis, lex_extension_perm)`,
    /// used to place a generic point inside a simplex
    /// (`RealChiro.cc:35-56`).
    ///
    /// # Errors
    ///
    /// As [`sign`](Self::sign); `prebasis` must have `rank - 1` labels.
    fn lex_extension_sign(&self, prebasis: &Simplex, order: &[usize]) -> Result<Sign> {
        for &x in order {
            let s = self.sign_appended(prebasis, x)?;
            if !s.is_zero() {
                return Ok(s);
            }
        }
        Ok(Sign::Zero)
    }
}

impl<T: OrientedMatroid + ?Sized> OrientedMatroid for &T {
    fn n(&self) -> usize {
        (**self).n()
    }

    fn rank(&self) -> usize {
        (**self).rank()
    }

    fn sign(&self, basis: &Simplex) -> Result<Sign> {
        (**self).sign(basis)
    }

    fn sign_appended(&self, facet: &Simplex, x: usize) -> Result<Sign> {
        (**self).sign_appended(facet, x)
    }

    fn first_basis(&self) -> Result<Option<Simplex>> {
        (**self).first_basis()
    }

    fn lex_extension_sign(&self, prebasis: &Simplex, order: &[usize]) -> Result<Sign> {
        (**self).lex_extension_sign(prebasis, order)
    }
}

/// Check that `basis` is a well-formed query for an oriented matroid with
/// `n` elements and rank `rank`: exactly `rank` labels, all below `n`.
///
/// For implementers of [`OrientedMatroid::sign`].
///
/// # Errors
///
/// [`Error::DimensionMismatch`] or [`Error::IndexOutOfRange`].
pub fn check_basis(basis: &Simplex, n: usize, rank: usize) -> Result<()> {
    let len = basis.len();
    if len != rank {
        return Err(Error::DimensionMismatch {
            expected: rank,
            found: len,
        });
    }
    match basis.last() {
        Some(last) if last >= n => Err(Error::IndexOutOfRange {
            index: last,
            len: n,
        }),
        _ => Ok(()),
    }
}

/// The chirotope in TOPCOM's text format, as `points2chiro` prints it:
/// `"n,r:\n"` followed by one of `+`, `-`, `0` per basis in lexicographic
/// order, and a newline.
///
/// Every basis is evaluated, so this is `C(n, r)` sign computations.
///
/// # Errors
///
/// Whatever [`OrientedMatroid::sign`] returns.
pub fn to_topcom_string(om: &impl OrientedMatroid) -> Result<String> {
    let (n, r) = (om.n(), om.rank());
    let mut out = format!("{n},{r}:\n");
    for basis in combinations(n, r) {
        out.push(match om.sign(&basis)? {
            Sign::Positive => '+',
            Sign::Negative => '-',
            _ => '0',
        });
    }
    out.push('\n');
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PointConfig;

    #[test]
    fn sign_appended_accounts_for_the_permutation() {
        let tri = PointConfig::from_row_major(&[0, 0, 1, 0, 0, 1], 3, 2).unwrap();
        let chi = tri.chirotope().unwrap();
        // (1, 2, 0) is an even permutation of (0, 1, 2); (0, 2, 1) is odd.
        assert_eq!(
            chi.sign_appended(&Simplex::from([1, 2]), 0),
            Ok(Sign::Positive)
        );
        assert_eq!(
            chi.sign_appended(&Simplex::from([0, 2]), 1),
            Ok(Sign::Negative)
        );
        assert_eq!(chi.sign_appended(&Simplex::from([0, 2]), 2), Ok(Sign::Zero));
    }

    #[test]
    fn first_basis_skips_degenerate_ones() {
        // Points 0, 1, 2 are collinear.
        let p = PointConfig::from_row_major(&[0, 0, 1, 1, 2, 2, 0, 1], 4, 2).unwrap();
        let chi = p.chirotope().unwrap();
        assert_eq!(chi.first_basis(), Ok(Some(Simplex::from([0, 1, 3]))));
    }

    #[test]
    fn check_basis_rejects_malformed_queries() {
        assert!(check_basis(&Simplex::from([0, 1, 2]), 3, 3).is_ok());
        assert_eq!(
            check_basis(&Simplex::from([0, 1]), 3, 3),
            Err(Error::DimensionMismatch {
                expected: 3,
                found: 2
            })
        );
        assert_eq!(
            check_basis(&Simplex::from([0, 1, 5]), 3, 3),
            Err(Error::IndexOutOfRange { index: 5, len: 3 })
        );
    }

    #[test]
    fn topcom_string_format() {
        let square = PointConfig::from_row_major(&[0, 0, 1, 0, 1, 1, 0, 1], 4, 2).unwrap();
        let chi = square.chirotope().unwrap();
        assert_eq!(to_topcom_string(&chi).unwrap(), "4,3:\n++++\n");
    }
}
