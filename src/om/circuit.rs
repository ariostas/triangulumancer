//! Signed circuits.
//!
//! Phase 2 needs only the circuit of a dependent `(rank + 1)`-set and its two
//! triangulations, for the fine triangulation's flip-in. Phase 3 adds the flip
//! predicates (`containment_ok`, `link_ok`) here, in the oriented-matroid layer
//! rather than in `triang`, per `PLAN.md` section 6.1.

use super::OrientedMatroid;
use crate::arith::Sign;
use crate::error::{Error, Result};
use crate::simplex::{Simplex, SimplexSet};

/// A signed circuit `Z = (Z⁺, Z⁻)`: the unique minimal linear dependency
/// among some elements, split by the sign of each coefficient.
///
/// The orientation is TOPCOM's: built by [`of_dependent_set`], the *positive*
/// part is the side of the smallest element of the support. The negation
/// `(Z⁻, Z⁺)` is the same circuit, oriented the other way.
///
/// [`of_dependent_set`]: Self::of_dependent_set
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Circuit {
    positive: Simplex,
    negative: Simplex,
}

impl Circuit {
    /// The circuit with the given parts. They should be disjoint; nothing is
    /// checked.
    pub fn new(positive: Simplex, negative: Simplex) -> Self {
        Self { positive, negative }
    }

    /// The circuit contained in a set of `rank + 1` elements of `om`.
    ///
    /// For the `i`-th element `e` of `set`, in increasing order, the
    /// coefficient of `e` has sign `(-1)^i · χ(set \ {e})`; elements where that
    /// is zero are not in the support. The side of the first element with a
    /// non-zero coefficient is [`positive`](Self::positive). This is TOPCOM's
    /// `Circuit(chiro, dependent_set)` (`Circuits.cc:43-72`).
    ///
    /// If `set \ {e}` is dependent for every `e`, the result is empty on both
    /// sides.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] unless `set` has `rank + 1` labels, and
    /// whatever [`OrientedMatroid::sign`] returns.
    pub fn of_dependent_set(om: &impl OrientedMatroid, set: &Simplex) -> Result<Self> {
        let rank = om.rank();
        if set.len() != rank + 1 {
            return Err(Error::DimensionMismatch {
                expected: rank + 1,
                found: set.len(),
            });
        }
        let mut positive = Simplex::new();
        let mut negative = Simplex::new();
        let mut reference = Sign::Zero;
        let mut basis = set.clone();
        for (i, e) in set.iter().enumerate() {
            basis.remove(e);
            let chi = om.sign(&basis)?;
            basis.insert(e);
            let coefficient = if i % 2 == 0 { chi } else { -chi };
            if coefficient.is_zero() {
                continue;
            }
            if reference.is_zero() {
                reference = coefficient;
            }
            if coefficient == reference {
                positive.insert(e);
            } else {
                negative.insert(e);
            }
        }
        Ok(Self { positive, negative })
    }

    /// `Z⁺`, TOPCOM's `first`.
    pub fn positive(&self) -> &Simplex {
        &self.positive
    }

    /// `Z⁻`, TOPCOM's `second`.
    pub fn negative(&self) -> &Simplex {
        &self.negative
    }

    /// `Z⁺ ∪ Z⁻`.
    pub fn support(&self) -> Simplex {
        &self.positive | &self.negative
    }

    /// The same circuit with its sides exchanged.
    pub fn negated(&self) -> Self {
        Self {
            positive: self.negative.clone(),
            negative: self.positive.clone(),
        }
    }

    /// `{ supp(Z) \ {z} : z ∈ Z⁻ }`, the triangulation of the support that
    /// uses every element of `Z⁺`. TOPCOM's `upper_facets`.
    pub fn upper_facets(&self) -> SimplexSet {
        self.facets_dropping(&self.negative)
    }

    /// `{ supp(Z) \ {z} : z ∈ Z⁺ }`, the triangulation of the support that
    /// uses every element of `Z⁻`. TOPCOM's `lower_facets`.
    pub fn lower_facets(&self) -> SimplexSet {
        self.facets_dropping(&self.positive)
    }

    fn facets_dropping(&self, side: &Simplex) -> SimplexSet {
        let support = self.support();
        side.iter()
            .map(|z| {
                let mut f = support.clone();
                f.remove(z);
                f
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{PointConfig, VectorConfig};

    #[test]
    fn interior_point_of_a_triangle() {
        // 3 is inside the triangle 0, 1, 2: 3·p3 = p0 + p1 + p2.
        let p = PointConfig::from_row_major(&[0, 0, 3, 0, 0, 3, 1, 1], 4, 2).unwrap();
        let chi = p.chirotope().unwrap();
        let z = Circuit::of_dependent_set(&chi, &Simplex::from([0, 1, 2, 3])).unwrap();
        assert_eq!(z.positive(), &Simplex::from([0, 1, 2]));
        assert_eq!(z.negative(), &Simplex::from([3]));
        assert_eq!(z.lower_facets().len(), 3);
        assert_eq!(
            z.upper_facets().to_vecs(),
            vec![vec![0, 1, 2]],
            "the triangulation without the interior point"
        );
    }

    #[test]
    fn square_diagonals() {
        let p = PointConfig::from_row_major(&[0, 0, 1, 0, 1, 1, 0, 1], 4, 2).unwrap();
        let chi = p.chirotope().unwrap();
        let z = Circuit::of_dependent_set(&chi, &Simplex::from([0, 1, 2, 3])).unwrap();
        assert_eq!(z.positive(), &Simplex::from([0, 2]));
        assert_eq!(z.negative(), &Simplex::from([1, 3]));
        assert_eq!(z.negated().positive(), &Simplex::from([1, 3]));
    }

    #[test]
    fn special_position_and_one_sided_circuits() {
        // 0, 1, 2 collinear with 1 in the middle: 3 is not in the support.
        let p = PointConfig::from_row_major(&[0, 0, 1, 0, 2, 0, 0, 1], 4, 2).unwrap();
        let chi = p.chirotope().unwrap();
        let z = Circuit::of_dependent_set(&chi, &Simplex::from([0, 1, 2, 3])).unwrap();
        assert_eq!(z.support(), Simplex::from([0, 1, 2]));
        assert_eq!(z.negative(), &Simplex::from([1]));

        // v0 + v1 = 0 for vectors: a circuit with an empty negative part.
        let v = VectorConfig::from_row_major(&[1, -1, 2], 3, 1).unwrap();
        let chi = v.chirotope().unwrap();
        let z = Circuit::of_dependent_set(&chi, &Simplex::from([0, 1])).unwrap();
        assert_eq!(z.positive(), &Simplex::from([0, 1]));
        assert!(z.negative().is_empty());
        assert!(z.upper_facets().is_empty());

        assert!(Circuit::of_dependent_set(&chi, &Simplex::from([0, 1, 2])).is_err());
    }
}
