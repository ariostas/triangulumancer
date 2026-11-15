//! Signed circuits, and the predicates that say when a circuit is realised in
//! a simplicial complex.
//!
//! The circuit of a dependent `(rank + 1)`-set is what the fine triangulation
//! flips in and what every bistellar flip is supported on.
//! The flip predicates [`Circuit::containment_ok`] and [`Circuit::link_ok`]
//! live here, in the oriented-matroid layer rather than in `triang`, per
//! `PLAN.md` section 6.1: they need nothing but a circuit and a set of
//! simplices.

use std::collections::BTreeMap;

use super::OrientedMatroid;
use crate::arith::Sign;
use crate::error::{Error, Result};
use crate::simplex::{Simplex, SimplexSet, combinations};

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

    /// Whether `complex` contains the triangulation of the support that uses
    /// every element of `Z⁻`: every `supp(Z) \ {z}` for `z ∈ Z⁺`.
    ///
    /// For a circuit in *general position* (support of `rank + 1` elements)
    /// this is the condition for a bistellar flip on `Z` to apply to a
    /// triangulation containing `complex`'s simplices: the side of the flip
    /// that is present is exactly [`lower_facets`](Self::lower_facets). True
    /// when `Z⁺` is empty. TOPCOM's `TriangNode::containment_ok`
    /// (`TriangNode.cc:20-32`).
    pub fn containment_ok(&self, complex: &SimplexSet) -> bool {
        let support = self.support();
        self.positive.iter().all(|z| {
            let mut facet = support.clone();
            facet.remove(z);
            complex.contains(&facet)
        })
    }

    /// The *special-position* analogue of
    /// [`containment_ok`](Self::containment_ok), for a circuit whose support
    /// has fewer than `rank + 1` elements: whether every `supp(Z) \ {z}` for
    /// `z ∈ Z⁺` has the same non-empty link in `complex`.
    ///
    /// Then the flip replaces the lower facets joined with that common link
    /// ([`common_link`](Self::common_link)) by the upper facets joined with
    /// it. Always false when `Z⁻` is empty, and always true for a circuit in
    /// general position, whose common link is the empty simplex — both as in
    /// TOPCOM's `TriangNode::link_ok` (`TriangNode.cc:34-76`), which this is.
    pub fn link_ok(&self, complex: &SimplexSet, rank: usize) -> bool {
        if self.negative.is_empty() {
            return false;
        }
        if self.support().len() == rank + 1 {
            return true;
        }
        self.common_link(complex).is_some()
    }

    /// The common link of the lower facets in `complex`: the link of
    /// `supp(Z) \ {z}` for the first `z ∈ Z⁺`, provided it is non-empty and
    /// every other `z ∈ Z⁺` gives the same one. `None` otherwise, and when
    /// `Z⁺` is empty.
    ///
    /// This does not look at `Z⁻`; [`link_ok`](Self::link_ok) does.
    pub fn common_link(&self, complex: &SimplexSet) -> Option<SimplexSet> {
        let support = self.support();
        let mut sides = self.positive.iter();
        let first = sides.next()?;
        let link = complex.link(&(&support - &Simplex::from([first])));
        if link.is_empty() {
            return None;
        }
        for z in sides {
            if complex.link(&(&support - &Simplex::from([z]))) != link {
                return None;
            }
        }
        Some(link)
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

/// Every circuit of `om`, one per support, ordered by support.
///
/// Each `(rank + 1)`-subset of the ground set contains exactly one circuit (or
/// none, when every `rank`-subset of it is dependent); collecting them and
/// dropping repeated supports gives every circuit, each in the orientation of
/// [`Circuit::of_dependent_set`]. This is TOPCOM's `Circuits` and the output
/// of `points2circuits`, which prints the same set in hash-table order. Empty
/// when `n == rank`.
///
/// # Errors
///
/// Whatever [`OrientedMatroid::sign`] returns.
pub fn circuits(om: &impl OrientedMatroid) -> Result<Vec<Circuit>> {
    let (n, rank) = (om.n(), om.rank());
    let mut by_support = BTreeMap::new();
    if n > rank {
        for set in combinations(n, rank + 1) {
            let z = Circuit::of_dependent_set(om, &set)?;
            let support = z.support();
            if !support.is_empty() {
                by_support.entry(support).or_insert(z);
            }
        }
    }
    Ok(by_support.into_values().collect())
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

    fn set(simplices: &[&[usize]]) -> SimplexSet {
        simplices.iter().map(|s| Simplex::from(*s)).collect()
    }

    #[test]
    fn containment_in_the_square() {
        let z = Circuit::new(Simplex::from([0, 2]), Simplex::from([1, 3]));
        let diagonal_02 = set(&[&[0, 1, 2], &[0, 2, 3]]);
        let diagonal_13 = set(&[&[0, 1, 3], &[1, 2, 3]]);
        // Z⁺ = {0, 2}: the lower facets drop 0 and 2, leaving the 1-3 diagonal.
        assert!(z.containment_ok(&diagonal_13));
        assert!(!z.containment_ok(&diagonal_02));
        assert!(z.negated().containment_ok(&diagonal_02));
        assert!(Circuit::new(Simplex::new(), Simplex::from([0])).containment_ok(&diagonal_02));
    }

    #[test]
    fn common_links_in_special_position() {
        // 0, 1, 2 on a line with 1 in the middle, 3 and 4 on either side:
        // Z = ({0, 2}, {1}), and the edge 0-2 has link {3, 4}.
        let z = Circuit::new(Simplex::from([0, 2]), Simplex::from([1]));
        let t = set(&[&[0, 2, 3], &[0, 2, 4]]);
        // The lower facets are the edges 1-2 and 0-1, which are not in t.
        assert_eq!(z.common_link(&t), None);
        assert!(!z.link_ok(&t, 3));
        // The other way round: drop 1 from the support, leaving the edge
        // 0-2, linked to 3 and 4 in t.
        let w = z.negated();
        assert_eq!(w.common_link(&t), Some(set(&[&[3], &[4]])));
        assert!(w.link_ok(&t, 3));
        // An empty negative side is never realised.
        assert!(!Circuit::new(Simplex::from([0, 2]), Simplex::new()).link_ok(&t, 3));
        // General position is vacuously fine.
        assert!(Circuit::new(Simplex::from([0, 2]), Simplex::from([1, 3])).link_ok(&t, 3));
    }

    #[test]
    fn circuits_of_a_square_with_a_collinear_point() {
        // (0,0) (1,0) (0,1) (1,1) (1,2): 1, 3, 4 lie on x = 1.
        let p = PointConfig::from_row_major(&[0, 0, 1, 0, 0, 1, 1, 1, 1, 2], 5, 2).unwrap();
        let chi = p.chirotope().unwrap();
        let found: Vec<(Vec<usize>, Vec<usize>)> = circuits(&chi)
            .unwrap()
            .iter()
            .map(|z| (z.positive().to_vec(), z.negative().to_vec()))
            .collect();
        // As `points2circuits` prints them, up to order.
        assert_eq!(
            found,
            vec![
                (vec![0, 3], vec![1, 2]),
                (vec![0, 4], vec![1, 2]),
                (vec![0, 4], vec![2, 3]),
                (vec![1, 4], vec![3]),
            ]
        );
    }
}
