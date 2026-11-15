//! Cocircuits, the facets of a configuration, and which facets of each basis
//! are interior.
//!
//! These are what the extension-graph enumeration (phase 5) needs to tell
//! when a set of simplices closes up into a triangulation: a facet of a
//! simplex either lies on the boundary of the configuration — in one of its
//! [`Facets`] — or is *interior*, and then a triangulation must cover it
//! exactly twice. They are oriented-matroid objects and live here rather than
//! in `triang`, per `PLAN.md` section 6.1; they are also what TOPCOM's
//! `points2cocircuits` and `points2facets` print.

use std::collections::BTreeMap;

use super::OrientedMatroid;
use crate::arith::Sign;
use crate::error::{Error, Result};
use crate::simplex::{Simplex, combinations};

/// A signed cocircuit `Y = (Y⁺, Y⁻)`: the elements strictly on either side of
/// a hyperplane spanned by elements of the configuration.
///
/// The orientation is TOPCOM's: built by [`of_hyperplane`], the *positive*
/// part is the side of the smallest element off the hyperplane. The negation
/// `(Y⁻, Y⁺)` is the same cocircuit, oriented the other way. A cocircuit with
/// an empty side is *positive* ([`is_positive`]): its hyperplane supports the
/// configuration, and the elements on it form a facet.
///
/// [`of_hyperplane`]: Self::of_hyperplane
/// [`is_positive`]: Self::is_positive
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Cocircuit {
    positive: Simplex,
    negative: Simplex,
}

impl Cocircuit {
    /// The cocircuit with the given parts. They should be disjoint; nothing is
    /// checked.
    pub fn new(positive: Simplex, negative: Simplex) -> Self {
        Self { positive, negative }
    }

    /// The cocircuit of the hyperplane spanned by `coplanar`, a set of
    /// `rank - 1` elements of `om`.
    ///
    /// Element `x` is on the side given by the orientation of the ordered
    /// basis `(coplanar, x)` ([`OrientedMatroid::sign_appended`]), and on the
    /// hyperplane when that is zero. The side of the first element off the
    /// hyperplane is [`positive`](Self::positive). This is TOPCOM's
    /// `Cocircuit(chiro, coplanar_set)` (`Cocircuits.cc:20-57`).
    ///
    /// If `coplanar` does not span a hyperplane (its elements are dependent),
    /// every element is "on" it and the result is empty on both sides.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] unless `coplanar` has `rank - 1` labels,
    /// and whatever [`OrientedMatroid::sign`] returns.
    pub fn of_hyperplane(om: &impl OrientedMatroid, coplanar: &Simplex) -> Result<Self> {
        let rank = om.rank();
        if rank == 0 || coplanar.len() != rank - 1 {
            return Err(Error::DimensionMismatch {
                expected: rank.saturating_sub(1),
                found: coplanar.len(),
            });
        }
        let mut positive = Simplex::new();
        let mut negative = Simplex::new();
        let mut reference = Sign::Zero;
        for x in 0..om.n() {
            if coplanar.contains(x) {
                continue;
            }
            let s = om.sign_appended(coplanar, x)?;
            if s.is_zero() {
                continue;
            }
            if reference.is_zero() {
                reference = s;
            }
            if s == reference {
                positive.insert(x);
            } else {
                negative.insert(x);
            }
        }
        Ok(Self { positive, negative })
    }

    /// `Y⁺`, TOPCOM's `first`.
    pub fn positive(&self) -> &Simplex {
        &self.positive
    }

    /// `Y⁻`, TOPCOM's `second`.
    pub fn negative(&self) -> &Simplex {
        &self.negative
    }

    /// `Y⁺ ∪ Y⁻`, the elements off the hyperplane.
    pub fn support(&self) -> Simplex {
        &self.positive | &self.negative
    }

    /// The same cocircuit with its sides exchanged.
    pub fn negated(&self) -> Self {
        Self {
            positive: self.negative.clone(),
            negative: self.positive.clone(),
        }
    }

    /// Whether one side is empty: the hyperplane supports the configuration.
    /// TOPCOM's "positive cocircuit". False for the empty cocircuit.
    pub fn is_positive(&self) -> bool {
        (self.positive.is_empty() || self.negative.is_empty()) && !self.support().is_empty()
    }
}

/// Every cocircuit of `om`, one per support, ordered by support; with
/// `only_positive`, only those with an empty side.
///
/// Each `(rank - 1)`-subset of the ground set spans a hyperplane or is
/// dependent; collecting the cocircuits of the hyperplanes and dropping
/// repeated supports gives every cocircuit, each in the orientation of
/// [`Cocircuit::of_hyperplane`]. This is TOPCOM's `Cocircuits(chiro,
/// only_positive)` (`Cocircuits.cc:60-102`), which keeps the same set in a
/// hash table keyed by support.
///
/// # Errors
///
/// [`Error::EmptyConfiguration`] for rank zero, and whatever
/// [`OrientedMatroid::sign`] returns.
pub fn cocircuits(om: &impl OrientedMatroid, only_positive: bool) -> Result<Vec<Cocircuit>> {
    let rank = om.rank();
    if rank == 0 {
        return Err(Error::EmptyConfiguration);
    }
    let mut by_support = BTreeMap::new();
    for coplanar in combinations(om.n(), rank - 1) {
        let y = Cocircuit::of_hyperplane(om, &coplanar)?;
        if y.support().is_empty() || (only_positive && !y.is_positive()) {
            continue;
        }
        by_support.entry(y.support()).or_insert(y);
    }
    Ok(by_support.into_values().collect())
}

/// The facets of a configuration: for each supporting hyperplane, the set of
/// elements on it.
///
/// A positive cocircuit `(Y⁺, ∅)` gives the facet `groundset \ Y⁺`. This is
/// TOPCOM's `Facets(cocircuits)` (`Facets.cc:17-62`) and the output of
/// `points2facets`. A facet in this sense is a set of labels, not a simplex:
/// it may hold more than `rank - 1` elements, and a configuration whose cone
/// is the whole space (a vector configuration that is not pointed) has none.
///
/// Kept in canonical order.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Facets {
    n: usize,
    rank: usize,
    facets: Vec<Simplex>,
}

impl Facets {
    /// The facets of `om`, from its positive cocircuits.
    ///
    /// # Errors
    ///
    /// As [`cocircuits`].
    pub fn new(om: &impl OrientedMatroid) -> Result<Self> {
        Ok(Self::from_cocircuits(
            om.n(),
            om.rank(),
            &cocircuits(om, true)?,
        ))
    }

    /// The facets given by the positive members of `cocircuits`, of an
    /// oriented matroid with `n` elements and rank `rank`. Cocircuits with
    /// both sides non-empty are ignored.
    pub fn from_cocircuits(n: usize, rank: usize, cocircuits: &[Cocircuit]) -> Self {
        let ground = Simplex::full(n);
        let mut facets: Vec<Simplex> = cocircuits
            .iter()
            .filter(|y| y.is_positive())
            .map(|y| &ground - &y.support())
            .collect();
        facets.sort();
        facets.dedup();
        Self { n, rank, facets }
    }

    /// The number of elements of the ground set.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The rank of the oriented matroid.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The number of facets.
    pub fn len(&self) -> usize {
        self.facets.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.facets.is_empty()
    }

    /// The facets, in canonical order.
    pub fn iter(&self) -> std::slice::Iter<'_, Simplex> {
        self.facets.iter()
    }

    /// The facet containing `face`, if any.
    ///
    /// For a face spanning a hyperplane — a facet of a basis — there is at
    /// most one: the facet of that hyperplane.
    pub fn containing(&self, face: &Simplex) -> Option<&Simplex> {
        self.facets.iter().find(|f| f.is_superset(face))
    }
}

impl<'a> IntoIterator for &'a Facets {
    type Item = &'a Simplex;
    type IntoIter = std::slice::Iter<'a, Simplex>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// How the facets of one basis meet the boundary of the configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Incidence {
    /// The facets of the basis that lie in no facet of the configuration, in
    /// canonical order.
    pub interior: Vec<Simplex>,
    /// The configuration facets that contain some facet of the basis, in
    /// canonical order.
    pub boundary: Vec<Simplex>,
}

/// For every basis of an oriented matroid, which of its facets are interior
/// and which configuration facets hold the others.
///
/// A facet of a basis is *interior* when no configuration facet contains
/// it. In a triangulation every interior facet of a simplex is shared with
/// exactly one other simplex, and every other facet with none; that is what
/// lets the extension graph recognise a complete triangulation. This is the
/// `_intfacets` / `_conffacets` half of TOPCOM's `Incidences`
/// (`Incidences.cc:22-108`); the reverse maps it also builds feed only
/// non-default search options and regularity checks, and are not kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Incidences {
    rank: usize,
    by_basis: BTreeMap<Simplex, Incidence>,
}

impl Incidences {
    /// The incidences of every basis of `om` with `facets`, which should be
    /// [`Facets::new(om)`](Facets::new).
    ///
    /// # Errors
    ///
    /// [`Error::EmptyConfiguration`] for rank zero, [`Error::DimensionMismatch`]
    /// if `facets` belongs to an oriented matroid of a different size or rank,
    /// and whatever [`OrientedMatroid::sign`] returns.
    pub fn new(om: &impl OrientedMatroid, facets: &Facets) -> Result<Self> {
        if om.rank() == 0 {
            return Err(Error::EmptyConfiguration);
        }
        Self::of_bases(om, facets, combinations(om.n(), om.rank()))
    }

    /// The incidences of the given bases only. Dependent sets among them are
    /// skipped.
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new), and as [`OrientedMatroid::sign`] for a set of
    /// the wrong size.
    pub fn of_bases(
        om: &impl OrientedMatroid,
        facets: &Facets,
        bases: impl IntoIterator<Item = Simplex>,
    ) -> Result<Self> {
        if (facets.n(), facets.rank()) != (om.n(), om.rank()) {
            return Err(Error::DimensionMismatch {
                expected: om.n(),
                found: facets.n(),
            });
        }
        let mut by_basis = BTreeMap::new();
        for basis in bases {
            if om.sign(&basis)?.is_zero() {
                continue;
            }
            let mut incidence = Incidence::default();
            for facet in basis.facets() {
                match facets.containing(&facet) {
                    Some(f) => incidence.boundary.push(f.clone()),
                    None => incidence.interior.push(facet),
                }
            }
            incidence.interior.sort();
            incidence.boundary.sort();
            incidence.boundary.dedup();
            by_basis.insert(basis, incidence);
        }
        Ok(Self {
            rank: om.rank(),
            by_basis,
        })
    }

    /// The rank of the oriented matroid.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The number of bases recorded.
    pub fn len(&self) -> usize {
        self.by_basis.len()
    }

    /// Whether no basis is recorded.
    pub fn is_empty(&self) -> bool {
        self.by_basis.is_empty()
    }

    /// The incidence of `basis`, or `None` if it is not a recorded basis.
    pub fn get(&self, basis: &Simplex) -> Option<&Incidence> {
        self.by_basis.get(basis)
    }

    /// The interior facets of `basis`, TOPCOM's `intfacets(basis)`.
    pub fn interior_facets(&self, basis: &Simplex) -> Option<&[Simplex]> {
        self.get(basis).map(|i| i.interior.as_slice())
    }

    /// Every recorded basis and its incidence, in canonical order.
    pub fn iter(&self) -> std::collections::btree_map::Iter<'_, Simplex, Incidence> {
        self.by_basis.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{PointConfig, VectorConfig};

    fn vecs(sets: &[Simplex]) -> Vec<Vec<usize>> {
        sets.iter().map(Simplex::to_vec).collect()
    }

    #[test]
    fn cocircuits_of_a_triangle_with_an_interior_point() {
        // 3 is inside the triangle 0, 1, 2.
        let p = PointConfig::from_row_major(&[0, 0, 3, 0, 0, 3, 1, 1], 4, 2).unwrap();
        let chi = p.chirotope().unwrap();
        let y = Cocircuit::of_hyperplane(&chi, &Simplex::from([0, 1])).unwrap();
        assert_eq!(y.positive(), &Simplex::from([2, 3]));
        assert!(y.negative().is_empty());
        assert!(y.is_positive());
        // The line through 0 and 3 separates 1 from 2.
        let y = Cocircuit::of_hyperplane(&chi, &Simplex::from([0, 3])).unwrap();
        assert_eq!(y.support(), Simplex::from([1, 2]));
        assert!(!y.is_positive());
        assert_eq!(y.negated().positive(), y.negative());

        assert_eq!(cocircuits(&chi, false).unwrap().len(), 6);
        assert_eq!(cocircuits(&chi, true).unwrap().len(), 3);
        let facets = Facets::new(&chi).unwrap();
        assert_eq!(
            vecs(&facets.iter().cloned().collect::<Vec<_>>()),
            vec![vec![0, 1], vec![0, 2], vec![1, 2]]
        );
        assert!(Cocircuit::of_hyperplane(&chi, &Simplex::from([0])).is_err());
    }

    #[test]
    fn facets_hold_every_point_on_their_hyperplane() {
        // A square with the midpoint 4 of the edge 0-1.
        let p = PointConfig::from_row_major(&[0, 0, 2, 0, 2, 2, 0, 2, 1, 0], 5, 2).unwrap();
        let chi = p.chirotope().unwrap();
        let facets = Facets::new(&chi).unwrap();
        assert_eq!(facets.len(), 4);
        assert_eq!(
            facets.containing(&Simplex::from([0, 4])),
            Some(&Simplex::from([0, 1, 4]))
        );
        assert_eq!(facets.containing(&Simplex::from([0, 2])), None);

        let inc = Incidences::new(&chi, &facets).unwrap();
        // 0, 1, 4 are collinear, so of the ten triples nine are bases.
        assert_eq!(inc.len(), 9);
        let i = inc.get(&Simplex::from([0, 2, 4])).unwrap();
        assert_eq!(vecs(&i.interior), vec![vec![0, 2], vec![2, 4]]);
        assert_eq!(vecs(&i.boundary), vec![vec![0, 1, 4]]);
        assert!(inc.get(&Simplex::from([0, 1, 4])).is_none());
        assert_eq!(
            inc.interior_facets(&Simplex::from([0, 1, 2])).map(vecs),
            Some(vec![vec![0, 2]])
        );
    }

    #[test]
    fn a_configuration_spanning_everything_has_no_facets() {
        // ±e1, ±e2 positively span the plane.
        let v = VectorConfig::from_row_major(&[1, 0, 0, 1, -1, 0, 0, -1], 4, 2).unwrap();
        let chi = v.chirotope().unwrap();
        assert!(Facets::new(&chi).unwrap().is_empty());
        assert_eq!(cocircuits(&chi, false).unwrap().len(), 2);
        let inc = Incidences::new(&chi, &Facets::new(&chi).unwrap()).unwrap();
        assert!(inc.iter().all(|(_, i)| i.boundary.is_empty()));
    }
}
