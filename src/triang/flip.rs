//! Bistellar flips: finding them, expanding them, and applying them.
//!
//! A port of TOPCOM's `FlipRep`, `Flip` and `TriangFlips` (`lib-src/Flip.cc`,
//! `lib-src/TriangFlips.cc`; `TOPCOM.md` section 5), with the trivial
//! symmetry group.
//!
//! A flip of a triangulation `T` is supported on a circuit `Z = (Z⁺, Z⁻)`: the
//! two triangulations of `supp(Z)` are the facets dropping each element of
//! `Z⁺` and those dropping each element of `Z⁻`, and when one of them, joined
//! with a common link, is a subcomplex of `T`, replacing it by the other is a
//! flip. [`FlipRep`] is the circuit oriented so that the present side comes
//! first, and [`Flip`] is that expanded into the simplices it removes and
//! adds.
//!
//! [`flip_reps`] recomputes the whole flip set of a triangulation from
//! scratch. TOPCOM updates it incrementally along the flip graph
//! (`TriangFlips.cc:20-35`, `TOPCOM.md` section 5.3); that is a phase 7
//! optimisation, and this naive version is to remain its test oracle.

use std::collections::BTreeSet;

use super::Triangulation;
use crate::error::{Error, Result};
use crate::hash::{FastHashMap, FastHashSet};
use crate::om::{Circuit, OrientedMatroid};
use crate::simplex::{Simplex, SimplexSet};

/// A flip, represented by its circuit: [`removed_side`](Self::removed_side)
/// is TOPCOM's `first`, the side whose facets (joined with the common link)
/// are present in the triangulation, and [`added_side`](Self::added_side) is
/// `second`.
///
/// Both sides are non-empty. The representation does not depend on the
/// triangulation beyond which side is present, so the same `FlipRep` can be
/// valid in many triangulations; [`expand`](Self::expand) needs the one it
/// is applied to. Ordered by `(removed_side, added_side)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FlipRep {
    removed: Simplex,
    added: Simplex,
}

impl FlipRep {
    /// The flip on the circuit in the `(rank + 1)`-set `set`, if `t` realises
    /// either orientation of it. TOPCOM's `FlipRep(chiro, dependent_set, tn)`
    /// (`Flip.cc:22-61`).
    ///
    /// For a circuit in general position (`rank + 1` elements in its support)
    /// the side `Z⁺` is present when [`Circuit::containment_ok`] holds; in
    /// special position, when [`Circuit::link_ok`] does. `Z⁺` is tried first,
    /// then `Z⁻`. `None` when neither is present, or when the present side
    /// leaves nothing to flip to (a circuit with an empty side, which a
    /// configuration of vectors can have).
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] unless `set` has `rank + 1` labels, and
    /// whatever [`OrientedMatroid::sign`] returns.
    pub fn of_dependent_set(
        om: &impl OrientedMatroid,
        set: &Simplex,
        t: &Triangulation,
    ) -> Result<Option<Self>> {
        let z = Circuit::of_dependent_set(om, set)?;
        let rank = om.rank();
        let general = z.support().len() == rank + 1;
        let present = |c: &Circuit| {
            if general {
                c.containment_ok(t.simplices())
            } else {
                c.link_ok(t.simplices(), rank)
            }
        };
        let oriented = if present(&z) {
            Some(z)
        } else {
            let w = z.negated();
            present(&w).then_some(w)
        };
        Ok(oriented.and_then(|c| Self::new(c.positive().clone(), c.negative().clone())))
    }

    /// The flip that removes the facets dropping each element of `removed`
    /// and adds those dropping each element of `added`, or `None` if either
    /// side is empty. The sides should be disjoint and form a circuit;
    /// nothing else is checked.
    pub fn new(removed: Simplex, added: Simplex) -> Option<Self> {
        (!removed.is_empty() && !added.is_empty()).then_some(Self { removed, added })
    }

    /// TOPCOM's `first`: the side of the circuit whose facets are removed.
    pub fn removed_side(&self) -> &Simplex {
        &self.removed
    }

    /// TOPCOM's `second`: the side of the circuit whose facets are added.
    pub fn added_side(&self) -> &Simplex {
        &self.added
    }

    /// The circuit, as `(removed_side, added_side)`.
    pub fn circuit(&self) -> Circuit {
        Circuit::new(self.removed.clone(), self.added.clone())
    }

    /// The support of the circuit.
    pub fn support(&self) -> Simplex {
        &self.removed | &self.added
    }

    /// The flip back: the same circuit, the other side present.
    pub fn inverse(&self) -> Self {
        Self {
            removed: self.added.clone(),
            added: self.removed.clone(),
        }
    }

    /// Whether the flip removes a vertex: the added side is a single element,
    /// which every added simplex omits. TOPCOM's `kills_vertex`.
    pub fn kills_vertex(&self) -> bool {
        self.added.len() == 1
    }

    /// Whether the flip adds a vertex. TOPCOM's `adds_vertex`.
    pub fn adds_vertex(&self) -> bool {
        self.removed.len() == 1
    }

    /// Whether both sides have the same size, so the flip keeps the number of
    /// simplices. TOPCOM's `is_balanced`.
    pub fn is_balanced(&self) -> bool {
        self.removed.len() == self.added.len()
    }

    /// The simplices this flip removes from `t` and adds to it. TOPCOM's
    /// `Flip::_construct` (`Flip.cc:65-115`).
    ///
    /// In general position the removed simplices are the facets of the support
    /// dropping each element of [`removed_side`](Self::removed_side), and the
    /// added ones those dropping each element of
    /// [`added_side`](Self::added_side). In special position both are joined
    /// with the link in `t` of the first removed facet, which is the common
    /// link [`Circuit::link_ok`] checked.
    ///
    /// Meaningful only for a flip of `t` (one [`flip_reps`] returned); for
    /// anything else the result is unspecified but well-formed.
    pub fn expand(&self, t: &Triangulation) -> Flip {
        let z = self.circuit();
        let support = self.support();
        let (removed, added) = if support.len() == t.rank() + 1 {
            (z.lower_facets(), z.upper_facets())
        } else {
            let first = self.removed.first().unwrap_or(0);
            let link = t.simplices().link(&(&support - &Simplex::from([first])));
            (z.lower_facets().join(&link), z.upper_facets().join(&link))
        };
        Flip {
            rep: self.clone(),
            removed,
            added,
        }
    }
}

/// A flip expanded against a particular triangulation: the simplices it
/// removes and the simplices it adds.
///
/// These are TOPCOM's `Flip`, and 0.3's `Flip.pre` and `Flip.post`. Neither is
/// a triangulation of the whole configuration; they are the two triangulations
/// of the region the flip changes. Ordered by `(removed, added)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Flip {
    removed: SimplexSet,
    added: SimplexSet,
    rep: FlipRep,
}

impl Flip {
    /// The simplices removed, in canonical order.
    pub fn removed(&self) -> &SimplexSet {
        &self.removed
    }

    /// The simplices added, in canonical order.
    pub fn added(&self) -> &SimplexSet {
        &self.added
    }

    /// The circuit the flip is supported on.
    pub fn rep(&self) -> &FlipRep {
        &self.rep
    }

    /// `t` with this flip applied: `t − removed + added`, the neighbour of
    /// `t` across the flip. TOPCOM's `TriangNode(ID, tn, flip)`
    /// (`TriangNode.hh:111-119`).
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if an added simplex does not have `t`'s
    /// rank, which cannot happen for a flip [`expand`](FlipRep::expand)ed
    /// against `t`.
    pub fn apply(&self, t: &Triangulation) -> Result<Triangulation> {
        let simplices = t.simplices().difference(&self.removed).union(&self.added);
        Triangulation::new(t.n(), t.rank(), simplices)
    }
}

/// Every flip of `t`, as circuits, in order.
///
/// The candidates are TOPCOM's (`TriangFlips::_add_new_flips`,
/// `TriangFlips.cc:37-240`): the `(rank + 1)`-sets `S ∪ {v}` for a simplex
/// `S` of `t` and an element `v` that `t` does not use, and `S ∪ S'` for two
/// simplices sharing a facet. Each gives at most one flip
/// ([`FlipRep::of_dependent_set`]); different candidates can give the same
/// one, which is kept once.
///
/// With `forbid_vertex_removal`, flips that would remove a vertex are left
/// out ([`FlipRep::kills_vertex`]). Starting from a fine triangulation, that
/// keeps a walk along flips among fine triangulations, which is how the flip
/// graph enumerates them (`TOPCOM.md` section 6.3).
///
/// # Errors
///
/// [`Error::DimensionMismatch`] unless `om` has `t`'s size and rank, and
/// whatever [`OrientedMatroid::sign`] returns.
pub fn flip_reps(
    om: &impl OrientedMatroid,
    t: &Triangulation,
    forbid_vertex_removal: bool,
) -> Result<BTreeSet<FlipRep>> {
    check_matches(om, t)?;
    let rank = t.rank();
    let unused = t.unused();

    // Simplices by facet, to find the ones sharing a facet without TOPCOM's
    // quadratic scan; the candidates are the same.
    let mut by_facet: FastHashMap<Simplex, Vec<&Simplex>> = FastHashMap::default();
    for s in t.iter() {
        for f in s.facets() {
            by_facet.entry(f).or_default().push(s);
        }
    }

    let mut candidates: FastHashSet<Simplex> = FastHashSet::default();
    for s in t.iter() {
        for v in &unused {
            let mut set = s.clone();
            set.insert(v);
            candidates.insert(set);
        }
    }
    for adjacent in by_facet.values() {
        for (i, a) in adjacent.iter().enumerate() {
            for b in &adjacent[i + 1..] {
                let set = *a | *b;
                if set.len() == rank + 1 {
                    candidates.insert(set);
                }
            }
        }
    }

    let mut out = BTreeSet::new();
    for set in &candidates {
        if let Some(rep) = FlipRep::of_dependent_set(om, set, t)? {
            if forbid_vertex_removal && rep.kills_vertex() {
                continue;
            }
            out.insert(rep);
        }
    }
    Ok(out)
}

/// Every flip of `t`, expanded, ordered by the simplices removed and then
/// those added. This is 0.3's `Triangulation.flips()`.
///
/// # Errors
///
/// As [`flip_reps`].
pub fn flips(om: &impl OrientedMatroid, t: &Triangulation) -> Result<Vec<Flip>> {
    let mut out: Vec<Flip> = flip_reps(om, t, false)?
        .iter()
        .map(|rep| rep.expand(t))
        .collect();
    out.sort();
    Ok(out)
}

/// Every triangulation one flip away from `t`, in canonical order. This is
/// 0.3's `Triangulation.neighbors()`.
///
/// # Errors
///
/// As [`flip_reps`].
pub fn neighbors(om: &impl OrientedMatroid, t: &Triangulation) -> Result<Vec<Triangulation>> {
    let mut out = flip_reps(om, t, false)?
        .iter()
        .map(|rep| rep.expand(t).apply(t))
        .collect::<Result<Vec<_>>>()?;
    out.sort();
    Ok(out)
}

fn check_matches(om: &impl OrientedMatroid, t: &Triangulation) -> Result<()> {
    for (expected, found) in [(om.n(), t.n()), (om.rank(), t.rank())] {
        if expected != found {
            return Err(Error::DimensionMismatch { expected, found });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{PointConfig, VectorConfig};
    use crate::triang::{fine, placing};

    fn vecs(t: &SimplexSet) -> Vec<Vec<usize>> {
        t.to_vecs()
    }

    #[test]
    fn the_square_has_one_flip() {
        let p = PointConfig::from_row_major(&[0, 0, 1, 0, 1, 1, 0, 1], 4, 2).unwrap();
        let chi = p.chirotope().unwrap();
        let t = placing(&chi).unwrap();
        let fs = flips(&chi, &t).unwrap();
        assert_eq!(fs.len(), 1);
        assert_eq!(vecs(fs[0].removed()), vec![vec![0, 1, 2], vec![0, 2, 3]]);
        assert_eq!(vecs(fs[0].added()), vec![vec![0, 1, 3], vec![1, 2, 3]]);
        assert!(fs[0].rep().is_balanced());
        let n = neighbors(&chi, &t).unwrap();
        assert_eq!(n.len(), 1);
        assert_eq!(n[0].check(&chi), Ok(()));
        // And back again.
        assert_eq!(neighbors(&chi, &n[0]).unwrap(), vec![t]);
    }

    #[test]
    fn interior_point_in_and_out() {
        let p = PointConfig::from_row_major(&[0, 0, 3, 0, 0, 3, 1, 1], 4, 2).unwrap();
        let chi = p.chirotope().unwrap();
        let t = placing(&chi).unwrap();
        let reps = flip_reps(&chi, &t, false).unwrap();
        assert_eq!(reps.len(), 1);
        let rep = reps.first().unwrap();
        assert!(rep.adds_vertex());
        assert!(rep.inverse().kills_vertex());
        let f = fine(&chi).unwrap();
        assert_eq!(rep.expand(&t).apply(&t).unwrap(), f);
        assert_eq!(flip_reps(&chi, &f, false).unwrap().len(), 1);
        assert!(flip_reps(&chi, &f, true).unwrap().is_empty());
    }

    #[test]
    fn special_position_joins_the_link() {
        // A square with the midpoint 4 of the edge 1-2 (x = 2): the circuit
        // ({1, 2}, {4}) is in special position, linked to 0 or 3.
        let p = PointConfig::from_row_major(&[0, 0, 2, 0, 2, 2, 0, 2, 2, 1], 5, 2).unwrap();
        let chi = p.chirotope().unwrap();
        let t = fine(&chi).unwrap();
        assert!(t.is_fine());
        for f in flips(&chi, &t).unwrap() {
            let next = f.apply(&t).unwrap();
            assert_eq!(next.check(&chi), Ok(()), "{f:?}");
            assert_eq!(
                f.rep().inverse().expand(&next).apply(&next).unwrap(),
                t,
                "{f:?} is reversible"
            );
        }
    }

    #[test]
    fn a_single_simplex_has_no_flips() {
        let v = VectorConfig::from_row_major(&[1, 0, 0, 1], 2, 2).unwrap();
        let chi = v.chirotope().unwrap();
        let t = placing(&chi).unwrap();
        assert!(flips(&chi, &t).unwrap().is_empty());
    }

    #[test]
    fn mismatched_sizes_are_errors() {
        let p = PointConfig::from_row_major(&[0, 0, 1, 0, 1, 1, 0, 1], 4, 2).unwrap();
        let chi = p.chirotope().unwrap();
        let t = Triangulation::new(5, 3, SimplexSet::new()).unwrap();
        assert!(matches!(
            flips(&chi, &t),
            Err(Error::DimensionMismatch { .. })
        ));
    }
}
