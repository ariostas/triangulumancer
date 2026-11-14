//! The symmetry seam.
//!
//! TOPCOM's headline feature for large configurations is enumerating
//! triangulations *up to symmetry*. The 0.3 package always used the trivial
//! group, under which every symmetry loop in TOPCOM runs zero times and every
//! orbit-representative test passes (`TOPCOM.md` section 7.9), so that layer
//! could simply be deleted. It is threaded through the enumerations instead,
//! as the [`Symmetry`] trait, so that adding real symmetric enumeration later
//! means writing one more implementer rather than rediscovering where TOPCOM
//! consults the group (`PLAN.md` section 6.2).
//!
//! [`TrivialGroup`] is the only implementer. Every method is a constant, so
//! after monomorphisation the generated code is the same as if the layer did
//! not exist.

use crate::triang::Triangulation;

/// A group of label permutations acting on triangulations.
///
/// The enumerations of phases 4 and 5 are generic over this at exactly the
/// points TOPCOM consults its `SymmetryGroup`.
pub trait Symmetry {
    /// The stabiliser of a triangulation, used to prune flip enumeration.
    type Stabilizer;

    /// True when every method below is constant, letting callers skip orbit
    /// bookkeeping entirely rather than merely computing a trivial answer.
    fn is_trivial(&self) -> bool {
        false
    }

    /// Whether `t` is the canonical member of its orbit.
    fn is_orbit_representative(&self, t: &Triangulation) -> bool;

    /// The subgroup fixing `t`.
    fn stabilizer(&self, t: &Triangulation) -> Self::Stabilizer;

    /// The number of triangulations in the orbit of `t`.
    fn orbit_size(&self, t: &Triangulation) -> usize;
}

/// The trivial group: no symmetries but the identity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct TrivialGroup;

impl Symmetry for TrivialGroup {
    type Stabilizer = ();

    fn is_trivial(&self) -> bool {
        true
    }

    fn is_orbit_representative(&self, _: &Triangulation) -> bool {
        true
    }

    fn stabilizer(&self, _: &Triangulation) {}

    fn orbit_size(&self, _: &Triangulation) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplex::{Simplex, SimplexSet};

    #[test]
    fn the_trivial_group_is_trivial() {
        let t =
            Triangulation::new(3, 3, SimplexSet::from_iter([Simplex::from([0, 1, 2])])).unwrap();
        let g = TrivialGroup;
        assert!(g.is_trivial());
        assert!(g.is_orbit_representative(&t));
        assert_eq!(g.orbit_size(&t), 1);
        g.stabilizer(&t);
    }
}
