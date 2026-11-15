//! Triangulations, and the algorithms that build them from an oriented matroid.
//!
//! Everything here takes an [`OrientedMatroid`], never a
//! [`Config`](crate::config::Config): a triangulation is a purely
//! combinatorial object once the chirotope is known. Keeping to that is what
//! makes TOPCOM's `chiro2*` entry points free later (`PLAN.md` section 6.1).
//!
//! * [`placing`] — the placing (pushing) triangulation.
//! * [`fine`] — the placing triangulation with every unused point flipped in.
//! * [`flip`] — bistellar flips: [`flips`], [`neighbors`].
//! * [`flipgraph`] — every triangulation connected to a seed by flips:
//!   [`FlipGraph`], [`all_connected`].
//! * [`Triangulation::check`] — whether a set of simplices is a triangulation.

mod check;
mod fine;
pub mod flip;
pub mod flipgraph;
mod placing;

pub use fine::fine;
pub use flip::{Flip, FlipRep, flips, neighbors};
pub use flipgraph::{FlipGraph, all_connected};
pub use placing::placing;

use crate::error::{Error, Result};
use crate::om::OrientedMatroid;
use crate::simplex::{Simplex, SimplexSet};

/// A triangulation of a configuration of `n` points or vectors: a set of
/// simplices of `rank` labels each.
///
/// This is plain data. It records the ground-set size and the simplex
/// cardinality so that it can answer questions like [`is_fine`] on its own,
/// but not the configuration or oriented matroid it triangulates; algorithms
/// that need signs take one alongside. That lets one representation serve
/// every source of triangulations — placing and fine (phase 2), flips and the
/// enumerations (phases 3–5), and regular triangulations (phase 6), whose
/// simplices may have fewer labels than the ambient rank when the
/// configuration was first reduced to its affine hull.
///
/// The simplices are kept in canonical order (`PLAN.md` section 5.6), so two
/// triangulations with the same simplices are equal, hash equally, and list
/// their simplices identically however they were built.
///
/// [`is_fine`]: Self::is_fine
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Triangulation {
    n: usize,
    rank: usize,
    simplices: SimplexSet,
}

impl Triangulation {
    /// A triangulation of a configuration of `n` elements whose simplices each
    /// have `rank` labels.
    ///
    /// Only the shape is checked here; whether the simplices actually
    /// triangulate anything is [`check`](Self::check).
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] for a simplex without exactly `rank`
    /// labels, and [`Error::IndexOutOfRange`] for a label not below `n`.
    pub fn new(n: usize, rank: usize, simplices: SimplexSet) -> Result<Self> {
        for s in &simplices {
            check_shape(s, n, rank)?;
        }
        Ok(Self { n, rank, simplices })
    }

    /// A triangulation from a row-major table of labels, one simplex per row
    /// of `width` entries: the shape of the Python `simplices` array.
    ///
    /// Rows may list their labels in any order, and the rows themselves may
    /// come in any order; the result is canonical either way.
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new), plus [`Error::InvalidTriangulation`] for a
    /// negative label, a label repeated within a row, or a simplex listed
    /// twice, and [`Error::DimensionMismatch`] if `labels.len()` is not a
    /// multiple of `width` or `width` is not `rank`.
    pub fn from_labels(n: usize, rank: usize, labels: &[i64], width: usize) -> Result<Self> {
        if width != rank {
            return Err(Error::DimensionMismatch {
                expected: rank,
                found: width,
            });
        }
        if width == 0 {
            if labels.is_empty() {
                return Ok(Self {
                    n,
                    rank,
                    simplices: SimplexSet::new(),
                });
            }
            return Err(Error::DimensionMismatch {
                expected: 0,
                found: labels.len(),
            });
        }
        if labels.len() % width != 0 {
            return Err(Error::DimensionMismatch {
                expected: labels.len().next_multiple_of(width),
                found: labels.len(),
            });
        }
        let mut simplices = SimplexSet::new();
        for row in labels.chunks_exact(width) {
            let mut s = Simplex::new();
            for &label in row {
                let label = usize::try_from(label).map_err(|_| {
                    Error::InvalidTriangulation(format!("negative label {label} in {row:?}"))
                })?;
                if label >= n {
                    return Err(Error::IndexOutOfRange {
                        index: label,
                        len: n,
                    });
                }
                if !s.insert(label) {
                    return Err(Error::InvalidTriangulation(format!(
                        "label {label} repeated in {row:?}"
                    )));
                }
            }
            if !simplices.insert(s) {
                return Err(Error::InvalidTriangulation(format!(
                    "simplex {row:?} listed twice"
                )));
            }
        }
        Ok(Self { n, rank, simplices })
    }

    /// The number of points or vectors in the configuration.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The number of labels in every simplex.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The simplices, in canonical order.
    pub fn simplices(&self) -> &SimplexSet {
        &self.simplices
    }

    /// The simplices, consuming the triangulation.
    pub fn into_simplices(self) -> SimplexSet {
        self.simplices
    }

    /// The number of simplices.
    pub fn len(&self) -> usize {
        self.simplices.len()
    }

    /// Whether there are no simplices, which a valid triangulation never is.
    pub fn is_empty(&self) -> bool {
        self.simplices.is_empty()
    }

    /// The simplices, in canonical order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &Simplex> + '_ {
        self.simplices.iter()
    }

    /// The simplices as sorted label lists, in canonical order.
    pub fn to_vecs(&self) -> Vec<Vec<usize>> {
        self.simplices.to_vecs()
    }

    /// The labels used by some simplex.
    pub fn support(&self) -> Simplex {
        self.simplices.support()
    }

    /// The labels used by no simplex.
    ///
    /// For a regular triangulation these are CGAL's *hidden points* (`PLAN.md`
    /// section 10); for a placing triangulation, the points that fell inside
    /// the hull of those placed before them.
    pub fn unused(&self) -> Simplex {
        &Simplex::full(self.n) - &self.support()
    }

    /// Whether every point or vector is used: `support == {0, ..., n - 1}`.
    ///
    /// The one definition of fineness, shared by every algorithm that filters
    /// on it (`TOPCOM.md` section 10.4).
    pub fn is_fine(&self) -> bool {
        self.unused().is_empty()
    }

    /// Check that this is a triangulation of `om`.
    ///
    /// This is TOPCOM's `CheckTriang` (`--checktriang`), and needs nothing but
    /// chirotope signs:
    ///
    /// 1. there is at least one simplex, and every simplex is a basis;
    /// 2. every facet of a simplex lies either in exactly one simplex, and then
    ///    no element of the configuration is strictly beyond it (it is on the
    ///    boundary of the configuration's hull or cone), or in exactly two,
    ///    which lie on opposite sides of it;
    /// 3. a generic point inside the first simplex — a lexicographic
    ///    extension — lies in no other simplex.
    ///
    /// (2) makes the number of simplices covering a generic point the same
    /// everywhere, and (3) makes that number one, which together characterise
    /// a triangulation (De Loera, Rambau and Santos, *Triangulations*,
    /// section 4.5).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidTriangulation`] naming the first violation found, and
    /// [`Error::DimensionMismatch`] if `om` does not have this triangulation's
    /// `n` and `rank`.
    pub fn check(&self, om: &impl OrientedMatroid) -> Result<()> {
        check::check(om, self)
    }
}

fn check_shape(s: &Simplex, n: usize, rank: usize) -> Result<()> {
    if s.len() != rank {
        return Err(Error::DimensionMismatch {
            expected: rank,
            found: s.len(),
        });
    }
    match s.last() {
        Some(last) if last >= n => Err(Error::IndexOutOfRange {
            index: last,
            len: n,
        }),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_checks_the_shape() {
        let t = Triangulation::from_labels(4, 3, &[2, 1, 0, 3, 2, 0], 3).unwrap();
        assert_eq!(t.to_vecs(), vec![vec![0, 1, 2], vec![0, 2, 3]]);
        assert_eq!((t.n(), t.rank(), t.len()), (4, 3, 2));
        assert!(t.is_fine());

        let t = Triangulation::from_labels(5, 3, &[0, 1, 2], 3).unwrap();
        assert!(!t.is_fine());
        assert_eq!(t.unused(), Simplex::from([3, 4]));

        assert!(matches!(
            Triangulation::from_labels(4, 3, &[0, 1, -1], 3),
            Err(Error::InvalidTriangulation(_))
        ));
        assert!(matches!(
            Triangulation::from_labels(4, 3, &[0, 1, 1], 3),
            Err(Error::InvalidTriangulation(_))
        ));
        assert!(matches!(
            Triangulation::from_labels(4, 3, &[0, 1, 2, 2, 1, 0], 3),
            Err(Error::InvalidTriangulation(_))
        ));
        assert_eq!(
            Triangulation::from_labels(4, 3, &[0, 1, 4], 3),
            Err(Error::IndexOutOfRange { index: 4, len: 4 })
        );
        assert!(Triangulation::from_labels(4, 3, &[0, 1], 2).is_err());
        assert!(Triangulation::from_labels(4, 3, &[0, 1, 2, 3], 3).is_err());
        assert!(Triangulation::from_labels(4, 3, &[], 3).unwrap().is_empty());

        assert!(Triangulation::new(4, 3, SimplexSet::from_iter([Simplex::from([0, 1])])).is_err());
    }
}
