//! Compute and work with triangulations of point and vector configurations.
//!
//! A pure-Rust implementation of what `triangulumancer` 0.3 obtained from
//! [CGAL] (regular and Delaunay triangulations) and [TOPCOM] (placing and fine
//! triangulations, bistellar flips, and the enumeration of all triangulations,
//! either connected by flips or all of them). There are no native
//! dependencies, every predicate is exact, and nothing panics or exits on bad
//! input: every failure is an [`Error`].
//!
//! # Example
//!
//! ```
//! use triangulumancer::config::PointConfig;
//! use triangulumancer::regular::{self, Scalar};
//! use triangulumancer::triang::{self, ExtensionGraph};
//!
//! # fn main() -> triangulumancer::Result<()> {
//! // The 3×3 grid, row-major: point i is coords[2 * i..2 * i + 2].
//! let coords = [0, 0, 1, 0, 1, 1, 0, 1, -1, 1, -1, 0, -1, -1, 0, -1, 1, -1];
//! let config = PointConfig::from_row_major(&coords, 9, 2)?;
//!
//! // Regular triangulations need the coordinates.
//! let delaunay = regular::delaunay(&config)?;
//! assert_eq!(delaunay.len(), 8);
//! let heights: Vec<Scalar> = (1..=9).map(Scalar::from).collect();
//! let t = regular::with_heights(&config, &heights)?;
//! assert!(t.check(&config.chirotope()?).is_ok());
//!
//! // Everything else works on the oriented matroid.
//! let chiro = config.chirotope()?;
//! let placing = triang::placing(&chiro)?;
//! assert_eq!(triang::neighbors(&chiro, &placing)?.len(), 6);
//! assert_eq!(triang::all_connected(&chiro, false)?.len(), 387);
//! assert_eq!(triang::all_triangulations(&chiro, true)?.len(), 64);
//!
//! // Or stream them, here counting without building any.
//! assert_eq!(ExtensionGraph::new(&chiro, false)?.count(), 387);
//! # Ok(())
//! # }
//! ```
//!
//! # Layout
//!
//! * [`arith`] — the exact determinant sign that every predicate in the crate
//!   reduces to (an `f64` filter with a proven error bound, then fraction-free
//!   elimination in `i128` and in arbitrary precision), and exact rank.
//! * [`config`] — integer point and vector configurations, and the exact
//!   reduction of a configuration to its affine hull.
//! * [`simplex`] — simplices as bitsets of labels, and sorted sets of them.
//! * [`om`] — the oriented-matroid layer: the [`OrientedMatroid`] trait and
//!   its implementation by a lazily evaluated [`Chirotope`], with circuits,
//!   cocircuits and the facets of the convex hull.
//! * [`triang`] — triangulations and everything built from an oriented
//!   matroid: placing and fine triangulations, bistellar flips, the flip
//!   graph ([`FlipGraph`]) and the extension graph ([`ExtensionGraph`]) —
//!   both streaming iterators, so enumerations run in bounded memory however
//!   many triangulations there are.
//! * [`regular`] — Delaunay and regular triangulations, from heights,
//!   weights or exact lifts (`f64`, integer or rational).
//! * [`symmetry`] — the seam for enumerating up to a symmetry group; only the
//!   trivial group exists so far.
//!
//! ## The oriented-matroid seam
//!
//! The combinatorial algorithms in [`triang`] take an `impl`
//! [`OrientedMatroid`] and never see coordinates: a triangulation is
//! determined by the chirotope alone. So they run unchanged on anything that
//! can answer chirotope queries — a configuration's [`Chirotope`], or an
//! oriented matroid given some other way — and the rest of TOPCOM's
//! `chiro2*` tools can be added without touching configurations. A regular
//! triangulation is *not* determined by the oriented matroid, which is why
//! [`regular`] is the one module that works on coordinates.
//!
//! # Features
//!
//! * `python` (off by default) — the PyO3 bindings that make up the Python
//!   package. The Rust library itself does not depend on PyO3.
//!
//! [CGAL]: https://www.cgal.org/
//! [TOPCOM]: https://www.wm.uni-bayreuth.de/de/team/rambau_joerg/TOPCOM/
//! [`OrientedMatroid`]: om::OrientedMatroid
//! [`Chirotope`]: om::Chirotope
//! [`FlipGraph`]: triang::FlipGraph
//! [`ExtensionGraph`]: triang::ExtensionGraph

pub mod arith;
pub mod config;
pub mod error;
mod hash;
pub mod om;
pub mod regular;
pub mod simplex;
pub mod symmetry;
pub mod triang;

#[cfg(feature = "python")]
pub mod python;

#[doc(inline)]
pub use error::{Error, Result};

/// Whether a configuration is a point configuration or a vector configuration.
///
/// The distinction is not cosmetic. A point configuration is homogenised — each
/// point carries an implicit trailing `1` — so a full-dimensional simplex of a
/// `d`-dimensional point configuration has `d + 1` vertices, while one of a
/// `d`-dimensional vector configuration has `d`.
///
/// The 0.3 C++ implementation threaded this through a dozen call sites as
/// `1 * is_pc` integer arithmetic. Keeping it a type makes the rank and simplex
/// sizes derivable in one place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ConfigKind {
    /// Points in affine space, homogenised with a trailing `1`.
    Point,
    /// Vectors in linear space, used as-is.
    Vector,
}

impl ConfigKind {
    /// The number of rows the configuration matrix carries for an ambient
    /// dimension of `dim`: `dim + 1` for points, `dim` for vectors.
    pub fn rows_for_dim(self, dim: usize) -> usize {
        match self {
            Self::Point => dim + 1,
            Self::Vector => dim,
        }
    }

    /// The ambient dimension implied by a configuration matrix with `rows` rows.
    ///
    /// Returns `0` for an empty matrix rather than underflowing.
    pub fn dim_for_rows(self, rows: usize) -> usize {
        match self {
            Self::Point => rows.saturating_sub(1),
            Self::Vector => rows,
        }
    }

    /// Whether this is a point configuration.
    pub fn is_point(self) -> bool {
        matches!(self, Self::Point)
    }

    /// The noun used for members of this kind of configuration, for messages.
    pub fn noun(self) -> &'static str {
        match self {
            Self::Point => "point",
            Self::Vector => "vector",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_and_dim_round_trip() {
        for kind in [ConfigKind::Point, ConfigKind::Vector] {
            for dim in 0..8 {
                assert_eq!(kind.dim_for_rows(kind.rows_for_dim(dim)), dim);
            }
        }
    }

    #[test]
    fn empty_matrix_has_dimension_zero() {
        assert_eq!(ConfigKind::Point.dim_for_rows(0), 0);
        assert_eq!(ConfigKind::Vector.dim_for_rows(0), 0);
    }

    #[test]
    fn point_configurations_are_homogenised() {
        assert_eq!(ConfigKind::Point.rows_for_dim(3), 4);
        assert_eq!(ConfigKind::Vector.rows_for_dim(3), 3);
    }
}
