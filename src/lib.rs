//! Compute and work with triangulations of point and vector configurations.
//!
//! This crate is a pure-Rust reimplementation of the functionality that
//! `triangulumancer` 0.3 obtained from [CGAL] (regular and Delaunay
//! triangulations) and [TOPCOM] (placing and fine triangulations, bistellar
//! flips, and the two triangulation enumerations).
//!
//! # Status
//!
//! **The port is in progress.** See `PLAN.md` in the repository for the phase
//! breakdown; [`CGAL.md`] and [`TOPCOM.md`] document the algorithms being
//! reimplemented. Entry points that are not yet ported are absent from the Rust
//! API and raise `NotImplementedError` from the Python bindings.
//!
//! # Layout
//!
//! The crate is organised in three layers, and the boundary between them is
//! load-bearing (see `PLAN.md` section 6):
//!
//! * `config` — integer point/vector configurations and exact linear algebra
//!   over them. Coordinates live here and nowhere else, apart from `regular`.
//! * `om` — the oriented-matroid layer: chirotope, circuits, cocircuits. Built
//!   *from* a configuration, and the only thing the combinatorial algorithms
//!   consume.
//! * `triang` — triangulations, flips, and enumerations. These take an
//!   oriented matroid, never a configuration. Keeping that boundary is what
//!   makes the rest of TOPCOM's `chiro2*` functionality cheap to add later.
//!
//! `regular` sits beside them: a regular triangulation is *not* determined by
//! the oriented matroid, so it is the one algorithm that genuinely needs
//! coordinates.
//!
//! [CGAL]: https://www.cgal.org/
//! [TOPCOM]: https://www.wm.uni-bayreuth.de/de/team/rambau_joerg/TOPCOM/
//! [`CGAL.md`]: https://github.com/ariostas/triangulumancer/blob/main/CGAL.md
//! [`TOPCOM.md`]: https://github.com/ariostas/triangulumancer/blob/main/TOPCOM.md

pub mod error;

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
