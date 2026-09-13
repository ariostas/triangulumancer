//! The crate's error type.
//!
//! TOPCOM calls `exit(1)` on many internal error paths, which from inside a
//! Python extension kills the interpreter with no traceback. Every one of those
//! paths becomes a variant here, and the Python bindings map them to ordinary
//! exceptions. See `PLAN.md` section 5.2.

use std::fmt;

/// The result type used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Anything that can go wrong constructing configurations or triangulations.
///
/// Marked `#[non_exhaustive]`: later phases and later TOPCOM functionality will
/// add variants, and doing so should not be a breaking change.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// A point or vector was added whose length does not match the
    /// configuration's ambient dimension.
    DimensionMismatch {
        /// The dimension the configuration already has.
        expected: usize,
        /// The dimension of the offending input.
        found: usize,
    },

    /// More points or vectors were added after the configuration was used to
    /// build a triangulation.
    ///
    /// Triangulations refer to their configuration by index, so the
    /// configuration is frozen once any triangulation exists.
    Locked,

    /// The configuration does not span its ambient space.
    ///
    /// The 0.3 C++ implementation silently produced a malformed simplices array
    /// in this case for regular triangulations (bug B4 in `PLAN.md`).
    NotFullRank {
        /// The rank the configuration actually has.
        rank: usize,
        /// The number of rows in the configuration matrix.
        rows: usize,
    },

    /// The configuration is empty, or has rank zero, where that is not allowed.
    EmptyConfiguration,

    /// The rank exceeds the number of points or vectors.
    RankExceedsCount {
        /// The rank of the configuration.
        rank: usize,
        /// The number of points or vectors.
        count: usize,
    },

    /// A per-point parameter list (weights or heights) has the wrong length.
    ParameterCountMismatch {
        /// The number of points in the configuration.
        expected: usize,
        /// The number of parameters supplied.
        found: usize,
    },

    /// An exact computation exceeded even the arbitrary-precision path.
    ///
    /// This should be unreachable; it exists so that a bug surfaces as an error
    /// rather than a wrong answer.
    Overflow(&'static str),

    /// A triangulation supplied by the caller is not a valid triangulation of
    /// its configuration.
    InvalidTriangulation(String),

    /// An internal invariant was violated. Always a bug in this crate.
    Internal(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, found } => {
                write!(f, "dimension mismatch: expected {expected}, found {found}")
            }
            Self::Locked => write!(
                f,
                "the configuration is locked because a triangulation of it exists, \
                 so more points or vectors cannot be added"
            ),
            Self::NotFullRank { rank, rows } => {
                write!(f, "configuration is not full rank: rank {rank} < {rows}")
            }
            Self::EmptyConfiguration => {
                write!(
                    f,
                    "the number of points or vectors and the rank must both be at least one"
                )
            }
            Self::RankExceedsCount { rank, count } => write!(
                f,
                "rank {rank} must not be larger than the number of points or vectors ({count})"
            ),
            Self::ParameterCountMismatch { expected, found } => write!(
                f,
                "the number of parameters must match the number of points: \
                 expected {expected}, found {found}"
            ),
            Self::Overflow(what) => write!(f, "arithmetic overflow in {what}"),
            Self::InvalidTriangulation(why) => write!(f, "invalid triangulation: {why}"),
            Self::Internal(what) => {
                write!(
                    f,
                    "internal error in {what}; this is a bug, please report it"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_are_not_empty() {
        let errors = [
            Error::DimensionMismatch {
                expected: 3,
                found: 2,
            },
            Error::Locked,
            Error::NotFullRank { rank: 2, rows: 3 },
            Error::EmptyConfiguration,
            Error::RankExceedsCount { rank: 4, count: 3 },
            Error::ParameterCountMismatch {
                expected: 9,
                found: 8,
            },
            Error::Overflow("determinant"),
            Error::InvalidTriangulation("simplices overlap".into()),
            Error::Internal("chirotope cache"),
        ];
        for e in errors {
            assert!(!e.to_string().is_empty());
        }
    }
}
