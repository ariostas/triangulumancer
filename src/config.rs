//! Integer point and vector configurations.
//!
//! A [`Config`] is a column-major integer matrix with one column per point or
//! vector. Point configurations are *homogenised*: each column carries a
//! trailing `1`, so that a `d`-dimensional point configuration is stored as a
//! vector configuration in dimension `d + 1`. That is the only difference
//! between the two kinds, and it is made once, here: everything downstream sees
//! a vector configuration of some rank `r`, exactly as TOPCOM does
//! (`TOPCOM.md` section 10.2).
//!
//! The kind is a type parameter rather than a flag ([`Config<Points>`] versus
//! [`Config<Vectors>`]), so the homogenisation rule lives in one place instead
//! of being threaded through call sites as `1 * is_pc` arithmetic.
//!
//! Coordinates live here and nowhere else apart from the regular-triangulation
//! code: the combinatorial algorithms take an oriented matroid built *from* a
//! configuration, never the configuration itself (`PLAN.md` section 6.1). The
//! entry point for that is [`Config::sign_of_simplex`], the chirotope of the
//! homogenised matrix.

pub mod hull;

use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::ConfigKind;
use crate::arith::{self, Sign};
use crate::error::{Error, Result};
use crate::simplex::Simplex;

mod sealed {
    pub trait Sealed {}
}

/// The kind of a configuration, at the type level: [`Points`] or [`Vectors`].
///
/// Sealed; the two implementers are the only ones.
pub trait Kind:
    sealed::Sealed
    + Copy
    + Default
    + fmt::Debug
    + PartialEq
    + Eq
    + std::hash::Hash
    + Send
    + Sync
    + 'static
{
    /// The same distinction as a value.
    const KIND: ConfigKind;
}

/// Marker for point configurations: points in affine space, homogenised with
/// a trailing `1`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Points;

/// Marker for vector configurations: vectors in linear space, used as-is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Vectors;

impl sealed::Sealed for Points {}
impl sealed::Sealed for Vectors {}

impl Kind for Points {
    const KIND: ConfigKind = ConfigKind::Point;
}

impl Kind for Vectors {
    const KIND: ConfigKind = ConfigKind::Vector;
}

/// A point configuration.
pub type PointConfig = Config<Points>;

/// A vector configuration.
pub type VectorConfig = Config<Vectors>;

/// A configuration of integer points or vectors.
///
/// Points and vectors are labelled `0..len()` in insertion order, and those
/// labels are what simplices refer to.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Config<K: Kind> {
    /// The ambient dimension.
    dim: usize,
    /// The number of points or vectors.
    len: usize,
    /// The homogenised matrix, column-major: column `j` is
    /// `data[j * rows .. (j + 1) * rows]`.
    data: Vec<i64>,
    kind: PhantomData<K>,
}

impl<K: Kind> Default for Config<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Kind> Config<K> {
    /// An empty configuration.
    ///
    /// Its dimension is `0` until the first point or vector is added, which
    /// then sets it — the 0.3 behaviour of `PointConfiguration()`.
    pub fn new() -> Self {
        Self::with_dim(0)
    }

    /// An empty configuration of ambient dimension `dim`.
    pub fn with_dim(dim: usize) -> Self {
        Self {
            dim,
            len: 0,
            data: Vec::new(),
            kind: PhantomData,
        }
    }

    /// A configuration of `n` points or vectors of dimension `dim`, given
    /// row-major: point `i` is `coords[i * dim .. (i + 1) * dim]`.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if `coords.len() != n * dim`.
    pub fn from_row_major(coords: &[i64], n: usize, dim: usize) -> Result<Self> {
        let mut c = Self::with_dim(dim);
        c.extend_row_major(coords, n, dim)?;
        Ok(c)
    }

    /// A configuration from an array of shape `(n, dim)`, as numpy hands it
    /// over: row-major data plus a shape.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidShape`] unless the shape is two-dimensional, and
    /// [`Error::DimensionMismatch`] if the data does not match it.
    pub fn from_array(data: &[i64], shape: &[usize]) -> Result<Self> {
        match *shape {
            [n, dim] => Self::from_row_major(data, n, dim),
            _ => Err(Error::InvalidShape {
                expected: "a two-dimensional array",
                found: shape.len(),
            }),
        }
    }

    /// Append from an array: a single point or vector of shape `(dim,)`, or
    /// several of shape `(n, dim)`.
    ///
    /// Nothing is appended unless everything is valid.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidShape`] for any other number of dimensions, and
    /// [`Error::DimensionMismatch`] if the dimension does not match.
    pub fn extend_from_array(&mut self, data: &[i64], shape: &[usize]) -> Result<()> {
        match *shape {
            [dim] => self.extend_row_major(data, 1, dim),
            [n, dim] => self.extend_row_major(data, n, dim),
            _ => Err(Error::InvalidShape {
                expected: "a one- or two-dimensional array",
                found: shape.len(),
            }),
        }
    }

    /// Append one point or vector.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if its length is not the configuration's
    /// dimension (any length is accepted into an empty configuration).
    pub fn push(&mut self, coords: &[i64]) -> Result<()> {
        self.extend_row_major(coords, 1, coords.len())
    }

    /// Append `n` points or vectors of dimension `dim`, given row-major.
    ///
    /// Into an empty configuration this sets the dimension. Nothing is appended
    /// unless everything is valid.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if `dim` is not the configuration's
    /// dimension, or `coords.len() != n * dim`.
    pub fn extend_row_major(&mut self, coords: &[i64], n: usize, dim: usize) -> Result<()> {
        let expected = n.checked_mul(dim).ok_or(Error::Overflow("array size"))?;
        if coords.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                found: coords.len(),
            });
        }
        if self.len > 0 && dim != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                found: dim,
            });
        }
        if self.len == 0 {
            self.dim = dim;
        }
        let rows = self.rows();
        self.data.reserve(n * rows);
        if dim == 0 {
            // Each column is empty, or just the homogenising 1.
            self.data.extend(std::iter::repeat_n(1, n * rows));
        } else {
            for point in coords.chunks_exact(dim) {
                self.data.extend_from_slice(point);
                if K::KIND.is_point() {
                    self.data.push(1);
                }
            }
        }
        self.len += n;
        Ok(())
    }

    /// Point or vector configuration.
    pub fn kind(&self) -> ConfigKind {
        K::KIND
    }

    /// The ambient dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The number of points or vectors.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no points or vectors.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of rows of the homogenised matrix: `dim + 1` for points,
    /// `dim` for vectors. The rank can be at most this.
    pub fn rows(&self) -> usize {
        K::KIND.rows_for_dim(self.dim)
    }

    /// The homogenised matrix, column-major.
    pub fn homogenised(&self) -> &[i64] {
        &self.data
    }

    /// Column `j` of the homogenised matrix: the coordinates, then `1` for a
    /// point configuration.
    pub fn column(&self, j: usize) -> Option<&[i64]> {
        let rows = self.rows();
        (j < self.len).then(|| &self.data[j * rows..(j + 1) * rows])
    }

    /// The coordinates of point or vector `j`, without homogenisation.
    pub fn coords(&self, j: usize) -> Option<&[i64]> {
        self.column(j).map(|c| &c[..self.dim])
    }

    /// The coordinates of every point or vector, in label order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[i64]> + '_ {
        (0..self.len).map(move |j| {
            let rows = self.rows();
            &self.data[j * rows..j * rows + self.dim]
        })
    }

    /// The coordinates as a row-major `len() x dim()` matrix, the shape of the
    /// Python `points` / `vectors` arrays.
    pub fn to_row_major(&self) -> Vec<i64> {
        self.iter().flatten().copied().collect()
    }

    /// The exact rank of the homogenised matrix.
    ///
    /// For a point configuration this is one more than the dimension of the
    /// affine hull; for a non-empty one it is at least `1`.
    pub fn rank(&self) -> usize {
        // Read row-major, the column-major data is the transpose, which has
        // the same rank. The shape is consistent by construction.
        arith::rank(&self.data, self.len, self.rows()).unwrap_or(0)
    }

    /// Whether the configuration spans its ambient space: rank equal to
    /// [`rows`](Self::rows). TOPCOM requires this.
    pub fn is_full_rank(&self) -> bool {
        self.rank() == self.rows()
    }

    /// Fail unless the configuration is non-empty and full rank.
    ///
    /// # Errors
    ///
    /// [`Error::EmptyConfiguration`] if there are no points or vectors or the
    /// rank is zero, and [`Error::NotFullRank`] if the rank is short of
    /// [`rows`](Self::rows).
    pub fn require_full_rank(&self) -> Result<()> {
        let rank = self.rank();
        if self.len == 0 || rank == 0 {
            return Err(Error::EmptyConfiguration);
        }
        if rank < self.rows() {
            return Err(Error::NotFullRank {
                rank,
                rows: self.rows(),
            });
        }
        Ok(())
    }

    /// The sign of the determinant of the homogenised columns `columns`, in
    /// the order given. There must be exactly [`rows`](Self::rows) of them.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] for the wrong number of columns, and
    /// [`Error::IndexOutOfRange`] for a label past the end.
    pub fn sign_of_columns<I>(&self, columns: I) -> Result<Sign>
    where
        I: IntoIterator<Item = usize>,
    {
        const INLINE: usize = 64;
        let rows = self.rows();
        let size = rows * rows;
        let mut inline = [0i64; INLINE];
        let mut heap = Vec::new();
        let buf: &mut [i64] = if size <= INLINE {
            &mut inline[..size]
        } else {
            heap.resize(size, 0);
            &mut heap
        };
        let mut count = 0;
        for j in columns {
            if count == rows {
                count += 1;
                break;
            }
            let col = self.column(j).ok_or(Error::IndexOutOfRange {
                index: j,
                len: self.len,
            })?;
            buf[count * rows..(count + 1) * rows].copy_from_slice(col);
            count += 1;
        }
        if count != rows {
            return Err(Error::DimensionMismatch {
                expected: rows,
                found: count,
            });
        }
        arith::sign_of_det(buf, rows)
    }

    /// The chirotope of the homogenised matrix at a basis: the sign of the
    /// determinant of the columns in `simplex`, taken in increasing label
    /// order. This is TOPCOM's `chiro(simplex)`, and the one place the
    /// oriented-matroid layer touches coordinates.
    ///
    /// # Errors
    ///
    /// As [`sign_of_columns`](Self::sign_of_columns): the simplex must have
    /// exactly [`rows`](Self::rows) labels, all in range.
    pub fn sign_of_simplex(&self, simplex: &Simplex) -> Result<Sign> {
        self.sign_of_columns(simplex.iter())
    }
}

impl<K: Kind> fmt::Display for Config<K> {
    /// The 0.3 `__repr__`, verbatim: e.g.
    /// `A 2-dimensional point configuration with 9 points`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let noun = K::KIND.noun();
        write!(
            f,
            "A {}-dimensional {noun} configuration with {} {noun}s",
            self.dim, self.len
        )
    }
}

impl<K: Kind> fmt::Debug for Config<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Config")
            .field("kind", &K::KIND)
            .field("dim", &self.dim)
            .field("coords", &self.iter().collect::<Vec<_>>())
            .finish()
    }
}

/// A configuration that can be extended until it is locked.
///
/// This models the 0.3 Python semantics: a configuration accepts new points
/// until the first triangulation of it is built, after which it is frozen,
/// because triangulations refer to it by label. Locking hands out a shared,
/// immutable [`Arc`]; since nothing can modify the configuration after that,
/// no further synchronisation is needed — this replaces the C++
/// `recursive_mutex` (`PLAN.md` section 5.3).
#[derive(Debug, Clone, Default)]
pub struct Lockable<K: Kind> {
    config: Arc<Config<K>>,
    locked: bool,
}

impl<K: Kind> Lockable<K> {
    /// Wrap an unlocked configuration.
    pub fn new(config: Config<K>) -> Self {
        Self {
            config: Arc::new(config),
            locked: false,
        }
    }

    /// The configuration.
    pub fn get(&self) -> &Config<K> {
        &self.config
    }

    /// A shared handle to the configuration as it is now, without locking it.
    ///
    /// For computing a triangulation without holding a borrow of `self`: run
    /// the computation on the snapshot, then lock with
    /// [`lock_snapshot`](Self::lock_snapshot).
    pub fn snapshot(&self) -> Arc<Config<K>> {
        Arc::clone(&self.config)
    }

    /// Mutable access, for adding points or vectors.
    ///
    /// If a [`snapshot`](Self::snapshot) is still alive, the configuration is
    /// copied first, so the snapshot never changes.
    ///
    /// # Errors
    ///
    /// [`Error::Locked`] once [`lock`](Self::lock) has been called.
    pub fn modify(&mut self) -> Result<&mut Config<K>> {
        if self.locked {
            return Err(Error::Locked);
        }
        Ok(Arc::make_mut(&mut self.config))
    }

    /// Lock the configuration and return a shared handle to it. Idempotent.
    pub fn lock(&mut self) -> Arc<Config<K>> {
        self.locked = true;
        Arc::clone(&self.config)
    }

    /// Lock the configuration, provided it is still the one `snapshot` was
    /// taken of, and return a shared handle to it.
    ///
    /// # Errors
    ///
    /// [`Error::ModifiedDuringComputation`] if it was modified since; it is
    /// then left as it was.
    pub fn lock_snapshot(&mut self, snapshot: &Arc<Config<K>>) -> Result<Arc<Config<K>>> {
        if !Arc::ptr_eq(&self.config, snapshot) {
            return Err(Error::ModifiedDuringComputation);
        }
        Ok(self.lock())
    }

    /// Whether [`lock`](Self::lock) has been called.
    pub fn is_locked(&self) -> bool {
        self.locked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pc(rows: &[&[i64]]) -> PointConfig {
        let mut c = PointConfig::new();
        for r in rows {
            c.push(r).unwrap();
        }
        c
    }

    #[test]
    fn homogenisation() {
        let p = pc(&[&[1, 2], &[3, 4]]);
        assert_eq!(p.homogenised(), &[1, 2, 1, 3, 4, 1]);
        assert_eq!(p.rows(), 3);
        assert_eq!(p.column(1), Some(&[3, 4, 1][..]));
        assert_eq!(p.coords(1), Some(&[3, 4][..]));
        assert_eq!(p.column(2), None);
        assert_eq!(p.to_row_major(), vec![1, 2, 3, 4]);

        let v = VectorConfig::from_row_major(&[1, 2, 3, 4], 2, 2).unwrap();
        assert_eq!(v.homogenised(), &[1, 2, 3, 4]);
        assert_eq!(v.rows(), 2);
        assert_eq!(v.kind(), ConfigKind::Vector);
    }

    #[test]
    fn dimension_is_set_by_the_first_addition() {
        let mut p = PointConfig::new();
        assert_eq!((p.dim(), p.len()), (0, 0));
        p.extend_from_array(&[2, 2], &[2]).unwrap();
        assert_eq!((p.dim(), p.len()), (2, 1));
        assert_eq!(
            p.push(&[1, 2, 3]),
            Err(Error::DimensionMismatch {
                expected: 2,
                found: 3
            })
        );
        // A failed addition changes nothing.
        assert_eq!(
            p.extend_from_array(&[1, 2, 3], &[1, 3]),
            Err(Error::DimensionMismatch {
                expected: 2,
                found: 3
            })
        );
        assert_eq!(p.len(), 1);
        assert!(p.extend_from_array(&[1], &[1, 1, 1]).is_err());
        assert!(PointConfig::from_array(&[1, 2], &[2]).is_err());
        assert!(PointConfig::from_array(&[1, 2, 3], &[2, 2]).is_err());
    }

    #[test]
    fn zero_dimensional_configurations() {
        let p = PointConfig::from_row_major(&[], 3, 0).unwrap();
        assert_eq!(p.homogenised(), &[1, 1, 1]);
        assert_eq!(p.rank(), 1);
        assert!(p.is_full_rank());
        let v = VectorConfig::from_row_major(&[], 3, 0).unwrap();
        assert_eq!(v.homogenised(), &[] as &[i64]);
        assert_eq!(v.rank(), 0);
        assert_eq!(v.require_full_rank(), Err(Error::EmptyConfiguration));
    }

    #[test]
    fn rank() {
        let square = pc(&[&[0, 0], &[1, 0], &[1, 1], &[0, 1]]);
        assert_eq!(square.rank(), 3);
        assert!(square.require_full_rank().is_ok());
        let line = pc(&[&[0, 0], &[1, 1], &[2, 2], &[3, 3]]);
        assert_eq!(line.rank(), 2);
        assert_eq!(
            line.require_full_rank(),
            Err(Error::NotFullRank { rank: 2, rows: 3 })
        );
        assert_eq!(pc(&[&[0, 0]]).rank(), 1);
        assert_eq!(PointConfig::new().rank(), 0);
        assert_eq!(
            PointConfig::new().require_full_rank(),
            Err(Error::EmptyConfiguration)
        );
        // Huge coordinates take the bigint path.
        let big = pc(&[&[i64::MAX, i64::MIN], &[i64::MIN, i64::MAX], &[1, 1]]);
        assert_eq!(big.rank(), 3);
    }

    #[test]
    fn chirotope_signs() {
        let tri = pc(&[&[0, 0], &[1, 0], &[0, 1]]);
        assert_eq!(
            tri.sign_of_simplex(&Simplex::from([0, 1, 2])),
            Ok(Sign::Positive)
        );
        assert_eq!(tri.sign_of_columns([1, 0, 2]), Ok(Sign::Negative));
        assert_eq!(
            tri.sign_of_columns([0, 1]),
            Err(Error::DimensionMismatch {
                expected: 3,
                found: 2
            })
        );
        assert!(tri.sign_of_columns([0, 1, 2, 0]).is_err());
        assert_eq!(
            tri.sign_of_columns([0, 1, 7]),
            Err(Error::IndexOutOfRange { index: 7, len: 3 })
        );
        let line = pc(&[&[0, 0], &[1, 1], &[2, 2]]);
        assert_eq!(line.sign_of_columns([0, 1, 2]), Ok(Sign::Zero));
    }

    #[test]
    fn repr_strings_match_0_3() {
        let p = pc(&[&[0, 0], &[1, 0], &[0, 1]]);
        assert_eq!(
            p.to_string(),
            "A 2-dimensional point configuration with 3 points"
        );
        let v = VectorConfig::from_row_major(&[1, 0, 0, 1], 2, 2).unwrap();
        assert_eq!(
            v.to_string(),
            "A 2-dimensional vector configuration with 2 vectors"
        );
        assert_eq!(
            PointConfig::new().to_string(),
            "A 0-dimensional point configuration with 0 points"
        );
    }

    #[test]
    fn locking() {
        let mut l = Lockable::new(PointConfig::new());
        l.modify().unwrap().push(&[1, 2]).unwrap();
        assert!(!l.is_locked());
        let frozen = l.lock();
        assert!(l.is_locked());
        assert_eq!(l.modify().err(), Some(Error::Locked));
        assert_eq!(frozen.len(), 1);
        assert!(Arc::ptr_eq(&frozen, &l.lock()));
        assert!(Arc::ptr_eq(&frozen, &l.lock_snapshot(&frozen).unwrap()));
    }

    #[test]
    fn snapshots() {
        let mut l = Lockable::new(PointConfig::new());
        l.modify().unwrap().push(&[1, 2]).unwrap();
        let snap = l.snapshot();
        // Modifying while a snapshot is alive copies; the snapshot is intact,
        // and locking against it fails without locking.
        l.modify().unwrap().push(&[3, 4]).unwrap();
        assert_eq!((snap.len(), l.get().len()), (1, 2));
        assert_eq!(
            l.lock_snapshot(&snap).err(),
            Some(Error::ModifiedDuringComputation)
        );
        assert!(!l.is_locked());
        let snap = l.snapshot();
        assert_eq!(l.lock_snapshot(&snap).unwrap().len(), 2);
        assert!(l.is_locked());
    }
}
