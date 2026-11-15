//! Python bindings.
//!
//! This module is a translation layer only: it converts between Python objects
//! and the crate's own types and contains no algorithmic logic. Each Rust
//! module gets one `register_*` function, so adding bindings for a new feature
//! touches exactly one place. See `PLAN.md` section 6.4.
//!
//! # Migration status
//!
//! The classes below carry the full 0.3 API surface so that the shape of the
//! package is fixed from the start. The configuration classes are real as of
//! phase 1, the placing and fine triangulations and the `Triangulation`
//! class as of phase 2, flips and neighbours as of phase 3, the flip-graph
//! enumeration as of phase 4, the regular triangulations as of phase 6 and
//! the extension-graph enumeration as of phase 5, which completes the 0.3
//! surface. Methods for phases that have not landed raise
//! `NotImplementedError` through [`todo`]; `python/tests/conftest.py` turns
//! that exception into a skip. None is left.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use numpy::ndarray::Array2;
use numpy::{AllowTypeChange, IntoPyArray, PyArray2, PyArrayLikeDyn};
use pyo3::exceptions::{
    PyIndexError, PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::PyFloat;

use crate::config::{Config, Kind, Lockable, PointConfig, Points, Vectors};
use crate::error::{Error, Result};
use crate::om::Chirotope;
use crate::regular::{self, Scalar};
use crate::triang;

impl From<Error> for PyErr {
    fn from(err: Error) -> Self {
        let msg = err.to_string();
        match err {
            // Bad input from the caller.
            Error::DimensionMismatch { .. }
            | Error::InvalidShape { .. }
            | Error::NotFullRank { .. }
            | Error::EmptyConfiguration
            | Error::RankExceedsCount { .. }
            | Error::ParameterCountMismatch { .. }
            | Error::InvalidParameter { .. }
            | Error::InvalidTriangulation(_) => PyValueError::new_err(msg),
            Error::IndexOutOfRange { .. } => PyIndexError::new_err(msg),
            // Invalid use, or our own bug.
            Error::Locked | Error::Overflow(_) | Error::Internal(_) => PyRuntimeError::new_err(msg),
        }
    }
}

/// Raise `NotImplementedError` naming the phase that will implement `what`.
///
/// Unused since phase 5 completed the 0.3 surface; kept for bindings that
/// land ahead of their implementation.
#[allow(dead_code)]
fn todo<T>(what: &str, phase: u8) -> PyResult<T> {
    Err(PyNotImplementedError::new_err(format!(
        "{what} is not implemented yet; it lands in phase {phase} of the Rust \
         migration (see PLAN.md)"
    )))
}

// ---------------------------------------------------------------------------
// Configurations
// ---------------------------------------------------------------------------

/// An integer array of any shape, converted by numpy (`np.asarray(x, int64)`),
/// as 0.3's `pybind11::array_t<int64_t>` did.
type IntArray<'py> = PyArrayLikeDyn<'py, i64, AllowTypeChange>;

/// The data, in row-major order, and the shape of an integer array.
fn unpack(array: &IntArray<'_>) -> (Vec<i64>, Vec<usize>) {
    let view = array.as_array();
    (view.iter().copied().collect(), view.shape().to_vec())
}

/// A configuration from an optional `(n, dim)` array.
fn new_config<K: Kind>(matrix: Option<IntArray<'_>>) -> PyResult<Lockable<K>> {
    let config = match matrix {
        None => Config::new(),
        Some(m) => {
            let (data, shape) = unpack(&m);
            Config::from_array(&data, &shape)?
        }
    };
    Ok(Lockable::new(config))
}

/// Append a `(dim,)` or `(n, dim)` array.
fn extend_config<K: Kind>(config: &mut Lockable<K>, matrix: &IntArray<'_>) -> PyResult<()> {
    let (data, shape) = unpack(matrix);
    Ok(config.modify()?.extend_from_array(&data, &shape)?)
}

/// The coordinates as an `int64` array of shape `(n, dim)`.
fn coordinates<'py, K: Kind>(
    py: Python<'py>,
    config: &Config<K>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let array = Array2::from_shape_vec((config.len(), config.dim()), config.to_row_major())
        .map_err(|_| Error::Internal("coordinate array shape"))?;
    Ok(array.into_pyarray(py))
}

/// Run a chirotope-based triangulation algorithm on a configuration, then lock
/// the configuration and wrap the result. A failure leaves it unlocked, as in
/// 0.3, where the lock was taken by the `Triangulation` constructor.
fn triangulate<K: Kind>(
    py: Python<'_>,
    config: &mut Lockable<K>,
    algorithm: fn(&Chirotope) -> Result<triang::Triangulation>,
) -> PyResult<Triangulation>
where
    SharedConfig: From<Arc<Config<K>>>,
{
    let frozen = config.get();
    let inner = py.detach(|| frozen.chirotope().and_then(|c| algorithm(&c)))?;
    Ok(Triangulation {
        config: config.lock().into(),
        inner,
    })
}

/// `all_connected_triangulations`, with the GIL released while it runs; locks
/// the configuration on success.
fn connected<K: Kind>(
    py: Python<'_>,
    config: &mut Lockable<K>,
    only_fine: bool,
) -> PyResult<Vec<Triangulation>>
where
    SharedConfig: From<Arc<Config<K>>>,
{
    let frozen = config.get();
    let all = py.detach(|| {
        frozen
            .chirotope()
            .and_then(|c| triang::all_connected(&c, only_fine))
    })?;
    let shared: SharedConfig = config.lock().into();
    Ok(all
        .into_iter()
        .map(|inner| Triangulation {
            config: shared.clone(),
            inner,
        })
        .collect())
}

/// `iter_all_connected_triangulations`: the seed is computed here, so that a
/// rank-deficient configuration fails at the call rather than on the first
/// `next`, and the configuration is locked once it succeeds.
fn connected_iter<K: Kind>(
    py: Python<'_>,
    config: &mut Lockable<K>,
    only_fine: bool,
) -> PyResult<TriangulationIterator>
where
    SharedConfig: From<Arc<Config<K>>>,
{
    let frozen = config.get();
    let graph = py.detach(|| {
        frozen
            .chirotope()
            .and_then(|c| triang::FlipGraph::new(c, only_fine))
    })?;
    Ok(TriangulationIterator {
        config: config.lock().into(),
        search: Mutex::new(Box::new(graph)),
    })
}

/// `all_triangulations`, with the GIL released while it runs; locks the
/// configuration on success.
fn all<K: Kind>(
    py: Python<'_>,
    config: &mut Lockable<K>,
    only_fine: bool,
) -> PyResult<Vec<Triangulation>>
where
    SharedConfig: From<Arc<Config<K>>>,
{
    let frozen = config.get();
    let all = py.detach(|| {
        frozen
            .chirotope()
            .and_then(|c| triang::all_triangulations(&c, only_fine))
    })?;
    let shared: SharedConfig = config.lock().into();
    Ok(all
        .into_iter()
        .map(|inner| Triangulation {
            config: shared.clone(),
            inner,
        })
        .collect())
}

/// `iter_all_triangulations`: the tables are built here, so that a
/// rank-deficient configuration fails at the call, and the configuration is
/// locked once that succeeds.
fn all_iter<K: Kind>(
    py: Python<'_>,
    config: &mut Lockable<K>,
    only_fine: bool,
) -> PyResult<TriangulationIterator>
where
    SharedConfig: From<Arc<Config<K>>>,
{
    let frozen = config.get();
    let graph = py.detach(|| {
        frozen
            .chirotope()
            .and_then(|c| triang::ExtensionGraph::new(&c, only_fine))
    })?;
    Ok(TriangulationIterator {
        config: config.lock().into(),
        search: Mutex::new(Box::new(graph.map(Ok))),
    })
}

/// Run a regular-triangulation algorithm on a point configuration, then lock
/// it, as [`triangulate`] does for the chirotope-based ones.
fn triangulate_regular<F>(
    py: Python<'_>,
    config: &mut Lockable<Points>,
    algorithm: F,
) -> PyResult<Triangulation>
where
    F: FnOnce(&PointConfig) -> Result<triang::Triangulation> + Send,
{
    let frozen = config.get();
    let inner = py.detach(|| algorithm(frozen))?;
    Ok(Triangulation {
        config: config.lock().into(),
        inner,
    })
}

/// A Python integer of any size.
fn to_ibig(x: &Bound<'_, PyAny>) -> PyResult<dashu_int::IBig> {
    if let Ok(v) = x.extract::<i64>() {
        return Ok(v.into());
    }
    let text = x.str()?;
    text.to_str()?
        .parse()
        .map_err(|_| PyTypeError::new_err("expected an integer"))
}

/// A weight or height: a `float` (or numpy float) exactly as the double it is,
/// an `int` or `numbers.Rational` such as `Fraction` exactly, and anything
/// else through `float()`, as 0.3 did.
fn to_scalar(x: &Bound<'_, PyAny>) -> PyResult<Scalar> {
    if x.is_instance_of::<PyFloat>() {
        return Ok(Scalar::Float(x.extract()?));
    }
    if let Ok(v) = x.extract::<i64>() {
        return Ok(Scalar::from(v));
    }
    if let (Ok(num), Ok(den)) = (x.getattr("numerator"), x.getattr("denominator")) {
        return Ok(Scalar::Rational {
            numerator: to_ibig(&num)?,
            denominator: to_ibig(&den)?,
        });
    }
    let f = x
        .call_method0("__float__")
        .map_err(|_| PyTypeError::new_err("weights and heights must be real numbers"))?;
    Ok(Scalar::Float(f.extract()?))
}

/// A sequence of weights or heights.
fn to_scalars(values: &Bound<'_, PyAny>) -> PyResult<Vec<Scalar>> {
    values.try_iter()?.map(|x| to_scalar(&x?)).collect()
}

/// A configuration of points in affine space.
#[pyclass(module = "triangulumancer")]
pub struct PointConfiguration {
    inner: Lockable<Points>,
}

#[pymethods]
impl PointConfiguration {
    #[new]
    #[pyo3(signature = (matrix = None))]
    fn new(matrix: Option<IntArray<'_>>) -> PyResult<Self> {
        Ok(Self {
            inner: new_config(matrix)?,
        })
    }

    fn __repr__(&self) -> String {
        self.inner.get().to_string()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.get().dim()
    }

    #[getter]
    fn n_points(&self) -> usize {
        self.inner.get().len()
    }

    #[getter]
    fn points<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        coordinates(py, self.inner.get())
    }

    fn add_point(&mut self, matrix: IntArray<'_>) -> PyResult<()> {
        extend_config(&mut self.inner, &matrix)
    }

    fn add_points(&mut self, matrix: IntArray<'_>) -> PyResult<()> {
        self.add_point(matrix)
    }

    fn placing_triangulation(&mut self, py: Python<'_>) -> PyResult<Triangulation> {
        triangulate(py, &mut self.inner, triang::placing)
    }

    fn fine_triangulation(&mut self, py: Python<'_>) -> PyResult<Triangulation> {
        triangulate(py, &mut self.inner, triang::fine)
    }

    /// Every triangulation connected to a seed by bistellar flips, in
    /// canonical order.
    ///
    /// The seed is the placing triangulation, or with `only_fine` the fine
    /// one. With `only_fine`, flips that would remove a point are never
    /// taken, so the result is the connected component of the fine seed *in
    /// the flip graph restricted to fine triangulations*: a fine
    /// triangulation reachable only through non-fine ones is not found
    /// (`all_triangulations(only_fine=True)` finds every one). With repeated
    /// points or zero vectors no fine triangulation exists and the result is
    /// empty. 0.3 seeded `only_fine` with the placing triangulation, which
    /// returned non-fine triangulations or did not terminate when some point
    /// lies inside the hull of earlier ones.
    #[pyo3(signature = (only_fine = false))]
    fn all_connected_triangulations(
        &mut self,
        py: Python<'_>,
        only_fine: bool,
    ) -> PyResult<Vec<Triangulation>> {
        connected(py, &mut self.inner, only_fine)
    }

    /// As `all_connected_triangulations`, but yielding the triangulations one
    /// at a time, in breadth-first order from the seed, as they are found.
    ///
    /// Use it to count or filter an enumeration too large to hold. The search
    /// itself still remembers every triangulation it has seen.
    #[pyo3(signature = (only_fine = false))]
    fn iter_all_connected_triangulations(
        &mut self,
        py: Python<'_>,
        only_fine: bool,
    ) -> PyResult<TriangulationIterator> {
        connected_iter(py, &mut self.inner, only_fine)
    }

    /// Every triangulation, connected to the others by flips or not, in
    /// canonical order; with `only_fine`, every fine one.
    ///
    /// A backtracking search over partial triangulations (TOPCOM's extension
    /// graph). With `only_fine` it only ever uses simplices that contain no
    /// other point, so it finds exactly the fine triangulations; with
    /// repeated points or zero vectors there are none and the result is
    /// empty.
    #[pyo3(signature = (only_fine = false))]
    fn all_triangulations(
        &mut self,
        py: Python<'_>,
        only_fine: bool,
    ) -> PyResult<Vec<Triangulation>> {
        all(py, &mut self.inner, only_fine)
    }

    /// As `all_triangulations`, but yielding the triangulations one at a
    /// time, in the order the search finds them, without holding them.
    ///
    /// The search needs memory for its tables, quadratic in the number of
    /// possible simplices, but none for the triangulations: counting or
    /// filtering an enumeration of any size takes constant memory.
    #[pyo3(signature = (only_fine = false))]
    fn iter_all_triangulations(
        &mut self,
        py: Python<'_>,
        only_fine: bool,
    ) -> PyResult<TriangulationIterator> {
        all_iter(py, &mut self.inner, only_fine)
    }

    fn delaunay_triangulation(&mut self, py: Python<'_>) -> PyResult<Triangulation> {
        triangulate_regular(py, &mut self.inner, regular::delaunay)
    }

    /// The regular triangulation for weights `w`, i.e. heights `|p|^2 - w`.
    /// Floats are taken exactly as the doubles they are; ints and
    /// `fractions.Fraction` are exact.
    fn triangulate_with_weights(
        &mut self,
        py: Python<'_>,
        weights: &Bound<'_, PyAny>,
    ) -> PyResult<Triangulation> {
        let weights = to_scalars(weights)?;
        triangulate_regular(py, &mut self.inner, move |c| {
            regular::with_weights(c, &weights)
        })
    }

    /// The regular triangulation for the given heights. Floats are taken
    /// exactly as the doubles they are; ints and `fractions.Fraction` are
    /// exact.
    fn triangulate_with_heights(
        &mut self,
        py: Python<'_>,
        heights: &Bound<'_, PyAny>,
    ) -> PyResult<Triangulation> {
        let heights = to_scalars(heights)?;
        triangulate_regular(py, &mut self.inner, move |c| {
            regular::with_heights(c, &heights)
        })
    }
}

/// A configuration of vectors in linear space.
#[pyclass(module = "triangulumancer")]
pub struct VectorConfiguration {
    inner: Lockable<Vectors>,
}

#[pymethods]
impl VectorConfiguration {
    #[new]
    #[pyo3(signature = (matrix = None))]
    fn new(matrix: Option<IntArray<'_>>) -> PyResult<Self> {
        Ok(Self {
            inner: new_config(matrix)?,
        })
    }

    fn __repr__(&self) -> String {
        self.inner.get().to_string()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.get().dim()
    }

    #[getter]
    fn n_vectors(&self) -> usize {
        self.inner.get().len()
    }

    #[getter]
    fn vectors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        coordinates(py, self.inner.get())
    }

    fn add_vector(&mut self, matrix: IntArray<'_>) -> PyResult<()> {
        extend_config(&mut self.inner, &matrix)
    }

    fn add_vectors(&mut self, matrix: IntArray<'_>) -> PyResult<()> {
        self.add_vector(matrix)
    }

    fn placing_triangulation(&mut self, py: Python<'_>) -> PyResult<Triangulation> {
        triangulate(py, &mut self.inner, triang::placing)
    }

    fn fine_triangulation(&mut self, py: Python<'_>) -> PyResult<Triangulation> {
        triangulate(py, &mut self.inner, triang::fine)
    }

    /// Every triangulation connected to a seed by bistellar flips, in
    /// canonical order.
    ///
    /// The seed is the placing triangulation, or with `only_fine` the fine
    /// one. With `only_fine`, flips that would remove a point are never
    /// taken, so the result is the connected component of the fine seed *in
    /// the flip graph restricted to fine triangulations*: a fine
    /// triangulation reachable only through non-fine ones is not found
    /// (`all_triangulations(only_fine=True)` finds every one). With repeated
    /// points or zero vectors no fine triangulation exists and the result is
    /// empty. 0.3 seeded `only_fine` with the placing triangulation, which
    /// returned non-fine triangulations or did not terminate when some point
    /// lies inside the hull of earlier ones.
    #[pyo3(signature = (only_fine = false))]
    fn all_connected_triangulations(
        &mut self,
        py: Python<'_>,
        only_fine: bool,
    ) -> PyResult<Vec<Triangulation>> {
        connected(py, &mut self.inner, only_fine)
    }

    /// As `all_connected_triangulations`, but yielding the triangulations one
    /// at a time, in breadth-first order from the seed, as they are found.
    ///
    /// Use it to count or filter an enumeration too large to hold. The search
    /// itself still remembers every triangulation it has seen.
    #[pyo3(signature = (only_fine = false))]
    fn iter_all_connected_triangulations(
        &mut self,
        py: Python<'_>,
        only_fine: bool,
    ) -> PyResult<TriangulationIterator> {
        connected_iter(py, &mut self.inner, only_fine)
    }

    /// Every triangulation, connected to the others by flips or not, in
    /// canonical order; with `only_fine`, every fine one.
    ///
    /// A backtracking search over partial triangulations (TOPCOM's extension
    /// graph). With `only_fine` it only ever uses simplices that contain no
    /// other point, so it finds exactly the fine triangulations; with
    /// repeated points or zero vectors there are none and the result is
    /// empty.
    #[pyo3(signature = (only_fine = false))]
    fn all_triangulations(
        &mut self,
        py: Python<'_>,
        only_fine: bool,
    ) -> PyResult<Vec<Triangulation>> {
        all(py, &mut self.inner, only_fine)
    }

    /// As `all_triangulations`, but yielding the triangulations one at a
    /// time, in the order the search finds them, without holding them.
    ///
    /// The search needs memory for its tables, quadratic in the number of
    /// possible simplices, but none for the triangulations: counting or
    /// filtering an enumeration of any size takes constant memory.
    #[pyo3(signature = (only_fine = false))]
    fn iter_all_triangulations(
        &mut self,
        py: Python<'_>,
        only_fine: bool,
    ) -> PyResult<TriangulationIterator> {
        all_iter(py, &mut self.inner, only_fine)
    }
}

// ---------------------------------------------------------------------------
// Triangulations and flips
// ---------------------------------------------------------------------------

/// A locked configuration of either kind, shared by every triangulation of it.
///
/// Triangulations refer to their configuration by label, so they keep it alive
/// and frozen (`Lockable::lock`).
#[derive(Clone)]
enum SharedConfig {
    Points(Arc<Config<Points>>),
    Vectors(Arc<Config<Vectors>>),
}

impl From<Arc<Config<Points>>> for SharedConfig {
    fn from(c: Arc<Config<Points>>) -> Self {
        Self::Points(c)
    }
}

impl From<Arc<Config<Vectors>>> for SharedConfig {
    fn from(c: Arc<Config<Vectors>>) -> Self {
        Self::Vectors(c)
    }
}

impl SharedConfig {
    fn dim(&self) -> usize {
        match self {
            Self::Points(c) => c.dim(),
            Self::Vectors(c) => c.dim(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Points(c) => c.len(),
            Self::Vectors(c) => c.len(),
        }
    }

    fn noun(&self) -> &'static str {
        match self {
            Self::Points(c) => c.kind().noun(),
            Self::Vectors(c) => c.kind().noun(),
        }
    }

    fn chirotope(&self) -> Result<Chirotope> {
        match self {
            Self::Points(c) => c.chirotope(),
            Self::Vectors(c) => c.chirotope(),
        }
    }

    /// Whether `self` and `other` are the same configuration object.
    fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Points(a), Self::Points(b)) => Arc::ptr_eq(a, b),
            (Self::Vectors(a), Self::Vectors(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// An unlocked configuration of either kind, borrowed from its Python object.
enum ConfigArg<'py> {
    Points(PyRefMut<'py, PointConfiguration>),
    Vectors(PyRefMut<'py, VectorConfiguration>),
}

impl<'py> ConfigArg<'py> {
    fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if obj.is_instance_of::<PointConfiguration>() {
            Ok(Self::Points(obj.extract()?))
        } else if obj.is_instance_of::<VectorConfiguration>() {
            Ok(Self::Vectors(obj.extract()?))
        } else {
            Err(PyTypeError::new_err(
                "expected a PointConfiguration or a VectorConfiguration",
            ))
        }
    }
}

/// `Triangulation(config, simplices)` for one kind of configuration: shape
/// checks always, the full triangulation check when `validate`, and the lock
/// last, so that a rejected triangulation leaves the configuration unlocked.
fn construct<K: Kind>(
    config: &mut Lockable<K>,
    labels: &[i64],
    shape: &[usize],
    validate: bool,
) -> PyResult<Triangulation>
where
    SharedConfig: From<Arc<Config<K>>>,
{
    let frozen = config.get();
    let rank = frozen.rank();
    let width = match *shape {
        [_, width] => width,
        // An empty list carries no width; take it to have the right one.
        [0] => rank,
        _ => {
            return Err(Error::InvalidShape {
                expected: "a two-dimensional array of simplices",
                found: shape.len(),
            }
            .into());
        }
    };
    let inner = triang::Triangulation::from_labels(frozen.len(), rank, labels, width)?;
    if validate {
        inner.check(&frozen.chirotope()?)?;
    }
    Ok(Triangulation {
        config: config.lock().into(),
        inner,
    })
}

/// A triangulation of a point or vector configuration.
///
/// Holds its configuration, locked, and the simplices in canonical order:
/// sorted within each simplex, then lexicographically.
#[pyclass(module = "triangulumancer", frozen)]
pub struct Triangulation {
    config: SharedConfig,
    inner: triang::Triangulation,
}

#[pymethods]
impl Triangulation {
    /// A triangulation of `config` with the given simplices, one row of labels
    /// each. It is checked to be a triangulation unless `validate` is false;
    /// either way the simplices must have the right size and labels in range.
    #[new]
    #[pyo3(signature = (config, simplices, validate = true))]
    fn new(config: &Bound<'_, PyAny>, simplices: IntArray<'_>, validate: bool) -> PyResult<Self> {
        let (labels, shape) = unpack(&simplices);
        match ConfigArg::extract(config)? {
            ConfigArg::Points(mut c) => construct(&mut c.inner, &labels, &shape, validate),
            ConfigArg::Vectors(mut c) => construct(&mut c.inner, &labels, &shape, validate),
        }
    }

    fn __repr__(&self) -> String {
        let noun = self.config.noun();
        format!(
            "A triangulation with {} simplices of a {noun} configuration with {} {noun}s",
            self.inner.len(),
            self.config.len()
        )
    }

    #[getter]
    fn dim(&self) -> usize {
        self.config.dim()
    }

    #[getter]
    fn n_simplices(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn simplices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let flat: Vec<i64> = self
            .inner
            .iter()
            .flat_map(|s| s.iter().map(|v| v as i64))
            .collect();
        let array = Array2::from_shape_vec((self.inner.len(), self.inner.rank()), flat)
            .map_err(|_| Error::Internal("simplices array shape"))?;
        Ok(array.into_pyarray(py))
    }

    #[getter]
    fn is_fine(&self) -> bool {
        self.inner.is_fine()
    }

    /// The labels no simplex uses, in increasing order. For a regular
    /// triangulation these are the hidden points: those lifted above the
    /// lower hull, and duplicates that lost the tie.
    #[getter]
    fn hidden_points(&self) -> Vec<usize> {
        self.inner.unused().to_vec()
    }

    /// Triangulations are equal when they have the same simplices and belong
    /// to the same configuration object.
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<Triangulation>()
            .is_ok_and(|o| self.config.same(&o.get().config) && self.inner == o.get().inner)
    }

    fn __hash__(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.inner.hash(&mut h);
        h.finish()
    }

    /// Every bistellar flip of this triangulation, ordered by the simplices
    /// each removes and then those it adds.
    fn flips(&self, py: Python<'_>) -> PyResult<Vec<Flip>> {
        let flips = py.detach(|| {
            let chi = self.config.chirotope()?;
            triang::flips(&chi, &self.inner)
        })?;
        flips
            .into_iter()
            .map(|f| {
                let (n, rank) = (self.inner.n(), self.inner.rank());
                Ok(Flip {
                    config: self.config.clone(),
                    pre: triang::Triangulation::new(n, rank, f.removed().clone())?,
                    post: triang::Triangulation::new(n, rank, f.added().clone())?,
                })
            })
            .collect()
    }

    /// The same as `flips`.
    fn bistellar_flips(&self, py: Python<'_>) -> PyResult<Vec<Flip>> {
        self.flips(py)
    }

    /// Every triangulation one bistellar flip away from this one, in
    /// canonical order.
    fn neighbors(&self, py: Python<'_>) -> PyResult<Vec<Triangulation>> {
        let neighbors = py.detach(|| {
            let chi = self.config.chirotope()?;
            triang::neighbors(&chi, &self.inner)
        })?;
        Ok(neighbors
            .into_iter()
            .map(|inner| Triangulation {
                config: self.config.clone(),
                inner,
            })
            .collect())
    }
}

/// A bistellar flip of a triangulation: `pre` is the set of simplices it
/// removes and `post` the set it adds. Neither is a triangulation of the whole
/// configuration; they are the two triangulations of the region the flip
/// changes.
#[pyclass(module = "triangulumancer", frozen)]
pub struct Flip {
    config: SharedConfig,
    pre: triang::Triangulation,
    post: triang::Triangulation,
}

#[pymethods]
impl Flip {
    fn __repr__(&self) -> String {
        let noun = self.config.noun();
        format!(
            "A bistellar flip of a triangulation of a {}-dimensional {noun} configuration \
             with {} {noun}s",
            self.config.dim(),
            self.config.len()
        )
    }

    /// The simplices the flip removes.
    #[getter]
    fn pre(&self) -> Triangulation {
        Triangulation {
            config: self.config.clone(),
            inner: self.pre.clone(),
        }
    }

    /// The simplices the flip adds.
    #[getter]
    fn post(&self) -> Triangulation {
        Triangulation {
            config: self.config.clone(),
            inner: self.post.clone(),
        }
    }
}

/// An iterator over the triangulations of an enumeration, returned by
/// `iter_all_connected_triangulations` (a flip-graph search) and
/// `iter_all_triangulations` (an extension-graph search).
///
/// Each `next` runs the search with the GIL released until it finds the next
/// triangulation. The search state is behind a mutex, so the iterator can be
/// shared between threads; they take turns.
#[pyclass(module = "triangulumancer", frozen)]
pub struct TriangulationIterator {
    config: SharedConfig,
    search: Mutex<Search>,
}

/// The search behind a [`TriangulationIterator`].
type Search = Box<dyn Iterator<Item = Result<triang::Triangulation>> + Send>;

#[pymethods]
impl TriangulationIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Triangulation>> {
        let next = py.detach(|| {
            let mut search = self
                .search
                .lock()
                .map_err(|_| Error::Internal("triangulation iterator lock poisoned"))?;
            search.next().transpose()
        })?;
        Ok(next.map(|inner| Triangulation {
            config: self.config.clone(),
            inner,
        }))
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register the configuration classes.
fn register_config(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PointConfiguration>()?;
    m.add_class::<VectorConfiguration>()?;
    Ok(())
}

/// Register the triangulation and flip classes.
fn register_triang(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Triangulation>()?;
    m.add_class::<Flip>()?;
    m.add_class::<TriangulationIterator>()?;
    Ok(())
}

/// The `triangulumancer._core` extension module.
#[pymodule(gil_used = false)]
pub fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    register_config(m)?;
    register_triang(m)?;
    Ok(())
}
