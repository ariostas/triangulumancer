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
//! phase 1, and the placing and fine triangulations and the `Triangulation`
//! class as of phase 2; the remaining methods are stubs that raise
//! `NotImplementedError` until their phase lands. `python/tests/conftest.py`
//! turns that exception into a skip, so the pytest suite lights up phase by
//! phase without any test edits.

use std::sync::Arc;

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

    #[pyo3(signature = (only_fine = false))]
    fn all_connected_triangulations(&self, only_fine: bool) -> PyResult<Vec<Triangulation>> {
        let _ = only_fine;
        todo("PointConfiguration.all_connected_triangulations", 4)
    }

    #[pyo3(signature = (only_fine = false))]
    fn all_triangulations(&self, only_fine: bool) -> PyResult<Vec<Triangulation>> {
        let _ = only_fine;
        todo("PointConfiguration.all_triangulations", 5)
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

    #[pyo3(signature = (only_fine = false))]
    fn all_connected_triangulations(&self, only_fine: bool) -> PyResult<Vec<Triangulation>> {
        let _ = only_fine;
        todo("VectorConfiguration.all_connected_triangulations", 4)
    }

    #[pyo3(signature = (only_fine = false))]
    fn all_triangulations(&self, only_fine: bool) -> PyResult<Vec<Triangulation>> {
        let _ = only_fine;
        todo("VectorConfiguration.all_triangulations", 5)
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

    fn flips(&self) -> PyResult<Vec<Flip>> {
        todo("Triangulation.flips", 3)
    }

    fn bistellar_flips(&self) -> PyResult<Vec<Flip>> {
        self.flips()
    }

    fn neighbors(&self) -> PyResult<Vec<Triangulation>> {
        todo("Triangulation.neighbors", 3)
    }
}

/// A bistellar flip between two triangulations.
#[pyclass(module = "triangulumancer")]
pub struct Flip {}

#[pymethods]
impl Flip {
    fn __repr__(&self) -> PyResult<String> {
        todo("Flip.__repr__", 3)
    }

    #[getter]
    fn pre(&self) -> PyResult<Triangulation> {
        todo("Flip.pre", 3)
    }

    #[getter]
    fn post(&self) -> PyResult<Triangulation> {
        todo("Flip.post", 3)
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
