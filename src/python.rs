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
//! phase 1; the triangulation methods are stubs that raise
//! `NotImplementedError` until their phase lands. `python/tests/conftest.py`
//! turns that exception into a skip, so the pytest suite lights up phase by
//! phase without any test edits.

use numpy::ndarray::Array2;
use numpy::{AllowTypeChange, IntoPyArray, PyArray2, PyArrayLikeDyn};
use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::config::{Config, Kind, Lockable, Points, Vectors};
use crate::error::Error;

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

    fn placing_triangulation(&self) -> PyResult<Triangulation> {
        todo("PointConfiguration.placing_triangulation", 2)
    }

    fn fine_triangulation(&self) -> PyResult<Triangulation> {
        todo("PointConfiguration.fine_triangulation", 2)
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

    fn delaunay_triangulation(&self) -> PyResult<Triangulation> {
        todo("PointConfiguration.delaunay_triangulation", 6)
    }

    fn triangulate_with_weights(&self, weights: Vec<f64>) -> PyResult<Triangulation> {
        let _ = weights;
        todo("PointConfiguration.triangulate_with_weights", 6)
    }

    fn triangulate_with_heights(&self, heights: Vec<f64>) -> PyResult<Triangulation> {
        let _ = heights;
        todo("PointConfiguration.triangulate_with_heights", 6)
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

    fn placing_triangulation(&self) -> PyResult<Triangulation> {
        todo("VectorConfiguration.placing_triangulation", 2)
    }

    fn fine_triangulation(&self) -> PyResult<Triangulation> {
        todo("VectorConfiguration.fine_triangulation", 2)
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

/// A triangulation of a point or vector configuration.
#[pyclass(module = "triangulumancer")]
pub struct Triangulation {}

#[pymethods]
impl Triangulation {
    #[new]
    fn new(config: &Bound<'_, PyAny>, simplices: &Bound<'_, PyAny>) -> PyResult<Self> {
        let _ = (config, simplices);
        todo("Triangulation", 2)
    }

    fn __repr__(&self) -> PyResult<String> {
        todo("Triangulation.__repr__", 2)
    }

    #[getter]
    fn dim(&self) -> PyResult<usize> {
        todo("Triangulation.dim", 2)
    }

    #[getter]
    fn n_simplices(&self) -> PyResult<usize> {
        todo("Triangulation.n_simplices", 2)
    }

    #[getter]
    fn simplices(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let _ = py;
        todo("Triangulation.simplices", 2)
    }

    #[getter]
    fn is_fine(&self) -> PyResult<bool> {
        todo("Triangulation.is_fine", 4)
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
