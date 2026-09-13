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
//! package is fixed from the start, but the methods are stubs that raise
//! `NotImplementedError` until their phase lands. `python/tests/conftest.py`
//! turns that exception into a skip, so the pytest suite lights up phase by
//! phase without any test edits.

use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::error::Error;

impl From<Error> for PyErr {
    fn from(err: Error) -> Self {
        let msg = err.to_string();
        match err {
            // Bad input from the caller.
            Error::DimensionMismatch { .. }
            | Error::NotFullRank { .. }
            | Error::EmptyConfiguration
            | Error::RankExceedsCount { .. }
            | Error::ParameterCountMismatch { .. }
            | Error::InvalidTriangulation(_) => PyValueError::new_err(msg),
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

/// A configuration of points in affine space.
#[pyclass(module = "triangulumancer")]
pub struct PointConfiguration {}

#[pymethods]
impl PointConfiguration {
    #[new]
    #[pyo3(signature = (matrix = None))]
    fn new(matrix: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let _ = matrix;
        todo("PointConfiguration", 1)
    }

    fn __repr__(&self) -> PyResult<String> {
        todo("PointConfiguration.__repr__", 1)
    }

    #[getter]
    fn dim(&self) -> PyResult<usize> {
        todo("PointConfiguration.dim", 1)
    }

    #[getter]
    fn n_points(&self) -> PyResult<usize> {
        todo("PointConfiguration.n_points", 1)
    }

    #[getter]
    fn points(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let _ = py;
        todo("PointConfiguration.points", 1)
    }

    fn add_point(&mut self, matrix: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = matrix;
        todo("PointConfiguration.add_point", 1)
    }

    fn add_points(&mut self, matrix: &Bound<'_, PyAny>) -> PyResult<()> {
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
pub struct VectorConfiguration {}

#[pymethods]
impl VectorConfiguration {
    #[new]
    #[pyo3(signature = (matrix = None))]
    fn new(matrix: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let _ = matrix;
        todo("VectorConfiguration", 1)
    }

    fn __repr__(&self) -> PyResult<String> {
        todo("VectorConfiguration.__repr__", 1)
    }

    #[getter]
    fn dim(&self) -> PyResult<usize> {
        todo("VectorConfiguration.dim", 1)
    }

    #[getter]
    fn n_vectors(&self) -> PyResult<usize> {
        todo("VectorConfiguration.n_vectors", 1)
    }

    #[getter]
    fn vectors(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let _ = py;
        todo("VectorConfiguration.vectors", 1)
    }

    fn add_vector(&mut self, matrix: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = matrix;
        todo("VectorConfiguration.add_vector", 1)
    }

    fn add_vectors(&mut self, matrix: &Bound<'_, PyAny>) -> PyResult<()> {
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
