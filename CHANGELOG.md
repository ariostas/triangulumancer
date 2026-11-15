# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: a minor version may break).

## [0.4.0] — unreleased

`triangulumancer` is now written in Rust. Versions up to 0.3.0 wrapped
[CGAL](https://www.cgal.org/) (regular and Delaunay triangulations) and
[TOPCOM](https://www.wm.uni-bayreuth.de/de/team/rambau_joerg/TOPCOM/) (everything
else) through pybind11; 0.4.0 reimplements both in pure Rust, exposed to Python
through PyO3. The Python API is the 0.3 API, plus a few additions, and the same
inputs give the same triangulations except where 0.3 was wrong. The sections
below list every difference a 0.3 user can observe.

The design, and the evidence for each claim below, is in
[`PLAN.md`](PLAN.md); the bugs are catalogued there as B1–B5 (section 3) and the
expected corrected values in
[`tests/fixtures/KNOWN_DIVERGENT.md`](tests/fixtures/KNOWN_DIVERGENT.md).

### Packaging

- No C or C++ dependency of any kind: no CMake, CGAL, Boost, Eigen, GMP, MPFR or
  cddlib. Building from source needs only a Rust toolchain (1.85 or newer); the
  build backend is maturin 1.12 or newer.
- **Windows is supported**, for the first time. Wheels are built for Linux
  (x86_64 and aarch64), macOS and Windows.
- One `abi3` wheel per platform covers CPython 3.10 and later, plus a separate
  wheel for free-threaded CPython 3.14 (`cp314t`), which has no stable ABI yet.
  The module does not need the GIL, as in 0.3.
- The project is also published as a Rust crate,
  [`triangulumancer`](https://crates.io/crates/triangulumancer), with the same
  functionality and no Python dependency (the bindings are behind the
  off-by-default `python` feature).
- The distributed packages no longer bundle CGAL or TOPCOM, so their licences no
  longer apply to them. `triangulumancer` itself remains GPL-3.0-or-later.

### Fixed

- **B1** — `all_connected_triangulations(only_fine=True)` started its search
  from the placing triangulation, which need not be fine, so it could return
  non-fine triangulations, miss fine ones, or (on the 3×3 grid with the origin
  listed fifth) never terminate. It now starts from the fine triangulation and
  returns the flip-connected component of fine triangulations; when no fine
  triangulation exists (repeated points, zero vectors) it returns an empty list.
- **B2** — `all_triangulations()` cleared a process-wide table that every
  existing triangulation depended on, so calling it while another
  `Triangulation` was alive was memory-unsafe. There is no global state now.
- **B3** — `triangulate_with_heights` overflowed for coordinates of absolute
  value 46 341 or more (the height-to-weight conversion squared in 32-bit
  integers). The lift is now computed exactly for any `int64` coordinates.
- **B4** — regular and Delaunay triangulations of a configuration that does not
  span its ambient space returned a malformed `simplices` array, silently padded
  with index `0`. Such configurations are now triangulated within their affine
  hull: the simplices have `dim(aff) + 1` vertices each, e.g. a set of collinear
  points in the plane gives segments.
- **B5** — vertices of regular and Delaunay triangulations were matched back to
  input points through rounded `float` coordinates. With repeated points an
  arbitrary copy was used, and coordinates above 2⁵³ could be matched to the
  wrong point. Now, of several copies of a point with equal weight the one with
  the lowest index is used, and the others are reported by `hidden_points`.
- Errors that TOPCOM handled by calling `exit(1)` — killing the Python
  interpreter without a traceback — are now exceptions.

### Changed

- **Exception types.** Bad input raises `ValueError` where 0.3 raised
  `RuntimeError`: wrong shapes and dimension mismatches when building or
  extending a configuration, configurations that are not full rank (for the
  placing, fine, flip and enumeration methods), wrong numbers of weights or
  heights, and invalid weights or heights. An out-of-range label in
  `Triangulation(config, simplices)` raises `IndexError`, and a value that is not
  a number raises `TypeError`. Adding points to a configuration that has been
  triangulated is still a `RuntimeError`, and so are internal errors (which
  would be bugs; please report them).
- **Empty configurations** raise `ValueError` from every triangulation method.
  0.3 returned an empty triangulation from `delaunay_triangulation` and the
  regular methods.
- **Deterministic output order.** Every simplex lists its vertices in increasing
  order, and the simplices of a triangulation are sorted lexicographically.
  `all_triangulations` and `all_connected_triangulations` return their
  triangulations sorted (by their sorted simplex lists), `neighbors()` likewise,
  and `flips()` is sorted by `(pre, post)`. In 0.3 these orders came from hash
  tables and could differ between runs and platforms. The sets are the same.
- **Ties in regular and Delaunay triangulations** (cospherical points, or lifted
  points on a common hyperplane) are broken by the same symbolic perturbation as
  CGAL's, so the result matches 0.3 wherever 0.3 was correct. It no longer
  depends on the insertion order.
- **Weights and heights are exact.** A `float` is taken as the exact value of the
  double, as before. An `int` of any size, or a `fractions.Fraction` (any
  `numbers.Rational`), is now taken exactly instead of being rounded to a
  double, so integer heights above 2⁵³ behave correctly. Other objects are
  converted with `float()`. Non-finite and subnormal floats, zero denominators,
  and inputs whose magnitudes span more than about a thousand binary orders of
  magnitude (after removing a common shift and scale) raise `ValueError`.
- **`Triangulation(config, simplices)` validates its input** by default: the
  simplices must form a triangulation of the configuration, or `ValueError` is
  raised. Pass `validate=False` to skip that check, as for the `pre` and `post`
  of a `Flip`, which are triangulations of part of the configuration only. The
  shape, the labels and repeated simplices are checked either way. The rows may
  be given in any order. It also accepts a `VectorConfiguration` now; 0.3
  accepted only a `PointConfiguration`.
- **Locking.** A configuration still becomes read-only once a triangulation of
  it has been computed, but only when the computation succeeds: a configuration
  whose triangulation failed (for example because it is not full rank) can still
  be extended. Computations release the GIL and work on a snapshot, so other
  threads can read, triangulate or modify the configuration meanwhile. If points
  are added while a computation is running, it raises `RuntimeError`
  ("modified during computation") instead of returning a triangulation of a
  configuration that has moved on, and the configuration is left unlocked.
- **Equality and hashing.** Two `Triangulation`s are equal (`==`) when they have
  the same simplices and belong to the same configuration *object*; equal
  coordinates in a different `PointConfiguration` are not enough. The hash is
  consistent with that, so triangulations can be put in sets and used as
  dictionary keys. In 0.3 equality was object identity.
- `all_connected_triangulations(only_fine=True)` is documented as what it
  computes: the fine triangulations reachable from the fine triangulation by
  flips *between fine triangulations*. A fine triangulation that is reachable
  only through non-fine ones is not included; `all_triangulations(only_fine=True)`
  finds every fine triangulation.

### Added

- `iter_all_triangulations(only_fine=False)` and
  `iter_all_connected_triangulations(only_fine=False)` on both configuration
  classes return iterators (`triangulumancer.TriangulationIterator`) that
  produce triangulations one at a time, in the search's own (deterministic,
  unsorted) order. The first works in constant
  memory however many triangulations there are; the second keeps the set already
  found. Each step runs with the GIL released.
- `Triangulation.is_fine`: whether every point or vector is a vertex.
- `Triangulation.hidden_points`: the sorted indices of the points no simplex
  uses. For a regular triangulation these are the points lifted above the lower
  hull, and duplicates; 0.3 dropped them silently.
- `Triangulation(..., validate=True)`, see above.
- `triangulumancer.__version__`.
- A Rust API: see the [crate documentation](https://docs.rs/triangulumancer).

### Unchanged

- Every 0.3 class, method and property, with its name, signature and `__repr__`
  string. `add_point`/`add_points` and `add_vector`/`add_vectors` remain aliases,
  as do `flips`/`bistellar_flips`.
- `points`, `vectors` and `simplices` are `int64` NumPy arrays of the same shapes.
- The triangulations themselves, wherever 0.3 was correct: the test corpus of
  0.3 outputs is reproduced exactly, as sets and, after sorting, in order.

## [0.3.0] and earlier

The C++ implementation. Its last commit is tagged `legacy-cpp`; see the
[GitHub releases](https://github.com/ariostas/triangulumancer/releases).

[0.4.0]: https://github.com/ariostas/triangulumancer/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ariostas/triangulumancer/releases/tag/v0.3.0
