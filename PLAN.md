# Migration plan: C++ (CGAL + TOPCOM) → Rust

This document is the overall plan for reimplementing `triangulumancer` in Rust and
retiring its two C++ dependencies. It is the top-level companion to two detailed
design-input documents produced by reading the upstream sources:

* [`CGAL.md`](CGAL.md) — the regular/Delaunay triangulation slice.
* [`TOPCOM.md`](TOPCOM.md) — the placing/fine triangulation, flip and enumeration slices.

Read those for the algorithms. This document covers *why*, *what order*, *what the
result looks like*, and *how we know it is correct*.

---

## 1. Motivation

The current package is ~1 200 lines of C++ that exist almost entirely to marshal data
between numpy, CGAL and TOPCOM. The scaffolding costs more than the code it wraps:

| Problem | Detail |
|---|---|
| **Heavy build** | CMake + scikit-build-core + pybind11 + FetchContent(TOPCOM) + system CGAL, Boost, Eigen, GMP, MPFR, cddlib. CI installs these through `yum`/`brew`/`micromamba`, three different ways. No Windows support at all. |
| **A fork we maintain** | TOPCOM is pulled from `ariostas/TOPCOM` at a pinned SHA, already carrying local patches for bugs upstream has not fixed (see [`TOPCOM.md` §12](TOPCOM.md)). |
| **`exit(1)` inside the interpreter** | TOPCOM hard-exits on many internal error paths ([`TOPCOM.md` §10.5](TOPCOM.md)). From inside a Python extension that kills the user's process with no traceback. |
| **Global mutable state** | `SymmetricExtensionGraphMaster::_init()` resets a *process-global* simplex intern table, invalidating every live `SimplicialComplex`. Calling `all_triangulations()` while another `Triangulation` is alive is unsound today ([`TOPCOM.md` §7.2](TOPCOM.md)). |
| **Live correctness bugs** | Five, listed in §3 below. Three are in our own 139-line `CGAL.cpp`. |
| **We use ~5% of each library** | Both `CGAL.md` §1 and `TOPCOM.md` §1 enumerate what is actually reachable. The estimate is ~16 600 lines of TOPCOM and a comparable slice of CGAL, replaced by ~7 000 lines of Rust. |
| **Nothing is reusable** | The algorithms are locked behind `pybind11::array_t` in the function signatures. There is no way to use them from Rust, and no crates.io presence. |

Two further facts make the rewrite unusually attractive here:

1. **Both libraries reduce to one shared primitive.** TOPCOM's chirotope is the sign of an
   `r × r` determinant of configuration columns ([`TOPCOM.md` §8.1](TOPCOM.md)). CGAL's
   `orientation` and `power_side_of_power_sphere` are *also* determinant signs — and the
   latter is exactly the former one dimension up, applied to lifted points
   ([`CGAL.md` §4.3](CGAL.md)). One well-tested `sign_of_det` function is the foundation of
   the entire package. That is not an accident we could exploit while the two libraries were
   separate black boxes.
2. **We can drop GMP entirely.** TOPCOM uses `mpq_class` with *fraction-producing* Gaussian
   elimination, which inflates intermediates far beyond the determinant's own size. Exact
   arithmetic is needed only for a determinant *sign*, and fraction-free Bareiss over `i128`
   with a bigint fallback is both correct and faster ([`TOPCOM.md` §8.3](TOPCOM.md)). With
   `CGAL_DISABLE_GMP ON` the current CGAL path is even worse — it runs `Eigen::PartialPivLU`
   over `cpp_rational` for `d ≥ 7` ([`CGAL.md` §5.1](CGAL.md)). A pure-Rust bigint on the
   rare fallback path beats both, and removes the GMP build-caching machinery that dominates
   CI time in comparable projects.

### Non-goals

* Porting anything from CGAL or TOPCOM we do not currently expose. Explicitly out:
  symmetry groups, GKZ/secondary polytopes, regularity checking (cddlib), volume tables,
  `remove()` from the triangulation, TOPCOM's CLI and file formats, persistence,
  and CGAL's kernel-traits metaprogramming layer. **Not** out of scope: making each of
  these cheap to add later — that is what §6 is for, and it is a design constraint on
  every phase.
* Changing the mathematics. Outputs must match the current package (except where the
  current package is wrong — §3).
* A general-purpose computational-geometry crate. The public Rust API is scoped to what
  this package needs, even though it will be published.

---

## 2. What must be preserved

The full public surface, derived from `tests/` and `src/triangulumancer/triangulumancer.cpp`.
The user has confirmed **all of it** is in scope — including flips and the flip graph, which
are the heaviest parts of the port.

| Class | Member | Backend today | Notes |
|---|---|---|---|
| `PointConfiguration` | `__init__()`, `__init__(array)` | — | Points stored with a homogenising `1` appended |
| | `dim`, `n_points`, `points` | — | `points` returns `int64` numpy, shape `(n, d)` |
| | `add_point`, `add_points` | — | Both names bound to the same method; accept 1-D or 2-D |
| | `placing_triangulation()` | TOPCOM | "Pushing" triangulation |
| | `fine_triangulation()` | TOPCOM | |
| | `all_connected_triangulations(only_fine=False)` | TOPCOM | Flip-graph BFS |
| | `all_triangulations(only_fine=False)` | TOPCOM | Extension-graph DFS |
| | `delaunay_triangulation()` | CGAL | |
| | `triangulate_with_weights(list[float])` | CGAL | |
| | `triangulate_with_heights(list[float])` | CGAL | `weight = −height + \|p\|²` |
| `VectorConfiguration` | mirror of the above, minus the three CGAL methods | TOPCOM | No homogenising `1`; simplices have `d` vertices, not `d+1` |
| `Triangulation` | `__init__(config, simplices)` | — | Direct construction; currently unvalidated |
| | `n_simplices`, `dim`, `simplices` | — | `simplices` is `int64` numpy, shape `(m, d+1)` or `(m, d)` |
| | `flips()`, `bistellar_flips()` | TOPCOM | Aliases |
| | `neighbors()` | TOPCOM | |
| `Flip` | `pre`, `post`, `__repr__` | TOPCOM | |

Plus two behaviours that are not methods but are tested:

* **Configuration locking.** Constructing any triangulation sets `is_locked`; subsequent
  `add_point`/`add_vector` raises. Triangulations share the configuration data via a
  `shared_ptr`, so a `Triangulation` keeps its configuration alive and frozen.
* **`__repr__` strings** on all four classes.

The regression corpus is the current `tests/` directory, whose concrete expected values
(387 and 64 triangulations of the 3×3 grid, 131 and 1 for the octagonal vector
configuration, the 46-simplex degenerate case, etc.) are the acceptance criteria.

---

## 3. Bugs found during research

These are real defects in the shipped package, found while reading the code. They shape the
plan: none are fixed in the C++, all are fixed by construction in Rust, and B1 changes how
the Phase 0 fixture corpus is interpreted.

| # | Where | Severity | Description |
|---|---|---|---|
| B1 | `TOPCOM.cpp:160` | **Wrong results** | `all_connected_triangulations(only_fine=True)` seeds with `PlacingTriang`, but `only_fine` is not an output filter — it is `forbid_vertex_removal`, which deletes flip-graph edges and *requires a fine seed*. TOPCOM's own driver uses `FineTriang` in that branch. Wrong whenever the configuration has points interior to the hull of earlier points. See [`TOPCOM.md` §6.3](TOPCOM.md). |
| B2 | `SymmetricExtensionGraphMaster.cc:267,283` | **Memory unsafety** | `_init()` unconditionally calls `SimplicialComplex::reset_index_table()`, clearing a process-global intern table and silently invalidating every previously created `SimplicialComplex`. Calling `all_triangulations()` while another `Triangulation` is alive is unsound. See [`TOPCOM.md` §7.2](TOPCOM.md). |
| B3 | `CGAL.cpp:45-46` | **Wrong results** | The height→weight conversion truncates `int64_t` to `int` and squares in `int` arithmetic. Overflows for `\|coord\| ≥ 46341`. See [`CGAL.md` §2.4](CGAL.md). |
| B4 | `CGAL.cpp:87-98` | **Silent corruption** | For rank-deficient input, null vertex handles hit `std::map::operator[]`, so the simplices array is silently padded with index `0` and has the wrong shape. See [`CGAL.md` §7.4](CGAL.md). |
| B5 | `CGAL.cpp:59-84` | **Wrong results, slow** | Vertices are matched back to input indices by an `O(n·V·d)` scan over `std::round(to_double(coord))`. Picks an arbitrary index for duplicate points and breaks above `2^53`. See [`CGAL.md` §7.6](CGAL.md). |

Also noted, not a bug but worth stating: `CGAL_assertion(t.is_valid())` compiles away under
`NDEBUG`, so released wheels perform no validity checking whatsoever
([`CGAL.md` §7.10](CGAL.md)).

**Decision: none of these are fixed in the C++.** B1 and B3 are one-line fixes and were
originally planned as a `0.3.1`, but the effort goes into the Rust port instead; B2, B4 and
B5 are structural and only the rewrite fixes them. All five are fixed by construction in the
Rust implementation — B1 in Phase 4, B3 and B5 in Phase 6, B4 in Phase 1 (the affine-hull
reduction), B2 in Phase 5 (no global interner, §5.3).

The consequence for Phase 0 is that the fixture oracle is `0.3.0` as shipped, so inputs
affected by B1 — `all_connected_triangulations(only_fine=True)` on configurations with
interior points — are recorded as **known-divergent**: captured for reference, and asserted
to *differ* from the Rust implementation rather than to match. Everything else is a
match-exactly fixture.

---

## 4. Target architecture

Mirrors [`ariostas/cygv`](https://github.com/ariostas/cygv): one crate at the repository
root, pure-Rust algorithms, Python bindings behind an off-by-default cargo feature, built by
maturin, published to both crates.io and PyPI.

```
triangulumancer/
├── Cargo.toml                  # crate-type = ["lib", "cdylib"]
├── Cargo.lock                  # committed; CI builds --locked
├── pyproject.toml              # build-backend = "maturin"
├── noxfile.py
├── AGENTS.md  CONTRIBUTING.md  CODE_OF_CONDUCT.md  README.md  LICENSE
├── docs-header.html            # KaTeX for rustdoc; there is a lot of maths here
├── src/
│   ├── lib.rs                  # re-exports; #[cfg(feature = "python")] pub mod python;
│   ├── arith.rs                # THE shared foundation: exact determinant sign
│   │   ├── bareiss.rs          #   fraction-free elimination, i128 and bigint
│   │   └── filter.rs           #   f64 error-bounded filter tier
│   ├── config.rs               # Config: column-major integer matrix, rank, point vs vector
│   │   └── hull.rs             #   exact HNF affine-hull reduction
│   ├── simplex.rs              # Simplex bitset, SimplexSet, link/join/support/boundary
│   ├── om/                     # ORIENTED MATROID LAYER — the reusable middle (§6)
│   │   ├── mod.rs              #   pub trait OrientedMatroid { fn sign(&self, basis) -> Sign }
│   │   ├── chirotope.rs        #   Chirotope: lazy, memoised, built from a Config
│   │   ├── circuit.rs          #   Circuit (Z⁺, Z⁻), containment_ok, link_ok
│   │   ├── cocircuit.rs        #   Cocircuits, Facets, Incidences
│   │   └── gale.rs             #   (later) Gale dual — chiro2dual, points2gale
│   ├── regular/                # the CGAL replacement
│   │   ├── mod.rs              #   delaunay / weights / heights entry points
│   │   ├── lift.rs             #   p ↦ (p, |p|² − w), exact scaling of f64 inputs
│   │   ├── predicates.rs       #   orient_d, power_side_d, symbolic perturbation
│   │   ├── tds.rs              #   arena of full cells, neighbours, mirror indices
│   │   └── insert.rs           #   walk, conflict BFS, hole retriangulation
│   ├── triang/                 # the TOPCOM replacement — consumes `om`, never `Config`
│   │   ├── mod.rs              #   Triangulation, is_fine, canonical ordering
│   │   ├── placing.rs          #   placing (pushing) triangulation
│   │   ├── fine.rs
│   │   ├── flip.rs             #   FlipRep, flip enumeration and application
│   │   ├── flipgraph.rs        #   BFS → all_connected_triangulations
│   │   └── extension/          #   DFS → all_triangulations
│   │       ├── mod.rs
│   │       └── admissibles.rs
│   ├── symmetry.rs             # Symmetry trait + TrivialGroup ZST — the seam (§6.2)
│   ├── error.rs                # #[non_exhaustive] Error enum; every C++ exit(1) is a variant
│   └── python.rs               # PyO3 bindings, feature-gated
│       └── (one `fn register_*(m)` per Rust module, so new bindings are additive)
├── python/
│   └── triangulumancer/
│       ├── __init__.py         # thin re-export layer
│       └── py.typed
├── python/tests/               # the existing pytest suite, moved
├── tests/                      # Rust integration + differential tests
│   └── fixtures/               # reference outputs from the C++ (Phase 0)
└── benches/                    # criterion
```

### Feature and dependency shape

```toml
[features]
default = []
python = ["dep:pyo3", "dep:numpy"]
# Reserved, not implemented in 0.4.0. Regularity checking is the one piece of future
# TOPCOM functionality that needs a solver (§6.3); keeping the slot named now means
# adding it never forces an LP dependency on library users who do not want it.
# lp    = ["dep:good_lp"]

[dependencies]
dashu-int = "0.6"         # bigint fallback tier; MIT/Apache-2.0, pure Rust
robust    = "1.2"         # d = 2, 3 predicate fast paths; MIT/Apache-2.0, zero deps
pyo3      = { version = "0.29", features = ["abi3-py310"], optional = true }
numpy     = { version = "0.29", optional = true }
```

No C dependency anywhere, everything MIT/Apache-2.0 or BSD-3, builds under maturin on
Linux, macOS **and Windows-MSVC** — which the current package cannot do at all. Note
`malachite` (LGPL-3.0-only) and `rug` (GMP, needs MSYS2 on Windows) are both ruled out for
wheel distribution; see [`CGAL.md` §10.4](CGAL.md).

`bnum` with a compile-time bit width is a possible faster alternative to `dashu-int` — but
only under the API decision in §5.1. Benchmark it in Phase 7, do not commit to it up front.

### Licensing

The crate stays **GPL-3.0-or-later**. Copyright protects expression rather than algorithms,
but this implementation is derived by closely reading GPL'd TOPCOM and CGAL sources, so
retaining the copyleft licence is both the safe and the simple choice. The practical win is
that the *distributed artifact* no longer bundles CGAL and TOPCOM, so the README's licensing
section collapses to a single paragraph.

---

## 5. Foundational decisions

### 5.1 Arithmetic, and the `f64` weight hazard

`i64` coordinates bound cleanly. By Hadamard, `bits(det) ≤ r·(log₂M + ½log₂r)`, so `i128`
covers `r ≤ 10, M ≤ 10³` — essentially all lattice-polytope work — with wide margin
([`TOPCOM.md` §8.2](TOPCOM.md)).

The hazard is elsewhere: **arbitrary `f64` weights convert to exact rationals with
denominators up to `2^1074`**, which blows any fixed-width bound wide open
([`CGAL.md` §10.7](CGAL.md)).

**Decision:** accept arbitrary `f64` (backwards compatibility is non-negotiable), carry the
lifted column as a rational with a shared denominator, track the exponent spread explicitly,
and use an arbitrary-precision fallback tier. **Additionally** accept exact integer/rational
heights in the Python API as a new, recommended input — for lattice-polytope work these are
the natural inputs anyway, and they keep the whole pipeline in `i128`. Reject subnormals and
absurd exponent spreads with a clear error rather than silently going quadratically slower.

The three-tier predicate stack, shared by both backends:

1. **`f64` with a derived error bound.** Rust has no stable rounding-mode control, so CGAL's
   `Interval_nt_advanced` is not portable; error-bounded round-to-nearest with `f64::mul_add`
   is the Rust idiom. Handles the overwhelming majority of calls.
2. **`i128` fraction-free Bareiss.** Every intermediate is itself a minor of the input, so
   the Hadamard bound applies to *every* intermediate, not just the result.
3. **Bigint Bareiss** on `checked_*` overflow. Astronomically rare for realistic input.

For `d = 2` and `d = 3`, delegate to `robust` — it will beat anything hand-written.

### 5.2 Errors, not exits

One `Error` enum. Every TOPCOM `exit(1)` path and every `std::runtime_error` becomes a
variant, mapped to `ValueError` (bad input) or `RuntimeError` (internal invariant violated)
at the PyO3 boundary. The Rust API returns `Result` throughout. This is on its own a
sufficient reason for the rewrite.

### 5.3 No global state

TOPCOM's process-global simplex intern table (B2) has no analogue. Any interning is owned by
the `Chirotope`/enumeration object that needs it. `Config` is `Send + Sync`; the
`recursive_mutex` in `PVConfigurationData` is replaced by ownership — the configuration is
consumed into an immutable frozen form when the first triangulation is built, which encodes
the existing "locking" behaviour in the type system rather than at runtime.

### 5.4 Streaming enumeration

`all_triangulations` and `all_connected_triangulations` currently materialise every
triangulation into a `Vec`. For configurations with 10⁶–10⁸ triangulations that is
hopeless. The Rust core exposes a callback/iterator as the *primary* API and collects only
on request; the Python side gains `iter_all_triangulations()` / `iter_all_connected_triangulations()`
alongside the existing list-returning methods. See [`TOPCOM.md` §11.1](TOPCOM.md) risk 5.

This is not a theoretical limit, and the threshold is lower than the figures above suggest.
The Phase 0 corpus contains a 16-vector rank-4 configuration (`vc_degenerate_flip`, taken
from the existing test suite) whose four enumerations all time out, and whose
`all_triangulations` run was stopped under system memory pressure several minutes in.
**Phase 5's acceptance criteria should include that configuration finishing** — streaming is
what makes that possible, so it is a requirement rather than an optimisation.

### 5.5 Free-threading

The current module declares `mod_gil_not_used()` and `per_interpreter_gil()`. PyO3 0.28+
supports free-threaded builds; use `#[pymodule(gil_used = false)]`. Note that free-threaded
3.14 has **no stable ABI** (`abi3t` arrives with 3.15 per PEP 803), so `abi3-py310` alone
does not cover it — the wheel matrix needs a version-specific `cp314-cp314t` build. cygv's
`python.yml` already has this shape.

### 5.6 Determinism

Specify and test a canonical output ordering (lexicographic on sorted simplex vertex lists,
and on the simplex list within a triangulation). The C++ output order is an artifact of hash
tables and is not reproducible across runs or platforms; pinning it makes differential
testing set-based *and* order-based, and makes user code reproducible.

---

## 6. Modularity: room for the rest of TOPCOM

We are porting six of TOPCOM's entry points. TOPCOM has around fifty. The goal of this
section is that adding any of the other forty-four later is *additive work on top of this
architecture*, not a refactor of it.

The guiding observation is that TOPCOM is already organised this way, and the naming makes
it obvious: almost every tool comes in a `points2X` / `chiro2X` pair, where `points2X`
is literally "compute the chirotope, then run `chiro2X`". **The chirotope is TOPCOM's
central abstraction, and most of its functionality is a function of the chirotope alone** —
not of the coordinates. If our port respects that boundary, the rest of TOPCOM lands cheaply.

### 6.1 The oriented-matroid layer is the seam

**Decision: every combinatorial algorithm takes `&impl OrientedMatroid`, never `&Config`.**

This is the single most important structural choice in this section, and it is cheap now and
expensive to retrofit. It is why `src/om/` exists as its own module in §4, and why
`circuit.rs` and `cocircuit.rs` live there rather than under `triang/` — circuits and
cocircuits are oriented-matroid concepts that happen to be *used* by the triangulation code.

Consequences:

* `Config` is needed only to *build* a `Chirotope`, and by the regular-triangulation code
  (which genuinely needs coordinates — a regular triangulation is not determined by the
  oriented matroid).
* Every `chiro2*` entry point becomes available the moment we accept a chirotope as input,
  with no changes to the algorithms.
* The layer is independently testable against `points2chiro`, `points2circuits`,
  `points2cocircuits` and `points2facets` output — which is worth doing in Phase 2 and
  Phase 5 regardless, as differential tests.

### 6.2 Symmetry: build the seam, not the feature

This resolves what was previously an open question. [`TOPCOM.md` §7.9](TOPCOM.md) shows the
symmetry layer can be *deleted* rather than stubbed, because with the trivial group every
symmetry loop is zero-iteration and the orbit-representative test returns `true`
unconditionally. Deleting it outright is the least code today, but it is also the change
that would be hardest to undo — symmetry is threaded through the enumerations at many
points, and rediscovering where is most of the work.

**Decision: thread a `Symmetry` trait through the enumerations at exactly the points TOPCOM
does, with a `TrivialGroup` zero-sized implementation as the only implementer.**

```rust
pub trait Symmetry {
    /// Stabiliser of a triangulation, used to prune flip enumeration.
    type Stabilizer;

    /// True when every method below is constant, letting callers skip orbit
    /// bookkeeping entirely rather than merely computing a trivial answer.
    fn is_trivial(&self) -> bool { false }
    /// Orbit representative test: is `t` the canonical member of its orbit?
    fn is_orbit_representative(&self, t: &Triangulation) -> bool;
    fn stabilizer(&self, t: &Triangulation) -> Self::Stabilizer;
    fn orbit_size(&self, t: &Triangulation) -> usize;
}

/// The only implementer in 0.4.0. Every method is a constant, so the compiler
/// monomorphises the whole layer away.
pub struct TrivialGroup;

impl Symmetry for TrivialGroup {
    type Stabilizer = ();
    fn is_trivial(&self) -> bool { true }
    fn is_orbit_representative(&self, _: &Triangulation) -> bool { true }
    fn stabilizer(&self, _: &Triangulation) {}
    fn orbit_size(&self, _: &Triangulation) -> usize { 1 }
}
```

Cost today: a few dozen lines and a generic parameter on three enumeration functions. The
compiler monomorphises `TrivialGroup` to nothing, so there is **zero runtime cost** and the
generated code is identical to the delete-it-entirely version. Benefit: the insertion points
are typed and documented rather than needing to be rediscovered from the C++, and adding
`points2symmetries` plus real symmetric enumeration later is a matter of writing one more
implementer.

This is the one place where I am deliberately spending effort now to buy optionality, and
it is worth it precisely because symmetric enumeration is the most likely thing to be wanted
next — it is TOPCOM's headline performance feature for large symmetric configurations.

### 6.3 What each future addition would actually cost

Given the architecture above, here is the honest accounting. "Free" means the code already
exists as an internal and only needs a public API plus bindings.

| TOPCOM functionality | Cost after this migration | Why |
|---|---|---|
| `points2n*` / `chiro2n*` counting variants | **Free** | `.count()` instead of `.collect()` on the streaming iterators of §5.4 |
| `points2chiro`, `chiro2*` entry points | **Free** | §6.1 — the algorithms already take a chirotope |
| `points2circuits`, `points2cocircuits`, `points2facets` | **Free** | Built in Phases 3 and 5, live in `om/` |
| `points2flips` | **Free** | Already exposed as `Triangulation.flips()` |
| `points2vertices` | **Trivial** | Read off the cocircuits |
| `points2volume` | **Trivial** | One determinant per simplex; Phase 1 gives exact determinants |
| Configuration generators (`cube`, `cross`, `cyclic`, `hypersimplex`, `lattice`, `permutahedron`) | **Trivial**, ~150 LOC total | Standalone constructors; worth adding early anyway as test fixtures |
| `chiro2dual`, `points2gale` | **Small**, ~150 LOC | Gale duality is exact linear algebra on `Config`; slot is reserved at `om/gale.rs` |
| `points2mintriang` | **Moderate**, ~300 LOC | Branch-and-bound reusing the Phase 5 extension graph |
| Symmetric enumeration, `points2symmetries` | **Moderate** given §6.2, large without it | One `Symmetry` implementer plus orbit machinery; the seams already exist |
| GKZ vectors, secondary polytope | **Moderate**, ~400 LOC | Needs the `Volumes` table we dropped, which is just determinants over Phase 1 |
| Regularity checking (`--regular`/`--nonregular`) | **Largest**, and the only one needing a new dependency | An LP feasibility problem per triangulation. Gate behind an `lp` cargo feature so it never becomes a mandatory dependency — see §4 |

Nothing in that table requires revisiting Phases 1–7. That is the test this section is
meant to pass.

### 6.4 Supporting conventions

Small decisions that keep the above true:

* **`#[non_exhaustive]` on `Error`** and on any public enum, so new variants are not
  semver-breaking.
* **Optional heavy pieces behind cargo features**, defaulting off (`lp` is the concrete
  case). The `python` feature is already shaped this way.
* **Point-vs-vector as a type-level distinction**, not a scattered `is_pc` boolean. The C++
  carries `1 * is_pc` arithmetic through a dozen call sites, which is exactly the kind of
  thing that makes adding a third configuration kind painful.
* **`python.rs` is a translation layer only** — one `register_*(m)` function per Rust
  module, no logic. Adding bindings for a new feature then touches one function.
* **Publish the Rust API from 0.4.0 with the `om` layer public.** Even if the Python side
  exposes none of it initially, having it public from the first release means later
  additions are not semver events.

---

## 7. Phases

Sequenced so that the low-risk majority of the work lands first, and so that each phase has
a testable deliverable. LOC estimates come from [`CGAL.md` §9](CGAL.md) and
[`TOPCOM.md` §11](TOPCOM.md).

### Phase 0 — Reference corpus and scaffolding *(no algorithms)*

* Tag the final C++ commit `legacy-cpp`; work proceeds on the `rust-migration` branch.
* **The C++ bugs of §3 are not fixed in the C++.** Decided: the effort goes into the Rust
  port instead. The consequence for this phase is that the oracle is `0.3.0` as shipped, so
  fixtures for the inputs affected by **B1** are recorded as **known-divergent** — captured
  for reference, but asserted to *differ* from the Rust implementation rather than to match.
  See `tests/fixtures/KNOWN_DIVERGENT.md`. B2–B5 do not affect fixture generation: B2 needs
  a liveness pattern the generator does not use, and B3–B5 need inputs the corpus flags
  separately.
* Build `scripts/gen_fixtures.py`, which installs `triangulumancer==0.3.0` in an isolated
  venv and dumps, for a corpus of configurations, the output of *every* public method.
  Store as compressed JSON in `tests/fixtures/`. The corpus:
  * everything in the current `tests/`;
  * all `0/1` configurations up to `n ≤ 8, d ≤ 4` (exhaustive small-case coverage — this is
    the mitigation for the two hardest risks, per [`TOPCOM.md` §11.1](TOPCOM.md));
  * cyclic polytopes, products of simplices, and configurations with known
    `n_triangulations` from TOPCOM's own `examples/`;
  * deliberately degenerate cases: duplicate points, cospherical sets, rank-deficient input,
    interior points, large coordinates near the `i128` boundary.
* New repository skeleton per §4: `Cargo.toml`, maturin `pyproject.toml`, `python/` package,
  `rust.yml` + `python.yml` + updated `deploy.yml`, pre-commit with `fmt`/`clippy`/`ruff`/
  `mypy`, `noxfile.py`, `AGENTS.md`.
* **The C++ is removed from the build in this phase.** It survives as the `legacy-cpp` tag
  and as the pinned PyPI `0.3.0` wheel used to generate fixtures. Not-yet-ported methods
  raise `NotImplementedError`. No PyPI release is cut until Phase 7; `0.3.0` remains what
  users install from PyPI throughout the migration.

  *Rationale:* a hybrid CMake+Cargo build is genuinely painful. Since no C++ bugfix release
  is planned, no `v0.3.x` maintenance branch is needed either — `legacy-cpp` is enough to
  recover the sources if that changes.

**Exit criteria:** `maturin develop` works; `import triangulumancer` succeeds; fixture corpus
committed and regenerable; CI green on all platforms including Windows.

### Phase 1 — Exact arithmetic and the configuration core  *(~750 LOC, high risk)*

* `arith::sign_of_det`: the three-tier stack of §5.1.
* `config::Config`: column-major integer matrix, exact rank, point-vs-vector homogenisation.
* `config::hull`: exact HNF affine-hull reduction. This is worth the 150–250 lines on its
  own — it deletes ~450 lines of CGAL's hardest code (`insert_increase_dimension`, all of
  `Coaffine.h`) and fixes B4 structurally ([`CGAL.md` §6.3](CGAL.md)).
* `simplex`: bitset `Simplex`, `SimplexSet`, `link`/`join`/`support`/`insert_boundary`.
* Ship the Python side of `PointConfiguration`/`VectorConfiguration`: `dim`, `n_points`/
  `n_vectors`, `points`/`vectors`, `add_point(s)`/`add_vector(s)`, locking, `__repr__`.

**Risk:** this is the highest-risk phase after the extension graph, because a wrong
determinant sign "manifests as a hang three thousand insertions later"
([`CGAL.md` §9](CGAL.md)). **Mitigation:** exhaustive differential testing against a naive
`BigInt` cofactor expansion (`O(n!)`, fine for `n ≤ 8`) over random matrices including
deliberately near-singular ones. Write the validation harness *before* the implementation.

**Exit criteria:** `sign_of_det` agrees with cofactor expansion on ≥10⁶ random and adversarial
matrices; `Config` rank matches the C++ on the whole fixture corpus; configuration tests in
`test_point_configuration.py` and `test_vector_configuration.py` pass unchanged.

### Phase 2 — Oriented-matroid layer, placing and fine triangulations  *(~550 LOC, low–medium risk)*

* `om::chirotope`: lazy, memoised chirotope over `sign_of_det`, behind the
  `OrientedMatroid` trait of §6.1. **From here on, no algorithm in `triang/` takes a
  `Config`** — this boundary is what makes every `chiro2*` entry point free later.
* `symmetry`: the `Symmetry` trait and `TrivialGroup` of §6.2. A few dozen lines, monomorphised
  away at compile time; the enumerations in Phases 4 and 5 are generic over it from the start
  rather than being retrofitted.
* `triang::placing`, `triang::fine`.
* **Ships** `placing_triangulation()` and `fine_triangulation()` for both configuration types.

**Exit criteria:** exact match against fixtures for both methods over the whole corpus,
including the 46-simplex degenerate vector configuration that "fails in TOPCOM 1.1.2".
Additionally, the chirotope itself matches `points2chiro` output — a differential test worth
having on its own, and the one that validates the §6.1 boundary.

### Phase 3 — Circuits, flips and neighbours  *(~750 LOC, medium–hard risk)*

* `om::circuit`: `Circuit`, `containment_ok`, `link_ok` — in `om/`, not `triang/`, per §6.1.
* `triang::flip`: `FlipRep`, naive (recompute-all) flip enumeration and application.
* **Ships** `flips()`, `bistellar_flips()`, `neighbors()`.

**Risk:** `containment_ok`/`link_ok` are ~80 lines of dense, uncommented C++ carrying the
entire degenerate-input correctness story, which our TOPCOM fork has already had to patch
twice. **Mitigation:** the exhaustive `n ≤ 8, d ≤ 4` fixture sweep from Phase 0, comparing
full flip sets rather than counts. Cross-check `om::circuit` against `points2circuits`.

### Phase 4 — Flip-graph enumeration  *(~250 LOC, medium risk)*

* `triang::flipgraph`: BFS with a single visited set (simpler than TOPCOM's two-layer
  scheme), streaming per §5.4.
* Seed with `FineTriang` when `only_fine` — fixing **B1**.
* Define `is_fine(T) == (support(T).len() == n)` once and post-filter, so both enumerators
  agree ([`TOPCOM.md` §10.4](TOPCOM.md)).
* **Ships** `all_connected_triangulations(only_fine)`.

**Exit criteria:** matches fixtures, including 387/64 for the 3×3 grid. Note the
known-divergent fixtures from B1 must match the *patched* C++, not `0.3.0`.

### Phase 5 — Extension-graph enumeration  *(~1 500 LOC, highest risk)*

* `om::cocircuit` (`Cocircuits`, `Facets`, `Incidences`) — in `om/`, per §6.1, since these
  are oriented-matroid objects and are what `points2cocircuits`/`points2facets` would expose.
* `extension::admissibles`, then the backtracking DFS.
* Symmetry enters only through the `Symmetry` trait of §6.2, instantiated at `TrivialGroup`.
  Every call collapses at compile time — with the trivial group TOPCOM's element vector is
  empty, every symmetry loop is zero-iteration, and the orbit-representative test returns
  `true` unconditionally, so the generated code matches a delete-it-entirely port. Uniqueness
  is carried independently by a lexicographic `forbidden`-set rule
  ([`TOPCOM.md` §7.9](TOPCOM.md)), which is *not* part of the trait and stays unconditional.
* `only_fine` restricts the *simplex universe* to empty simplices here — a genuine pruning,
  not an output filter ([`TOPCOM.md` §7.7](TOPCOM.md)).
* **Ships** `all_triangulations(only_fine)`.

**Risk:** ~3 200 lines of the densest code in TOPCOM, with a lex-order canonicity argument
that guarantees each triangulation is produced exactly once. Getting it subtly wrong yields
silently missing or duplicated results. **Mitigation:** differential-test *sets*, not just
counts, over the entire corpus; cross-check against Phase 4 (on configurations whose flip
graph is connected, the two enumerations must produce identical sets).

**Exit criteria:** matches fixtures as sets over the whole corpus, and `vc_degenerate_flip`
— where the oracle times out on all four enumerations and exhausts memory (§5.4) —
completes. That configuration is the phase's headline target, not a stretch goal: it is the
one input in the corpus the C++ cannot handle at all.

### Phase 6 — Regular and Delaunay triangulations  *(~3 000 LOC, high risk)*

The CGAL replacement, per [`CGAL.md` §6.6](CGAL.md): CGAL's data structure, the single
lifted predicate, preceded by the affine-hull reduction already built in Phase 1.

* `regular::lift`: `p ↦ (p, |p|² − w)`, with exact `f64` scaling per §5.1. Fixes **B3**.
* `regular::predicates`: `orient_d` and `power_side_d` as thin wrappers over `sign_of_det` —
  they are the *same call*, since
  `power_side(cell, q) = (−1)^d · orient_{d+1}(q̂, v̂₀, …, v̂_d)`. Plus index-based symbolic
  perturbation (Simulation of Simplicity).
* `regular::tds`: arena of cells with `verts`/`neigh`/`mirror`, infinite vertex.
* `regular::insert`: remembering walk, conflict-zone BFS, hash-map ridge-matching hole
  retriangulation (much safer than rotors). Vertex→index mapping is carried through the
  insertion rather than recovered by a coordinate scan — fixing **B5**.
* Hidden points fall out of hole retriangulation; **report them** to Python rather than
  dropping them silently.
* `remove()` is not ported (~320 lines of CGAL's hardest code, unused).
* Spatial sort (BRIO/Hilbert) last, and only if profiling asks for it.
* **Ships** `delaunay_triangulation()`, `triangulate_with_weights()`,
  `triangulate_with_heights()`.

**Risk:** hole retriangulation is the one place where a subtle combinatorial error produces a
structurally invalid mesh that only blows up much later. **Mitigation:** assert the TDS
invariants after *every* insertion in debug builds; fuzz with random point sets; add a
capped-iteration walk that returns a hard error rather than hanging. Use a flip-based
construction (start from the placing triangulation of Phase 2, flip down to regular) as an
independent oracle on small inputs.

### Phase 7 — Parity, performance, release

* Full differential suite green across the corpus, order-sensitive per §5.6.
* Criterion benchmarks against the `0.3.0` oracle. Watch two things specifically:
  triangulation equality/hashing (TOPCOM's interned bitsets make this very cheap — a naive
  `HashSet<Simplex>` port may be several times slower), and the `d ≥ 10` bigint predicate
  path (though the current CGAL configuration is already slow there, so the bar is low).
* Incremental `TriangFlips` maintenance as an *optimisation* over Phase 3's naive version,
  behind a debug-assert that the two agree ([`TOPCOM.md` §11.1](TOPCOM.md) risk 3).
* Optional: parallel flip-graph BFS with a sharded visited set. TOPCOM has no parallelism
  here to match, so this is upside, not parity.
* README, rustdoc with KaTeX, migration notes.
* Release **0.4.0** to PyPI *and* crates.io.

---

## 8. Validation strategy

Correctness is the whole risk of this project. Four independent layers:

1. **Differential testing against the C++ oracle.** The Phase 0 fixture corpus, compared as
   *sets and order*, on every phase. This is the primary gate.
2. **Internal oracles.** Cheap-but-slow reference implementations kept permanently as test-only
   code: `BigInt` cofactor expansion vs. `sign_of_det`; naive recompute-all flips vs.
   incremental; flip-based regular construction vs. incremental insertion; Phase 4's
   enumeration vs. Phase 5's on flip-connected configurations.
3. **Invariant assertions.** TDS validity after every insertion (debug builds); triangulation
   validity (simplices cover the configuration, pairwise interiors disjoint, boundary
   consistency) after every construction.
4. **Property-based fuzzing.** `proptest` over random integer configurations, checking
   invariants and oracle agreement rather than fixed outputs.

Layers 2–4 are what let the package be maintained after the C++ oracle is retired.

---

## 9. Packaging and CI

| | Now | After |
|---|---|---|
| Build backend | scikit-build-core + CMake + pybind11 | maturin + Cargo |
| System deps | CGAL, Boost, Eigen, GMP, MPFR, cddlib, CMake | none |
| Platforms | Linux (x86_64, arm64), macOS arm64 | + Windows |
| CI workflows | `test.yml` | `rust.yml` + `python.yml` |
| Wheels | cibuildwheel with `yum`/`brew` prologues | cibuildwheel, no prologue; abi3-py310 + `cp314-cp314t` |
| Publishes | PyPI | PyPI + crates.io |
| Lint | clang-format, cmake-format, ruff, codespell | rustfmt, clippy, ruff, mypy, codespell |

`deploy.yml` gains a `deploy-crate` job gated on the same release event, so the crate and the
wheels always ship together. `[tool.uv] cache-keys` must list `Cargo.toml` and `src/**/*.rs`,
or `uv run pytest` silently tests a stale extension.

Note that dropping GMP removes the single largest source of CI complexity in comparable
projects — no `GMP_MPFR_SYS_CACHE`, no `CC=clang` pinning, no 13-minute Windows builds.

---

## 10. Proposed API cleanups

Within the "mostly keep, allow cleanups" mandate. All are additive or bug fixes; nothing
listed here breaks existing code.

| Change | Rationale |
|---|---|
| `Triangulation.is_fine` property | A single explicit definition, used by both enumerators (§5.4, B1) |
| `iter_all_triangulations()`, `iter_all_connected_triangulations()` | Streaming; the list-returning forms remain (§5.4) |
| `triangulate_with_heights` / `_weights` accept ints and `Fraction` | Exact input path; avoids the `f64` precision hazard entirely (§5.1) |
| `Triangulation.hidden_points` | Regular triangulations can leave weighted points unused; currently dropped silently ([`CGAL.md` §7.2](CGAL.md)) |
| `Triangulation(config, simplices)` validates by default, `validate=False` to opt out | Currently accepts arbitrary garbage |
| `ValueError` for bad input, `RuntimeError` for internal errors | Currently everything is `RuntimeError`, or `exit(1)` (§5.2) |
| Canonical, deterministic output ordering | Reproducibility (§5.6) |
| `dim`/`n_points` etc. unchanged | No churn without cause |

---

## 11. Effort summary

| Phase | Component | Rust LOC | Risk |
|---|---|---:|---|
| 0 | Scaffolding, fixtures, CI | ~600 | low |
| 1 | Arithmetic, config, hull, simplex | ~750 | **high** |
| 2 | OM layer, symmetry seam, placing, fine | ~550 | low–medium |
| 3 | Circuits and flips | ~750 | medium–hard |
| 4 | Flip-graph BFS | ~250 | medium |
| 5 | Extension-graph DFS | ~1 500 | **highest** |
| 6 | Regular / Delaunay | ~3 000 | **high** |
| 7 | Bindings, benchmarks, polish | ~650 | low |
| | **Total** | **~8 050** | replacing ~16 600 lines of TOPCOM plus a comparable slice of CGAL |

Phases 1–4 cover four of the six triangulation methods and are the low-risk majority. The
schedule risk lives in Phases 5 and 6, which are independent of each other and could be
worked in either order — or in parallel, since they share only the Phase 1 foundation.

The modularity work of §6 costs ~50 LOC in total (the `Symmetry` trait and `TrivialGroup`);
everything else there is a matter of *where* code is put rather than how much of it there is.

---

## 12. Open questions

1. **Windows support** becomes free with this migration. Worth advertising in the README, and
   adding `windows-2022` to the CI matrix from Phase 0 so it never regresses.
2. **Should `all_connected_triangulations` keep its name?** It returns the flip-connected
   component of a seed, and with `only_fine` the component in the *induced subgraph on fine
   triangulations* — a fine triangulation reachable only through non-fine intermediates is
   not found. That is an intended TOPCOM limitation, but it is not obvious from the name and
   should at minimum be documented.
3. **`bnum` vs `dashu-int`** for the fallback tier — resolve by benchmark in Phase 7, after
   §5.1's API decision has fixed the bit bound.
4. **Which of §6.3's "free" items to actually expose in 0.4.0.** Counting variants, the
   chirotope, circuits, cocircuits and facets all exist as internals by Phase 5 and cost only
   an API surface plus bindings. Exposing them is cheap but each one is a compatibility
   commitment. My suggestion: ship the configuration generators (they make the test corpus
   better anyway) and the `n*` counting variants, hold the oriented-matroid objects behind
   the Rust API only until there is a concrete use for them in Python.

*(Symmetry was previously an open question here; §6.2 resolves it — build the seam, not the
feature.)*

---

## 13. Current status

**Phase 0 is complete** (branch `rust-migration`). Phases 1–7 are not started.

| | |
|---|---|
| Branch | `rust-migration` — nothing is on `main` |
| The old C++ | tag `legacy-cpp` (`c643162`); `git show legacy-cpp:src/triangulumancer/CGAL.cpp` etc. |
| PyPI | still serves `0.3.0`; the crate version is `0.4.0-dev` |
| Rust | `cargo build` / `test` / `clippy` clean on both feature sets; 4 unit tests |
| Python | `import triangulumancer` works; `pytest` gives 16 skips, each naming its phase |
| Fixtures | `tests/fixtures/core.json.gz`, 31 configurations, 25 KiB, oracle `0.3.0` |

### What exists in `src/`

Only `lib.rs` (the `ConfigKind` point/vector distinction, plus the layering doc comment),
`error.rs` (the `#[non_exhaustive] Error` enum), and `python.rs` (the full 0.3 API surface as
`todo()` stubs). **No algorithms.** The module tree in §4 is the target, not the current state.

### Verified vs not

Verified locally on macOS arm64, Python 3.14: both cargo feature sets build, clippy is clean
with `-D warnings`, `cargo fmt --check` passes, `prek -a` passes all hooks, the extension
imports, pytest collects and skips cleanly, and the fixture generator runs end to end.

**Verified in CI** (both workflows green on `rust-migration`, 16/16 jobs):

* **Windows works.** `cargo build`/`test`/`clippy` and `pip install .[test]` + `pytest` all
  pass on `windows-2022`, which the C++ package could never do. This was the single largest
  open question about the migration and it is now settled.
* **Free-threading works.** `python.yml` builds and tests `3.14t` alongside `3.10` and `3.14`
  on all four platforms, so `abi3-py310` plus a version-specific free-threaded build is the
  right shape (§5.5) and PyO3's `gil_used = false` compiles under it.
* **All four platforms × three interpreters.** ubuntu-24.04, ubuntu-24.04-arm, macos-15,
  windows-2022.

**Not verified:**
* **crates.io publish.** `deploy-crate` needs a `CARGO_REGISTRY_TOKEN` secret and the
  `crates.io` environment on the repository; neither is set up. The name
  `triangulumancer` is unclaimed on crates.io (checked 2026-11-12) — worth claiming early,
  since the whole point of §4's layout is that the Rust library is publishable.
* **`cargo package`.** Never run. The `exclude` list in `Cargo.toml` is a guess.
* **`deploy.yml`.** Only runs on a published release or `workflow_dispatch`, so none of the
  cibuildwheel configuration has executed. Expect to iterate on it at Phase 7, not before.

### CI runs on this branch

Both workflows trigger on pushes to `rust-migration` as well as `main`, so every phase
commit is checked without needing an open pull request. Remove those two lines from the
trigger lists when the branch merges.

Note that `macos-14` — used by the pre-migration workflow, and still used on `main` — has
been retired by GitHub; jobs on it fail in under two seconds with "The configuration
'macos-14-arm64-us-default' is not supported". This branch uses `macos-15`. **`main`'s own
CI is broken until someone makes the same change there**, independently of this migration.

---

## 14. Picking this up on another machine

### Prerequisites

```bash
# Rust: 1.85+ (edition 2024). Nothing else to install — no CMake, no CGAL, no GMP.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# uv, for the dev venv and for the fixture generator's throwaway oracle venv.
curl -LsSf https://astral.sh/uv/install.sh | sh

# prek, for linting (the project uses it in place of `pre-commit run -a`).
uv tool install prek
```

### Setup

```bash
git clone git@github.com:ariostas/triangulumancer.git
cd triangulumancer
git checkout rust-migration

cargo test                                      # should pass, 4 tests
cargo clippy --all-targets -- -D warnings
cargo clippy --features python --all-targets -- -D warnings

uv venv .venv-dev
VIRTUAL_ENV=.venv-dev uv pip install -e '.[test]'
.venv-dev/bin/python -m pytest                  # should give 16 skips, 0 failures

prek -a --quiet                                 # all hooks should pass
```

`.venv-dev/` and `target/` are gitignored, so both are rebuilt from scratch. The first
`uv pip install -e` compiles the extension through maturin and takes a minute.

### The upstream sources you will need

Both research documents cite `path:line` into checkouts that are **not** in this repository
and must be recreated. The commands are in each document's preamble; in short:

```bash
# TOPCOM — pinned to an exact revision, so its line numbers are exact.
git clone https://github.com/ariostas/TOPCOM.git topcom
cd topcom && git checkout a4333293e9c3cbcf5b8420b89ee30f51078d47fc

# CGAL — sparse, pinned to the nearest release to when the notes were written.
# Line numbers are approximate; grep for the quoted symbol names instead.
git clone --filter=blob:none --sparse https://github.com/CGAL/cgal.git cgal
cd cgal && git checkout v6.2.1 && git sparse-checkout set \
    Triangulation/include NewKernel_d/include Kernel_d/include \
    Spatial_sorting/include Number_types/include Installation/include
```

Put them wherever you like, outside the repository.

### Regenerating the fixture corpus

Only needed if the corpus changes; the committed one is complete.

```bash
python scripts/gen_fixtures.py --list        # inspect, installs nothing
python scripts/gen_fixtures.py               # ~15 min, needs network and uv
python scripts/gen_fixtures.py --only grid   # re-run a subset, merged into the corpus
```

It installs `triangulumancer==0.3.0` from PyPI into a throwaway venv, so it needs a platform
that 0.3.0 has a wheel for. That is macOS **arm64** and Linux x86_64/arm64 only — there are
no Windows wheels and no Intel-macOS wheels, and building 0.3.0 from source means installing
CGAL, TOPCOM's dependencies, GMP and cddlib, which is the whole thing we are escaping. On
those platforms, take the committed corpus as given; it is complete and does not need
regenerating to do any of phases 1–7.

Do not raise the timeout for `vc_degenerate_flip` hoping for better data; see §5.4.

### Starting Phase 1

In order, and note that the harness comes first:

1. `tests/` — the differential harness for `sign_of_det`: a naive `BigInt` cofactor expansion
   (`O(n!)`, fine for `n ≤ 8`) over random and deliberately near-singular matrices. Write it
   before the implementation. §6 of `CGAL.md` and §8 of `TOPCOM.md` are the references.
2. `src/arith/` — the three-tier stack of §5.1: f64 filter, `i128` Bareiss, bigint Bareiss.
   Add `dashu-int = "0.6"` to `Cargo.toml` at this point (§4 explains why it is not there yet).
3. `src/config.rs` + `src/config/hull.rs` — the configuration matrix, exact rank, and the HNF
   affine-hull reduction. This is what fixes B4.
4. `src/simplex.rs` — the bitset `Simplex` and `SimplexSet`.
5. `src/python.rs` — replace the `todo()` stubs for `dim`, `n_points`/`n_vectors`,
   `points`/`vectors`, `add_point(s)`/`add_vector(s)`, `__repr__`, and the locking behaviour.
   Add `numpy` usage here; the dependency is already declared.

Phase 1 is done when `test_point_configuration.py` and `test_vector_configuration.py` pass
rather than skip, and the `sign_of_det` harness is green on ≥10⁶ matrices.

### A warning about the `__repr__` strings

The 0.3 strings are asserted nowhere in the test suite but are in the README's doctest-style
example, e.g. `A 2-dimensional point configuration with 9 points`. Reproduce them exactly;
they are recoverable from `git show legacy-cpp:src/triangulumancer/PVConfiguration.cpp`.
