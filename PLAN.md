# Migration plan: C++ (CGAL + TOPCOM) → Rust

This document is the overall plan for reimplementing `triangulumancer` in Rust and
retiring its two C++ dependencies. It is the top-level companion to two detailed
design-input documents produced by reading the upstream sources:

* [`CGAL.md`](CGAL.md) — the regular/Delaunay triangulation slice.
* [`TOPCOM.md`](TOPCOM.md) — the placing/fine triangulation, flip and enumeration slices.

Read those for the algorithms. This document covers *why*, *what order*, *what the
result looks like*, and *how we know it is correct*.

> ## ▶ Status: Phases 0–2 complete. **Phase 3 is next.**
>
> Branch `rust-migration`. `src/` has the foundation — exact determinant signs (`arith`),
> integer configurations with exact rank and affine-hull reduction (`config`), bitset
> simplices (`simplex`) — and, from Phase 2, the oriented-matroid layer (`om`: the
> `OrientedMatroid` trait, a lazy `Chirotope`, `Circuit`), the `Symmetry` seam, and
> `triang` (the `Triangulation` type with a full validity check, placing and fine).
> Python ships `placing_triangulation`, `fine_triangulation` and a real `Triangulation`
> class; flips, neighbours, the enumerations and the CGAL methods are still
> `NotImplementedError` stubs. `regular/`, `om/cocircuit.rs` and `triang/flip*` in §4 do
> not exist yet.
>
> * **§7** — the phase table and what each phase ships.
> * **§13** — what exists, what CI has verified, and **the Phase 1 and Phase 2 decisions
>   later phases build on**.
> * **§14** — setup on a fresh machine, and the ordered first steps for Phase 3.
>
> *Finishing a phase means updating this banner and the table in §7, alongside §13.*

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
robust    = "1.2"         # d = 2, 3 predicate fast paths — not adopted in Phase 1,
                          # reconsider in Phase 6 (§13, "Phase 1 decisions")
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
*(Phase 1 did not: `robust` takes `f64` points, so it only applies to integer input below
`2^53` and to point predicates, not general determinants. Revisit in Phase 6; see §13.)*

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
| `points2volume` | **Trivial** | One determinant per simplex. Phase 1 exposes only the exact *sign*; the Bareiss code computes the value and needs only a public entry point |
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

| | Phase | Ships | Risk | Status |
|---|---|---|---|---|
| 0 | Reference corpus and scaffolding | *(nothing user-visible)* | low | ✅ **done** |
| 1 | Exact arithmetic and configuration core | the configuration classes | high | ✅ **done** |
| 2 | Oriented-matroid layer, placing and fine | `placing_triangulation`, `fine_triangulation` | low–med | ✅ **done** |
| 3 | Circuits, flips and neighbours | `flips`, `bistellar_flips`, `neighbors` | med–hard | ▶ **next** |
| 4 | Flip-graph enumeration | `all_connected_triangulations` | medium | ⬜ |
| 5 | Extension-graph enumeration | `all_triangulations` | **highest** | ⬜ |
| 6 | Regular and Delaunay triangulations | `delaunay_triangulation`, `triangulate_with_*` | high | ✅ **done** |
| 7 | Parity, performance, release | the 0.4.0 release | low | ⬜ |

Phases 5 and 6 are independent of each other and share only the Phase 1 foundation, so they
can be taken in either order, or in parallel.

### ✅ Phase 0 — Reference corpus and scaffolding — **DONE** *(no algorithms)*

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

### ✅ Phase 1 — Exact arithmetic and the configuration core — **DONE** *(~2 500 lines of `src/` including docs and unit tests, high risk)*

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

**Met.** `tests/sign_of_det.rs` (`differential_million`, `#[ignore]`d) checked 1 000 098
matrices of sizes 0–8 across eleven families in 37 s (release build), with zero disagreements
for the full stack and for each tier on its own; the bigint tier was needed on 290 383 of
them and the `f64` filter decided 538 360. `tests/fixture_rank.rs` checks the rank implied by
every trusted result in the corpus (19 713 simplices, 24 "not full rank" rejections). Both
pytest files pass unchanged. The decisions this phase made for later ones are in §13.

### ✅ Phase 2 — Oriented-matroid layer, placing and fine triangulations — **DONE** *(~1 300 lines of `src/` including docs and unit tests, low–medium risk)*

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

**Met.** `tests/fixture_triang.rs` matches both methods on all 31 corpus entries — 56
triangulations identical simplex for simplex in canonical order, and the 6 "not full rank"
rejections — including the 46-simplex `vc_degenerate_flip`. TOPCOM itself was built from
the pinned checkout (GMP only; recipe in `scripts/gen_topcom_fixtures.py`) and
`tests/topcom.rs` checks, against its `points2chiro`, `points2placingtriang` and
`points2finetriang`, 306 configurations (6 named, 300 seeded-random and mostly
degenerate): 300 chirotopes (38 660 signs) and 612 triangulations, all identical. The same
test reruns placing and fine on a chirotope *parsed from TOPCOM's text*, with no
configuration in sight, which is the §6.1 boundary exercised. Beyond the brief,
`Triangulation::check` (a port of TOPCOM's `CheckTriang`) accepts all 3 353 whole
triangulations in the corpus and rejects all 77 871 corruptions of them (one simplex
dropped, or one foreign simplex added). The Phase 2 decisions are in §13.

### ▶ Phase 3 — Circuits, flips and neighbours — **NEXT** *(~750 LOC, medium–hard risk)*

* `om::circuit`: `Circuit`, `containment_ok`, `link_ok` — in `om/`, not `triang/`, per §6.1.
* `triang::flip`: `FlipRep`, naive (recompute-all) flip enumeration and application.
* **Ships** `flips()`, `bistellar_flips()`, `neighbors()`.

**Risk:** `containment_ok`/`link_ok` are ~80 lines of dense, uncommented C++ carrying the
entire degenerate-input correctness story, which our TOPCOM fork has already had to patch
twice. **Mitigation:** the exhaustive `n ≤ 8, d ≤ 4` fixture sweep from Phase 0, comparing
full flip sets rather than counts. Cross-check `om::circuit` against `points2circuits`.

### ⬜ Phase 4 — Flip-graph enumeration  *(~250 LOC, medium risk)*

* `triang::flipgraph`: BFS with a single visited set (simpler than TOPCOM's two-layer
  scheme), streaming per §5.4.
* Seed with `FineTriang` when `only_fine` — fixing **B1**.
* Define `is_fine(T) == (support(T).len() == n)` once and post-filter, so both enumerators
  agree ([`TOPCOM.md` §10.4](TOPCOM.md)).
* **Ships** `all_connected_triangulations(only_fine)`.

**Exit criteria:** matches fixtures, including 387/64 for the 3×3 grid. Note the
known-divergent fixtures from B1 must match the *patched* C++, not `0.3.0`.

### ⬜ Phase 5 — Extension-graph enumeration  *(~1 500 LOC, highest risk)*

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

### ✅ Phase 6 — Regular and Delaunay triangulations — **DONE** *(~1 350 lines of `src/regular/` including docs and unit tests, plus ~150 in `arith`; high risk)*

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

**Met.** `tests/regular.rs` matches all 155 trusted `delaunay_triangulation` and
`triangulate_with_*` results in the corpus exactly, in canonical order — the cospherical and
lattice cases included (unit square, 3×3 grid, cubes, cross-polytopes, products of
simplices) — and asserts the B3, B4 and B5 values now written in `KNOWN_DIVERGENT.md` for
the 34 untrusted ones. Every result, corpus and random, passes an independent regularity
certificate (`tests/common/regular.rs`: `Triangulation::check` on the hull-reduced
chirotope, and every lifted point weakly above every lifted simplex), and the TDS
invariants after every insertion in debug builds. The tie-breaking is pinned by a
brute-force oracle — every simplex whose lifted hyperplane has all other points strictly
above it, for lifts perturbed *numerically* by the rule the implementation applies
symbolically — on 215 small random configurations by default and 5 043 of 20 000 in the
`#[ignore]`d release run (`cargo test --release --test regular -- --ignored`, ~45 s),
all identical. The flip-based oracle was not used: flips are Phase 3, being built in
parallel; the brute-force oracle is the stronger check on small inputs anyway, since it
also fixes the tie-breaking. Decisions are in §13.

### ⬜ Phase 7 — Parity, performance, release

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

**Phases 0, 1 and 2 are complete; Phase 3 is next.** Nothing in phases 3–7 has been started.
The phase table in §7 is the at-a-glance version; this section is the detail.

| | |
|---|---|
| Branch | `rust-migration` — nothing is on `main` |
| The old C++ | tag `legacy-cpp` (`c643162`); `git show legacy-cpp:src/triangulumancer/CGAL.cpp` etc. |
| PyPI | still serves `0.3.0`; the crate version is `0.4.0-dev` |
| Rust | `cargo build` / `test` / `clippy` / `doc` clean on both feature sets; 69 tests, plus the `#[ignore]`d 10⁶-matrix run |
| Python | `pytest`: 16 pass, 8 skip, each skip naming its phase |
| Fixtures | `tests/fixtures/core.json.gz`, 31 configurations, 25 KiB, oracle `0.3.0`; `tests/fixtures/topcom.json.gz`, 306 configurations, 25 KiB, TOPCOM's own CLI tools |

### What exists in `src/`

| Module | Contents |
|---|---|
| `lib.rs` | `ConfigKind`, the layering doc comment |
| `error.rs` | the `#[non_exhaustive] Error` enum; Phase 1 added `InvalidShape` and `IndexOutOfRange` |
| `arith.rs`, `arith/{filter,bareiss}.rs` | `Sign`, `SquareMatrix`, `sign_of_det`, `rank`; the three tiers, each public on its own |
| `config.rs`, `config/hull.rs` | `Config<K: Kind>` with markers `Points`/`Vectors`, `Lockable<K>`, `reduce_to_hull` |
| `simplex.rs` | `Simplex` (bitset), `SimplexSet` (canonically ordered set of simplices), `combinations` (lex-order `k`-subsets) |
| `om/mod.rs` | the `OrientedMatroid` trait; `check_basis`; `to_topcom_string` (`points2chiro` format) |
| `om/chirotope.rs` | `Chirotope`, lazy and memoised; `Config::chirotope()` |
| `om/circuit.rs` | `Circuit` with TOPCOM's orientation, `upper_facets`/`lower_facets` |
| `symmetry.rs` | `Symmetry`, `TrivialGroup` — as §6.2, verbatim |
| `triang/mod.rs`, `triang/check.rs` | `Triangulation`; `check`, the port of TOPCOM's `CheckTriang` |
| `triang/placing.rs`, `triang/fine.rs` | `triang::placing`, `triang::fine` |
| `python.rs` | configuration classes, `placing_triangulation`, `fine_triangulation`, `Triangulation`; flips, neighbours, enumerations, CGAL methods and `Flip` still `todo()` stubs |

`regular/`, `om/cocircuit.rs` and `triang/{flip,flipgraph,extension}` from §4 do not exist
yet.

Tests: `tests/sign_of_det.rs` (the differential harness; run the full 10⁶ with
`cargo test --release --test sign_of_det -- --ignored --nocapture`, about 40 s),
`tests/fixture_rank.rs` (rank against the corpus, and the B4 hulls),
`tests/fixture_triang.rs` (placing and fine against the corpus; `check` over every corpus
triangulation and its corruptions) and `tests/topcom.rs` (chirotope, placing and fine
against TOPCOM's CLI). The corpus reader is `tests/common/mod.rs`; each test file includes
it with `mod common;`.

### Phase 1 decisions later phases depend on

**Determinants (`arith`).**

* The API is `arith::sign_of_det(&[i64], n) -> Result<Sign>`; the only error is a
  wrong-length slice. Row- or column-major does not matter (transpose). `n ≤ 2` is answered
  directly in `i128`; from `n = 3` the three tiers run in order:
  `filter::sign_of_det` → `bareiss::sign_of_det_i128` → `bareiss::sign_of_det_bigint`.
  Up to `n = 8` nothing allocates before the bigint tier.
* The filter is **interval Gaussian elimination**, rounding to nearest and widening each
  result outward by one ulp (sound without rounding-mode control; inputs beyond `2^53` enter
  as one-ulp intervals). It declines every singular matrix by construction, so **an exact zero
  always costs one `i128` Bareiss run**. Chirotopes of degenerate configurations are full of
  zeros; if Phase 7 profiling shows that matters, a cheap exact-zero pre-check is the lever.
* **Entries are `i64` only.** Phase 6 needs more: `|p|²` needs `i128`, and scaled `f64`
  weights need bigints (§5.1). The Bareiss code is already generic over an `Exact` trait with
  `i128` and `IBig` instances; Phase 6 should add an entry-type parameter (or a separate
  `sign_of_det` over `i128`/`IBig` entries) rather than squeezing lifts into `i64`. The
  filter's `Interval::from_i64` generalises the same way.
* **`robust` is not used.** Its predicates take `f64` coordinates — exact for integers only
  up to `2^53` — and are point predicates (`orient2d(a, b, c)`), not general determinant
  signs. A `2 × 2` integer determinant is exact in `i128` with no help, and `3 × 3` gains
  little over the filter. It may still pay off in Phase 6 for `orient2d`/`orient3d`/`incircle`
  /`insphere` on lifted lattice points with small coordinates; benchmark it then rather than
  adding it on spec.

**Configurations (`config`).**

* `Config<K: Kind>` with zero-sized markers `Points` and `Vectors`, aliases `PointConfig` and
  `VectorConfig`. Storage is the **homogenised** matrix, column-major `i64`, so a column is the
  `r`-vector the chirotope sees. `ConfigKind` remains as the value-level mirror (`K::KIND`).
* **The OM layer's entry point is `Config::sign_of_simplex(&Simplex)`**: the sign of the
  determinant of the homogenised columns in *increasing label order*, i.e. TOPCOM's `χ(B)`.
  It requires exactly `rows()` labels. `Config::sign_of_columns` takes an explicit order.
  Phase 2's `Chirotope` should wrap a `&Config<K>` (or an `Arc`) and memoise this; it must
  call `require_full_rank()` first, which yields the oracle's "not full rank" rejection as
  `Error::NotFullRank` (the corpus records that rejection for every TOPCOM method on
  rank-deficient input, so the TOPCOM methods should keep it rather than reduce).
* `rank()` is exact (fraction-free elimination, `i128` then bigint) and recomputed per call;
  cache it in the frozen form if it shows up in profiles.
* **Locking** is `config::Lockable<K>`: an `Arc<Config<K>>` plus a flag. The Python classes
  hold one; Phase 2's triangulation methods call `lock()` (which needs `&mut self` on the
  pyclass) and keep the returned `Arc`. The `Triangulation` pyclass will need an enum over
  `Arc<Config<Points>>` / `Arc<Config<Vectors>>`, since it serves both kinds.
* An empty configuration takes its dimension from the first addition (0.3 did this when
  `dim == 0`; the rule is now `len == 0`). Shape and dimension errors are `ValueError`
  (0.3: `RuntimeError`), per §10; adding to a locked configuration is still `RuntimeError`.

**Affine hull (`config::hull`).**

* `reduce_to_hull()` re-expresses the configuration, same labels and order, in the basis of
  the lattice **generated by** the differences `p_i − p_0` (vectors: by the vectors) — the
  column-style HNF of the generators. Not the saturated lattice `aff(P) ∩ Z^d`: normalised
  volumes after reduction are relative to the configuration's own lattice. A full-rank
  configuration is returned unchanged. Coordinates that do not fit `i64` are
  `Error::Overflow`, not a panic.
* It preserves the oriented matroid up to one global sign (tested by embedding random
  full-rank configurations and comparing all chirotope signs), but **not the metric**.
  **Phase 6 must compute lift heights from the original coordinates** (`|p|² − w`) and
  triangulate the reduced points with those heights; that is sound because the regular
  subdivision induced by fixed heights is invariant under affine isomorphisms. Delaunay is the
  same with `w = 0`.

**Simplices (`simplex`).**

* `Simplex` is a bitset stored **inline as a `u128` for labels below 128, spilling to a boxed
  word slice above**; the representation is canonical, so derived `Eq`/`Hash` are correct. It
  is `Clone`, not `Copy` (because of the spill), and cheap to clone inline. There is **no cap
  on the number of points**: the corpus needs at most 16, the exhaustive tier 8, but placing
  and fine triangulations of configurations with hundreds of points are realistic.
  `as_bits()` exposes the `u128` for hand-written fast paths. Phase 7 can measure whether a
  const-generic fixed width beats this.
* `Ord` is **lexicographic on sorted label lists** — the canonical order of §5.6, the same
  order Python gives `sorted(simplex)` lists — computed with bit operations, not by
  materialising lists.
* `SimplexSet` is a `BTreeSet<Simplex>` newtype, so iteration is canonical and two
  triangulations with the same simplices are equal and hash equally. `support`, `star`,
  `link`, `join`, `join_simplex`, `insert_boundary` follow TOPCOM's `SimplicialComplex`
  semantics (including the empty simplex as the link of a listed simplex). This is option (1)
  of [`TOPCOM.md` §2.4](TOPCOM.md); interning simplices per enumeration is Phase 5/7's call.

**Fixtures.** `single_point`'s Delaunay and weighted results are B4-affected but were not
marked `untrusted` (B4 was marked by tag only). `gen_fixtures.py` now also marks a point
configuration whose placing triangulation the oracle rejected as not full rank, and the entry
was regenerated with `--only single_point`; its results are byte-identical, only `untrusted`
changed.

### Phase 2 decisions later phases depend on

**The oriented-matroid layer (`om`).**

* `OrientedMatroid` has three required methods — `n()`, `rank()`,
  `sign(&Simplex) -> Result<Sign>` — and three provided ones that every algorithm so far is
  written in: `sign_appended(facet, x)` (the orientation of the *ordered* basis
  `(facet ascending, x)`, TOPCOM's `perm.sign() * chiro(perm)` idiom), `first_basis()`
  (TOPCOM's `find_non_deg_basis`, lex order) and `lex_extension_sign(prebasis, order)`
  (TOPCOM's `chiro(prebasis, lex_extension_perm)`, used by the triangulation check now and
  by the extension graph in Phase 5). `sign` returns a `Result` so that a malformed query is
  an error rather than a panic; implementers validate with `om::check_basis`. There is a
  blanket impl for `&T`.
* `Chirotope` **owns a copy of the homogenised matrix** (as a `VectorConfig`), so it has no
  lifetime, is `'static`, and does not care whether it came from points or vectors. Its
  cache is a `RwLock<HashMap<Simplex, Sign>>`: `Send + Sync`, owned per chirotope, nothing
  global. `Chirotope::new` (or `Config::chirotope()`) calls `require_full_rank()`, so every
  TOPCOM method rejects rank-deficient input with `Error::NotFullRank`, as the oracle does.
  Every query takes the read lock; if Phase 5 or 7 profiling shows that, the levers are an
  unsynchronised variant behind the same trait, or an eager table indexed by combination
  rank when `C(n, r)` is small.
* `Circuit` (in `om/circuit.rs`, per §6.1) uses TOPCOM's orientation: `positive()` is the
  side of the smallest element of the support. `upper_facets()` drops each element of
  `negative()` and `lower_facets()` each of `positive()`, as `Circuits.cc` does.
* `om::to_topcom_string` prints TOPCOM's chirotope format. Parsing it takes a dozen lines
  (`tests/topcom.rs` has a test-only `Table` implementer); a public parser is all that
  stands between the crate and TOPCOM's `chiro2*` entry points.

**`Triangulation` (`triang`).**

* **Plain data: `{ n, rank, simplices: SimplexSet }`, with no reference to a chirotope or
  configuration.** Algorithms that need signs take an `&impl OrientedMatroid` alongside.
  `rank` is the simplex cardinality, not the configuration's row count, so Phase 6's
  triangulations of hull-reduced rank-deficient configurations (B4) fit unchanged.
* `Triangulation::new(n, rank, set)` checks **shape only** (cardinality, labels in range);
  `from_labels` does the same from a row-major `i64` table and also rejects negative and
  repeated labels and repeated simplices. `check(&om)` is the geometric check. Partial
  triangulations — 0.3's `Flip.pre`/`Flip.post`, which are the simplices a flip removes and
  adds — are therefore representable, and **Phase 3 must not `check` them**.
* `check` is TOPCOM's `CheckTriang`: every simplex a basis; every facet in one simplex with
  nothing beyond it, or in two on opposite sides; and a lexicographic-extension point
  inside the first simplex lies in no other. It is the test oracle for Phases 3–6 too:
  assert it on every flip result, enumerated triangulation and regular triangulation (§8,
  layer 3).
* `unused()` is the complement of the support and `is_fine()` is `unused().is_empty()` —
  the single definition Phases 4 and 5 should filter on. For a regular triangulation
  `unused()` **is** the hidden-point set, so Phase 6's `hidden_points` needs no new field.
* `Eq`/`Hash`/`Ord` are derived and canonical, so a `HashSet<Triangulation>` is a correct
  visited set for Phase 4 (Phase 7 decides whether it is a fast enough one).

**Placing and fine.**

* `placing` keeps the boundary as a `BTreeMap<facet, apex>` instead of recomputing each
  facet's link in the whole triangulation as TOPCOM does: the same result without an
  `O(|T|)` scan per facet. `fine` on 200 random points in dimension 4 takes 0.3 s through
  the Python bindings.
* `fine` scans simplices in canonical order where TOPCOM scans its intern table. The doc
  comment proves the two equivalent (the face an unused point subdivides is unique), and
  306 configurations against the TOPCOM binary confirm it.
* **TOPCOM's fine triangulation is not always fine**, and the port reproduces that: a zero
  vector is never flipped in, and a duplicate *replaces* the copy already used (the two
  form a two-element circuit). In both situations no fine triangulation exists at all, so
  **Phase 4's B1 fix** (seeding `only_fine` with `fine`) should return nothing when the seed
  is not fine — which is exactly `KNOWN_DIVERGENT.md`'s `duplicate_points → 0`.
* Fine's flip-in is a special bistellar flip (the stellar subdivision of a face). Phase 3's
  general flip application can subsume it, but `fine` must keep producing the same output.

**Python.**

* `Triangulation` is a `frozen` pyclass `{ config: SharedConfig, inner: triang::Triangulation }`,
  where `SharedConfig` is an enum over `Arc<Config<Points>>` / `Arc<Config<Vectors>>` taken
  from `Lockable::lock()`. **Phase 3**'s neighbours and `Flip.pre`/`.post` clone the
  `SharedConfig` and wrap a new `inner`; **Phase 6** wraps its result the same way and adds
  `hidden_points` as `inner.unused()`. Methods that need signs build a chirotope from the
  shared configuration per call, which is cheap; caching one in a `OnceLock` on the pyclass
  is the option if flips end up being called in a loop.
* The configuration methods go through one helper, `triangulate(py, &mut lockable, f)`, which
  runs `f` on the chirotope with the GIL released and **locks only on success**: a
  configuration whose triangulation fails (rank-deficient) stays extendable, as in 0.3,
  where the lock was taken by the `Triangulation` constructor.
* `Triangulation(config, simplices, validate=True)` accepts either configuration class (0.3
  accepted only `PointConfiguration`). Simplices must have `config.rank()` labels;
  `validate=True` runs `check` and therefore also needs a full-rank configuration
  (`ValueError` otherwise); `validate=False` keeps only the shape checks. Rows may come in
  any order and are canonicalised. Locking again happens only on success. `__repr__` is
  0.3's, verbatim, and `simplices` is canonical, as the fixtures are.
* There is no `__eq__`/`__hash__` yet; Phase 4 may want them for deduplicating in Python.

**Fixtures.** `tests/fixtures/topcom.json.gz` is generated by
`scripts/gen_topcom_fixtures.py --topcom-bin DIR` from TOPCOM binaries built out of the
pinned checkout. The recipe, a ten-line Makefile needing only GMP (no cddlib), is in the
script's docstring; the build takes about a minute on 8 cores. The CLI tools differ from
0.3's bindings in rejecting zero vectors and rank below two and in silently reducing
rank-deficient input, so the generator only feeds them full-rank configurations without
zero columns (zero vectors are covered by unit tests instead). Extending it with
`points2circuits`, `points2flips` and `points2ntriangs` is the cheapest strong oracle for
Phases 3–5.

### Phase 6 decisions later phases depend on

**Arithmetic.** `arith::sign_of_det_of<T: Entry>` runs the three tiers over `i64`, `i128`
or `IBig` entries (sealed `Entry` trait; the `i64` API is unchanged).
`filter::sign_of_det_of`, `bareiss::sign_of_det_i128_of` and `bareiss::sign_of_det_bigint_of`
are the tiers; `tests/sign_of_det.rs` checks all of them on the existing families and on a
new family of genuinely wide entries. `robust` is still not used: every predicate here is a
general determinant on integers, and the filter decides the generic cases.

**Lifts (`regular::lift`).** The core takes integer lifts (`regular::from_lifts`); heights,
weights and Delaunay are thin wrappers. A `Scalar` is `Float(f64)` (its exact dyadic
value), `Integer(IBig)` or `Rational`; all are brought to a common denominator (lcm) and
carried as `IBig`, kept as `i64` inside the predicates when they fit. `|p|²` comes from the
*original* coordinates. Rejected with `Error::InvalidParameter` (new; `ValueError`):
non-finite and subnormal floats, zero denominators, and scaled lifts wider than
`lift::MAX_LIFT_BITS = 1280` bits — about a thousand binary orders of magnitude of
spread, e.g. `1e-200` next to `1e200`. That cap is a judgement call; revisit it if a real
input hits it.

**Ties: CGAL's rule, not input index.** `CGAL.md` §4.4 recommended perturbing by input
index; that would have produced valid triangulations differing from 0.3 on every
cospherical fixture. CGAL's `perturbed_power_side_of_power_sphere` is itself a consistent
symbolic perturbation of the lifts (`ℓᵢ + ε^π(i)`, the lexicographically largest point by
coordinates raised most), so its output is the *unique* regular triangulation of the
perturbed lifts, independent of CGAL's randomised insertion order. Implementing the same
rule reproduces the oracle on every trusted fixture. The order compares the original
coordinates and breaks ties between duplicates by label (higher label raised more), so
among duplicates with equal lifts the lowest label survives — the deterministic B5 choice.
The infinite-cell tie (query on a hull facet's hyperplane) defers to the finite cell
across the facet, as CGAL does. The cospherical fixtures are therefore *not* ambiguous for
this port: the rule is part of the output contract, and changing it changes results.

**Algorithm.** Affine-hull reduction first (`Config::reduce_to_hull`), then: dimension 0
is the lowest lift; otherwise an initial simplex (first affinely independent points in
label order) with its `d + 1` infinite cells, and every other point inserted in label
order. No spatial sort: 20 000 random points in the plane take about a second through
Python, 1 000 in dimension 4 under half a second; add BRIO/Hilbert ordering only if
profiling asks. The walk is CGAL's remembering visibility walk with a seeded xorshift
start index, capped at one step per live cell (regular triangulations are acyclic for the
in-front relation, so exact predicates never need more); exceeding it is `Error::Internal`.
Hole retriangulation links new cells by hashing their facets
(`Tds::link_unlinked`); an unmatched facet is `Error::Internal`, never a corrupt mesh. The
TDS stores mirror indices as `u8`, hence a dimension cap of 254 (`Error::Overflow`).

**Results.** A plain `triang::Triangulation` with `rank = dim(aff P) + 1`; hidden points
are `unused()`. An empty configuration is `Error::EmptyConfiguration` (0.3 returned an
empty array). Python: `PointConfiguration.delaunay_triangulation()`,
`triangulate_with_weights(seq)`, `triangulate_with_heights(seq)` — a `float` (or numpy
float) is taken as its exact double, an `int` (any size) or `numbers.Rational` such as
`Fraction` exactly, anything else through `__float__` — and `Triangulation.hidden_points`,
a sorted `list[int]` (`unused()`, so for placing triangulations it lists the unused
points). Locking happens only on success, as for the other methods.

### Verified vs not

Verified locally on macOS arm64, Python 3.14: both cargo feature sets build, clippy is clean
with `-D warnings`, `cargo fmt --check` passes, `prek -a` passes all hooks, the extension
imports, pytest collects and skips cleanly, and the fixture generator runs end to end.

Phase 1 was verified locally on Linux x86_64, Python 3.13: `cargo test`, clippy with
`-D warnings` on both feature sets, `cargo fmt --check`, `prek -a`, pytest (6 pass, 10
skip), the 10⁶-matrix harness in release, and `gen_fixtures.py --only single_point`. Phase 1
has **not** yet been through CI; nothing in it is platform-specific, but the `f64` filter's
soundness argument assumes IEEE round-to-nearest, which holds on all four CI targets.

Phase 6 was verified locally on Linux x86_64: `cargo test` (85 tests, 2 ignored),
the ignored 20 000-configuration release fuzz, clippy with `-D warnings` and `cargo doc`
on both feature sets, `cargo fmt --check`, and pytest (28 pass, 5 skip, all skips
Phase 3–5). `prek -a` reports ruff findings only in files this phase did not touch
(`test_triangulation_api.py`, `gen_topcom_fixtures.py`), which are unchanged from Phase 2.

Phase 2 was verified locally on Linux x86_64, Python 3.13: `cargo test` (69 tests),
clippy with `-D warnings` and `cargo doc` with `-D warnings` on both feature sets,
`cargo fmt --check`, `prek -a`, and pytest (16 pass, 8 skip). It has not been through CI
either; it adds nothing platform-specific.

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

cargo test                                      # should pass, 69 tests (1 ignored)
cargo clippy --all-targets -- -D warnings
cargo clippy --features python --all-targets -- -D warnings

uv venv .venv-dev
VIRTUAL_ENV=.venv-dev uv pip install -e '.[test]'
.venv-dev/bin/python -m pytest                  # should give 16 passes, 8 skips

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

### Starting Phase 3

Phases 1 and 2 are done (recorded in §7, decisions in §13). For Phase 3, in order:

1. Extend `scripts/gen_topcom_fixtures.py` with `points2circuits` and `points2flips`, and
   build those two binaries the same way (the recipe is in the script's docstring; the
   TOPCOM checkout is the one from "The upstream sources you will need" above). The core
   corpus has `flips_of_placing` and `neighbors_of_placing` for 31 configurations;
   TOPCOM's own tools give hundreds more, including degenerate ones of the kind that broke
   the fork twice.
2. `src/om/circuit.rs` — `containment_ok` and `link_ok` (`lib-src/TriangNode.cc:20-76`;
   [`TOPCOM.md` §2.5](TOPCOM.md)), next to the `Circuit` that already exists; and a
   `circuits(om)` iterator (TOPCOM's `Circuits`, lex order over `(rank + 1)`-sets) checked
   against `points2circuits`.
3. `src/triang/flip.rs` — `FlipRep` enumeration for a `Triangulation`
   ([`TOPCOM.md` §5.1](TOPCOM.md)), the expanded flip (`Flip.cc:65-115`), and applying one
   to get a neighbour. Naive recompute-all is the plan; incremental `TriangFlips` is
   Phase 7. Assert `Triangulation::check` on every neighbour in the tests.
4. `src/python.rs` — a `Flip` pyclass whose `pre`/`post` are `Triangulation`s sharing the
   `SharedConfig`, built with `triang::Triangulation::new` and **not** checked (they are
   partial), `__repr__` from `git show legacy-cpp:src/triangulumancer/Flip.cpp`; then
   `flips`, `bistellar_flips` and `neighbors`.

Phase 3 is done when `flips_of_placing` and `neighbors_of_placing` match the corpus (as
sets, in canonical order) and the circuits match `points2circuits`.

### A warning about the `__repr__` strings

The 0.3 strings are asserted nowhere in the test suite but are in the README's doctest-style
example, e.g. `A 2-dimensional point configuration with 9 points`. Reproduce them exactly;
they are recoverable from `git show legacy-cpp:src/triangulumancer/PVConfiguration.cpp`.
The configuration strings are done (Phase 1, `impl Display for Config`, unit-tested); the
`Triangulation` and `Flip` ones are in `Triangulation.cpp` and `Flip.cpp`.
