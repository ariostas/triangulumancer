# Migration plan: C++ (CGAL + TOPCOM) → Rust

This document is the overall plan for reimplementing `triangulumancer` in Rust and
retiring its two C++ dependencies. It is the top-level companion to two detailed
design-input documents produced by reading the upstream sources:

* [`CGAL.md`](CGAL.md) — the regular/Delaunay triangulation slice.
* [`TOPCOM.md`](TOPCOM.md) — the placing/fine triangulation, flip and enumeration slices.

Read those for the algorithms. This document covers *why*, *what order*, *what the
result looks like*, and *how we know it is correct*.

> ## ▶ Status: Phases 0–6 complete; Phase 7's engineering is done. **What remains is the 0.4.0 release** — the owner's decisions listed in §13 under "Before releasing" (version and date, the crates.io token and environments, a `workflow_dispatch` dry run of `deploy.yml`, merging to `main`).
>
> Branch `rust-migration`. `src/` has the foundation — exact determinant signs (`arith`),
> integer configurations with exact rank and affine-hull reduction (`config`),
> simplices (`simplex`) — and, from Phase 2, the oriented-matroid layer (`om`: the
> `OrientedMatroid` trait, a lazy `Chirotope`, `Circuit`), the `Symmetry` seam, and
> `triang` (the `Triangulation` type with a full validity check, placing and fine), and,
> from Phase 3, circuits with TOPCOM's flip predicates and `triang::flip` (flip sets,
> expansion, application), from Phase 4 `triang::flipgraph` (a streaming BFS, B1 fixed),
> from Phase 5 `om::cocircuit` (cocircuits, facets, incidences) and `triang::extension`
> (a streaming, splittable DFS over partial triangulations, B2 gone by construction), and
> from Phase 6 `regular/` (Delaunay and regular triangulations replacing CGAL, B3–B5
> fixed). Python ships the whole 0.3 surface — `placing_triangulation`,
> `fine_triangulation`, `Triangulation` with `__eq__`/`__hash__`,
> `flips`/`bistellar_flips`/`neighbors`, `Flip`, `all_connected_triangulations` and
> `all_triangulations` each with a streaming `iter_*` twin, `delaunay_triangulation`,
> `triangulate_with_weights`/`_heights` (exact int/`Fraction` inputs accepted) and
> `Triangulation.hidden_points`. No `NotImplementedError` stub is left. Phase 7 added
> Criterion benchmarks and a comparison against 0.3.0, the performance work of §13
> ("Phase 7 performance": everything TOPCOM did is 3–65× faster, and Delaunay is at or
> ahead of CGAL except in 3-D), and the release preparation (README, CHANGELOG,
> CONTRIBUTING, packaging, `deploy.yml`, crate documentation).
>
> * **§7** — the phase table and what each phase ships.
> * **§13** — what exists, what CI has verified, and **the per-phase decisions later
>   phases build on**.
> * **§14** — setup on a fresh machine, and notes for Phase 5.
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
reduction), B2 in Phase 5 (no global interner, §5.3). All five are now done.

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
│   ├── simplex.rs              # Simplex, SimplexSet, link/join/support/boundary
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
   is the Rust idiom. Handles the overwhelming majority of calls. *(As built: Phases 1–6
   used `[lo, hi]` intervals widened by one ulp after every operation; Phase 7 replaced
   them with Gaussian elimination on `f64` midpoints with a running error bound, whose
   proof is in `src/arith/filter.rs` — see §13, "Phase 7 performance". No `mul_add`:
   without the `fma` target feature it is a library call.)*
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
*(Phase 5: it streams in constant memory, but has more than 8.3 billion triangulations, so a
full count takes hours; see §7.)*

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
| 3 | Circuits, flips and neighbours | `flips`, `bistellar_flips`, `neighbors` | med–hard | ✅ **done** |
| 4 | Flip-graph enumeration | `all_connected_triangulations` | medium | ✅ **done** |
| 5 | Extension-graph enumeration | `all_triangulations` | **highest** | ✅ **done** |
| 6 | Regular and Delaunay triangulations | `delaunay_triangulation`, `triangulate_with_*` | high | ✅ **done** |
| 7 | Parity, performance, release | the 0.4.0 release | low | ◐ **engineering done; release pending** |

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

### ✅ Phase 3 — Circuits, flips and neighbours — **DONE** *(~700 lines of `src/` including docs and unit tests, medium–hard risk)*

* `om::circuit`: `Circuit`, `containment_ok`, `link_ok` — in `om/`, not `triang/`, per §6.1.
* `triang::flip`: `FlipRep`, naive (recompute-all) flip enumeration and application.
* **Ships** `flips()`, `bistellar_flips()`, `neighbors()`.

**Risk:** `containment_ok`/`link_ok` are ~80 lines of dense, uncommented C++ carrying the
entire degenerate-input correctness story, which our TOPCOM fork has already had to patch
twice. **Mitigation:** the exhaustive `n ≤ 8, d ≤ 4` fixture sweep from Phase 0, comparing
full flip sets rather than counts. Cross-check `om::circuit` against `points2circuits`.

**Met.** `tests/fixture_flips.rs` matches `flips_of_placing` and `neighbors_of_placing` on
all 31 corpus entries as full sets in canonical order — 65 flips and 65 neighbours, among
them the 11 of `vc_degenerate_flip` — plus the 6 "not full rank" rejections; every
neighbour passes `check`, and flipping back returns the placing triangulation.
`tests/topcom.rs` grew to 504 configurations (the 306 of Phase 2 plus every full-rank 0/1
configuration of the exhaustive tier — `n ≤ 8`, `d ≤ 4` — standing in for the 0.3
exhaustive sweep, which was never generated) and checks 464 circuit sets (39 600 circuits)
against `chiro2circuits`, and 1 662 flip sets (6 001 flips) against `points2flips` — of
the placing and fine triangulations and up to three more per configuration, from TOPCOM's
own flip-graph enumeration — with `check` on every one of the 6 001 neighbours. Nothing
differed. The Phase 3 decisions are in §13.

### ✅ Phase 4 — Flip-graph enumeration — **DONE** *(~300 lines of `src/` including docs and unit tests, medium risk)*

* `triang::flipgraph`: BFS with a single visited set (simpler than TOPCOM's two-layer
  scheme), streaming per §5.4.
* Seed with `FineTriang` when `only_fine` — fixing **B1**.
* Define `is_fine(T) == (support(T).len() == n)` once and post-filter, so both enumerators
  agree ([`TOPCOM.md` §10.4](TOPCOM.md)).
* **Ships** `all_connected_triangulations(only_fine)`.

**Exit criteria:** matches fixtures, including 387/64 for the 3×3 grid. Note the
known-divergent fixtures from B1 must match the *patched* C++, not `0.3.0`.

**Met.** `tests/flipgraph.rs` matches both methods on every trusted corpus entry the oracle
finished, as sets in canonical order (48 results, 6 "not full rank" rejections), and all
six B1 values in `KNOWN_DIVERGENT.md` — including `square_plus_axes`, where 0.3 never
terminated (64) and `duplicate_points` (0) — each also equal *as a set* to the oracle's
`all_triangulations(only_fine=True)`. Against TOPCOM, the default run checks the 375
components of at most 500 triangulations (302 of them as whole sets, every triangulation
`check`ed); the `#[ignore]`d run checks all 442 `points2ntriangs` counts and all 451
`points2nfinetriangs` counts — 2.9 million triangulations in about four minutes, release
build. In 169 of the latter TOPCOM's fine seed is not fine (duplicate points) and the port
correctly returns nothing (§13). `vc_degenerate_flip` is not enumerated by any test. The
Phase 4 decisions are in §13.

### ✅ Phase 5 — Extension-graph enumeration — **DONE** *(~1 500 lines of `src/` including docs and unit tests, highest risk)*

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

*Revised:* "completes" was written before anyone knew the size of the answer. It has more
than 8.3 × 10⁹ triangulations (below), so no enumeration completes in reasonable time. The
criterion that was met instead is that it **streams in constant memory** — starts at once,
never grows, can be counted, filtered or abandoned, and `par_count` spends every core on it.
The full count is the `#[ignore]`d `vc_degenerate_flip_count` test.

**Met.** `tests/extension.rs` matches all 54 trusted `all_triangulations[_fine]` results of
the core corpus exactly, in canonical order (1 576 triangulations, every one `check`ed; 6
"not full rank" rejections), and the six B1 configurations give the same set as the fixed
flip graph. Against TOPCOM, the default run checks 388 `points2nalltriangs` and 432
`points2nallfinetriangs` counts of at most 1 000 (716 of them as whole sets from
`points2all[fine]triangs`), and that the flip graph gives the same set on all 796 of those;
the `#[ignore]`d run checks every count TOPCOM finished — 456 and 461, 15.7 million
triangulations, 92 s in release and 300 MB at peak (results above 200 000 are counted
rather than collected; collecting two million `Triangulation`s takes gigabytes). No configuration in either corpus has a disconnected flip
graph: TOPCOM's own `points2ntriangs` and `points2nalltriangs` agree wherever both
finished, and the extension and flip graphs give identical sets. `tests/topcom.rs` adds
cocircuits (`chiro2cocircuits`) and facets (`points2facets`) for all 504 configurations.
**`vc_degenerate_flip`: streams and counts in constant memory, but does not finish in
reasonable time.** It is not a 0.3-sized problem: it has **more than 8.3 billion
triangulations and more than 4.3 billion fine ones** — the counts `par_count` reached in
44 minutes on five and three threads (about 3.1 M/s and 1.6 M/s on a heavily loaded
8-core machine) before the runs were stopped, with resident memory flat at about 4 MB
throughout. An earlier 68-minute six-thread run was also stopped unfinished. TOPCOM
itself managed 75 million in 22 minutes (one thread, `-O1`) and was still on its first
top-level branch. So the oracle's failure was never only memory: the full count is
hours of CPU even here. What the phase delivers is the property that matters — the
enumeration of an arbitrarily large configuration starts at once, never grows, and can
be counted, filtered or abandoned — plus `par_count` to spend all cores on it. The full
count is the `#[ignore]`d `vc_degenerate_flip_count`; the default suite checks its tables
(1 434 simplices, as TOPCOM) and streams its first 1 000 triangulations, and pytest
streams 100 through Python.

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

### ◐ Phase 7 — Parity, performance, release — **engineering DONE; release pending**

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

**Status.** Everything but the release itself is done: the differential suites are green,
the benchmarks exist and are recorded against 0.3.0, incremental flips landed behind a
debug-assert, the documentation and packaging are ready (§13, "Phase 7 release
preparation" and "Phase 7 performance"). The parallel flip-graph BFS was considered and
not done (reasons in §13). The release steps are the owner's: see "Before releasing" in
§13.

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
   should at minimum be documented. *(Phase 4: documented in the Python docstring and in
   `triang::flipgraph`; the name is unchanged.)*
3. **`bnum` vs `dashu-int`** for the fallback tier — resolve by benchmark in Phase 7, after
   §5.1's API decision has fixed the bit bound.
   *(Phase 7: keep `dashu-int`. Bareiss over `bnum::BInt` was slower at every size measured,
   and a fixed width cannot cover scaled `f64` weights anyway; see "Phase 7 performance" in
   §13.)*
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

**Phases 0–6 are complete, and so is Phase 7's engineering**: its release preparation
(see "Phase 7 release preparation" below) and its performance work ("Phase 7
performance" below). What remains is the release itself, which needs the owner's
decisions ("Before releasing", below).
Phase 6 was developed in parallel on a separate branch and merged.
The phase table in §7 is the at-a-glance version; this section is the detail.

| | |
|---|---|
| Branch | `rust-migration` — nothing is on `main` |
| The old C++ | tag `legacy-cpp` (`c643162`); `git show legacy-cpp:src/triangulumancer/CGAL.cpp` etc. |
| PyPI | still serves `0.3.0`; the crate version is `0.4.0-dev` |
| Rust | `cargo build` / `test` / `clippy` / `doc` clean on both feature sets; 135 tests (with the crate doctest), plus the `#[ignore]`d 10⁶-matrix, full flip-graph, 20 000-configuration regular-fuzz, full extension-graph, `vc_degenerate_flip`, 600-configuration brute-force, long invariance and 50 000-point Delaunay-memory runs |
| Python | `pytest`: 62 pass, no skips |
| Fixtures | `tests/fixtures/core.json.gz`, 31 configurations, 25 KiB, oracle `0.3.0`; `tests/fixtures/topcom.json.gz`, 504 configurations, ~245 KiB, TOPCOM's own CLI tools |

### What exists in `src/`

| Module | Contents |
|---|---|
| `lib.rs` | `ConfigKind`, and the crate documentation (docs.rs front page: layout, the OM seam, a doctested example, features) |
| `error.rs` | the `#[non_exhaustive] Error` enum; Phase 1 added `InvalidShape` and `IndexOutOfRange`, Phase 6 `InvalidParameter` |
| `arith.rs`, `arith/{filter,bareiss}.rs` | `Sign`, `SquareMatrix`, `sign_of_det`, `rank`; the three tiers, each public on its own |
| `config.rs`, `config/hull.rs` | `Config<K: Kind>` with markers `Points`/`Vectors`, `Lockable<K>`, `reduce_to_hull` |
| `simplex.rs` | `Simplex` (inline `u128` bitmask, sorted label list above 128), `SimplexSet` (canonically ordered set of simplices), `combinations` (lex-order `k`-subsets) |
| `om/mod.rs` | the `OrientedMatroid` trait; `check_basis`; `to_topcom_string` (`points2chiro` format) |
| `om/chirotope.rs` | `Chirotope`, lazy and memoised; `Config::chirotope()` |
| `om/circuit.rs` | `Circuit` with TOPCOM's orientation, `upper_facets`/`lower_facets`, `containment_ok`, `link_ok`, `common_link`; `circuits(om)` |
| `om/cocircuit.rs` | `Cocircuit` with TOPCOM's orientation, `cocircuits(om, only_positive)`, `Facets`, `Incidence`/`Incidences` (interior and boundary facets of each basis) |
| `symmetry.rs` | `Symmetry`, `TrivialGroup` — as §6.2, verbatim |
| `triang/mod.rs`, `triang/check.rs` | `Triangulation`; `check`, the port of TOPCOM's `CheckTriang` |
| `triang/placing.rs`, `triang/fine.rs` | `triang::placing`, `triang::fine` |
| `triang/flip.rs` | `FlipRep`, `Flip`, `flip_reps` (naive), `flips`, `neighbors` |
| `triang/flipgraph.rs` | `FlipGraph` (streaming BFS iterator, generic over `Symmetry`), `all_connected` |
| `triang/extension/{mod,admissibles,bits}.rs` | `ExtensionGraph` (streaming DFS iterator, generic over `Symmetry`, `count` and multi-threaded `par_count`), `all_triangulations`; `Admissibles` (the bitset compatibility table) |
| `regular/{mod,lift,predicates,tds,insert}.rs` | `delaunay`, `with_weights`, `with_heights`, `from_lifts`; `Scalar` exact lifts; CGAL-compatible symbolic perturbation; cell arena; walk + conflict-zone insertion |
| `python.rs` | configuration classes, `placing_triangulation`, `fine_triangulation`, `Triangulation` with `flips`/`bistellar_flips`/`neighbors`/`__eq__`/`__hash__`, `Flip`, `all_connected_triangulations`, `iter_all_connected_triangulations`, `all_triangulations`, `iter_all_triangulations` and `TriangulationIterator` (over either search); CGAL-replacement methods and `hidden_points`; no `todo()` stub is left |

§4's layout is complete except `om/gale.rs` (reserved); `benches/` landed in Phase 7.

Tests: `tests/sign_of_det.rs` (the differential harness; run the full 10⁶ with
`cargo test --release --test sign_of_det -- --ignored --nocapture`, about 40 s),
`tests/fixture_rank.rs` (rank against the corpus, and the B4 hulls),
`tests/fixture_triang.rs` (placing and fine against the corpus; `check` over every corpus
triangulation and its corruptions), `tests/fixture_flips.rs` (flips and neighbours of
placing against the corpus), `tests/flipgraph.rs` (the flip graph against the corpus, the
B1 values and TOPCOM's enumerators), `tests/extension.rs` (the extension graph against the
corpus, TOPCOM's `points2[n]all[fine]triangs` and the flip graph; `vc_degenerate_flip`) and `tests/regular.rs` (regular/Delaunay against the corpus, B3–B5 values, brute-force oracle fuzz) and `tests/topcom.rs` (chirotope, placing, fine,
circuits, flip sets, cocircuits and facets against TOPCOM's CLI). The corpus reader is `tests/common/mod.rs`; each test file includes
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
  as one-ulp intervals). *(Phase 7 replaced the intervals by midpoints with a running
  error bound; same decisions, about a third of the cost. See "Phase 7 performance".)* It declines every singular matrix by construction, so **an exact zero
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
  hold one behind a `Mutex` (`ConfigCell` in `python.rs`; the pyclasses are `frozen`, every
  method takes `&self`). A computation takes `snapshot()` (an `Arc` clone) under the mutex,
  releases it, computes with the GIL released, and then calls `lock_snapshot(&snapshot)`,
  which locks only if the configuration is still that `Arc` and otherwise fails with
  `Error::ModifiedDuringComputation` (`RuntimeError`), leaving it unlocked. `modify()` is
  copy-on-write (`Arc::make_mut`), so adding points mid-computation never changes the
  snapshot. Earlier phases held a `&mut self` pyclass borrow across `py.detach`, so any
  other thread touching the configuration during a computation got `RuntimeError: Already
  mutably borrowed`; `python/tests/test_threads.py` covers reads, concurrent computations
  and modification during a computation. The `Triangulation` pyclass holds
  `SharedConfig`, an enum over `Arc<Config<Points>>` / `Arc<Config<Vectors>>`.
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

* `Simplex` is stored **inline as a `u128` bitmask for labels below 128, spilling to a boxed
  sorted label list (`Box<[usize]>`) above**; the representation is canonical, so derived
  `Eq`/`Hash` are correct. It is `Clone`, not `Copy` (because of the spill), and cheap to
  clone inline. There is **no cap on the number of points**: the corpus needs at most 16, the
  exhaustive tier 8, but placing and fine triangulations of configurations with hundreds of
  points, and Delaunay triangulations of hundreds of thousands, are realistic.
  `as_bits()` exposes the `u128` for hand-written fast paths.
* **A spilled simplex costs O(cardinality), not O(n).** The spill used to be an `n`-bit
  bitset, so every simplex of a large triangulation was as wide as the configuration and a
  Delaunay triangulation cost O(n²): 40 000 random 3-D points peaked at 1.1 GB through
  Python. With label lists the same run peaks at 78 MB, and 100 000 points at 150 MB. Set
  operations on spilled simplices are sorted merges (inline/spilled mixes take a bitmask
  shortcut where possible), `contains` is a binary search, `Extend`/`FromIterator` sort
  once, and `SimplexSet::support` marks labels in a flat table rather than taking repeated
  unions. Guarded by `simplex::tests::spilled_simplices_cost_their_cardinality`, the
  `#[ignore]`d `large_delaunay_memory_is_linear` in `tests/regular.rs` (50 000 points, peak
  64 MiB in release) and `test_large_delaunay_memory_is_linear` in pytest (20 000 points,
  peak under 200 MiB, in a subprocess).
* `Ord` is **lexicographic on sorted label lists** — the canonical order of §5.6, the same
  order Python gives `sorted(simplex)` lists — computed with bit operations for two inline
  simplices, and by comparing the label lists otherwise.
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
* The configuration methods go through one helper, `ConfigCell::compute(py, f)`, which
  runs `f` on a snapshot with the GIL released and **locks only on success**: a
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

### Phase 3 decisions later phases depend on

**Circuits (`om::circuit`).**

* `containment_ok`, `link_ok` and `common_link` are methods on `Circuit` that take a
  `&SimplexSet`, not a `Triangulation` — `om` must not depend on `triang`. `link_ok` takes
  the rank because TOPCOM's returns true for a general-position circuit; the flip code only
  calls it in special position, so that early return is reproduced, not relied on.
  `common_link` returns the link it checked, so expanding a flip does not recompute it.
* `om::circuits(om)` is TOPCOM's `Circuits`: every `(rank + 1)`-subset, one circuit per
  support, ordered by support. The reference is **`chiro2circuits`, not
  `points2circuits`**: given points, this TOPCOM revision enumerates circuits with a
  subset-graph search whose output is oriented and labelled differently (it is not a
  relabelling we could undo cheaply), while `chiro2circuits` uses the same
  `Circuit(chiro, dependent_set)` the flip code does. Phase 5's `Admissibles` should be
  checked against `chiro2circuits` too.

**Flips (`triang::flip`).**

* `FlipRep { removed, added }` is TOPCOM's `(first, second)`: the circuit oriented so that
  the removed side's facets are present. Constructing one guarantees both sides non-empty,
  so `kills_vertex`/`adds_vertex`/`is_balanced` are TOPCOM's. `FlipRep` is independent of
  the triangulation; `FlipRep::expand(&t)` gives `Flip { removed, added, rep }` (the
  simplices), and `Flip::apply(&t)` the neighbour.
* `flip_reps(om, t, forbid_vertex_removal) -> BTreeSet<FlipRep>` recomputes the whole set
  from TOPCOM's candidate sets — `S ∪ {v}` for unused `v`, and `S ∪ S'` for facet-adjacent
  simplices — but finds adjacent pairs through a facet → simplices map rather than
  TOPCOM's `O(|T|²)` scan (its `count > rank` early break never fires on a triangulation, so
  the candidates are the same). Candidates are deduplicated before the circuit is built.
  This is the oracle **Phase 7's incremental `TriangFlips`** must agree with; it takes the
  same `forbid_vertex_removal` flag Phase 4 needs, and a restriction set would be the
  natural extension.
* Every flip of every triangulation in the tests is applied and the result `check`ed; no
  TOPCOM flip has produced an invalid triangulation, including the special-position ones.
  `FlipRep::expand` against a triangulation the flip does not belong to is well-formed but
  meaningless, and Python never does it.
* Order: `flips()` is sorted by `(removed, added)` simplex lists — exactly how the fixtures
  sort `pre`/`post` — and `neighbors()` canonically. `flip_reps` is ordered by
  `(removed side, added side)`, which also fixes the BFS order in Phase 4.

**Python.** `Flip` is a frozen pyclass `{ config: SharedConfig, pre, post }` whose getters
wrap new `Triangulation`s (unchecked, partial). `flips`/`neighbors` build a chirotope per
call and run with the GIL released. `Triangulation.flips()` on a triangulation built with
`validate=False` is whatever TOPCOM's predicates make of it — no panic, no promise.

**Fixtures.** `gen_topcom_fixtures.py` now also needs `chiro2circuits`, `points2flips`,
`points2ntriangs`, `points2nfinetriangs` and `points2triangs` (the last three for Phase 4;
recorded now so the file is generated once). Those link `ComputeTriangs.cc`, which pulls in
cddlib's headers through `RegularityCheck.hh`; the docstring describes a stub header that
avoids cddlib, since regularity is never requested. Each TOPCOM process runs under a 1 GiB
address-space cap and enumerations under a 20 s timeout (`null` when either is hit) —
**a parallel run without the cap exhausted a 13 GB machine**. Configurations with
`n == rank` are skipped by the enumeration tools, which fail outright on them. Generation
takes about 5 minutes on 4 cores.

### Phase 4 decisions later phases depend on

**The search (`triang::flipgraph`).**

* `FlipGraph<O, S = TrivialGroup>` **owns** its oriented matroid (`O` may be `&Chirotope`
  through the blanket impl, or an owned `Chirotope` — which the Python iterator needs) and
  is an `Iterator<Item = Result<Triangulation>>`. It is lazy: a triangulation's flips are
  computed only when the yielded queue runs dry, so `next()` does one node's work at most.
  Order is breadth-first, neighbours in `flip_reps` order, hence deterministic.
  `all_connected(om, only_fine)` collects and sorts canonically; it is what the list API
  returns. After an error the iterator yields nothing more.
* One `HashSet<Triangulation>` of everything seen; no flip marking. Memory is therefore
  proportional to the output (plus the frontier), which streaming does not change — only a
  canonical spanning tree (reverse search) would. Throughput is about 80 µs per
  triangulation in release on the TOPCOM corpus, with naive flip sets and
  `BTreeSet`-based triangulations; **Phase 7**'s incremental flips and interning are the
  levers, and `FlipGraph::expand` is where an incremental `TriangFlips` slots in (it has
  the parent, the flip and the child at hand).
* `only_fine`: seed with `fine`, pass `forbid_vertex_removal`, and **yield nothing when the
  seed is not fine** — `FlipGraph::from_seed` applies the same rule to any seed, so the
  result is always the seed's component in the subgraph induced on fine triangulations.
  This is B1 fixed, and it is why the result for repeated points is empty where
  `points2nfinetriangs` is not. Phase 5's `only_fine` must agree with this wherever the
  fine flip graph is connected.
* **Symmetry**: generic over `S: Symmetry`; `stabilizer` is called where TOPCOM adds
  equivalent flips and `is_orbit_representative` filters output. The visited set is not
  orbit-aware, so a real group would also need an orbit-canonical form (not in the §6.2
  trait). With `TrivialGroup` both calls are constants.

**Python.**

* `all_connected_triangulations(only_fine=False)` runs with the GIL released, returns
  canonical order, and locks the configuration on success.
* `iter_all_connected_triangulations(only_fine=False)` computes the chirotope and seed
  eagerly (so a rank-deficient configuration raises at the call and stays unlocked), locks,
  and returns a `TriangulationIterator`: a frozen pyclass holding the `SharedConfig` and the
  `FlipGraph<Chirotope>` behind a `Mutex`. Each `__next__` releases the GIL and runs the
  search to the next triangulation; threads sharing one iterator take turns. Order is BFS,
  not canonical.
* `Triangulation.__eq__` is true for the same simplices of the **same configuration
  object** (`Arc::ptr_eq`) — configurations have identity, not value, semantics in 0.3 —
  and `__hash__` hashes the simplices only. Comparison with a non-`Triangulation` is
  `False`.

### Phase 5 decisions later phases depend on

**Cocircuits, facets, incidences (`om::cocircuit`).**

* `Cocircuit` mirrors `Circuit`: TOPCOM's orientation (`positive()` is the side of the
  smallest element off the hyperplane), built by `Cocircuit::of_hyperplane(om, coplanar)`
  from `sign_appended`. `cocircuits(om, only_positive)` is one per support, ordered by
  support, and matches `chiro2cocircuits` exactly (all 504 TOPCOM configurations).
* `Facets` is the sorted list of facets (label sets, possibly larger than `rank - 1`) from
  the positive cocircuits; it matches `points2facets` on all 504 TOPCOM configurations,
  which computes them a different way (from the placing triangulation's boundary). A
  non-pointed vector configuration has none, and then every facet is interior.
* `Incidences` keeps only TOPCOM's `_intfacets`/`_conffacets` half (per basis: interior
  facets, and the configuration facets holding the others). The reverse maps feed only
  `--fullextensioncheck` and regularity checks; add them with those.

**The search (`triang::extension`).**

* `Admissibles` holds the universe (every basis, or every empty one) in canonical order —
  which *is* TOPCOM's simplex index order — and a `|U| × |U|` bit table built per circuit
  as `star(Z⁺) × star(Z⁻)` rather than TOPCOM's `|U| × #circuits` loop. Memory is
  `|U|²/8` bytes (264 KB for `vc_degenerate_flip`'s 1 434 simplices; ~100 MB at
  `|U| ≈ 27 000`, e.g. 30 points in rank 4). That is the scaling limit, as in TOPCOM.
* `ExtensionGraph` owns its tables and **not** the oriented matroid: all chirotope work
  happens in the constructor, so the search cannot fail and yields plain `Triangulation`s
  (`Item = Triangulation`, unlike `FlipGraph`'s `Result`). Python wraps it with `map(Ok)`.
* The DFS is an explicit stack of per-level bitsets (admissibles over the universe, free
  interior facets over the canonically numbered interior facets). The `forbidden` set is
  not materialised: since every sibling before `s` is forbidden and the parent's
  admissibles before `s` are exactly those siblings, a child's admissibles are
  `parent ∩ Adm[s] ∩ (s, ∞)`. Uniqueness is therefore structural and was confirmed by
  deduplication tests. The lexicographic cut and TOPCOM's `_noncoverable_freefacet_lex`
  both reduce to integer comparisons via a precomputed `first_coverable[s]` (the number of
  interior facets below `s`'s smallest facet). TOPCOM compares against the facet of
  smallest *index*; using the true lex-minimum is the same or stronger and still sound
  (any free facet below `s`'s smallest facet is uncoverable from `s` on).
* `only_fine` is the empty-simplex universe, TOPCOM's rule verbatim (including that a
  circuit with an empty side — a zero vector — marks every simplex non-empty). On top of
  it, when some label is in no empty simplex (duplicates, zero vectors) the search is
  empty, and every emitted triangulation is checked fine anyway; both agree with Phase 4's
  "nothing when no fine triangulation exists", and `points2nallfinetriangs` returns 0 in
  every such case.
* Output order of the iterator is DFS order (deterministic, not canonical);
  `all_triangulations` sorts. `Iterator::count` is overridden to skip building
  triangulations (~1.6 M/s on `vc_degenerate_flip`, against ~0.5 M/s when materialised and
  ~88 k/s for TOPCOM at `-O1`).
* **`par_count(threads)`** (new, not in TOPCOM's defaults): `std::thread::scope` workers
  share a queue; a busy worker checks every 4 096 nodes whether anyone is idle and, if so,
  donates the untried siblings of its shallowest open level as a new search
  (`Search::split`). No dependency was added. Scaling was not measured cleanly: five threads
  reached about 2.5 times the single-thread rate on an 8-core machine already at load 15;
  the same splitting is the natural basis for a parallel *enumeration* (with a
  channel) in Phase 7 if wanted. Only counting is parallel today.
* **Symmetry**: generic over `S: Symmetry`; `is_orbit_representative` filters output, and
  `count` takes the fast path when `is_trivial()`. TOPCOM's real-group pruning
  (`child_is_lexmin` on *partial* triangulations) has no trait method yet; whoever adds a
  group should add one (`is_lexmin_extension(&partial, new_simplex)` or similar) and call
  it where `ExtensionGraph` builds a child.
* `Triangulation::from_trusted` (crate-private) skips the shape check, debug-asserted.

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
input hits it. The cap is **invariant under a common shift and a common positive scale**:
`scale` subtracts the smallest lift and divides out the numerators' common factor before
taking the common denominator (neither changes any predicate sign nor the symbolic
perturbation), so heights all equal to `2**1300` are accepted as all-zero heights, and
`[2**1300, 2**1300 + 5, …]` as `[0, 5, …]`. (It was first checked on the raw lifts, which
rejected those.) The common denominator is checked as it grows — past
`MAX_LIFT_BITS + max denominator bits` every non-zero lift is already too wide — so many
large coprime denominators fail fast instead of building a huge lcm. `from_heights` and
`from_weights` therefore return normalised lifts (minimum zero); `from_lifts` itself takes
any integers, uncapped.

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
`Fraction` exactly (ints are read through `__index__` in hexadecimal, so Python's
4 300-digit limit on decimal conversion does not apply), anything else through
`__float__` — and `Triangulation.hidden_points`,
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

Phase 3 was verified the same way: `cargo test` (78 tests), clippy and `cargo doc` with
`-D warnings` on both feature sets, `cargo fmt --check`, `prek -a` (which also needed
three pre-existing `PT011`/formatting fixes in `test_triangulation_api.py`), and pytest
(20 pass, 7 skip).

Phase 4 likewise: `cargo test` (85 tests; the flip-graph test takes about 6 s in a debug
build), the `#[ignore]`d full TOPCOM comparison in release, clippy and `cargo doc` with
`-D warnings` on both feature sets, `cargo fmt --check`, `prek -a`, and pytest (28 pass,
7 skip). The two 0.3 tests asserting 387/64 still report as skipped because they go on to
call `all_triangulations`; their flip-graph assertions run and pass first.

Phase 5 likewise: `cargo test` (119 tests, 5 ignored; `tests/extension.rs` takes about
30 s in a debug build, most of it the flip-graph cross-check), the `#[ignore]`d full
TOPCOM comparison in release (92 s), clippy and `cargo doc` with `-D warnings` on both
feature sets, `cargo fmt --check`, `prek -a`, and pytest (52 pass, 0 skip; the 387/64 and
131/1 tests now run to the end). Every heavy command ran under
`systemd-run --user --scope -p MemoryMax=3G` — see the warning in §14.

**Post-Phase-6 review fixes** (Linux x86_64, Python 3.13): `Simplex` spills to a sorted
label list (O(rank) per simplex; Delaunay of 40 000 random 3-D points through Python went
from a 1 116 MB to a 78 MB peak, 100 000 points take 150 MB and 13 s; placing, fine and
flips on 200–300 points are about 10 % slower than with the bitset spill); configuration
pyclasses are `frozen` with the `Lockable` behind a mutex, computations run on a snapshot
(no more `Already mutably borrowed` from other threads); the lift cap is shift- and
scale-invariant; Python ints are converted through hexadecimal, so heights beyond 4 300
decimal digits no longer hit Python's int-to-str limit. Ported from an adversarial review:
`tests/brute_force.rs` (both enumerations, `check`, neighbour symmetry and the fine
subsets against an independent brute-force oracle on degenerate lattice configurations),
`tests/regular_invariance.rs` (Delaunay under similarities up to `2^60`, relabelling,
coordinates near the `i64` limit against brute force, `f64` weights over the whole
exponent range), `python/tests/test_bad_input.py` and `python/tests/test_threads.py`.
Verified: `cargo test` (128 tests, 8 ignored; the ignored brute-force, invariance and
memory runs pass in release), clippy with `-D warnings` on both feature sets,
`cargo fmt --check`, `prek -a`, pytest (62 pass, 0 skip).

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
* **`deploy.yml`.** Only runs on a published release or `workflow_dispatch`, so none of the
  cibuildwheel configuration has executed. It was reviewed and corrected in Phase 7 (below),
  and the wheel it builds was reproduced locally with maturin, but cibuildwheel itself has
  not run. Run it once with `workflow_dispatch` before the release; it publishes nothing
  then.

### Phase 7 release preparation

Done on Linux x86_64, Python 3.13, from a clean export of `db6f9ec` plus these changes:

* **`README.md`** rewritten for 0.4.0: Python usage for every method, including the 0.4
  additions, a Rust example, and the one-paragraph licence section of §4. The 0.3 example
  is kept with its `__repr__` strings verbatim (and its two missing echo lines restored, so
  it is a valid doctest). Every `>>>` example passes `python -m doctest -o ELLIPSIS` against
  the built wheel once the code fences are stripped; `python.yml` and `nox -s readme` now
  run that. The Rust example was compiled and run against the crate.
* **`CHANGELOG.md`**: the migration notes, listing every behaviour a 0.3 user can observe
  changing (§10, the B1–B5 fixes, and the per-phase Python decisions above).
* **`CONTRIBUTING.md`**: setup, checks, layout and conventions, for human contributors.
* **`cargo package`**: `exclude` replaced by an `include` whitelist — `src/`, the Rust
  tests and the two committed fixture files, `README.md`, `CHANGELOG.md`, `LICENSE`,
  `docs-header.html` and the lockfile. 50 files, 766 KiB (399 KiB compressed). `cargo test
  --locked` passes from the unpacked `.crate`, so the published source is self-testing.
  Benchmarks are left out; cargo drops their `[[bench]]` targets from the packaged
  manifest. `rust.yml` now runs `cargo package` and `cargo doc` with `-D warnings` and the
  KaTeX header (Ubuntu only), and `nox -s package` / `nox -s docs` do the same locally.
  The header is harmless but currently unused: no doc comment uses `$…$` or `\(…\)`.
* **Wheel and sdist**: `maturin build --release --locked` gives a 0.66 MB
  `cp310-abi3-manylinux_2_34_x86_64` wheel (the extension, `__init__.py`, `py.typed`,
  licence and SBOM only); installed into a fresh venv, it imports and passes the 62 pytest
  tests. `maturin sdist` (415 KB) includes the Python package but not `python/tests`;
  building and installing from it in a fresh venv also passes them.
* **`deploy.yml`**, as reviewed: `CIBW_BUILD` pinned to `cp310-* cp314-* cp314t-*` (the
  abi3 wheel built once on 3.10, re-tested by cibuildwheel on 3.14, plus the free-threaded
  wheel — §5.5 — and no `cp313t`); 32-bit Windows skipped (the runners' Rust has no
  `i686-pc-windows-msvc`, which would have failed the build); the container-only `PATH`
  moved to `CIBW_ENVIRONMENT_LINUX`, since `$HOME/.cargo/bin:$PATH` with a `:` would
  corrupt `PATH` on Windows; tests run `{project}/python/tests` with the `test` extra;
  `macos-15-intel` added for x86_64 macOS wheels, which 0.3 never had; musllinux wheels
  are built as cibuildwheel's default. `deploy-crate` passes the token as
  `CARGO_REGISTRY_TOKEN` instead of `cargo login` and drops the unused `id-token`.
* **`[tool.uv] cache-keys`** also lists `Cargo.lock` (§9).

**Before releasing** (the owner's decisions; nothing here was pushed, tagged or
published):

* Set the version: `Cargo.toml` is `0.4.0-dev`, which maturin publishes as `0.4.0.dev0`
  and `__version__` reports as `0.4.0-dev`. Fill in the date in `CHANGELOG.md`.
* Create the `crates.io` environment and the `CARGO_REGISTRY_TOKEN` secret (a new crate's
  first publish needs a token; trusted publishing can replace it afterwards), and check the
  `pypi` environment still has trusted publishing for this repository.
* Remove `rust-migration` from the triggers of `rust.yml` and `python.yml` on merge.
* Decide §12 question 4.

### Phase 7 performance

Measured on Linux x86_64 (8 cores, other work running), release builds, every command
under the memory cap of §14.

**Benchmarks.** `benches/` has three Criterion suites (`cargo bench --bench arith|triang|regular`)
over seeded workloads in `benches/common/mod.rs`: the determinant tiers separately and
together (generic, singular, wide and `IBig` entries, 3×3 to 12×12); placing and fine
(100 points in 3-D, 60 in 4-D, 300 in 3-D with spilled simplices); warm `flip_reps` and
cold `neighbors`; the flip-graph BFS (3×4 grid, all and fine); the extension graph (count
and iterate the 3×4 grid, setup and 100 000 items of `vc_degenerate_flip`); visited-set
hashing; Delaunay in 2, 3, 5 and 10 dimensions and on a 3-D lattice; regular with `f64`
weights. `benches/compare_oracle.py` times the same configurations (same SplitMix64 stream)
through Python, each workload in a subprocess, and prints a table from two runs; its
docstring shows how to set up the `0.3.0` venv.

**Against 0.3.0 through Python** (median; both include building the configuration). Taken
at the end of Phase 7 on an otherwise idle machine; 0.3.0's own numbers came out lower than
in an earlier, loaded run (e.g. 1.23 s rather than 1.67 s in 10-D), so only compare rows
within this table:

| workload | 0.3.0 | 0.4.0-dev | |
|---|---:|---:|---:|
| `placing_triangulation`, 100 random points in 3-D | 54.6 ms | 1.14 ms | 48× |
| `fine_triangulation`, same | 230 ms | 3.87 ms | 59× |
| `placing_triangulation`, 60 random points in 4-D | 162 ms | 2.75 ms | 59× |
| `fine_triangulation`, same | 269 ms | 4.78 ms | 56× |
| `neighbors()` of a fine triangulation, 40 points in 3-D | 8.04 ms | 0.744 ms | 11× |
| `all_connected_triangulations()`, 3×4 grid (14 842) | 555 ms | 189 ms | 2.9× |
| `all_connected_triangulations(only_fine=True)`, same | 21.6 ms | 5.73 ms | 3.8× |
| `all_triangulations()`, same (14 842) | 76.7 ms | 5.69 ms | 13× |
| `delaunay_triangulation()`, 10 000 random points in 2-D | 99.4 ms | 42.6 ms | 2.3× |
| same, 2 000 random points in 3-D | 32.0 ms | 37.6 ms | 0.85× |
| same, 2 000 points of a 3-D lattice (range 8) | 12.4 ms | 22.3 ms | 0.56× |
| same, 300 random points in 5-D | 164 ms | 162 ms | 1.0× |
| same, 40 random points in 10-D | 1.23 s | 1.09 s | 1.1× |
| `triangulate_with_weights` (`f64`), 2 000 points in 3-D | 28.7 ms | 42.3 ms | 0.68× |

Everything TOPCOM did is faster. Delaunay is ahead of CGAL in 2-D and level with it in 5-D
and 10-D (it was 2.6× behind there before the filter was replaced; see below); CGAL is still
ahead in 3-D, on lattices and with `f64` weights. Interleaved runs pinned to one CPU under
load agreed: 1.09× in 5-D, 1.13× in 10-D. `vc_degenerate_flip` has no 0.3 column: 0.3
runs out of memory on it.

**Rust before and after Phase 7** (Criterion, the same machine; the Delaunay and regular
"after" column is before the filter replacement below, which has its own table):

| benchmark | before | after |
|---|---:|---:|
| `construction/placing/random_100x3` | 2.81 ms | 0.96 ms |
| `construction/fine/random_100x3` | 12.2 ms | 3.70 ms |
| `construction/fine/random_60x4` | 13.0 ms | 8.93 ms |
| `construction/fine/random_300x3` (spilled) | 111 ms | 64.2 ms |
| `flips/flip_reps/random_40x3_warm` | 191 µs | 135 µs |
| `flips/neighbors/random_40x3_cold` | 1.05 ms | 0.41 ms |
| `flip_graph/bfs_all/grid_3x4` | 686 ms | 181 ms |
| `flip_graph/bfs_fine/grid_3x4` | 20.4 ms | 5.63 ms |
| `extension/count/grid_3x4` | 2.47 ms | 1.97 ms |
| `extension/iterate/grid_3x4` | 7.08 ms | 2.79 ms |
| `extension/iterate_100k/vc_degenerate_flip` | 175 ms | 65.1 ms |
| `hashing/hashset_insert/grid_3x4` (std hasher) | 10.8 ms | 5.47 ms |
| `delaunay/random_10000x2` | 259 ms | 42.9 ms |
| `delaunay/random_2000x3` | 110 ms | 39.1 ms |
| `delaunay/lattice_2000x3_range8` | 83.7 ms | 23.7 ms |
| `delaunay/random_300x5` | 485 ms | 473 ms |
| `delaunay/random_40x10` | 4.30 s | 4.31 s |
| `regular/float_weights_2000x3` | 118 ms | 60.8 ms |

Iterating the extension graph materialises about 1.5 M triangulations/s on
`vc_degenerate_flip` (was 0.57 M/s); counting is unchanged at its 1.6 M/s order, so building
a `Triangulation` is no longer the iterator's bottleneck. The flip-graph BFS is about 12 µs
per triangulation on the 3×4 grid (was 46 µs). The 50 000-point Delaunay memory test peaks
at 53 MiB (was 64).

**What changed, and the decisions later work builds on.**

* **`SimplexSet` is a strictly increasing `Vec<Simplex>`**, not a `BTreeSet`. Building
  (`from_sorted_vec`, crate-private, for already-sorted input such as the extension
  graph's chosen indices), comparing, hashing and dropping triangulations are flat; set
  operations are sorted merges; `contains` is a binary search; `insert`/`remove` are
  O(len), which no algorithm does more than a bounded number of times per pass.
  `iter()` now returns `std::slice::Iter` and there is `as_slice()` — a public API change
  of the unreleased crate. Equality, order and hashing are unchanged in meaning. No
  interning was needed: with a flat representation, hashing the visited set is no longer
  significant next to computing flips.
* **`crate::hash`**: an FxHash-style hasher (`FastHashMap`/`FastHashSet`, crate-private)
  for every internal table keyed by data the algorithms built — the chirotope cache, flip
  candidates, the flip-graph visited set, `check`, placing's boundary, the TDS facet map.
  SipHash was about 10 % of the BFS. A dozen lines, no dependency. Iteration order of
  these maps is deterministic (no random state) but never reaches output.
* **Incremental flips (`flip::flip_reps_after`)**, TOPCOM's `TriangFlips` update: a
  parent's flip `G` is kept unless some removed simplex contains `supp(G) \ {z}` for a `z`
  on `G`'s removed side (its removed simplices are the whole star of those faces, so this
  is exact without expanding `G`); new candidates are only those containing an added
  simplex. The BFS frontier carries `(Arc<parent flips>, Flip)`. A `debug_assert_eq!`
  against `flip_reps` runs on every node in debug builds; the full TOPCOM flip-graph
  comparison (504 configurations) and the 600-configuration brute-force run were run in
  release **with debug assertions enabled** (`CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true`,
  separate target dir) and pass.
* **Tier 0 of `sign_of_det`** for 2×2 to 4×4 (`arith::det_small`): cofactor expansion in
  `i128`. If every entry fits in `i64` and the column bit lengths sum to at most 120, no
  intermediate can overflow (every term takes one entry per column) and the products are
  plain widening multiplications; otherwise checked arithmetic, falling through to the
  three tiers on overflow. Exact, and it decides exact zeros, which the interval filter
  never does — the "cheap exact-zero pre-check" Phase 1 asked about, for the small sizes.
  It serves chirotopes of rank ≤ 4 as well as the 2-D/3-D predicates. Tested against
  bigint Bareiss on 20 000 matrices with mixed column widths, plus the 10⁶ harness.
* **Regular triangulations insert in a spatial order** (`regular::insertion_order`): a
  seeded shuffle, then rounds of halving size (BRIO), each sorted along a Z-order curve of
  the hull-reduced coordinates. The output is the unique regular triangulation of the
  perturbed lifts, so it is order-independent; the corpus, the invariance suites and the
  long random fuzz all pass unchanged. A single sorted pass (no rounds) was measured and
  is slower.
* **Fine**: the search for the simplex an unused point subdivides tests the circuit
  coefficients one at a time and stops at the first on the point's side — the same
  decision as building the whole circuit, at one or two signs per rejected simplex.
* `Flip::apply` shape-checks only the added simplices.

**Not done, and why.**

* **3-D Delaunay** is at 0.85× of CGAL; the remaining cost is spread over the 4×4
  determinants (on the exact cofactor fast path) and the hole retriangulation. Lattices
  (0.56×) spend extra time in symbolic-perturbation ties, and `f64` weights (0.68×) have
  lifts too wide for the cofactor fast path, so their 4×4 tests go through the filter.
* **Parallel flip-graph BFS**: not attempted. A level-synchronous parallel BFS can keep
  the yielded *set* deterministic but not the streaming order without buffering whole
  levels, and the per-node work is now small enough (≈ 12 µs) that a sharded visited set
  would eat much of the gain. The extension graph's `par_count` remains the parallel path.
* **`bnum` vs `dashu-int`** (§12 question 3): Bareiss on singular `n × n` matrices of
  62-bit entries, 2 000 each — 4×4: dashu 1.7 µs, `BInt<16>` (1024-bit) 5.6 µs; 8×8: dashu
  37 µs, `BInt<16>` 82 µs; 512-bit `bnum` is also slower than dashu and already overflows at
  8×8. dashu stays.

#### Closing the high-dimensional Delaunay gap

Profiling `delaunay/random_40x10` (`perf record` on the Criterion binary) showed 59 % in
the interval filter's `eliminate`, 4 % in the software `i128 → f64` conversion, 3 % in a
heap allocation for every `11 × 11` matrix (the filter's inline buffer held 64 entries),
and 15–20 % in `Tds::link_unlinked`, which allocated a sorted `Vec<u32>` key for every
facet. What changed:

* **The filter (`arith/filter.rs`) is midpoint-radius Gaussian elimination**: partial
  pivoting on `f64` midpoints, with a radius per entry bounding its distance from the value
  exact elimination (same row exchanges) would produce. The radius recurrences are
  first-order error propagation plus the rounding of the midpoint operations; they are
  evaluated in round to nearest, then inflated by `C = 1 + 2⁻⁴⁰` and padded by
  `E = 2⁻¹⁰⁰⁰`, which provably absorbs every rounding and underflow error made computing
  them (`C ≥ (1 + u)¹⁰`; `E` far above any `η = 2⁻¹⁰⁷⁵` loss). Pivots must satisfy
  `r/|m| ≤ 1/4` (the last one only `r < |m|`); non-finite values provably reach a pivot and
  make the filter decline. The full proof is the module's doc comment. Per updated entry:
  two operations for the midpoint and about a dozen branch-free, vectorising ones for the
  radius, against four products, six comparisons and four `next_up`/`next_down` before.
  Same decisions in practice: on the default differential run it decides 14 680
  near-singular matrices to the old filter's 14 644, and 6 fewer of the `extremes` family.
  `Entry::bracket` (an interval) became `Entry::approx` (a midpoint and radius). Stack
  buffers now go to `12 × 12`; `eliminate` is monomorphised for `n = 5..=11`.
* **The predicates feed the filter directly in `f64`** (`regular/predicates.rs`) when every
  coordinate and lift is below `2⁵²` — differences are then integers below `2⁵³`, exact in
  `f64` — for matrices of size five and up. Only if the filter declines is the `i128`
  matrix built, and then the filter is skipped (`arith::sign_of_det_exact_of`). Lifts that
  fit in `i128` (below `2¹²⁶`, so their differences fit too) are kept as `i128` instead of
  `IBig`, which keeps scaled `f64` weights off the heap; and `det_small` no longer attempts
  checked arithmetic when the column bit lengths sum past 140, where it would overflow and
  then the filter would decide anyway.
* **Hole retriangulation hashes facets without allocating**: each facet's hash is the sum of
  a mixing function over its vertices (the cell's sum minus the opposite vertex, `O(d)` per
  cell); the unlinked facets go into a reused buffer, are sorted by hash, and are matched
  within each run of equal hashes by an exact vertex-set comparison (per-point stamps,
  `O(d)`), so a collision costs time, never correctness. `Builder::insert` reuses its zone
  and new-cell vectors.

Release builds on the same idle machine, before and after (best of 10, or of 3 in 10-D):

| Delaunay / regular | before | after | |
|---|---:|---:|---:|
| 10 000 random points in 2-D | 38.1 ms | 36.7 ms | 1.04× |
| 2 000 random points in 3-D | 36.8 ms | 31.7 ms | 1.16× |
| 2 000 points of a 3-D lattice | 22.3 ms | 21.0 ms | 1.06× |
| `f64` weights, 2 000 points in 3-D | 57.6 ms | 40.6 ms | 1.42× |
| 300 random points in 5-D | 457 ms | 146 ms | 3.1× |
| 40 random points in 10-D | 4.19 s | 1.07 s | 3.9× |

and Criterion's `sign_of_det/filter_only_n×n` (1 000 random matrices, entries up to
1 000): 3×3 242 → 153 µs, 4×4 493 → 250 µs, 6×6 1.49 → 0.44 ms, 8×8 3.23 → 0.81 ms.

**Validation.** `tests/sign_of_det.rs` gained `filter_threshold_sweep`: singular rank-`n − 1`
matrices up to `12 × 12`, one entry moved by `±2^t` for every `t` in `0..63`, so the
determinant sweeps from far below the filter's error bound to far above it, with the whole
matrix optionally scaled by `2^30`, `2^64`, `2^100` or `2^300` first (inexact and
beyond-`i128` entries); the filter must decline or agree with the exact oracle, must give
the same answer on `i64`, `i128` and `IBig` entries, and must both decide and decline at
every size (9 885 decided and 17 835 declined in a four-repetition release run; the
default is two). The `#[ignore]`d 10⁶-matrix run passes (538 472 filter decisions).
`tests/regular.rs` gained `high_dimensional_random_sets` — dimensions 5–7, coordinates in
`{−1, 0, 1}`, `{−2..2}`, `2²⁰` and `2⁴⁰` — so the `f64` fast path, its exact fallback on
degenerate input, and the wide-lift path are all certified by the regularity oracle. The
long regular fuzz, the long invariance suite and the 50 000-point memory test (peak
54 MiB) pass.

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
# Rust: 1.85+ (edition 2024) for the library, 1.86+ for the tests and benchmarks
# (criterion). Nothing else to install — no CMake, no CGAL, no GMP.
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

cargo test                                      # should pass, 128 tests (8 ignored)
cargo clippy --all-targets -- -D warnings
cargo clippy --features python --all-targets -- -D warnings

uv venv .venv-dev
VIRTUAL_ENV=.venv-dev uv pip install -e '.[test]'
.venv-dev/bin/python -m pytest                  # should give 62 passes, no skips

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

### Picking up after Phase 7

Phases 0–6 and Phase 7's engineering are done (recorded in §7, decisions in §13); what
remains is the release ("Before releasing" in §13). Notes from the earlier phases that
still apply to any further work:

* **Cap memory on this kind of machine.** Run every cargo, test, Python and TOPCOM command
  under `systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 timeout <s> …`.
  Two development sessions on a 13 GB machine were OOM-killed. The enumerations
  themselves are not the risk any more (the extension graph streams in constant memory),
  but collecting `vc_degenerate_flip`'s triangulations into a list, in Rust or Python,
  never finishes: only count or stream it.
* **Performance baselines** are the Criterion suites in `benches/` and the 0.3.0
  comparison in `benches/compare_oracle.py`; §13 "Phase 7 performance" has the numbers.
* `ExtensionGraph::par_count` is the template for parallel enumeration (and for the
  optional parallel flip-graph BFS): `Search::split` donates subtrees on demand. A
  parallel *iterator* would need a channel and gives up DFS order.
* §12 question 4 still stands: `om::cocircuits`, `Facets`, `Incidences` and
  `ExtensionGraph::count` are public in Rust but not bound in Python.
* The fixture generator's `--only-extension` refreshes just the Phase 5 fields; run it
  on an idle machine, since a loaded one turns borderline 20 s enumerations into
  timeouts (recorded as `null`, which the tests tolerate).

### A warning about the `__repr__` strings

The 0.3 strings are asserted nowhere in the test suite but are in the README's doctest-style
example, e.g. `A 2-dimensional point configuration with 9 points`. Reproduce them exactly;
they are recoverable from `git show legacy-cpp:src/triangulumancer/PVConfiguration.cpp`.
The configuration strings are done (Phase 1, `impl Display for Config`, unit-tested); the
`Triangulation` and `Flip` ones are in `Triangulation.cpp` and `Flip.cpp`.
