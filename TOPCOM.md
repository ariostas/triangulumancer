# TOPCOM → Rust: design input

Research notes for replacing the TOPCOM C++ dependency of `triangulumancer` with a
pure-Rust implementation (exposed to Python through PyO3/maturin).

Everything below was read out of the exact pinned fork/revision the project builds:

* repo `https://github.com/ariostas/TOPCOM.git`, rev `a4333293e9c3cbcf5b8420b89ee30f51078d47fc`
  (pinned at `CMakeLists.txt:17-20` of the `legacy-cpp` tag; that file is gone from
  this branch)
* sources under `lib-src/` (library) and `lib-src-reg/` (regularity/LP, cddlib)

Recreate the checkout with:

```bash
git clone https://github.com/ariostas/TOPCOM.git topcom
cd topcom && git checkout a4333293e9c3cbcf5b8420b89ee30f51078d47fc
```

File references below are written `lib-src/File.cc:123` and are relative to the TOPCOM
checkout root; because the revision is pinned exactly, they are exact. References into
this project are written `src/triangulumancer/TOPCOM.cpp:42` and point at the
`legacy-cpp` tag, where the C++ still exists.

---

## 1. Scope

### 1.1 What we need

From `src/triangulumancer/TOPCOM.cpp` the entire TOPCOM surface used is:

| Project function | TOPCOM entry point |
|---|---|
| `PVConfiguration` storage | `topcom::PointConfiguration` / `Matrix` / `Vector` (`src/triangulumancer/PVConfiguration.cpp:41-53,120-158`) |
| `validate_configuration` | `.rank()`, `.rowdim()`, `.coldim()`, `.no()` |
| `triangulate_placing` | `Chirotope(points,false)` + `PlacingTriang` |
| `triangulate_fine` | `Chirotope(points,false)` + `FineTriang` |
| `find_flips` / `find_neighbors` | `TriangNode`, `TriangFlips`, `MarkedFlips`, `FlipRep`, `Flip` |
| `find_all_connected_triangulations` | `SymmetricFlipGraph` |
| `find_all_triangulations` | `SymmetricExtensionGraphMaster` |
| result marshalling | `SimplicialComplex` / `Simplex` iteration |

So the Rust port needs exactly six algorithmic pieces:

1. **Chirotope** (sign of every rank-subset determinant), lazily evaluated.
2. **Placing (beneath–beyond / pushing) triangulation.**
3. **Fine triangulation** = placing + flip-in of the unused points.
4. **Circuits and bistellar flips** of a given triangulation, plus the flipped neighbours.
5. **Flip-graph BFS** — the flip-connected component of a seed.
6. **Extension-graph backtracking** — *all* triangulations, including flip-disconnected ones.

### 1.2 What we explicitly do not need

The project **always** uses the trivial symmetry group: `topcom::SymmetryGroup symmetries(no)`
with no generators (`src/triangulumancer/TOPCOM.cpp:106,161,196`). Confirmed in
`lib-src/Symmetry.hh:391-397`: that constructor leaves the element vector *empty*
(`SymmetryGroup` stores only non-identity elements; `order() == size()+1`), sets
`_complete = true`, and stores an identity permutation. Consequently
`symmetries.stabilizer_ptrs(seed)` (`lib-src/Symmetry.hh:573-595`) iterates an empty
container and returns an empty `symmetryptr_datapair`.

**Therefore the entire symmetry layer can be deleted from the Rust port**, not merely
stubbed. See §6.4 and §7.9 for the exhaustive list of what collapses.

Also droppable:

* **cddlib / regularity checking.** cddlib enters only via
  `lib-src-reg/LPinterface.hh:24-25` (`setoper.h`, `cdd.h`) → `RegularityCheck`. In the
  flip graph, `Facets`/`Incidences`/`RegularityCheck` are built only under
  `check_regular() || check_nonregular() || check_sometimes()`
  (`lib-src/SymmetricFlipGraph.cc:1047-1091`), all of which default to `false`
  (`lib-src/CommandlineOptions.cc:49-52`) and are never set by this project (it never calls
  `CommandlineOptions::init`). The extension graph builds `Incidences` unconditionally
  (`lib-src/SymmetricExtensionGraphMaster.cc:417`) but uses `RegularityCheck` only under the
  same flags. So `lib-src-reg/` is dead weight and the `checkreg` target can go.
* I/O: `read`/`write`/`print_string` on every class, `ContainerIO.hh`, `Graphics.*`.
* `Statistics.*` (`--stats`), `CheckTriang.*` (`--checktriang`).
* Checkpointing/persistence (`--dumpstatus`/`--readstatus`,
  `lib-src/SymmetricFlipGraph.cc:988-1006,1167-1178,1212-1378`).
* Caching knobs: `--memopt`, `--chirocache`, `--localcache`.
* GKZ vectors and the `Volumes` table (§6.5).
* All the non-STL container reimplementations (`HashSet.hh`, `HashMap.hh`, `PlainHash*.hh`,
  `Array.hh`, `PlainArray.hh`, `RefCount.hh`, `Pair.hh`) — the build defines
  `STL_CONTAINERS`, `STL_CHIROTOPE`, `STL_FLIPS`, `STL_SYMMETRIES`
  (`CMakeLists.txt:85` of TOPCOM), so those branches are not even compiled in.

### 1.3 Build configuration that is actually in effect

`CMakeLists.txt:85` of TOPCOM sets
`STL_SYMMETRIES STL_CONTAINERS STL_FLIPS STL_CHIROTOPE GMPRATIONAL PERMLIB_DOMAIN_INT USE_LEXORDER`
and this project additionally sets `USE_PHIT_PREPROCESSING=false` (`CMakeLists.txt:21`).

Notably **not** defined: `PREPROCESS_CHIRO`, `MAXNOIS64`, `USE_SPARSEINTSET`,
`COMPRESSED_INTEGERSET`, `SPARSE`. Hence:

* `Chirotope = VirtualChiro` (`lib-src/Chirotope.hh:17-21`) — the *lazy* chirotope.
* `basis_type = Simplex = LabelSet = IntegerSet` (`lib-src/LabelSet.hh:17-21`,
  `lib-src/Simplex.hh:15`) — a heap-allocated variable-length bitset, not the 64-bit one.
* `SimplicialComplex = FastSimplicialComplex = SimplicialComplexTemplate<IntegerSet>`
  (`lib-src/SimplicialComplex.hh:17-21`, `lib-src/FastSimplicialComplex.hh:17-23`).
* `Field = Rational = mpq_class` (`lib-src/Field.hh:17`, `wrap-gmp-gmpxx/Rational.h:18`).

Since `CommandlineOptions::init()` is never called, every option keeps its static default
(`lib-src/CommandlineOptions.cc:18-85`). The ones that matter:

```
verbose=false  debug=false  memopt=false        no_of_threads=1
check_regular=false  check_nonregular=false  check_sometimes=false  check_unimodular=false
use_volumes=false    use_gkz=false           use_switch_tables=false
parallel_symmetries=false  simpidx_symmetries=false  dont_add_points=false
no_of_simplices=0    max_no_of_simplices=0   chirocache=12582917
```

**Net effect: TOPCOM runs single-threaded, without cddlib, without volume tables, without
symmetry, with a fully lazy but unbounded-cache chirotope.**

#### What `USE_PHIT_PREPROCESSING=false` disables

The macro is used in exactly one place: `lib-src/PlainHashIndexTable.hh:130-144`.

```cpp
if (_preprocessed && USE_PHIT_PREPROCESSING) {
  auto finder = key_table_type::find(key);
  if (finder != end()) return finder->second;
  else { std::cerr << "... not preprocessed properly ..."; exit(1); }   // <-- hard exit
}
else { /* dynamic path: look up, else append a new index */ }
```

`SimplicialComplex` interns every simplex in a **process-global static** index table
(`lib-src/SimplicialComplexTemplate_Declarations.hh:99`, `lib-src/IndexTableArray.hh:34-175`).
`SymmetricExtensionGraphMaster::_init()` pre-fills that table for *its* configuration and
marks it `preprocessed` (`lib-src/SymmetricExtensionGraphMaster.cc:262-285`). With the
upstream default, a *second* point configuration in the same process would then hand the
table an unknown simplex and TOPCOM would `exit(1)` — killing the Python interpreter.
Setting the macro to `false` forces the dynamic path always, so the table simply grows.

Two consequences the Rust port must be aware of:

* the global table **never shrinks** — it is a monotonic leak across repeated calls in a
  long-lived Python process;
* `SymmetricExtensionGraphMaster::_init()` calls `SimplicialComplex::reset_index_table()`
  (`lib-src/SymmetricExtensionGraphMaster.cc:267,283`;
  `lib-src/SimplicialComplexTemplate_Definitions.hh:146-148`) which **clears the global
  table and thereby silently invalidates every `SimplicialComplex` created earlier in the
  process** (their bitsets now index into a different table). Calling
  `all_triangulations()` while another `Triangulation` object is alive is unsound in the
  current C++ code. *The Rust port must not use a global intern table, or must make it
  per-computation.*

---

## 2. Core data structures

### 2.1 Configuration matrix — `Matrix` / `Vector` / `PointConfiguration`

`lib-src/Matrix.hh:38`, `lib-src/Vector.hh:37`, `lib-src/PointConfiguration.hh:78`.

* `Vector : std::vector<Field>` — one **column** of the configuration.
* `Matrix : std::vector<Vector>` — **column-major**: `m(i,j)` is row `i`, column `j`;
  `push_back(v)` appends a *column*; `rowdim()`/`coldim()` are as expected.
* `PointConfiguration : Matrix` adds `no() == coldim()` (number of points) and
  `rank() == _compute_rank()` (`lib-src/PointConfiguration.cc:228-245`), which is computed by
  incremental `StrictStairCaseMatrix` elimination — **recomputed from scratch on every
  call**, so `points.rank()` inside a loop is O(n·d²) each time.

The project fills it at `src/triangulumancer/PVConfiguration.cpp:41-53`: for a *point*
configuration each column is `(p_0,…,p_{d-1}, 1)` (homogenising `1` appended), for a
*vector* configuration the raw `(v_0,…,v_{d-1})`. So `rowdim() = d + 1` resp. `d`, and
`rank()` must equal `rowdim()` for the configuration to be accepted
(`src/triangulumancer/TOPCOM.cpp:61-75`).

**Proposed Rust representation**

```rust
/// Column-major, exact integer entries. rows = homogenised dimension, cols = #points.
pub struct Config {
    rows: usize,
    cols: usize,
    data: Vec<i64>,          // column-major: data[col*rows + row]
}
```

Integer, not rational: the Python API only accepts `int64` arrays
(`include/triangulumancer/PVConfiguration.hpp` uses `pybind11::array_t<int64_t>` everywhere),
so the configuration entries are always integers. TOPCOM stores them as `mpq_class`
purely because its `Field` is fixed at compile time. See §8.

### 2.2 Chirotope — `VirtualChiro`

`lib-src/VirtualChiro.hh:33-128`, backed by `RealChiro` (`lib-src/RealChiro.hh:61-121`).

Mathematically: for a configuration of `n` vectors in rank `r`, the chirotope is the map

```
χ : {r-subsets of [n]}  →  {-1, 0, +1},    χ(B) = sign det( columns of B, in increasing order )
```

`Chirotope(points, /*immediate=*/false)` (the project's call,
`src/triangulumancer/TOPCOM.cpp:82`) produces the **lazy** variant: nothing is computed up
front, and `operator()(basis)` (`lib-src/VirtualChiro.hh:130-184`) computes
`StairCaseMatrix::det()` of the `r×r` submatrix on demand and memoises
`basis → (sign, det)` in an `unordered_map<IntegerSet, pair<short,Field>>`
(`lib-src/RealChiro.hh:39`). With `memopt=false` the cache is **unbounded** (no eviction,
`lib-src/VirtualChiro.hh:176-179`).

The eager variant `RealChiro(points)` (`lib-src/RealChiro.cc:23-33,262-346`) enumerates all
`C(n,r)` bases by a DFS over prefixes, maintaining a `StairCaseMatrix` incrementally and
short-circuiting whole subtrees to sign 0 once a prefix becomes rank-deficient
(`_recursive_chiro`, `already_dependent` flag). That incremental-prefix trick is worth
reproducing if a dense chirotope is ever wanted.

Two extra entry points are used:

* `operator()(prebasis, lex_extension_perm)` (`lib-src/RealChiro.cc:35-56`) — used by the
  extension-graph lex machinery.
* `find_non_deg_basis()` (`lib-src/RealChiro.cc:67-82`) — first (lexicographically) basis with
  nonzero sign; seeds the placing triangulation.

**Proposed Rust representation**

```rust
pub struct Chirotope<'a> {
    cfg: &'a Config,
    rank: usize,
    cache: RefCell<HashMap<Simplex, i8>>,   // or DashMap if threaded
}
impl Chirotope<'_> { fn sign(&self, basis: &Simplex) -> i8; }
```

Only the *sign* is ever needed by the algorithms we port (dets are used only by `Volumes`
and the unimodularity predicate, both droppable). Keeping only `i8` shrinks the cache by a
large factor versus TOPCOM's `(short, mpq_class)`.

### 2.3 `Simplex` = `LabelSet` = `IntegerSet`

`lib-src/IntegerSet.hh:53-419` (+ 2596 lines of `.cc`). A dynamically sized bitset over
`uint64_t` blocks with a cached `_invariant` word for fast inequality rejection
(`lib-src/IntegerSet.hh:60-63`). Provides `card()`, `min_elem()`, `max_elem()`,
`contains`, `subset`/`superset`/`disjoint`, `+= -= *= ^=` (union/difference/intersection/
symmetric difference), `lexmin_subset`, `permute`, iteration over set bits, and hash keys
`keysize() == no_of_blocks()`, `key(n) == bitrep[n]`.

A `Simplex` is a set of **point labels** (cardinality `rank` for a full-dimensional simplex).
`n` is small in practice (tens to low hundreds), so:

**Proposed Rust representation**: `u128` bitmask for `n ≤ 128` with a `Vec<u64>` /
`smallvec` fallback, or unconditionally a fixed `[u64; K]`. A `#[derive(Copy)]` 128-bit
simplex removes essentially all of TOPCOM's allocation traffic. Note TOPCOM's own
`IntegerSet64` (`lib-src/IntegerSet64.hh`) is exactly this specialisation, enabled by
`MAXNOIS64` — which this build does *not* enable, so TOPCOM here pays heap allocation per
simplex operation.

### 2.4 `SimplicialComplex` — the triangulation representation

`lib-src/SimplicialComplexTemplate_Declarations.hh:40-313`,
definitions in `lib-src/SimplicialComplexTemplate_Definitions.hh`.

This is the single most surprising piece of TOPCOM's design:

* Simplices are **interned** into a process-global `IndexTableArray`, one
  `PlainHashIndexTable<LabelSet>` per cardinality
  (`lib-src/IndexTableArray.hh:24-34`, `lib-src/PlainHashIndexTable.hh:50`). Interning
  assigns consecutive `size_type` indices in first-seen order (or in lex order when
  pre-processed, §1.3).
* A `SimplicialComplex` is then **a bitset over those indices**, one `IntegerSet` per
  cardinality (`_index_set`), plus `_is_pure`/`_pure_rank`/`_mincard`/`_maxcard`
  (`Declarations.hh:93-99`). A pure complex (all simplices of the same cardinality — which
  every triangulation is) stores a single index set.
* Therefore set operations on whole complexes (`+=`, `-=`, `*=`, `^=`,
  `Definitions.hh:1388-1524`) are **word-parallel bitset ops**, and equality/hashing are
  bitset compares (`keysize`/`key` at `Definitions.hh:293-320`, hashed by the Murmur-ish
  `Hash<Key>` at `lib-src/HashKey.hh:29-49`).

Operations actually used by the ported algorithms:

| op | definition |
|---|---|
| `support()` — union of all simplices | `Definitions.hh:587-594` |
| `link(simp)` | `Definitions.hh:638-679` |
| `star(simp)` | `Definitions.hh:596-636` |
| `join(SimplicialComplex)` / `join(Simplex)` | `Definitions.hh:2152-2250` |
| `insert_boundary(simp)` — add all facets of `simp` | `Definitions.hh:1054-1067` |
| `contains`, `card()`, iteration | `Definitions.hh:325-363` |

**Proposed Rust representation.** Two options:

1. *Direct*: `HashSet<Simplex>` (or `BTreeSet<Simplex>` for deterministic iteration). Simple,
   no global state, no interning hazard (§1.3). Costs: `+=`/`-=`/`^=` become O(k) instead of
   O(words), and hashing a triangulation is O(k) rather than O(#interned/64).
2. *TOPCOM-like*: a per-computation `Vec<Simplex>` + `HashMap<Simplex, u32>` interner plus a
   `FixedBitSet` over simplex ids. Worth it only for the big enumerations, where a
   triangulation is compared/hashed millions of times.

Recommendation: start with (1) behind a trait; move the two enumerators to (2) if profiling
demands it. Whatever you choose, the interner must be **owned by the enumeration**, never a
global.

### 2.5 `TriangNode`

`lib-src/TriangNode.hh:23-80`. `SimplicialComplex` + `(_ID, _no, _rank)`. `_ID` is a serial
number for printing only — `operator==` ignores it (`lib-src/TriangNode.hh:148-150`), so the
hash-map identity of a node is purely its set of simplices.

Two non-trivial predicates live here and are **essential to flip validity**:

* `containment_ok(circuit)` (`lib-src/TriangNode.cc:20-32`): for a *general-position* circuit
  `Z = (Z⁺, Z⁻)` with `|Z⁺ ∪ Z⁻| = rank+1`, checks that all `rank`-subsets
  `supp(Z) \ {z}` for `z ∈ Z⁺` are simplices of the triangulation. That is exactly the
  condition "the lower/upper triangulation of the circuit is a subcomplex of `T`".
* `link_ok(circuit)` (`lib-src/TriangNode.cc:34-76`): the *special-position* analogue, where
  `|supp(Z)| < rank+1`. It computes the link of `supp(Z) \ {z}` for the first `z ∈ Z⁺` and
  verifies that *every other* `z ∈ Z⁺` has the **same** link (same set, same cardinality).
  This is the "common link" condition that makes the flip well defined.

**Rust**: a plain struct `Triang { simplices: SimplexSet }`, plus free functions
`containment_ok` / `link_ok`.

### 2.6 `Circuit`, `FlipRep`, `Flip`, `MarkedFlips`

* `Circuit : pair<LabelSet,LabelSet>` — the signed circuit `(Z⁺, Z⁻)` of a dependent set
  (`lib-src/Circuits.hh:40-57`). Constructor `Circuit(chiro, dependent_set)`
  (`lib-src/Circuits.cc:43-72`) derives it from the chirotope by alternating signs of the
  `rank`-subsets of the `(rank+1)`-set: for the `i`-th element `e` (in increasing order),
  `σ_e = (-1)^i · χ(D \ {e})`; elements with `σ_e > 0` go to `Z⁺`, `σ_e < 0` to `Z⁻`,
  `σ_e = 0` to neither (they are not in the circuit's support). `upper_facets()` /
  `lower_facets()` (`lib-src/Circuits.cc:19-41`) produce the two triangulations of `supp(Z)`.
* `FlipRep : Circuit` (`lib-src/Flip.hh:47-97`) — a circuit *oriented so that its "lower"
  side is present in the triangulation*. Truthy iff both parts are non-empty
  (`Flip.hh:77`). Helpers: `kills_vertex() == (second.card()==1)`,
  `adds_vertex() == (first.card()==1)`, `is_balanced()`.
* `Flip : pair<SimplicialComplex,SimplicialComplex>` (`lib-src/Flip.hh:99-135`) — the
  *expanded* flip: `first` = simplices to remove, `second` = simplices to add. Built by
  `Flip::_construct` (`lib-src/Flip.cc:65-115`); handles the general-position case
  (facets of `supp(Z)`) and the special-position case (join with the common link).
* `MarkedFlips : unordered_map<FlipRep,bool>` (`lib-src/MarkedFlips.hh:30-164`) — the flip
  set of a node with a "already traversed" mark per flip, plus `_no_of_marked`.
* `TriangFlips` (`lib-src/TriangFlips.hh:27-74`) — `MarkedFlips` + `(no, rank)`, with the
  *incremental* update logic (§5.2).

**Rust sketch**

```rust
struct Circuit { pos: Simplex, neg: Simplex }         // Z⁺, Z⁻
struct Flip    { out: SimplexSet, inn: SimplexSet }   // remove, add
struct FlipSet { flips: HashMap<Circuit, bool> }      // circuit -> marked
```

---

## 3. Algorithm: placing (pushing) triangulation

`lib-src/PlacingTriang.hh:22-40`, `lib-src/PlacingTriang.cc:19-99`. Entry point in the
project: `src/triangulumancer/TOPCOM.cpp:77-87`.

This is classical **beneath–beyond** on the *chirotope only* — no coordinates are touched.

State: the triangulation `T` built so far, and `_bd_triang` = the set of boundary facets of
`T` (cardinality-`(rank-1)` simplices) maintained as a `SimplicialComplex`.

```
start = chiro.find_non_deg_basis()              # PlacingTriang.cc:26
T     = { start }
bd    = boundary(start)                          # insert_boundary, .cc:38
for i in [0..n) \ start, in increasing order:    # .cc:39-49
    new_facets = ∅
    for F in bd:                                 # .cc:58-75
        if visible(F, i):
            T   += F ∪ {i}
            new_facets ^= boundary(F ∪ {i})      # symmetric difference!
    bd ^= new_facets                             # .cc:76
```

The **symmetric-difference** trick (`^=`) is the whole boundary-maintenance algorithm: a
facet that appears in exactly one simplex stays on the boundary; one that appears in two
cancels out. Note `new_facets` is accumulated with `^=` too, so a facet shared by two newly
created simplices also cancels.

**Visibility test** (`lib-src/PlacingTriang.cc:80-99`) — again purely chirotope:

```
visible(F, i):
    if χ(F ∪ {i}) == 0: return false               # i is coplanar with F
    p = the unique vertex of link_T(F)             # the apex of F's interior simplex
    return sign(perm(F,p)) * χ(F∪{p}) != sign(perm(F,i)) * χ(F∪{i})
```

i.e. `i` and the interior neighbour `p` lie on **opposite** sides of the hyperplane spanned
by `F`. `perm(F,x).sign()` is the sign of the permutation that sorts the sequence
`(F in increasing order, x)` (`lib-src/Permutation.cc:40-50`), converting the *ordered*
orientation into the chirotope's *sorted-basis* convention.

### 3.1 Important semantics

**The placing triangulation is generally *not* fine.** If `i` lies in the interior of the
convex hull of the already-placed points, no boundary facet is visible from it and `i` is
simply skipped. This is exactly why `FineTriang` exists.

### 3.2 Complexity

`O(n · |bd| · cost(χ))`. The chirotope queries dominate; each is one `rank × rank`
determinant unless cached. `find_non_deg_basis` scans bases in lex order
(`lib-src/RealChiro.cc:67-82`) — cheap for generic input, worst case `C(n,r)`.

### 3.3 Rust port

Direct translation, ~120 LOC. The only subtlety is `link(F)` returning a one-element
complex whose single element is a one-element simplex (`*facet_lk.begin()->begin()`,
`lib-src/PlacingTriang.cc:87`) — with a `HashSet<Simplex>` representation this is
"find the unique `S ∈ T` with `S ⊃ F`, take `S \ F`".

---

## 4. Algorithm: fine triangulation

`lib-src/FineTriang.hh:23-46`, `lib-src/FineTriang.cc:16-65`. Entry point:
`src/triangulumancer/TOPCOM.cpp:89-99`.

```
T = PlacingTriang(chiro)                             # FineTriang.hh:44
not_used = [0..n) \ support(T)                       # FineTriang.cc:17-18
while not_used non-empty:
    i = min(not_used)
    flip_in(T, i)                                    # FineTriang.cc:20-21
    not_used -= i
```

`flip_in(T, i)` (`lib-src/FineTriang.cc:26-65`) scans the simplices of `T`:

```
for S in T:
    D = S ∪ {i}                       # a (rank+1)-set, hence dependent
    Z = Circuit(chiro, D)
    U = Z.upper_facets(); L = Z.lower_facets()
    if i ∈ supp(U) and i ∉ supp(L):
        if L empty: continue                         # degenerate, fork fix
        lk = link_T(first element of L)
        T -= L.join(lk);  T += U.join(lk)            # the bistellar flip
        break
    if i ∉ supp(U) and i ∈ supp(L):   symmetric
```

Geometrically: find a simplex `S` of the current triangulation such that `i` lies in the
relative interior of (a face of) `S`; the circuit of `S ∪ {i}` then puts `i` alone on one
side, and the flip replaces the star of the other side by the star of `i`'s side, inserting
`i` as a vertex. The `break` is mandatory — `T` was just mutated.

**Fork patch.** Upstream had `assert(!lower.empty())` there; the fork replaced both asserts
by `continue` (commit `e9494f3`, "Fixed issue with degenerate flips while finding a fine
triangulation"). Reproduce that: with degenerate configurations the circuit can have an
empty side and the assertion fires.

Also note commit `a433329` ("Fixed issue with dangling reference") which copies
`*lower.begin()` into a local `Simplex` before calling `link()` — a pure artefact of
TOPCOM's global intern table reallocating under the iterator. A Rust port with owned
simplices does not have this class of bug.

### 4.1 Correctness note

The loop terminates because each `flip_in` strictly adds `i` to `support(T)` and never
removes a previously added point (the flip replaces a star by a star that contains `i`).
The result is fine by construction: `support(T) == [0..n)`.

**Rust port**: ~80 LOC on top of the placing triangulation.

---

## 5. Algorithm: bistellar flips and neighbours

Entry points: `src/triangulumancer/TOPCOM.cpp:101-142` (`find_flips`, `find_neighbors`),
both via `compute_triang_flips`, which builds
`TriangFlips(chiro, tn, seed_symmetryptrs, /*forbid_vertex_removal=*/false)`.

### 5.1 Enumerating the flips of a triangulation

`TriangFlips::_add_new_flips` (`lib-src/TriangFlips.cc:37-240`). With the trivial group, all
the `tn_symmetryptrs` loops are empty and the algorithm is:

```
tn_support       = support(T)
missing_vertices = [0..n) \ tn_support
dependent_sets   = ∅        # memo of (rank+1)-sets already tried
done_set         = ∅        # memo of simplices already used as simp1

for simp1 in restriction:                          # = T for a fresh node
    if simp1 ∈ done_set: continue
    done_set += simp1

    # (a) candidate flips that add a currently unused point
    for v in missing_vertices:                     # .cc:104-151
        D = simp1 ∪ {v}
        if D ∈ dependent_sets: continue
        try_flip(D)

    # (b) candidate flips across a shared facet
    count = 0
    for simp2 in T:                                # .cc:154-222
        D = simp1 ∪ simp2
        if |D| != rank+1: continue                 # not facet-adjacent
        if D ∈ dependent_sets: continue
        try_flip(D)
        if ++count > rank: break                   # at most rank+1 neighbours
```

where

```
try_flip(D):
    fr = FlipRep(chiro, D, T)                      # Flip.cc:22-61
    dependent_sets += D
    if fr:                                          # both sides non-empty
        if forbid_vertex_removal and fr.kills_vertex(): return
        flips[fr] = false                           # unmarked
```

`FlipRep(chiro, D, T)` (`lib-src/Flip.cc:22-61`):

```
Z = Circuit(chiro, D)
if |supp(Z)| == rank+1:                # general position
    if T.containment_ok(Z):            (first,second) = (Z⁺, Z⁻)
    elif T.containment_ok(Z.inverse()) (first,second) = (Z⁻, Z⁺)
else:                                  # special position
    if T.link_ok(Z):                   (first,second) = (Z⁺, Z⁻)
    elif T.link_ok(Z.inverse())        (first,second) = (Z⁻, Z⁺)
# otherwise the FlipRep stays empty == not a valid flip
```

So a `(rank+1)`-subset `D` of the ground set induces a valid flip of `T` iff one of the two
orientations of its circuit is *realised* in `T` (all its "lower" simplices, joined with a
common link, are in `T`). That's exactly the standard characterisation of bistellar flips
via circuits.

The **`count > rank` break** at `lib-src/TriangFlips.cc:218-221` is an important
optimisation: a `rank`-simplex has `rank` facets, so at most `rank` simplices of `T` can be
facet-adjacent to it; once that many candidate dependent sets have been processed, stop
scanning `T`.

### 5.2 Expanding a flip and applying it

`Flip::_construct(T, Z)` (`lib-src/Flip.cc:65-115`):

* **general position** (`|supp(Z)| == rank+1`): `out = { supp(Z) \ {z} : z ∈ Z⁺ }`,
  `in = { supp(Z) \ {z} : z ∈ Z⁻ }`. Cardinalities `|Z⁺|` and `|Z⁻|`, summing to `rank+1`
  — this is the (|Z⁺|, |Z⁻|) bistellar flip.
* **special position** (`|supp(Z)| < rank+1`): let `σ = supp(Z) \ {first z ∈ Z⁺}`; the
  *common link* is `{ S \ σ : S ∈ T, S ⊃ σ }`; then
  `out = { (supp(Z)\{z}) ∪ ℓ : z ∈ Z⁺, ℓ ∈ link }`,
  `in  = { (supp(Z)\{z}) ∪ ℓ : z ∈ Z⁻, ℓ ∈ link }`.

Applying it: `TriangNode(ID, T, flip)` (`lib-src/TriangNode.hh:111-119`) is just
`T - flip.first + flip.second`. That is exactly what
`src/triangulumancer/TOPCOM.cpp:135-139` uses to build the neighbours.

The project's `find_flips` returns, for each flip, the pair `(flip.first, flip.second)`
*rendered as two triangulations* — note that these are **not** triangulations of the whole
configuration, they are the two sub-complexes of the flip
(`src/triangulumancer/TOPCOM.cpp:120-127`). `find_neighbors` returns the actual flipped
triangulations.

### 5.3 Incremental flip-set maintenance

Used by the flip-graph BFS, not by the standalone `find_flips`.
`TriangFlips(chiro, T_before, F_before, T_after, flip, syms, forbid)`
(`lib-src/TriangFlips.hh:89-105`):

```
F_after = copy of F_before
F_after._remove_destroyed_flips(T_before, flip)     # TriangFlips.cc:20-35
F_after.unmark_all()
F_after._add_new_flips(chiro, T_after, flip.second /* restriction */, syms, forbid)
```

`_remove_destroyed_flips` erases every old `FlipRep` whose expanded `out` set intersects
`flip.first` — i.e. whose "lower" simplices were just removed. `_add_new_flips` then
re-scans only the *newly inserted* simplices (`restriction = flip.second`) for new flips,
because every new flip must involve at least one of them.

**This is the performance core of the whole enumeration.** A naive "recompute all flips
from scratch at every node" is correct but much slower.

⚠️ `lib-src/TriangFlips.cc:22` declares `static MarkedFlips old_flips;` — a function-local
static used as a scratch buffer. It is **not thread-safe**. In Rust this is a local.

### 5.4 Complexity

Per node, `_add_new_flips` costs roughly
`O(|restriction| · (|missing| + |T|) · cost(FlipRep))` where `cost(FlipRep)` is
`O(rank)` chirotope lookups plus an `O(|T| · rank)` `containment_ok`/`link_ok` scan.
`|T|` grows like the number of simplices in the triangulation.

### 5.5 Rust port

~350 LOC for circuits + FlipRep + Flip + incremental FlipSet. This is the piece to get
right first: both enumerators are built on it, and `find_flips`/`find_neighbors` are direct
users.

---

## 6. Algorithm: flip-graph enumeration (`SymmetricFlipGraph`)

Declaration `lib-src/SymmetricFlipGraph.hh:473-483`, definition `:532-573`, driver
`lib-src/SymmetricFlipGraph.cc`. Project call site: `src/triangulumancer/TOPCOM.cpp:144-181`.

### 6.1 The call, decoded

```cpp
SymmetricFlipGraph sfg(no, rank, points, chiro, symmetries, seed,
                       seed_symmetryptrs, voltableptr /*=nullptr*/,
                       true /* print_triangs */, only_fine, callback);
```

Positional parameter 9 is **`print_triangs`**, not something else
(`lib-src/SymmetricFlipGraph.hh:481`). It must be `true` or the callback never fires: the
callback is invoked from `__sbfs_cout_triang::operator()`
(`lib-src/SymmetricFlipGraph.hh:85-88`), and `_cout_triang_ptr` is set to the *no-op base
class* when `print_triangs == false` (`lib-src/SymmetricFlipGraph.cc:1127-1132`). Good
thing the project passes `true`.

All work happens in the constructor (`_init()` at `:572` → `_bfs()` at
`lib-src/SymmetricFlipGraph.cc:1206`).

### 6.2 The BFS

Containers (`lib-src/SymmetricFlipGraph.hh:56-61,422-428`):

```
_previous_triangs : unordered_map<TriangNode, TriangFlips>   # layer L_k
_new_triangs      : unordered_map<TriangNode, TriangFlips>   # layer L_{k+1}
_stabilizers      : unordered_map<TriangNode, symmetryptr_datapair>   # empty w/ trivial group
```

```
_bfs():                                        # .cc:968-1008
    while _previous_triangs non-empty:
        _bfs_step()
        swap(_previous_triangs, _new_triangs)

_bfs_step():                                   # .cc:924-966
    while _previous_triangs non-empty:
        (T, F) = copy of *_previous_triangs.begin()
        _process_flips(T, F)
        _previous_triangs.erase(T)

_process_flips(T, F):                          # .cc:890-922
    for (fr, marked) in F:
        if marked: continue
        flip = Flip(T, fr)
        T'   = T - flip.first + flip.second
        _process_newtriang(T, F, fr, T')

_process_newtriang(...):                       # .cc:778-888
    if T' ∈ _previous_triangs:  mark fr.inverse() in that stored entry; return
    if T' ∈ _new_triangs:       mark fr.inverse() in that stored entry; return
    F' = TriangFlips(chiro, T, F, T', flip, {}, only_fine)   # incremental, §5.3
    F'.mark(fr.inverse())
    emit(T') via callback
    _new_triangs.insert(T', F')
```

Only **two layers** are held at once. That is valid because in an unweighted graph every
edge joins nodes whose BFS layers differ by at most 1, so `L_{k-1}` (already erased) can
only be reached through the *reverse* of a flip, and that reverse flip was already marked
when the child was created.

**The marking is load-bearing, not an optimisation.** It is what prevents re-visiting
`L_{k-1}` after it has been discarded. If you keep a single global
`HashSet<Triangulation>` of all visited nodes instead (more memory, much simpler), you can
drop the marking entirely — recommended for the Rust port unless memory becomes the binding
constraint.

**Completeness / non-duplication**: the graph is the flip graph restricted to the seed's
connected component; BFS visits every vertex of a connected graph exactly once, and
duplicate detection is exact set equality of the triangulation (`TriangNode::operator==`
ignores `_ID`, `lib-src/TriangNode.hh:148-150`). There is no canonicalisation anywhere —
TOPCOM stores the first-seen node and compares against the stored set.

### 6.3 `only_fine` in the flip graph — a real trap

There is **no "is fine" predicate anywhere in this code path**. `only_fine_triangs` is
forwarded verbatim as `forbid_vertex_removal` to `TriangFlips`
(`lib-src/SymmetricFlipGraph.cc:834,1195`), which at `lib-src/TriangFlips.cc:116-118,186-188`
does `if (forbid_vertex_removal && fliprep.kills_vertex()) continue;`, i.e. it **deletes
edges from the flip graph**; the output is never filtered
(`_output_pred_ptr` is the always-true base predicate under default options,
`lib-src/SymmetricFlipGraph.cc:1133-1146`).

Consequences:

1. If the seed *is* fine, no vertex can be added either (`missing_vertices` is empty,
   `lib-src/TriangFlips.cc:104-151`), so every visited node is fine. You get the connected
   component of the seed **in the induced subgraph on fine triangulations** — a fine
   triangulation reachable only through non-fine intermediates is *not* found. This is a
   known, intended limitation of `points2finetriangs`.
2. **If the seed is not fine, `only_fine=true` returns non-fine triangulations.** TOPCOM's
   own driver therefore picks the seed accordingly
   (`lib-src/ComputeTriangs.cc:314-319`):

   ```cpp
   if (fine_only) seed = FineTriang(chiro);
   else           seed = PlacingTriang(chiro);
   ```

   **`src/triangulumancer/TOPCOM.cpp:160` unconditionally uses `PlacingTriang(chiro)`** —
   and §3 shows the placing triangulation omits interior points. So
   `all_connected_triangulations(only_fine=True)` is currently wrong whenever the
   configuration has interior points. *The Rust port should use the fine seed in that
   branch (and it would be worth fixing in the C++ meanwhile).*

### 6.4 Parallelism

`_no_of_threads = min(symmetries.size(), CommandlineOptions::no_of_threads())`
(`lib-src/SymmetricFlipGraph.hh:564`). With the trivial group `symmetries.size() == 0`, so
**zero threads**, and threads are only ever created under `--parallelsymmetries`
(`lib-src/SymmetricFlipGraph.cc:1122-1124,1010-1046`, default false). The BFS itself
(`_bfs`, `_bfs_step`, `_process_flips`, `TriangFlips`) is entirely sequential. The only
parallel regions are the orbit scan/construction inside `_old_symmetry_class`
(`lib-src/SymmetricFlipGraph.cc:435-493,611-684`).

**There is no parallelism to match.** A Rust port is free to add its own — the natural place
is a work-stealing frontier with a concurrent visited set (`DashMap`/sharded `HashSet`),
which TOPCOM never does.

### 6.5 `voltableptr` / GKZ

`Volumes` (`lib-src/Volumes.hh:34-55`, `lib-src/Volumes.cc:25-44`) is
`unordered_map<Simplex, Field>` mapping every nonzero-volume `rank`-subset to its normalised
volume, built by brute force over all `C(n, rank)` subsets. Inside `SymmetricFlipGraph` it is
used for exactly one thing: **GKZ vectors as a cheap symmetry fingerprint**
(`_gkz`, `lib-src/SymmetricFlipGraph.cc:378-396`), and every use is guarded by
`use_gkz() && _voltableptr` (`:68,347,365,504,782,883,941,975,1181,1201`). The project passes
`nullptr` and `use_gkz` defaults to false.

**Drop `Volumes` entirely.** The class never allocates one and never deletes the pointer
(the caller owns it, `lib-src/ComputeTriangs.cc:270-281,392`).

### 6.6 Rust sketch

```rust
fn flip_graph(chiro: &Chirotope, seed: Triang, only_fine: bool,
              mut emit: impl FnMut(&Triang)) {
    let mut visited: HashSet<Triang> = HashSet::new();
    let mut queue:   VecDeque<(Triang, FlipSet)> = VecDeque::new();
    let f0 = FlipSet::compute(chiro, &seed, only_fine);       // TriangFlips.cc:37
    emit(&seed); visited.insert(seed.clone()); queue.push_back((seed, f0));
    while let Some((cur, flips)) = queue.pop_front() {
        for fr in flips.iter() {
            let flip = expand(&cur, fr);                      // Flip.cc:65
            let next = cur.apply(&flip);                      // TriangNode.hh:111
            if !visited.insert(next.clone()) { continue; }
            let nf = flips.updated(chiro, &cur, &next, &flip, only_fine); // §5.3
            emit(&next);
            queue.push_back((next, nf));
        }
    }
}
```

With the trivial group, TOPCOM's `totalcount == symcount ==` number of emitted
triangulations (`_orbitsize` is always 1, `lib-src/SymmetricFlipGraph.cc:591`).

---

## 7. Algorithm: extension-graph enumeration (`SymmetricExtensionGraphMaster`)

Declaration `lib-src/SymmetricExtensionGraphMaster.hh:174-182`, definition `:250-290`;
search in `lib-src/SymmetricExtensionGraph.cc`; node state in `lib-src/PartialTriang.hh`.
Project call site: `src/triangulumancer/TOPCOM.cpp:183-208`.

This enumerator does **not** use flips at all. It is a backtracking search over *partial
triangulations*, so it finds every triangulation including those that are flip-disconnected
from the placing triangulation.

### 7.1 The call, decoded

```cpp
SymmetricExtensionGraphMaster segm(no, rank, points, chiro, symmetries,
                                   true /* print_triangs */,
                                   only_fine,
                                   false /* find_minimal_triang */,
                                   callback);
```

(`lib-src/SymmetricExtensionGraphMaster.hh:250-258`.) As in §6.1, `print_triangs` must be
`true` or the callback is never reached — the callback lives inside
`if (this->_print_triangs)` at `lib-src/SymmetricExtensionGraph.cc:1411,1415-1418`.

`find_minimal_triang = false` disables the branch-and-bound that `points2mintriang` uses
(`lib-src/SymmetricExtensionGraph.cc:1053-1064,1220-1230,1399-1410`); with `true` it finds
one minimum-cardinality triangulation instead of enumerating all
(`lib-src/ComputeTriangs.cc:46` vs `:65`).

All work happens in the constructor: `_init(); _run();`
(`lib-src/SymmetricExtensionGraphMaster.hh:288-289`).

### 7.2 One-time table construction (`_init`, `lib-src/SymmetricExtensionGraphMaster.cc:249-499`)

In order:

1. `SimplicialComplex::reset_index_table()` then
   `preprocess_index_table(no, 0, min(no, rank+1), chiro, /*only_independent=*/true)`
   (`:283-284`; implementation `lib-src/SimplicialComplexTemplate_Definitions.hh:229-261`).
   This enumerates **all** `C(n, rank)` subsets in lex order, keeps those with `χ ≠ 0`, and
   interns them — so **simplex index order == lexicographic order on the sorted vertex
   tuple**. It then recursively interns every subset of every cardinality `< rank` and
   `rank+1`. (See §1.3 for the global-state hazard this creates.)
2. `Circuits(chiro)` (`:298`, `lib-src/Circuits.cc:75-95`) — all `C(n, rank+1)` circuits.
3. `Admissibles(circuits, points, chiro, only_fine)` (`:303`) — see §7.4. `Circuits` is
   deleted right after (`:317`).
4. `Cocircuits(chiro, /*only_positive=*/true)` (`:402`, `lib-src/Cocircuits.cc:60-102`) —
   all `C(n, rank-1)` cocircuits, keeping only the ones with an empty side.
5. `Facets(cocircuits)` (`:409`, `lib-src/Facets.cc:17-62`) — a positive cocircuit
   `(Z⁺, ∅)` gives the configuration facet `groundset \ Z⁺`. `Cocircuits` deleted at `:413`.
6. `Incidences(chiro, facets)` (`:417`) — see §7.5. `Facets` deleted at `:439`.

Volumes are built only under `--usevolumes` (`:250-260`), which is off.

### 7.3 The node: `PartialTriang`

`lib-src/PartialTriang.hh:32-42`. A `SimplicialComplex` plus four incrementally maintained
fields:

| field | update | invariant |
|---|---|---|
| the complex `T` | `insert(simp)` (`:252`) | a set of pairwise-compatible full-dimensional simplices |
| `_admissibles` | `*= Adm[simp]; -= forbidden` (`:264-287`) | every simplex compatible with **all** of `T` and not lex-excluded |
| `_freeintfacets` | `^= inctable->intfacets(simp)` (`:290`) | **XOR**: interior facets covered an *odd* number of times |
| `_freeconffacets` | erase each covered configuration facet (`:295-300`) | boundary facets not yet touched |
| `_covered_volume` | `:304-308` | only if a `Volumes` table exists — not here |

The `_freeintfacets` XOR is the key invariant: in a triangulation every **interior** facet
is shared by exactly two simplices, so it is covered an even number of times. Hence

> `T` is a complete triangulation ⟺ `_freeintfacets` is empty.

That is literally the termination test (`lib-src/SymmetricExtensionGraph.cc:1210`):

```cpp
// next partial triang is new - check for completeness:
if (next_partial_triang.freeintfacets().empty()) { /* report */ }
```

`_freeconffacets` is *not* checked at completion — a closed odd-cover of all interior facets
by pairwise-compatible full-dimensional simplices necessarily fills the configuration.

The root (`lib-src/PartialTriang.hh:88-121`, built at
`lib-src/SymmetricExtensionGraphMaster.cc:460`) is the empty complex with `_admissibles` =
all keys of the `Admissibles` table and `_freeconffacets` = all configuration facets.

(`PartialTriang::forbid()` at `:227-232` and `PartialTriang::complete()` at `:233-247` are
dead code — nothing calls them. Same for `SymmetricExtensionGraph::_bfs`
at `lib-src/SymmetricExtensionGraph.cc:678-943`.)

### 7.4 `Admissibles` — the compatibility table

`lib-src/Admissibles.hh:31` — `unordered_map<Simplex, SimplicialComplex>`.
Construction `lib-src/Admissibles.cc:66-216`:

1. Build the **universe** of candidate simplices: all `rank`-subsets with `χ ≠ 0`
   (`:114`), optionally filtered to *empty* simplices when `only_fine` (`:116-131`).
2. `VertexFacetTable vertexfacets(universe)` (`:140`) — a vertex → containing-simplices index.
3. For every `s` in the universe (`:191-214`), start from the whole universe and, for every
   circuit `(Z⁺, Z⁻)`: if `Z⁺ ⊆ s` remove the star of `Z⁻`; if `Z⁻ ⊆ s` remove the star of
   `Z⁺` (`:202-208`). Finally remove `s` itself (`:209`).

So `Adm[s]` = **all simplices whose interiors are disjoint from `s`'s** (they intersect in a
common face). Two simplices conflict exactly when a circuit witnesses overlapping interiors.
Pruning then costs one bitset AND per node.

### 7.5 `Incidences`

`lib-src/Incidences.hh:40-43`, built in `lib-src/Incidences.cc:22-183`. Four maps:

* `_intfacets : Simplex → SimplicialComplex` — the **interior** facets of each simplex
  (a facet is interior iff no configuration facet contains it, `:57-73`);
* `_conffacets : Simplex → set<facet>` — which configuration facets absorb its exterior facets;
* `_intcofacets : interior facet → simplices containing it` (`:109-142`);
* `_confcofacets : configuration facet → simplices with a facet in it` (`:143-176`).

Only `_intfacets`/`_conffacets` are needed for the default search
(`lib-src/PartialTriang.hh:290-301`); the cofacet maps feed the stronger, non-default
obstruction propagation (`--fullextensioncheck`,
`lib-src/SymmetricExtensionGraph.cc:284-447,518-544`) and `RegularityCheck`.

`lib-src/InteriorFacets.{hh,cc}` is the pre-1.0 predecessor of `Incidences` and is
**never called** — drop it.

### 7.6 The DFS

`SymmetricExtensionGraph::_dfs(node, depth)`
(`lib-src/SymmetricExtensionGraph.cc:945-1507`), entered once from
`lib-src/SymmetricExtensionGraph.hh:367`. `depth` tracks `card(T)`.

```
_dfs(T):
    forbidden = ∅                                   # :948, one per node
    for s in T.admissibles(), in ASCENDING SIMPLEX INDEX ORDER:   # :986-988
        # lex pruning: can s still cover the lex-minimal uncovered interior facet?
        min_coverable = s.lexmin_subset(rank-1)      # :1013  (drop the largest vertex)
        if T.freeintfacets() non-empty
           and min(T.freeintfacets()).lexsmaller(min_coverable):
            break                                    # :1027  -- BREAK, not continue

        forbidden += s                               # :1030
        T' = PartialTriang(T, s, forbidden)          # :1043

        if T'.freeintfacets() non-empty:
            if T'.admissibles().empty():  continue   # :1092-1111  branching dead end
            if noncoverable_freefacet(T'): continue  # :1123-1138  early dead end
        if T'.freeintfacets().empty():
            emit(T')                                 # :1210, callback at :1415
            continue
        _dfs(T')                                     # :1493-1498
```

Two things carry the whole algorithm:

**(a) The `forbidden` set is the canonicity device.** `_update_admissibles` does
`_admissibles *= Adm[s]; _admissibles -= forbidden` (`lib-src/PartialTriang.hh:279-283`),
and `forbidden` accumulates every already-tried sibling *before* the child is built
(`lib-src/SymmetricExtensionGraph.cc:1030`). Since admissibles only ever shrink, every
root-to-node path adds simplices in **strictly increasing simplex index**. A triangulation
`T = {t_1 < t_2 < … < t_k}` therefore has exactly one possible generating path
(add `t_1`, then `t_2`, …), and that path is never blocked (each `t_{i+1}` is compatible
with all of `t_1..t_i`, so it survives the intersections, and its index exceeds every
forbidden sibling). **Hence each triangulation is produced exactly once, and all of them
are produced.** No flip-connectivity is assumed anywhere.

**(b) The lex-facet rule is the pruning that makes it tractable.**
`lexmin_subset(rank-1)` (`lib-src/IntegerSet.cc:2192-2195`) drops the largest vertex, i.e.
returns the lex-smallest facet of `s`, and is monotone non-decreasing in the lex order of
simplices. So once the lex-minimal uncovered interior facet is lex-smaller than the
lex-smallest facet of `s`, no remaining (lex-larger) sibling can cover it either — hence
`break`, cutting the entire sibling suffix rather than one child. This is the standard
"the lex-minimal uncovered facet must be covered by the next simplex added" rule, and it
is the single highest-leverage optimisation in the file. It is guarded by
`!no_extension_check && !use_volume_order && !use_random_order`
(`lib-src/SymmetricExtensionGraph.cc:1007-1010`), all true by default.

The same predicate reappears as a node-level test, `_noncoverable_freefacet_lex`
(`:455-477`); with default options `_noncoverable_freefacet` dispatches to it (`:546-559`).

### 7.7 `only_fine` in the extension graph

Unlike the flip graph, here `only_fine` **restricts the universe of simplices** and so
prunes the search tree correctly and unconditionally. From
`lib-src/Admissibles.cc:114-135`:

```cpp
if (chiro[simp] != 0) {
  bool is_empty(true);
  if (only_empty_simplices) {
    for (circuit : circuits)
      if ((simp.superset(circuit.first)  && circuit.second.card() == 1)
       || (simp.superset(circuit.second) && circuit.first.card()  == 1)) { is_empty = false; break; }
  }
  if (is_empty) universe += simp;
}
```

A simplex `S` is **empty** iff no point of the configuration lies in its relative interior;
that is witnessed by a circuit with `Z⁺ ⊆ S` and `|Z⁻| = 1` (the singleton being a point
inside `conv(Z⁺)`), or symmetrically. And a triangulation is fine ⟺ all its simplices are
empty (any point not a vertex of the simplex containing it would make that simplex
non-empty). So excluding non-empty simplices from the universe makes every enumerated
triangulation fine, with **no** extra test at the completion check. This is the semantics
the Rust port should adopt for both enumerators (§10.4).

Note `only_fine` is also forwarded to `Volumes(points, only_fine)` at
`lib-src/SymmetricExtensionGraphMaster.cc:256` — where the parameter is called
`only_unimodular_simplices`. That is dead here (`use_volumes()` is false) and in any case
only affects a sort key (`CompareSimplexVolumes`, `lib-src/Volumes.hh:87-101`), not
correctness.

### 7.8 Parallelism

Two independent layers, both off by default:

* **Enumeration parallelism** — `SymmetricExtensionGraphMaster::Worker`, gated by
  `parallel_enumeration()` (default false, `lib-src/CommandlineOptions.cc:439-440`).
* **Symmetry-check parallelism** — `SymmetricExtensionGraph::SymmetryWorker`, gated by
  `parallel_symmetries()` (default false).

With the project's defaults `_run()` takes the serial branch
(`lib-src/SymmetricExtensionGraphMaster.cc:200-246`): `_node_budget = SIZE_MAX`, the root is
pushed once, and a single `SymmetricExtensionGraph` recursion does the whole enumeration —
`_open_nodes` never grows, the loop runs once, no threads are created.

The parallel design, should you want to copy it: a master owns a FIFO of *unexplored partial
triangulations*; workers get a finite `_node_budget` and, when they exceed it, stop
recursing and push the node back (`lib-src/SymmetricExtensionGraph.cc:1482-1498`), which the
master splices into its deque (`lib-src/SymmetricExtensionGraphMaster.cc:132-134`) and
re-hands-out round-robin (`:167-183`). Budget policy
`max(1, |open|² · scale/100)` clamped to `[min_nodebudget, max_nodebudget]`
(`lib-src/SymmetricExtensionGraphMaster.hh:297-305`). One mutex `_main_waitmutex` protects
all worker state; `IO_sync::mutex` serialises output **and the callback**
(`lib-src/SymmetricExtensionGraph.cc:1415-1417`), so callbacks are mutually excluded but can
arrive on different threads.

This design (budgeted DFS with node stealing) translates very naturally to `rayon` or a
hand-rolled work-stealing deque, and is the obvious place to beat the C++, which ships with
it disabled.

### 7.9 What the trivial symmetry group removes

`SymmetryGroup symmetries(no)` has `size() == 0`, so:

| site | effect |
|---|---|
| `lib-src/SymmetricExtensionGraphMaster.cc:318-398` | `_simpidx_symmetriesptr`, `_classified_symmetriesptr`, `_switch_tableptr` all stay `nullptr` |
| `lib-src/SymmetricExtensionGraphNode.cc:36-44` | root `_critsimpidx_table` is zero-length |
| `lib-src/SymmetricExtensionGraph.cc:1174-1188` | dispatches to `child_is_lexmin_lean` |
| `lib-src/SymmetricExtensionGraphNode.cc:195-294` | the orbit loop never runs → **`return true` unconditionally** (`:293`) |
| `lib-src/SymmetricExtensionGraph.cc:1427-1449` | `_orbitsize = (0+1)/(0+1) = 1` → `_totalcount == _symcount` |
| `SymmetryWorker`, `ClassifiedExtensionSymmetries`, `SwitchTable`, `PermutationPartialComparator` | never invoked |

So the orbit-representative test collapses to a constant `true`, and uniqueness is carried
entirely by the increasing-index `forbidden` rule (§7.6a). Deleting the symmetry layer is
sound.

One residual cost worth removing in the port:
`_symmetry_images_by_element.resize(no_of_simplices(rank), std::vector<size_type>(0, …))`
(`lib-src/SymmetricExtensionGraphNode.hh:146`) allocates one empty `std::vector` (24 bytes)
per full-dimensional simplex even when the group is trivial.

### 7.10 Performance

**Setup dominates for moderate `n`**, and is exponential:

| step | cost |
|---|---|
| `preprocess_index_table` | `C(n, rank)` chirotope sign evaluations + interning; plus all subsets of every smaller cardinality |
| `Circuits` | `C(n, rank+1)` circuit computations |
| `Admissibles` | **asymptotic hot spot**: `O(#universe × #circuits) ≈ C(n,rank) × C(n,rank+1)`, and the result is `#universe` bitsets of `#universe` bits ⇒ `Θ(C(n,rank)²/8)` bytes |
| `Cocircuits` | `C(n, rank-1)` |
| `Incidences` | `O(C(n,rank) × rank × #facets)` plus the reverse maps |

**Search**: per-node work is dominated by the `PartialTriang` **copy** at
`lib-src/SymmetricExtensionGraph.cc:1043` — it copies the complex, `_admissibles`,
`_freeintfacets` and the `std::set` `_freeconffacets`, then does three bitset ops. Node
count is exponential; `_deadendcount`/`_earlydeadendcount` measure the pruning. Recursion
depth equals the size of a triangulation and each frame owns a full `PartialTriang` plus a
`forbidden` complex.

**Rust wins available for free**: carry `_admissibles` and `_freeintfacets` as
fixed-width `Vec<u64>` bitsets over the (now known-size) universe and undo them on backtrack
— XOR is self-inverse for `freeintfacets`, and `admissibles` can be restored from a trail or
a single saved copy per depth — instead of copying a whole `PartialTriang` per node.

---

## 8. Arithmetic and numerics

### 8.1 Where exact arithmetic is genuinely required

Exactly one place: **the sign of an `r × r` determinant of columns of the configuration**
(`VirtualChiro::operator()`, `lib-src/VirtualChiro.hh:157-164`;
`StairCaseMatrix::det()`, `lib-src/StairCaseMatrix.cc:178-198`).

Everything downstream — circuits, `containment_ok`/`link_ok`, flips, both enumerations,
`Facets`, `Incidences`, `Admissibles` — is **purely combinatorial** and consumes only
`{-1,0,+1}`. There is no other numeric computation in the slice we need. (`Volumes`,
`points.volume()`, the unimodularity predicate and `RegularityCheck` do use magnitudes, and
all three are droppable.)

Even the sign must be *exactly* right: a single wrong sign silently changes the oriented
matroid and yields a wrong (possibly non-)triangulation with no diagnostic.

### 8.2 Entry-size growth

Input entries are `int64` (`pybind11::array_t<int64_t>` throughout the project's API),
plus the homogenising `1`. Let `M = max |entry|` and `r = rank = d+1` (point config) or `d`
(vector config). By Hadamard,

```
|det| ≤ ∏ ‖column‖₂ ≤ (√r · M)^r
```

so `bits(det) ≤ r · (log2 M + ½ log2 r)`.

Practical numbers for lattice polytope work (the project's use case, `M` small):

| `M` | `r = 5` | `r = 10` | `r = 20` |
|---|---|---|---|
| 1 | 6 bits | 17 | 43 |
| 10 | 23 | 45 | 107 |
| 10³ | 56 | 112 | 240 |
| 2³¹ | 166 | 327 | 663 |

So `i128` accumulators are safe for `r·log2 M ≲ 120`, which covers essentially all
realistic lattice-polytope inputs (`r ≤ 10`, `M ≤ 10³`) with a wide margin, but **not**
adversarial `int64` coordinates.

TOPCOM itself is far more wasteful than the bound suggests: `StairCaseMatrix::_eliminate`
(`lib-src/StairCaseMatrix.cc:81-86`) does *fraction-producing* Gaussian elimination
(`multiplier = delinquent / eraser`) over `mpq_class`, so intermediate numerators and
denominators blow up well beyond the determinant's own size. It even carries a commented-out
division-free variant at `lib-src/StairCaseMatrix.cc:66-80`. A Rust port should **not**
copy this.

### 8.3 Recommended Rust numerics

Compute determinant *signs* over the integers, never rationals:

1. **Fraction-free Bareiss elimination** on `i64`/`i128`. Every intermediate is itself a
   minor of the input matrix, so the Hadamard bound above applies to *every* intermediate,
   not just the result — no growth beyond `(√r·M)^r`.
2. Run in `i128` with `checked_mul`/`checked_sub`; on overflow, fall back to the same
   Bareiss routine over `num_bigint::BigInt` for that basis only. Overflow will be
   astronomically rare for realistic inputs, so the fallback costs nothing in practice.
3. Do **not** pull in `rug` (GMP FFI, build friction, licensing footprint). `num-bigint` is
   pure Rust and only ever exercised on the rare fallback path. If profiling ever shows the
   fallback is hot, `malachite` is the faster pure-Rust option.

Optional extra speed: a floating-point filter (compute the determinant in `f64` with an
error bound; only redo exactly when the bound straddles zero). Standard in CGAL-style
predicates; worth it only if the chirotope turns out to dominate.

### 8.4 cddlib

Used only by `lib-src-reg/LPinterface.hh:24-25` → `RegularityCheck`, which is only
constructed under `--regular`/`--nonregular`/`--checkfreq`/`--heights`
(`lib-src/SymmetricFlipGraph.hh:143,202,248,285`;
`lib-src/SymmetricExtensionGraph.cc:817,844,875,1246,1302,1347`). None of those is reachable
from this project. **cddlib can be dropped entirely.** If regularity checking is ever wanted
in Rust, it is a single LP feasibility problem per triangulation (does a height vector exist
whose lower hull induces `T`?), solvable with an exact rational simplex or by delegating to
`good_lp`/`clarabel` with an exactness caveat.

---

## 9. Dependency slice: files needed vs droppable

Computed as the transitive `#include` closure from the entry points.

### 9.1 Needed for the flip-graph / placing / fine / flips features

| file | role |
|---|---|
| `lib-src/Global.hh` | `block_type`, `size_type`, `parameter_type`, `_binomial` |
| `lib-src/Field.hh` | `Field = Rational`, `sign()` — replaced by integer sign in Rust |
| `lib-src/Vector.hh/.cc` | one configuration column |
| `lib-src/Matrix.hh/.cc` | column-major matrix, `row_echelon_form`, `left_upper_det` |
| `lib-src/StairCaseMatrix.hh/.cc` | incremental staircase form + `det()` — **the determinant kernel** |
| `lib-src/StrictStairCaseMatrix.hh/.cc` | strict variant, used only by `PointConfiguration::_compute_rank` |
| `lib-src/StrictStairCaseMatrixTrans.hh/.cc` | needed only by `PointConfiguration::gale()` — droppable |
| `lib-src/PointConfiguration.hh/.cc` | `no()`, `rank()`, `volume()` |
| `lib-src/IntegerSet.hh/.cc` | the bitset (`Simplex`, index sets) |
| `lib-src/LabelSet.hh`, `lib-src/Simplex.hh` | typedefs |
| `lib-src/Permutation.hh/.cc` | `lexnext()` subset enumeration, `sign()` |
| `lib-src/RealChiro.hh/.cc` | dense chirotope + `find_non_deg_basis` |
| `lib-src/VirtualChiro.hh` | **lazy** chirotope (this is what `Chirotope` resolves to) |
| `lib-src/Chirotope.hh` | typedef switch |
| `lib-src/SimplicialComplexTemplate_*.hh` | the interned-bitset complex; `link`, `join`, `support`, `insert_boundary` |
| `lib-src/FastSimplicialComplex.hh/.cc`, `lib-src/SimplicialComplex.hh` | typedef layer |
| `lib-src/IndexTableArray.hh`, `lib-src/PlainHashIndexTable.hh` | the global simplex interner (§1.3) |
| `lib-src/HashKey.hh` | the Murmur-ish hash over `keysize()/key(n)` |
| `lib-src/Circuits.hh/.cc` | `Circuit(chiro, depset)`, `upper_facets`, `lower_facets` (the `Circuits` *table* is only needed by the extension graph) |
| `lib-src/PlacingTriang.hh/.cc` | §3 |
| `lib-src/FineTriang.hh/.cc` | §4 |
| `lib-src/Flip.hh/.cc` | `FlipRep`, `Flip` |
| `lib-src/TriangNode.hh/.cc` | node + `containment_ok` / `link_ok` |
| `lib-src/TriangFlips.hh/.cc` | flip-set construction and incremental update |
| `lib-src/MarkedFlips.hh` | marked flip map |
| `lib-src/SymmetricFlipGraph.hh/.cc` | the BFS |
| `lib-src/CommandlineOptions.hh/.cc` | replace with a small config struct or constants |

### 9.2 Additionally needed for the extension graph

| file | role |
|---|---|
| `lib-src/Cocircuits.hh/.cc` | cocircuits from the chirotope (`Cocircuit(chiro, coplanar_set)`) |
| `lib-src/Facets.hh/.cc` | facets of the configuration, from positive cocircuits |
| `lib-src/Incidences.hh/.cc` | simplex ↔ interior-facet / configuration-facet incidences |
| `lib-src/Admissibles.hh/.cc` | per-simplex set of simplices compatible with it |
| `lib-src/PartialTriang.hh` | a partial triangulation node |
| `lib-src/SymmetricExtensionGraphNode.hh/.cc` | search-tree node |
| `lib-src/SymmetricExtensionGraph.hh/.cc` | the backtracking search |
| `lib-src/SymmetricExtensionGraphMaster.hh/.cc` | setup + work distribution |
| `lib-src/VertexFacetTable.hh/.cc` | helper for `Admissibles` (`remove_star`) |

### 9.3 Droppable outright

* `lib-src-reg/*` (`LPinterface`, `SPXinterface`, `QSOinterface`, `RegularityCheck`) — cddlib.
* `lib-src/Symmetry.hh/.cc`, `lib-src/Permutation*Comparator.*`, `lib-src/SwitchTable.*`,
  `lib-src/ClassifiedSubsetSymmetries.*`, `lib-src/ClassifiedExtensionSymmetries.*`,
  `lib-src/PermlibPermutationGroup.hh` — trivial group (§1.2, §6.4, §7.9).
* `lib-src/Volumes.hh/.cc` — GKZ only (§6.5).
* `lib-src/Statistics.hh/.cc` — `--stats`; not referenced by `SymmetricFlipGraph` at all.
* `lib-src/Graphics.hh/.cc` — Asymptote output.
* `lib-src/CheckTriang.hh/.cc` — `--checktriang` validation.
* `lib-src/ComputeTriangs.*`, `lib-src/ComputeCircuits.*`, `lib-src/ComputeCocircuits.*` —
  CLI drivers (still useful as *reference* for how the pieces are wired).
* `lib-src/SymmetricSubsetGraph*.hh` — the generic subset-graph machinery, not on our path.
* `lib-src/InteriorFacets.*`, `lib-src/VertexFacetTableArray.*` — not reached.
* `lib-src/SparseIntegerSet.*`, `lib-src/CompressedIntegerSet.*`,
  `lib-src/SparseSimplicialComplex.*`, `lib-src/IntegerSet64.*` — alternative
  representations behind macros this build does not set.
* `lib-src/HashSet.hh`, `lib-src/HashMap.hh`, `lib-src/PlainHash*.hh`,
  `lib-src/PlainArray.hh`, `lib-src/Array.hh`, `lib-src/Pair.hh`, `lib-src/RefCount.hh`,
  `lib-src/ContainerIO.hh` — non-STL container reimplementations, compiled out.
* `lib-src/Vertices.*` — Asymptote graphics only.
* `src/*.cc`, `src-reg/*.cc` — the ~60 CLI binaries.
* `wrap-gmp-gmpxx/`, `external/` — GMP/cddlib/qsopt bundling.
* All `read`/`write`/`print_string`/`operator<<`/`operator>>` members everywhere.

---

## 10. Edge cases and validation

### 10.1 What the project already validates

`validate_configuration` (`src/triangulumancer/TOPCOM.cpp:61-75`) rejects:

* `rank() < rowdim()` — "Configuration is not full rank";
* `no() < 1 || rank() < 1`;
* `rank() > no()`.

Keep these. Note `PointConfiguration::rank()` recomputes the rank by elimination on every
call (`lib-src/PointConfiguration.cc:87,228-245`) — three calls per validation.

### 10.2 Point vs vector configurations

The only difference is the homogenising `1`
(`src/triangulumancer/PVConfiguration.cpp:45-52,128-133,145-152`):

* point config in `R^d` → columns in `R^{d+1}`, `rank = d+1`, simplices have `d+1` vertices;
* vector config in `R^d` → columns in `R^d`, `rank = d`, simplices have `d` vertices.

TOPCOM itself is agnostic: it only ever sees a rank-`r` vector configuration and its
chirotope. **Everything in the port should be written against "vector configuration of
rank r"**, with homogenisation done once at the API boundary — exactly as the project does
today.

Consequence for output shape: `simplicial_complex_to_triangulation`
(`src/triangulumancer/TOPCOM.cpp:15-40`) writes `dim + is_pc` labels per simplex and
**assumes every simplex is full-dimensional** (has exactly `rank` vertices). That holds
because every triangulation TOPCOM produces is pure of rank `rank`. If the Rust port ever
produces a non-pure complex, this marshalling breaks silently.

### 10.3 Degenerate configurations

* **Duplicate points.** `PointConfiguration::read` rejects the all-zero column
  (`lib-src/PointConfiguration.hh:143-153`) but that path is not used by the project
  (the matrix is filled directly). Duplicates are *not* rejected: two equal columns give
  `χ(B) = 0` for every basis containing both. A duplicated point can never be a simplex
  vertex together with its twin, and interacts with `only_fine`: a fine triangulation must
  use *all* points, which is impossible if two points coincide. Worth an explicit check in
  the Rust API.
* **Points on a common hyperplane through the origin / not full rank.** Rejected by
  `validate_configuration`. `Chirotope` would otherwise be identically zero and
  `find_non_deg_basis` would `assert` (`lib-src/RealChiro.cc:80`).
* **Special-position circuits** (`|supp(Z)| < rank+1`): handled explicitly everywhere
  (`lib-src/Flip.cc:47-57,84-114`, `lib-src/TriangNode.cc:34-76`). This is the code path
  that is easiest to get wrong and that the fork already had to patch
  (`lib-src/FineTriang.cc:39-63`, commit `e9494f3`). **Build a regression suite of
  degenerate configurations before porting.**
* **Circuits with an empty side.** `Circuit::Circuit` (`lib-src/Circuits.cc:43-72`) can
  produce `Z⁻ = ∅` when the dependent set is degenerate; `FlipRep::operator bool` rejects
  those (`lib-src/Flip.hh:77`), and `FineTriang` skips them (fork patch).
* **`rank == no`** — a single simplex; `Circuits::Circuits` returns empty
  (`lib-src/Circuits.cc:78-80`), the flip graph has one node.

### 10.4 What "fine" means

A triangulation is **fine** (= *full*, = *using all points*) iff `support(T) == [0..n)`.
TOPCOM encodes this two different ways in the two enumerators:

* **Flip graph**: `only_fine` ⇒ `forbid_vertex_removal` ⇒ drop flips with
  `|Z⁻| == 1` (`lib-src/TriangFlips.cc:116-118,186-188`,
  `lib-src/Flip.hh:82-84`). *Requires a fine seed* (§6.3).
* **Extension graph**: `only_fine` ⇒ restrict the simplex universe to **empty simplices**
  (`Admissibles(circuits, points, chiro, only_fine)` at
  `lib-src/SymmetricExtensionGraphMaster.cc:303`; `only_empty_simplices` at
  `lib-src/Admissibles.cc:69,114-135`). A simplex `S` is *not* empty iff some circuit
  `(Z⁺,Z⁻)` has `Z⁺ ⊆ S` with `|Z⁻| = 1` (or symmetric) — i.e. some point lies in the
  convex hull of a subset of `S`. And indeed: a triangulation is fine ⟺ all its simplices
  are empty (any point not a vertex of the simplex containing it would make that simplex
  non-empty). This is a genuine **pruning** of the search, not an output filter.

The Rust port should expose a single explicit `is_fine(T) == (support(T).len() == n)`
predicate and use it as a post-filter as well, so the two paths agree.

### 10.5 Hard exits

TOPCOM `exit(1)`s on many internal error paths
(`lib-src/RealChiro.hh:166`, `lib-src/PlainHashIndexTable.hh:141`,
`lib-src/IndexTableArray.hh:64`, `lib-src/SimplicialComplexTemplate_Declarations.hh:143,169`,
`lib-src/Symmetry.hh:555`, `lib-src/SymmetricFlipGraph.cc` various). Inside a Python
extension this kills the interpreter. **Every such path becomes a `Result::Err` in Rust,
surfaced as a Python exception.** This alone is a strong argument for the rewrite.

---

## 11. Port complexity assessment and risks

LOC estimates are for idiomatic Rust including tests; TOPCOM LOC is `.hh + .cc` of the
corresponding files (which include a lot of I/O and macro-branch noise).

| Component | TOPCOM LOC | Rust LOC | Difficulty | Risk |
|---|---:|---:|---|---|
| `Config` (column-major integer matrix, rank) | ~1 100 | 150 | trivial | — |
| `Simplex` bitset | ~3 000 | 200 | easy | — |
| Determinant sign (Bareiss, i128 + bigint fallback) | ~360 | 200 | easy–medium | overflow handling |
| `Chirotope` (lazy, memoised) | ~980 | 150 | easy | cache memory |
| `SimplexSet` / triangulation container | ~3 300 | 250 | easy–medium | perf of the naive version |
| `Circuit` + `FlipRep` + `Flip` | ~640 | 350 | **medium–hard** | special-position cases |
| `containment_ok` / `link_ok` | ~80 | 100 | **medium–hard** | subtle, poorly documented |
| `PlacingTriang` | ~170 | 120 | easy–medium | `^=` boundary maintenance |
| `FineTriang` | ~130 | 80 | medium | degenerate circuits |
| `TriangFlips` (incremental) | ~430 | 300 | **hard** | correctness of the incremental update |
| Flip-graph BFS | ~2 000 | 250 | medium | memory; only_fine seed bug |
| `Cocircuits` + `Facets` + `Incidences` | ~790 | 500 | medium | `C(n,r)` blow-up |
| `Admissibles` | ~430 | 250 | medium | — |
| Extension-graph backtracking | ~3 200 | 800–1 200 | **hardest** | completeness, lex-canonicity |
| PyO3 bindings + API surface | — | 400 | easy | — |
| **Total** | ~16 600 | **~4 000–4 500** | | |

### 11.1 Ranked risks

1. **Extension-graph enumeration.** ~3 200 lines of the densest code in TOPCOM, with the
   lex-order canonicity argument that guarantees each triangulation is produced exactly
   once. Getting this subtly wrong yields silently missing or duplicated triangulations.
   *Mitigation*: differential-test against the C++ (counts **and** sets) on every
   configuration in `tests/`, plus cyclic polytopes, products of simplices, and known
   `n_triangulations` values from the TOPCOM examples (`examples/`).
2. **Special-position circuits / `link_ok`.** ~80 lines of C++ carrying the entire
   degenerate-input correctness story, with no comments and at least one fork bug fix
   already. *Mitigation*: exhaustive small-case tests (all `0/1` configurations up to
   `n ≤ 8`, `d ≤ 4`) comparing the full flip set against the C++.
3. **Incremental `TriangFlips` update.** If `_remove_destroyed_flips` /
   `_add_new_flips`-with-restriction is wrong, the BFS silently under-explores.
   *Mitigation*: implement the naive "recompute all flips" first, use it as the oracle, then
   add the incremental version behind a debug-assert that they agree.
4. **`only_fine` semantics.** Two different meanings in the two enumerators (§10.4), plus
   the existing seed bug (§6.3). *Mitigation*: define `is_fine` once, post-filter in both
   paths, and document that the flip-graph variant only explores the fine-induced component.
5. **Memory in the flip graph.** TOPCOM keeps two layers plus a `TriangFlips` per node.
   The single-visited-set simplification is simpler but stores every triangulation. For
   configurations with 10⁶–10⁸ triangulations neither fits; a Rust port should expose a
   *streaming* iterator (callback per triangulation) as the primary API and collect into a
   `Vec` only on request — the current Python API always collects
   (`src/triangulumancer/TOPCOM.cpp:150-152,190-192`).
6. **Performance regression.** TOPCOM's bitset-over-interned-indices representation makes
   triangulation equality/hashing very cheap. A naive `HashSet<Simplex>` port may be several
   times slower on the big enumerations. *Mitigation*: benchmark early; keep the interner
   option open behind a trait.

### 11.2 Suggested ordering

1. `Config` + determinant sign + `Chirotope` (with a `points2chiro`-style differential test).
2. `Simplex` / `SimplexSet` + `link`/`join`/`support`/`insert_boundary`.
3. `PlacingTriang`, `FineTriang` → ship `placing_triangulation` / `fine_triangulation`.
4. `Circuit`/`FlipRep`/`Flip`/`containment_ok`/`link_ok` (naive flip enumeration) → ship
   `flips` / `neighbors`.
5. Flip-graph BFS (naive visited set first) → ship `all_connected_triangulations`.
6. Incremental `TriangFlips` as an optimisation.
7. `Cocircuits`/`Facets`/`Incidences`/`Admissibles` + extension graph → ship
   `all_triangulations`.

Steps 1–5 cover four of the six public methods and are the low-risk majority of the work;
step 7 is where the schedule risk lives.

---

## 12. References

### TOPCOM source (checkout root as above)

| Topic | Path |
|---|---|
| Build defines | `CMakeLists.txt:85`; project `CMakeLists.txt:21` (`USE_PHIT_PREPROCESSING=false`) |
| Option defaults | `lib-src/CommandlineOptions.cc:18-85` |
| `Field = mpq_class` | `lib-src/Field.hh:17`, `wrap-gmp-gmpxx/Rational.h:18` |
| Column-major matrix | `lib-src/Matrix.hh:38-119` |
| Rank by elimination | `lib-src/PointConfiguration.cc:228-245` |
| Determinant kernel | `lib-src/StairCaseMatrix.cc:17-91,178-198` |
| Lazy chirotope | `lib-src/VirtualChiro.hh:130-184,270-292` |
| Dense chirotope DFS | `lib-src/RealChiro.cc:23-33,262-346` |
| Non-degenerate basis | `lib-src/RealChiro.cc:67-82` |
| Bitset | `lib-src/IntegerSet.hh:53-419` |
| Simplex interning | `lib-src/IndexTableArray.hh:34-175`, `lib-src/PlainHashIndexTable.hh:50-185` |
| `USE_PHIT_PREPROCESSING` | `lib-src/PlainHashIndexTable.hh:130-144` |
| Index-table preprocessing / reset | `lib-src/SimplicialComplexTemplate_Definitions.hh:146-261` |
| Complex representation | `lib-src/SimplicialComplexTemplate_Declarations.hh:40-313` |
| `link` / `star` / `join` / `support` / `insert_boundary` | `lib-src/SimplicialComplexTemplate_Definitions.hh:587-679,1054-1067,2152-2250` |
| Hash | `lib-src/HashKey.hh:29-49`; `Definitions.hh:293-320` |
| Circuit from chirotope | `lib-src/Circuits.cc:43-72` |
| Upper/lower facets | `lib-src/Circuits.cc:19-41` |
| Cocircuits | `lib-src/Cocircuits.cc:20-57` |
| Facets from cocircuits / boundary | `lib-src/Facets.cc:17-96` |
| Incidences | `lib-src/Incidences.cc:22-183` |
| Admissibles / empty simplices | `lib-src/Admissibles.cc:27-140` |
| Placing triangulation | `lib-src/PlacingTriang.cc:19-99` |
| Fine triangulation | `lib-src/FineTriang.cc:16-65` |
| `FlipRep` construction | `lib-src/Flip.cc:22-61` |
| `Flip` expansion | `lib-src/Flip.cc:65-115` |
| `containment_ok` / `link_ok` | `lib-src/TriangNode.cc:20-76` |
| Apply a flip | `lib-src/TriangNode.hh:111-119` |
| Flip enumeration | `lib-src/TriangFlips.cc:37-240` |
| Incremental flip update | `lib-src/TriangFlips.cc:20-35`, `lib-src/TriangFlips.hh:89-105` |
| Flip-graph BFS | `lib-src/SymmetricFlipGraph.cc:778-1008` |
| Flip-graph constructor | `lib-src/SymmetricFlipGraph.hh:473-483,532-573` |
| GKZ / volumes guard | `lib-src/SymmetricFlipGraph.cc:378-396` |
| Regularity guard | `lib-src/SymmetricFlipGraph.cc:1047-1091` |
| Seed choice (fine vs placing) | `lib-src/ComputeTriangs.cc:305-320` |
| Extension-graph constructor | `lib-src/SymmetricExtensionGraphMaster.hh:250-289` |
| Extension-graph setup | `lib-src/SymmetricExtensionGraphMaster.cc:249-320,400-425` |
| cddlib entry | `lib-src-reg/LPinterface.hh:24-25` |
| Trivial group | `lib-src/Symmetry.hh:391-397`, `lib-src/Symmetry.hh:573-595` |

### Fork-specific commits (relative to upstream TOPCOM 1.1.2)

| Commit | Change |
|---|---|
| `a433329` | `FineTriang.cc`: copy the simplex before `link()` — the global index table can reallocate under the reference |
| `e9494f3` | `FineTriang.cc`: replace `assert(!lower.empty())` / `assert(!upper.empty())` by `continue` (degenerate circuits) |
| `abd06a6` | `USE_PHIT_PREPROCESSING` flag (§1.3) |
| `b593277` | `std::function<void(const SimplicialComplex&)>` callback in `SymmetricFlipGraph`, `SymmetricExtensionGraph`, `SymmetricExtensionGraphMaster`, replacing stdout printing |

### Project source

| Topic | Path |
|---|---|
| TOPCOM call sites | `src/triangulumancer/TOPCOM.cpp` |
| Configuration storage / homogenisation | `src/triangulumancer/PVConfiguration.cpp:41-53,120-158` |
| Public API | `include/triangulumancer/PVConfiguration.hpp`, `include/triangulumancer/TOPCOM.hpp` |
| Build | `CMakeLists.txt:12-21` |

### Papers

* J. Rambau, *TOPCOM: Triangulations of Point Configurations and Oriented Matroids*,
  in Mathematical Software — ICMS 2002, pp. 330–340. The flip-graph BFS and the
  extension-graph enumeration are described here.
* J. A. De Loera, J. Rambau, F. Santos, *Triangulations: Structures for Algorithms and
  Applications*, Springer 2010. Chapter 2 (circuits, bistellar flips, the flip graph),
  Chapter 4 (placing/pushing, beneath–beyond), Chapter 8 (enumeration). This is the
  reference for every definition used above.
* J. Rambau, *Computing triangulations using oriented matroids*, in Algebra, Geometry and
  Software Systems, Springer 2003, pp. 77–96. The chirotope-only formulation the code
  implements.
* I. M. Gelfand, M. M. Kapranov, A. V. Zelevinsky, *Discriminants, Resultants and
  Multidimensional Determinants*, Birkhäuser 1994. GKZ vectors / secondary polytope
  (relevant only if regularity is ever reinstated).
