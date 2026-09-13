# CGAL → Rust: design input for reimplementing the regular-triangulation slice

**Status:** research / design input. No code in this repo has been changed.

**Audience:** whoever implements the Rust replacement for `src/triangulumancer/CGAL.cpp`.

All `path:line` references below are into a CGAL checkout. Clone with:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/CGAL/cgal.git cgal
cd cgal && git sparse-checkout set Triangulation/include NewKernel_d/include \
    Kernel_d/include Spatial_sorting/include Number_types/include Installation/include
```

Paths are written relative to that checkout root, e.g.
`Triangulation/include/CGAL/Regular_triangulation.h:965`.

---

## 1. Scope

### 1.1 What the project actually uses

`src/triangulumancer/CGAL.cpp` is 139 lines and uses exactly one CGAL capability:

| Used | Where |
|---|---|
| `CGAL::Epick_d<Dimension_tag<D>>`, `D = 1..10` | `CGAL.cpp:16`, `CGAL.cpp:109-127` |
| `CGAL::Epick_d<Dynamic_dimension_tag>` for `d > 10` | `CGAL.cpp:129-130` |
| `CGAL::Regular_triangulation<K>` | `CGAL.cpp:17` |
| `K::Weighted_point_d` construction from `(Point_d, double)` | `CGAL.cpp:19`, `CGAL.cpp:50` |
| `T::insert(first, last)` (range insert, i.e. with spatial sort) | `CGAL.cpp:56` |
| `t.vertices_begin()/vertices_end()`, `t.is_infinite(v)` | `CGAL.cpp:62-65` |
| `Regular_triangulation_traits_adapter<K>::Compute_coordinate_d` | `CGAL.cpp:20-24`, `CGAL.cpp:68` |
| `t.finite_full_cells_begin()/end()`, `cell->vertex(i)` | `CGAL.cpp:88`, `CGAL.cpp:93-98` |
| `t.is_valid()` (inside `CGAL_assertion`, so a no-op in release) | `CGAL.cpp:57` |

Input: `int64_t` coordinates (`PVConfiguration.hpp:33`, `pybind11::array_t<int64_t>`),
`double` weights or heights. Output: an `(n_simplices, d+1)` array of `int64_t` indices
into the input point list.

### 1.2 What we need to reimplement

1. **Regular (weighted Delaunay) triangulation** of a finite set of points with
   `i64` coordinates in `R^d`, given per-point real weights (or, equivalently, heights).
2. **Delaunay** as the special case "all weights zero".
3. Extraction of the top-dimensional simplices as index tuples into the input array.

### 1.3 What we explicitly do **not** need

These are big chunks of CGAL that the project never touches, and skipping them removes
most of the difficulty:

- **`remove()` / vertex deletion.** `Regular_triangulation::remove`
  (`Triangulation/include/CGAL/Regular_triangulation.h:474-790`) is the single hardest
  part of the package — it builds "dark" auxiliary triangulations and is explicitly
  documented as *incorrect* for regular triangulations ("Warning: these functions are not
  correct since they do not restore hidden vertices", `Regular_triangulation.h:181-182`).
  We never call it. **Do not port it.**
- `insert_if_in_star` (`Regular_triangulation.h:897-931`) — only used by `remove`.
- Nearest-neighbour / point-location queries as a public API.
- Constructions: circumcenters, `Power_center`, squared circumradius, dual/Voronoi.
- Kernel constructions in general (`Epick_d` is *inexact-constructions*; the project only
  uses predicates).
- Incremental / online insertion by the user after the triangulation is built.
- Periodic triangulations, meshing, alpha shapes, hierarchy/`Triangulation_hierarchy`.
- 2D/3D specialised packages (`Triangulation_2`, `Triangulation_3`) — the project uses
  the dD package uniformly.

---

## 2. The mathematics

### 2.1 Regular triangulations

A **weighted point** is a pair `(p, w)` with `p ∈ R^d`, `w ∈ R`. Its **power distance**
to a point `x` is

```
pow((p,w), x) = |x − p|² − w
```

Given weighted points `(p_0,w_0), …, (p_{n−1},w_{n−1})`, the **regular triangulation**
(a.k.a. weighted Delaunay triangulation, or coherent/regular subdivision) is the dual of
the power diagram: a simplex `σ = {p_{i_0},…,p_{i_d}}` is a cell iff there exists a point
`z` and a value `α` with `pow((p_{i_k}, w_{i_k}), z) = α` for all `k`, and
`pow((p_j,w_j), z) > α` for every other `j`. With all `w_i = 0` this is exactly the
Delaunay triangulation.

Two things distinguish it from Delaunay:

- Not every input point need be a vertex. A point whose weight is "too small" relative to
  its neighbours is **hidden** (also "redundant"). Delaunay has no hidden points (in
  general position every point is a vertex; only exact duplicates collapse).
- The vertex set, hence the triangulation, depends on the weights.

### 2.2 The lifting map

Define the **lift** of a weighted point as

```
ℓ(p, w) = |p|² − w        (a single real number)
p̂       = (p, ℓ(p,w)) ∈ R^{d+1}
```

**Theorem (standard; Brown 1979, Edelsbrunner–Seidel 1986, Ziegler *Lectures on
Polytopes* §5).** The regular triangulation of `{(p_i, w_i)}` is the vertical projection
to `R^d` of the **lower faces** of the convex hull `conv{ p̂_0, …, p̂_{n−1} } ⊂ R^{d+1}`,
where "lower" means the faces whose outward normal has negative last coordinate.

- A point `p_i` is a **vertex** of the regular triangulation iff `p̂_i` is a vertex of the
  lower hull. A **hidden point** is exactly a `p̂_i` that lies strictly above the lower
  hull (or on it but not as a vertex).
- If no `d+2` of the `p̂_i` lie on a common lower supporting hyperplane, every lower face
  is a simplex and the projection is a triangulation. Otherwise the projection is a
  **regular subdivision** with non-simplicial cells, and one must perturb (symbolically or
  otherwise) to obtain a triangulation. See §5.4.

This is the central fact for the port: **the regular triangulation is a convex-hull
problem in `d+1` dimensions**, and the only geometric predicate needed is the
`(d+1)`-dimensional orientation predicate applied to lifted points.

### 2.3 Weights vs. heights — the project's convention

`src/triangulumancer/CGAL.cpp:41-48` computes, when `use_heights` is set:

```cpp
double weight = params[i];   // the height h_i
weight *= -1;
for (j) { int sqrt_h0 = points_buf[i*dim+j]; weight += sqrt_h0 * sqrt_h0; }
// weight  ==  −h_i + |p_i|²
```

i.e. `w_i = |p_i|² − h_i`, equivalently `h_i = |p_i|² − w_i = ℓ(p_i, w_i)`.

> **The project's "height" *is* the lift coordinate.** There is no conversion to do in the
> Rust port: `triangulate_with_heights(h)` means "take the lower hull of
> `{(p_i, h_i)}`", and `triangulate_with_weights(w)` means "take the lower hull of
> `{(p_i, |p_i|² − w_i)}`". Delaunay is `h_i = |p_i|²`.

Design consequence: **the Rust core should take lift values `ℓ_i` as its only parameter.**
Weights are a thin wrapper (`ℓ_i = |p_i|² − w_i`); heights are the identity.

This also matches TOPCOM's language: `h` is a height function on the point
configuration, and the induced subdivision is the regular subdivision `S(h)`. The
regular-triangulation code and the TOPCOM-side regularity checks (`checkreg`, linked at
`CMakeLists.txt:47`) are talking about the same object.

### 2.4 Bug in the current height→weight conversion

`CGAL.cpp:45-46`:

```cpp
int sqrt_h0 = points_buf[i * dim + j];   // int64_t → int, truncating
weight += sqrt_h0 * sqrt_h0;             // int * int → int, overflows
```

Two defects: (a) the `int64_t` coordinate is truncated to 32 bits; (b) the square is
computed in `int` arithmetic, so it is UB/wrong for `|coord| ≥ 46341`. The Rust port must
compute `|p|²` in at least `i128` (or exactly, see §5).

---

## 3. CGAL's implementation

### 3.1 Class stack

```
Regular_triangulation<Traits, TDS>                       Triangulation/include/CGAL/Regular_triangulation.h:27
  └─ Triangulation<Regular_triangulation_traits_adapter<Traits>, TDS>   Triangulation/include/CGAL/Triangulation.h
       └─ Triangulation_data_structure<Dim, Vertex, Full_cell>          Triangulation/include/CGAL/Triangulation_data_structure.h:43
```

`Regular_triangulation_traits_adapter`
(`Triangulation/include/CGAL/Regular_triangulation_traits_adapter.h:32-278`) is pure glue:
it makes `Point_d ≡ Weighted_point_d` for the base class and forwards
`Orientation_d`, `Compare_lexicographically_d`, `Less_coordinate_d`,
`Contained_in_affine_hull_d` etc. through `Construct_point_d` (drop the weight).

### 3.2 The triangulation data structure

`Triangulation_data_structure` stores a triangulation of a **`d`-sphere**, not of a ball:
a single **infinite vertex** `infinity_` (`Triangulation.h:201`) compactifies the space.

- A **full cell** (`Triangulation_ds_full_cell.h:29-310`) stores
  `cur_dim+1` vertex handles and `cur_dim+1` neighbour handles, indexed 0..`cur_dim`.
  Invariant: `neighbor(i)` is the cell sharing the facet **opposite** vertex `i`.
  Storage is `S_or_D_array` (fixed array when `Dimension_tag<D>`, heap vector when
  `Dynamic_dimension_tag`), `TDS_full_cell_default_storage_policy.h:30-91`.
- **Mirror index.** `mirror_index(i)` is the index of `this` inside `neighbor(i)`. The
  default storage policy does *not* store it — it recomputes by linear scan
  (`TDS_full_cell_default_storage_policy.h:64-72`) and recovers the opposite vertex by an
  **XOR trick** over vertex pointers (`TDS_full_cell_default_storage_policy.h:73-83`).
  `TDS_full_cell_mirror_storage_policy.h` stores mirror indices explicitly instead.
  **For the Rust port: just store the mirror index (`u32`).** It is 4 bytes per facet and
  removes the entire XOR/pointer-reinterpretation hack.
- **Finite vs. infinite cells.** Cells containing the infinite vertex are the "outer"
  cells; `finite_full_cells_begin()` is a filter iterator skipping them
  (`Triangulation.h:465-474`). The convex hull of the point set is exactly the set of
  facets shared between a finite and an infinite cell.
- **`current_dimension()`** starts at `-2` (empty) and grows as affinely independent
  points are inserted, up to `maximal_dimension()`. Cells always have
  `current_dimension()+1` meaningful vertices; slots above that are null.
- Orientation invariant: **every finite full cell is positively oriented**
  (`Triangulation.h:1323-1360` is the validity check; `reorient_full_cells()` at
  `Triangulation.h:855` restores it). For an infinite cell, replacing the infinite vertex
  by the opposite vertex of the adjacent finite cell must give a **negatively** oriented
  simplex (`Triangulation.h:1334-1342`).

Rust sketch:

```rust
type CellId = u32;  type VertId = u32;
struct Cell { verts: [VertId; D1], neigh: [CellId; D1], mirror: [u8; D1] }  // D1 = dmax+1
struct Vertex { point: PointId, cell: CellId }       // one incident cell
const INFINITE_VERT: VertId = 0;
```

Use arena `Vec`s with a free list rather than CGAL's `Compact_container`. There is no
`Vertex_handle` identity requirement beyond stability of indices.

### 3.3 Insertion order: spatial sort

`Regular_triangulation::insert(first, last)`
(`Regular_triangulation.h:219-238`) copies the points into a vector, calls
`CGAL::spatial_sort(points.begin(), points.end(), geom_traits())`
(`Regular_triangulation.h:226`), then inserts one by one, passing the previous vertex's
cell as the `hint` for `locate`.

`spatial_sort` for `Point_d` (`Spatial_sorting/include/CGAL/spatial_sort.h:81-101`):

1. `random_shuffle` the whole range,
2. then `Multiscale_sort<Hilbert_sort_d>` with
   `threshold_hilbert = 10`, `threshold_multiscale = 500`, `ratio = 0.05`.

`Multiscale_sort` (`Spatial_sorting/include/CGAL/Multiscale_sort.h:37-45`) recursively
leaves the first `5%` of the range unsorted-at-this-level and Hilbert-sorts the rest,
which gives the "BRIO" (biased randomized insertion order) behaviour: good locality *and*
randomised `O(n log n)` expected behaviour. Default policy is
`Hilbert_sort_median_policy` (`spatial_sort.h:160-172`), i.e.
`Hilbert_sort_median_d` (`Spatial_sorting/include/CGAL/Hilbert_sort_median_d.h:47-160`),
which recursively splits by `nth_element` on cycling coordinates with alternating
orientation to follow a Hilbert curve.

Two consequences:

- **Spatial sort is a pure optimisation.** Any insertion order gives a valid
  triangulation. Skipping it costs an asymptotic factor (point location walks get long);
  a plain `random_shuffle` already gives expected `O(n^{⌈d/2⌉+1})`-ish worst case with
  good constants. Implement Hilbert sort only if profiling demands it.
- **Spatial sort destroys the input order**, which is *why* `CGAL.cpp:59-84` has to map
  vertices back to indices by coordinate comparison. In the Rust port, carry the index
  alongside the point and this problem vanishes. (CGAL even has
  `Triangulation::insert_and_index` at `Triangulation.h:727-752` that stores the index in
  `vertex->data()`; the wrapper does not use it.)

### 3.4 Point location: the remembering stochastic walk

`Triangulation::do_locate` (`Triangulation.h:1086-1255`), dispatched by
`Triangulation::locate` (`Triangulation.h:1262-1278`).

```
if current_dimension() == −1:               → OUTSIDE_AFFINE_HULL
if current_dimension() ==  0:               compare lexicographically with the single vertex
if current_dimension() < maximal_dimension():
    if !contained_in_affine_hull(cell points, p): → OUTSIDE_AFFINE_HULL
s := hint (or any finite cell); previous := none
loop:
    i := random in [0, cur_dim]                       # randomised start index
    for j in 0..=cur_dim:                             # cyclically from i
        n := s.neighbor(i)
        if n == previous: orientations[i] := POSITIVE; continue     # remembered
        orientations[i] := orientation( s.points with vertex i replaced by p )
        if orientations[i] != NEGATIVE: continue
        previous := s; s := n
        if is_infinite(n): → OUTSIDE_CONVEX_HULL, stop
        break
    if all d+1 facets passed: → p is in s, stop
classify by counting COPLANAR entries:
    0 coplanar               → IN_FULL_CELL
    1 coplanar               → IN_FACET
    cur_dim coplanar         → ON_VERTEX
    otherwise                → IN_FACE
```

Key points for the port:

- The predicate used is **`orientation` only** — `d+1` points, one of which is the query.
  The cost is `O(d+1)` orientation evaluations per step.
- "Remembering" (skip the facet you just came through) saves one predicate per step;
  "stochastic" (random starting index) prevents cycling in degenerate configurations.
- It never loops forever *provided the predicates are consistent*. With inconsistent
  (floating-point) predicates it can and does loop forever. See §5.6.
- `Regular_triangulation::insert` at `Regular_triangulation.h:797-834` only distinguishes
  `OUTSIDE_AFFINE_HULL`, `ON_VERTEX`, and *everything else*. So the full `Locate_type`
  classification (`IN_FACET` / `IN_FACE` / `IN_FULL_CELL` / `OUTSIDE_CONVEX_HULL`) is
  **not needed** for our use case. The port only needs:
  - `OUTSIDE_AFFINE_HULL` (triggers dimension increase),
  - `ON_VERTEX` (duplicate point handling),
  - otherwise "here is a cell to start the conflict search from".

  That is a substantial simplification over `do_locate`.

### 3.5 The conflict region

`Regular_triangulation::Conflict_predicate` (`Regular_triangulation.h:307-350`) decides
whether the new weighted point `p` is *in conflict* with a cell `s`:

```cpp
if (!is_infinite(s)) {
    o = power_side_of_power_sphere(s.points[0..cur_dim], p);
    if (o == ON_POSITIVE_SIDE) conflict = true;
    else if (o == ON_NEGATIVE_SIDE) conflict = false;
    else conflict = (perturbed_power_side_of_power_sphere(p, s, ori) == ON_POSITIVE_SIDE);
} else {
    // infinite cell: substitute p for the infinite vertex and take the orientation
    o = orientation(s.vertices with infinite vertex replaced by p);
    if (o == POSITIVE) conflict = true;
    else if (o == NEGATIVE) conflict = false;
    else conflict = recurse on s.neighbor(index_of_infinite_vertex);   // p is on the hull facet
}
```

That is the **star-of-the-boundary trick**: an infinite cell "contains the point at
infinity", so conflict with an infinite cell means "`p` sees the corresponding hull facet
from outside", which is an orientation test. In the lifted picture (§2.2), the infinite
vertex plays the role of the vertical direction, and the infinite cells are the *vertical*
facets of the lifted hull.

`compute_conflict_zone` (`Regular_triangulation.h:1094-1119`) is a BFS from the located
cell over `gather_full_cells` (`Triangulation_data_structure.h:665-704`):

```
queue := {start}; visited := {start}
while queue nonempty:
    s := pop
    emit s
    for i in 0..=cur_dim:
        n := s.neighbor(i)
        if not visited(n):
            mark visited(n)
            if traversal_predicate(Facet(s,i)):   # n is in conflict
                push n
            else:
                boundary_facet := Facet(s,i)      # remember one boundary facet
clear visited flags
return boundary_facet
```

The conflict zone is **star-shaped** with respect to `p` (that is the theorem that makes
this correct), so BFS from any one conflicting cell finds all of them, and the boundary of
the zone is a topological sphere.

> In lifted terms: the conflict zone is exactly the set of lower-hull facets **visible**
> from the lifted new point `p̂`, and `compute_conflict_zone` is the visibility-region
> search of the beneath-beyond convex hull algorithm.

`is_in_conflict` (`Regular_triangulation.h:1071-1092`) evaluates the predicate on a single
cell; it is the "is `p` hidden?" test.

### 3.6 Insertion proper

`Regular_triangulation::insert_in_conflicting_cell`
(`Regular_triangulation.h:962-1008`):

```
if !is_in_conflict(p, s):
    m_hidden_points.push_back(p);     # p is HIDDEN, no vertex is created
    return null
cs  := compute_conflict_zone(p, s, ...)        # list of cells, plus one boundary facet ft
cz_vertices := process_conflict_zone(cs)       # collect all vertices of the zone,
                                               # and set vertex.full_cell := null for each
v   := insert_in_hole(p, cs, ft)               # retriangulate the hole by coning to p
process_cz_vertices_after_insertion(cz_vertices)
     # any vertex whose full_cell is STILL null was not re-attached
     # => it disappeared from the triangulation => it becomes a hidden point
return v
```

`process_conflict_zone` / `process_cz_vertices_after_insertion` are at
`Regular_triangulation.h:406-449`. This is the **entire** hidden-point mechanism: it is a
side effect of retriangulating the hole. A vertex that was only incident to conflict-zone
cells and that does not end up on the boundary of the hole gets deleted
(`tds().delete_vertex(vh)`, `Regular_triangulation.h:441`) and its weighted point is
pushed into `m_hidden_points` (`Regular_triangulation.h:461`).

Note that the base `Triangulation::insert` (`Triangulation.h:889-918`) does dispatch to
`insert_in_full_cell` / `insert_in_facet` / `insert_in_face` / `insert_outside_convex_hull`.
**`Regular_triangulation` does not**: it funnels everything through
`insert_in_conflicting_cell`, which handles the outside-convex-hull case uniformly because
the conflict predicate handles infinite cells. So `Triangulation::insert_outside_convex_hull`
(`Triangulation.h:993-1023`) is dead code for our path and need not be ported.

#### `insert_in_hole` — coning the hole to the new vertex

`Triangulation_data_structure::insert_in_hole` (`Triangulation_data_structure.h:1035-1052`)
marks the hole cells "visited", allocates the new vertex, calls `insert_in_tagged_hole`,
and deletes the old cells.

`insert_in_tagged_hole` (`Triangulation_data_structure.h:933-1032`) is the fiddliest
routine in the package. For each **boundary facet** of the hole it creates one new cell =
(the boundary facet) ∪ {v}, sets the outward neighbour link, and then walks **rotors** to
find the adjacent boundary facets and link the new cells to each other.

A **Rotor** (`Triangulation.h:249-272`, `Triangulation_data_structure.h:128`) is a triple
`(cell, i, j)` denoting a codimension-2 face (the face opposite to both `i` and `j`).
`rotate_rotor` (`Triangulation.h:266-272`) turns the rotor around that codim-2 face:

```cpp
Rotor rotate_rotor(Rotor & r) {
  int opposite  = get<0>(r)->mirror_index(get<1>(r));
  Full_cell_handle s = get<0>(r)->neighbor(get<1>(r));
  int new_second = s->index(get<0>(r)->vertex(get<2>(r)));
  return Rotor(s, new_second, opposite);
}
```

The algorithm rotates until it hits a boundary facet (`while (!is_boundary_facet(rot)) rot = rotate_rotor(rot);`,
`Triangulation_data_structure.h:996-997`), which is the neighbouring boundary facet across
the codim-2 ridge. That gives the adjacency between the new cells. The whole thing is
driven by an explicit `std::stack<IITH_task>` (`Triangulation_data_structure.h:392`,
`:942`) rather than recursion.

**This is the routine to be most careful about in the port.** An equivalent, and in my
view easier, formulation for a fresh implementation:

1. Collect the conflict cells `C`. Mark them.
2. For every `(c, i)` with `c ∈ C` and `c.neighbor(i) ∉ C`, that facet is a boundary
   facet; create a new cell `new(c,i)` with vertices `c.verts` but slot `i` replaced by
   `v`, and link `new.neigh[i] = c.neighbor(i)` (and the mirror back-link).
3. To link new cells to each other: for each new cell and each slot `k ≠ i`, the facet
   opposite `k` contains `v`; it must be matched with exactly one other new cell. Build a
   hash map from the sorted `(d−1)`-tuple of the *other* vertices (the ridge, i.e. the
   boundary facet's vertices minus one) to `(new_cell, k)` and match pairs.

   Cost: `O(|boundary facets| · d · d log d)` hashing versus CGAL's rotor walk. For
   `d ≤ 30` and typical conflict zones this is negligible and it is far easier to get
   right. Use CGAL's rotor walk only if this shows up in a profile.

#### Dimension growth: `insert_outside_affine_hull`

`Regular_triangulation::insert_outside_affine_hull`
(`Regular_triangulation.h:845-890`) →
`Triangulation_data_structure::insert_increase_dimension`
(`Triangulation_data_structure.h:1164-1203`) →
`do_insert_increase_dimension` (`Triangulation_data_structure.h:1090-1160`).

This is how CGAL bootstraps from an empty triangulation up to dimension `d`:

- `cur_dim = −2` → create the infinite vertex and one 0-cell.
- `cur_dim = −1` → the first finite point; create the 0-sphere `{v, ∞}`.
- `cur_dim ≥ 0` → `do_insert_increase_dimension`: extend every existing cell by the new
  vertex `x` in slot `cur_dim`, and for every *bounded* cell create its unbounded twin
  containing `∞`; then relink all neighbours by BFS; then fix parity by swapping the last
  two vertices in half the cells when `cur_dim` is even and `> 1`
  (`Triangulation_data_structure.h:1151-1159`).
- Afterwards, `Regular_triangulation.h:857-861` checks the orientation of one finite cell
  and calls `reorient_full_cells()` (`Triangulation.h:855-870`) if it is negative, to
  restore the global "all finite cells positive" invariant; plus a special fix-up for
  `cur_dim == 1` (`Regular_triangulation.h:866-888`).

This code is correct but hairy, and it exists to support the case where the input is
**not full-dimensional** and CGAL still wants to produce a triangulation of the affine
hull. **Recommendation for the Rust port: don't do this.** See §6.3 — compute the affine
hull up front with exact integer linear algebra, change basis, and run a purely
full-dimensional algorithm. That deletes `do_insert_increase_dimension`,
`Flat_orientation_d`, `Construct_flat_orientation_d`, `In_flat_orientation_d`,
`In_flat_power_side_of_power_sphere_d` and `Contained_in_affine_hull_d` from the port —
roughly a third of the geometric traits and all of `NewKernel_d/Coaffine.h`.

#### The "coaffine" predicates (what we are skipping)

While `current_dimension() < maximal_dimension()`, CGAL cannot use the `d`-dimensional
determinant (the points are degenerate in `R^d`). It instead picks, once, a set of
`cur_dim` coordinate axes onto which the current affine hull projects bijectively, caches
that choice as a `Flat_orientation` (`NewKernel_d/Coaffine.h:22-26`: a list `proj` of kept
coordinates, a list `rest`, and a `reverse` flag), and evaluates all orientations in that
projection (`Construct_flat_orientation` at `NewKernel_d/Coaffine.h:53-108`, found by
greedy rank-revealing determinants). It is reset (`reset_flat_orientation`) on every
dimension increase. The same trick gives
`In_flat_power_side_of_power_sphere_d`, wrapped by
`Power_side_of_power_sphere_for_non_maximal_dim_d` (`Regular_triangulation.h:130-146`).

### 3.7 Validity checks

- `Triangulation_data_structure::is_valid` — combinatorial: neighbour symmetry, vertex
  incidence, cell degree (`Triangulation_data_structure.h:1209+`).
- `Triangulation::is_valid` (`Triangulation.h:1321-1360`) — every finite cell positively
  oriented and non-flat; every infinite cell negatively oriented after substituting the
  mirror vertex for `∞`.
- `Regular_triangulation::is_valid` (`Regular_triangulation.h:1122-1163`) — the **empty
  power sphere** property: for every finite cell and every finite neighbour, the
  neighbour's opposite vertex must not be `ON_POSITIVE_SIDE` of the cell's power sphere.

These are `O(n · d)` predicate evaluations and are exactly the right acceptance tests for
the Rust port. Note `CGAL.cpp:57` wraps `is_valid()` in `CGAL_assertion`, so in a release
build (`NDEBUG`) **it never runs**.

---

## 4. Required predicates

Only two, and really only one.

### 4.1 `orientation` — sign of a `d × d` determinant

`CartesianDKernelFunctors::Orientation_of_points`
(`NewKernel_d/include/CGAL/NewKernel_d/function_objects_cartesian.h:42-77` generic;
`:80-111` unrolled for `d ≤ 6`; `:113-135` for `d = 1`).

Given `p_0, …, p_d ∈ R^d`:

```
orient(p_0,…,p_d) = sign det M,     M[i][j] = p_{i+1}[j] − p_0[j],   i,j = 0..d−1
```

(`function_objects_cartesian.h:56-64`; note the FIXME comment there — the differences are
written into **rows**.) Equivalently the `(d+1)×(d+1)` determinant

```
          | 1  p_0[0] … p_0[d−1] |
orient =  | 1  p_1[0] … p_1[d−1] |
          |     ⋮                |
          | 1  p_d[0] … p_d[d−1] |
```

with the same sign.

For `i64` input coordinates this is a **sign of an integer determinant**. Entries are
differences of `i64`, so they fit in `i65`; use `i128` inputs.

### 4.2 `power_side_of_power_sphere` — sign of a `(d+1) × (d+1)` determinant

`CartesianDKernelFunctors::Power_side_of_power_sphere`
(`NewKernel_d/include/CGAL/NewKernel_d/Types/Weighted_point.h:105-123`) drops weights and
calls `Power_side_of_power_sphere_raw`
(`function_objects_cartesian.h:482-530`). The unweighted version is
`Side_of_oriented_sphere` (`function_objects_cartesian.h:536-600`) — identical with all
weights zero.

Call convention (see `Regular_triangulation.h:328`): the range `[f, e)` is the **cell's
`d+1` weighted vertices** `(v_0,w_0)…(v_d,w_d)`, and the extra argument is the **query**
`(q, w_q)`. CGAL builds a `(d+1)×(d+1)` matrix `M` with one row per cell vertex
(`function_objects_cartesian.h:505-522`, the `else` branch, which is the one taken since
`Squared_distance_to_origin` is not cached in `Epick_d`):

```
M[i][j] = v_i[j] − q[j]                                    for j = 0 .. d−1
M[i][d] = (w_q − w_i) + Σ_j (v_i[j] − q[j])²
```

and returns

```
result = (−1)^d · sign(det M)         (function_objects_cartesian.h:524-527)
```

with `ON_POSITIVE_SIDE` (= `+1`) meaning **`q` is inside the power sphere of the cell**,
i.e. the cell is in conflict, *provided the cell is positively oriented*.

CGAL's other branch (`function_objects_cartesian.h:494-504`) uses
`M[i][d] = (|v_i|² − w_i) − (|q|² − w_q) = ℓ_i − ℓ_q`. The two differ by
`−2 q · (v_i − q)`, a linear combination of the first `d` columns, so the determinant is
identical. **Use the `ℓ_i − ℓ_q` form** — it is the lifted form and needs no `Σ` per row.

### 4.3 The one predicate you actually need

Write `p̂ = (p, ℓ(p,w))` as in §2.2. Then, with `M` as above and the `ℓ` form of the last
column, `M` **is** the orientation matrix of the `d+2` lifted points with `q̂` as the
reference point:

```
sign(det M) = orient_{d+1}( q̂, v̂_0, v̂_1, …, v̂_d )
```

and therefore

```
power_side_of_power_sphere( (v_i,w_i)_{i=0..d}, (q,w_q) )
      = (−1)^d · orient_{d+1}( q̂, v̂_0, …, v̂_d )
      = − sign D,
```

where `D` is the "textbook" `(d+2)×(d+2)` in-power-sphere determinant

```
      | 1  v_0[0] … v_0[d−1]  ℓ_0 |
      | 1  v_1[0] … v_1[d−1]  ℓ_1 |
D  =  |     ⋮                     |          ℓ_i = |v_i|² − w_i,  ℓ_q = |q|² − w_q
      | 1  v_d[0] … v_d[d−1]  ℓ_d |
      | 1  q[0]   … q[d−1]    ℓ_q |
```

(Proof: subtract the last row from all others and expand along the first column; the
cofactor sign is `(−1)^{d+1}`. Verified numerically for `d = 1..5` over 1500 random
instances, including the "inside the circumsphere ⇔ `ON_POSITIVE_SIDE`" direction.)

**Implication for the port:** implement *one* routine,

```rust
/// Sign of the determinant of an n×n exact-integer matrix.
fn sign_of_det(m: &mut [[Int; N]; N]) -> i8;
```

and derive both predicates from it. `orient_d` uses the `d×d` coordinate differences;
`power_side` uses the `(d+1)×(d+1)` lifted differences. There is no separate "in-sphere"
code path, no separate error analysis, no separate filter. That is the single biggest
simplification available.

### 4.4 Symbolic perturbation (`perturbed_power_side_of_power_sphere`)

`Regular_triangulation.h:1013-1069`. When the power test returns `0` (`d+2` points on a
common power sphere, i.e. `d+2` lifted points coplanar), CGAL resolves the tie with a
simulation-of-simplicity scheme:

1. Collect the `d+2` weighted points (cell vertices + `p`) and sort them by
   `Compare_points_for_perturbation`
   (`Triangulation/include/CGAL/Triangulation/internal/utilities.h:94-112`), which is
   lexicographic comparison of the **bare coordinates**.
2. Walk them from largest to smallest. If the current one is `p` itself, return
   `ON_NEGATIVE_SIDE` (`p` is outside).
3. Otherwise remove that point from the cell, put `p` in its place, and evaluate the
   `orientation` of the resulting `d+1` points. If non-zero, return
   `−adjust_sign · orientation`, where `adjust_sign` encodes the parity of the swap
   (`Regular_triangulation.h:1036-1048`).
4. If zero, continue to the next point.

**What this is, mathematically.** Perturb each lift value `ℓ_i → ℓ_i + ε^{π(i)}` with
distinct exponents `π(i)`. `det M` is multilinear in the last column, so

```
det M(ε) = det M⁰ + Σ_{i=0..d} ε^{π(i)} · A_i  −  ε^{π(q)} · Σ_i A_i
```

where `A_i` is the cofactor of entry `(i,d)` — which is, up to sign, exactly the
`orientation` of the `d+1` points obtained by deleting `v_i` and adding `q`, and
`Σ_i A_i` is (up to sign) the orientation of the cell itself. So: take the lowest `ε`
exponent whose coefficient is non-zero, and return its sign. That is precisely steps 2–4
above.

**Recommendation.** CGAL's comparator sorts by *coordinates*, which is not a strict weak
ordering when the configuration contains duplicate points, and makes the result depend on
`std::sort`'s behaviour on equal elements. Use **insertion index** instead:
`π(i) = index of the point in the input array`. It is guaranteed distinct, deterministic,
and free. The resulting triangulation is then a reproducible function of `(points, lifts)`
alone. (It will in general differ from CGAL's output on degenerate inputs — see §7.3.)

### 4.5 Predicates not needed

`Compare_lexicographically_d` (only for the `cur_dim == 0` case and the perturbation
comparator — replaceable), `Less_coordinate_d` (only for Hilbert sort),
`Contained_in_affine_hull_d`, `Construct_flat_orientation_d`,
`In_flat_orientation_d`, `In_flat_power_side_of_power_sphere_d` (all removable with the
HNF preprocessing of §6.3), and every *construction*.

---

## 5. Robustness & numerics — the crux

### 5.1 What CGAL actually does

`Epick_d` (`NewKernel_d/include/CGAL/Epick_d.h:28-80`) is a three-layer filtered kernel:

```
Epick_d
 └─ Kernel_d_interface< Cartesian_wrap< Epick_d_help3 > >
     └─ Epick_d_help3 = Cartesian_static_filters< Dim, Epick_d_help2, Epick_d_help3 >
         └─ Epick_d_help2 = Cartesian_filter_K< Epick_d_help1,
                                 AK = Cartesian_base_d<Interval_nt_advanced, Dim>,
                                 EK = Cartesian_base_d<Exact_ring_selector<double>::Type, Dim>,
                                 restricted to Functors_without_division<Dim> >
             └─ Epick_d_help1 = Cartesian_filter_K< Cartesian_base_d<double, Dim>,
                                     AK = Cartesian_base_d<Interval_nt_advanced, Dim>,
                                     EK = Cartesian_base_d<Exact_field_selector<double>::Type, Dim> >
```

Read from the inside out, a predicate call is:

1. **Static filter** (`NewKernel_d/Cartesian_static_filters.h`). Only wired up for
   `Orientation_2/3/4/5/6` and `Side_of_oriented_circle_2` / `Side_of_oriented_sphere_3`.
   **Never applies to `Power_side_of_power_sphere`.** So for the regular triangulation the
   static filter layer is inert.
2. **Interval filter** (`NewKernel_d/Filtered_predicate2.h:48-99`). Set the FPU rounding
   mode to `+∞` (`Protect_FPU_rounding`), evaluate the whole predicate in
   `Interval_nt_advanced` (a pair of doubles with directed rounding), and if the resulting
   sign interval does not straddle 0, return it. This catches the overwhelming majority of
   calls.
3. **Exact fallback** — the `EK` layer. Which exact type is used depends on whether the
   functor tag is in `Functors_without_division<Dim>`
   (`NewKernel_d/Cartesian_filter_K.h:23-41`):

   ```cpp
   template<class> struct Functors_without_division { typedef typeset<> type; };   // empty!
   template<> struct Functors_without_division<Dimension_tag<1..6>> {
     typedef typeset<Orientation_of_points_tag, Side_of_oriented_sphere_tag> type;
   };
   ```

   `Power_side_of_power_sphere_tag` is **never** in that list, and for
   `Dimension_tag<7..10>` and `Dynamic_dimension_tag` the list is **empty**. So the
   exact fallback for our predicates is always `Exact_field_selector<double>::Type`, the
   **field** (rational) type.

4. **Which exact type?** `Number_types/include/CGAL/Number_types/internal/Exact_type_selector.h:129-180`.
   `CMakeLists.txt:31` sets `CGAL_DISABLE_GMP ON`, which via
   `Installation/include/CGAL/Installation/internal/enable_third_party_libraries.h:19-26`
   undefines `CGAL_USE_GMP`/`CGAL_USE_MPFR`. `CGAL_USE_BOOST_MP` stays defined
   (`enable_third_party_libraries.h:51-57`). Hence
   `Default_exact_nt_backend = BOOST_BACKEND` (`Exact_type_selector.h:136-153`), i.e.

   | | with GMP | with `CGAL_DISABLE_GMP ON` (this project) |
   |---|---|---|
   | `Exact_ring_selector<double>` | `Mpzf` / `Gmpzf` | `CGAL::cpp_float` (boost `cpp_int` based) |
   | `Exact_field_selector<double>` | `Gmpq` | `boost::multiprecision::cpp_rational` |

   So **the exact fallback for the power test in this project is
   `boost::multiprecision::cpp_rational`** — header-only, pure C++, and considerably
   slower than GMP.

5. **How is the determinant computed exactly?** `CGAL_EIGEN3_ENABLED`
   (`CMakeLists.txt:52`) selects `LA_eigen` (`NewKernel_d/Cartesian_LA_base.h:28-32` —
   without it there is an `#error Eigen3 is required`). `LA_eigen::determinant`
   (`NewKernel_d/LA_eigen/LA.h:114-163`) hard-codes cofactor expansions for `n ≤ 7` and
   otherwise falls back to `Eigen::MatrixBase::determinant()`
   (`LA.h:107-109`, `LA.h:159-160`), which for `n > 4` is **`PartialPivLU`** — Gaussian
   elimination **with divisions**. Over `cpp_rational` that is exact but suffers the
   classic rational-coefficient blow-up. There is even a TODO in the source
   (`LA.h:111-113`): *"Implement a version of (sign_of_)determinant that works without
   (inexact) division in any dimension."*

   > **Practical consequence:** CGAL's exact fallback for `d ≥ 7` in this configuration is
   > slow. A Rust port using **Bareiss fraction-free elimination over big integers**
   > will be materially faster on the exact path than what the project has today.

6. Finally, note that `Epick_d` stores coordinates as **`double`**
   (`Epick_d.h:30`, `Cartesian_base_d<double, Dim>`). The `int64_t` inputs are converted
   at `CGAL.cpp:40`. For `|coord| > 2^53` the conversion is already lossy and CGAL's
   "exactness" is exactness *with respect to the rounded doubles*, not the integers. See
   §7.6.

### 5.2 Precision required: bit-growth estimates

Let `C = max |coordinate|` and `d` the dimension. Matrix entries in the first `d` columns
are differences, bounded by `2C`.

**Orientation** (`d × d`). Hadamard: `|det| ≤ ∏‖row‖₂ ≤ (2C√d)^d`, so

```
bits(orient) ≤ d · ( log₂(2C) + ½ log₂ d )
```

**Power test** (`(d+1) × (d+1)`). Expanding along the last column,
`|det| ≤ (d+1) · L · (2C√d)^d` where `L = max_i |ℓ_i − ℓ_q|`. With integer coordinates and
integer lifts scaled by `2^s` (see §5.3), `L ≤ 2^s · (2C² + Δℓ)`, giving

```
bits(power) ≤ log₂(d+1) + s + log₂(2C² + Δℓ) + d · ( log₂(2C) + ½ log₂ d )
```

Concrete numbers (integer lifts already, i.e. `s = 0`, `Δℓ ≤ 2C²`):

| `C` | `d` | orient bits | power bits |
|---|---|---|---|
| `2^20` (10^6) | 4 | 88 | 134 |
| `2^20` | 10 | 227 | 276 |
| `2^20` | 20 | 464 | 515 |
| `2^20` | 30 | 704 | 756 |
| `2^40` | 10 | 427 | 516 |
| `2^62` (near `i64` max) | 10 | 647 | 779 |
| `2^62` | 30 | 1964 | 2099 |

Add `s` bits for scaled `f64` lifts (§5.3): typically `s ≈ 52–60`, worst case `1074`.

Reading the table:

- `i64` accumulation is **never** enough beyond `d = 2` with small coordinates.
- `i128` covers `d ≤ 4` at `C = 2^20`, or `d ≤ 2` at `C = 2^40`. Useful as a *second*
  filter tier, not as the answer.
- `f64` can represent the exact determinant only when the result fits in 53 bits — never,
  in practice. Naive `f64` determinants are **wrong** whenever the sign is close to zero,
  which is exactly the case that matters.

### 5.3 Making the lift exact

Coordinates are `i64`, so `|p|²` is an exact integer (needs `i128`). Weights/heights are
`f64`, i.e. each is an **exact** dyadic rational `m · 2^e` with `|m| < 2^53`,
`e ∈ [−1074, 971]`.

The last column of the power-test matrix enters the determinant **linearly**, so
multiplying every `ℓ_i` by the same positive constant does not change the sign. Therefore:

```
s   := max over i of ( − exponent_of_trailing_bit(ℓ_i) ), clamped at 0
ℓ'_i := ℓ_i · 2^s          ∈ Z exactly
```

and run the predicate on the integers `ℓ'_i`. For "sane" weights (`|w| ≲ 2^30`, no
subnormals) `s` is about 52–60 and costs a single extra 64-bit limb. For adversarial input
(a subnormal weight) `s` can reach 1074; validate and either reject or renormalise.

An alternative that avoids the shared scale: represent each `ℓ_i` as an exact rational
`(num: BigInt, den: 2^k)` and clear denominators per-row inside the determinant. Since row
scaling by a positive constant does not change the sign, this is legitimate but it is more
code for no benefit. **Use the common power-of-two scale.**

A third option worth offering at the API level: **let the caller pass integer or rational
heights**. For the project's actual use case (heights coming from a lattice-polytope /
Kähler-cone computation) integer or rational heights are natural, and they make the whole
pipeline exact with no scaling at all. This is a small API addition with a large
robustness payoff.

### 5.4 Degeneracy: why exactness alone is not enough

Even with perfectly exact predicates, a `0` from the power test is a real possibility
(cospherical points; and in the *Delaunay* case, `d+2` points on a common sphere, which
happens constantly for lattice points — e.g. the four corners of a unit square). Exactness
turns "wrong answer" into "no answer"; §4.4's symbolic perturbation turns "no answer" into
a consistent answer. **Both are mandatory.** An implementation with exact predicates but no
tie-breaking will produce invalid combinatorics on the very first square.

### 5.5 Recommended Rust numeric stack

A three-tier filter, in order of increasing cost:

1. **`f64` with a forward error bound (static/semi-static filter).**
   Compute the determinant in `f64` by LU (or by the same Bareiss recursion in `f64`) and
   simultaneously an error bound. For a `n × n` matrix with entries bounded by `E`, a
   sound, cheap bound is the Hadamard-style
   `err ≤ n · ulp_factor(n) · (E√n)^n`, but that is far too pessimistic; prefer a
   **running error bound** accumulated alongside the elimination (the "error-free
   transformation" style used by Ozaki/Rump, or Shewchuk's `errbound` idiom). If
   `|det| > err`, return `sign(det)`. This catches ~99%+ of calls in practice.
   *Note:* Rust has no stable API to change the FPU rounding mode, so CGAL's
   `Interval_nt_advanced` is not directly portable. Round-to-nearest with an explicit error
   term is the idiomatic Rust substitute, and `f64::mul_add` (FMA) is available and stable,
   which makes exact two-products (and hence Shewchuk expansions) implementable.

2. **`i128` exact.** If `bits(power) ≤ 126` (a compile-/run-time check from `C` and `d`
   via the table in §5.2), run Bareiss in `i128` — no allocation, no branches, very fast.
   Worth having because a large fraction of realistic inputs (`d ≤ 5`, coordinates in the
   hundreds) land here.

3. **BigInt exact, Bareiss fraction-free elimination.** The Bareiss algorithm:

   ```
   M⁽⁰⁾ = M, prev = 1
   for k in 0..n−1:
       pivot = M[k][k]  (with row swaps; track sign)
       for i,j > k:  M[i][j] := (M[i][j]*pivot − M[i][k]*M[k][j]) / prev   # exact division
       prev = pivot
   det = ± M[n−1][n−1]
   ```

   Every intermediate entry is a **minor** of the original matrix, hence bounded by the
   Hadamard bound of the *whole* matrix — no coefficient explosion, unlike rational
   Gaussian elimination. `O(n³)` bigint multiplications on operands of at most
   `bits(det)` bits. This is strictly better than what CGAL does today for `d ≥ 7`
   (§5.1 item 5).

   Handle a zero pivot column by row search; if an entire column below `k` is zero, the
   determinant is `0` — return early.

Do **not** attempt to generalise Shewchuk's adaptive expansions to arbitrary `d`. His
`orient3d`/`insphere` derivations are hand-tuned per predicate; a general-`d` version is a
research project, and the `i128`+Bareiss tier already captures most of the win. (This is
also why every robust-predicate crate in Rust stops at 3D — see §10.3.)

For `d = 2` and `d = 3` specifically, short-circuit to the `robust` crate's
`orient2d`/`orient3d`/`incircle`/`insphere`: they are hand-tuned, exact, dependency-free,
and will beat a generic implementation. See §10.6.

### 5.6 Failure modes of naive `f64` — the main risk of the port

This is worth spelling out because the symptoms are *not* "slightly wrong output":

- **Inconsistent orientations.** The walk in `do_locate` (§3.4) moves to `neighbor(i)`
  when `orientation(...) == NEGATIVE`. If the predicate reports `NEGATIVE` for the facet
  from cell `A` into `B` *and* `NEGATIVE` for the same facet from `B` into `A` (evaluated
  with a different vertex ordering, hence a different rounding), the walk **ping-pongs
  forever**. The random starting index only masks this.
- **Non-star-shaped conflict zone.** `compute_conflict_zone` assumes the set of
  conflicting cells is connected and its boundary is a sphere. With an inconsistent power
  test, the "boundary" can be non-manifold, and `insert_in_tagged_hole` will build cells
  with wrong adjacencies — corrupting the mesh silently. Later insertions then walk into
  garbage.
- **Negatively oriented cells.** Breaks the sign convention that
  `Conflict_predicate` relies on (`ON_POSITIVE_SIDE ⇒ conflict` is only true for
  positively oriented cells). One flipped cell makes every subsequent conflict decision in
  its neighbourhood meaningless.
- **Hidden-point cascades.** A spurious "not in conflict" makes a point hidden that should
  be a vertex, and vice versa; the output triangulation then does not even cover the
  convex hull.
- **Non-termination / panic in `insert_in_tagged_hole`.** The rotor walk
  (`Triangulation_data_structure.h:996-997`) is `while (!is_boundary_facet(rot)) rot = rotate_rotor(rot);`
  — an unbounded loop that relies on the hole being a topological ball. In Rust this
  becomes an infinite loop or an index-out-of-bounds panic.

Mitigation, in order:
1. Exact predicates (not "more precise" — *exact*).
2. Symbolic perturbation for the exact zeros.
3. Debug-mode invariant checks after every insertion (§8).
4. Defensive loop bounds in the hole-retriangulation with a hard error rather than a hang.

---

## 6. Alternative approaches

### 6.1 (A) Direct port of CGAL's incremental regular triangulation

Port §3 more or less literally: infinite vertex, `locate` walk, conflict BFS,
`insert_in_hole`, hidden points, dimension growth.

- **Effort:** highest of the pure-Rust options (§9).
- **Robustness:** good, *if* the predicates are exact.
- **Speed:** best. Output-sensitive: work per insertion is proportional to the conflict
  zone plus the walk length; expected `O(n log n + n^{⌈d/2⌉})` with BRIO ordering.
- **Fidelity:** closest to current behaviour.

### 6.2 (B) Lower convex hull of the lifted points in `R^{d+1}`

Lift `p_i ↦ p̂_i = (p_i, ℓ_i)`, compute the lower hull of `{p̂_i} ⊂ R^{d+1}`, project.

The crucial observation from §3.5/§4.3 is that **(A) and (B) are the same algorithm**.
Under the lifting map:

| incremental regular triangulation | incremental (beneath-beyond) convex hull |
|---|---|
| full cell | lower facet |
| infinite vertex / infinite cell | vertical direction / upper or vertical facet |
| conflict zone of `p` | facets visible from `p̂` |
| power test | orientation in `R^{d+1}` |
| `insert_in_hole` | cone the visible region's boundary ridges to `p̂` |
| hidden point | point above the lower hull |

So (B) is not a *different* algorithm to implement — it is a *reframing* that eliminates
the second predicate (§4.3) and the weight bookkeeping. The infinite-vertex trick of (A)
is a genuinely good way to encode "the boundary of the hull" (it turns the special case
into the general one), so the recommendation is to keep (A)'s data structure and adopt
(B)'s predicate.

If instead you implement (B) as a *general* convex hull in `R^{d+1}` (all facets, both
lower and upper, e.g. Qhull-style or the Clarkson–Shor randomised incremental hull):

- **Pros:** the machinery is reusable for other things the project plausibly wants
  (secondary polytope / secondary fan, regularity testing, the TOPCOM side's chirotope).
  One primitive: `orient_{d+1}` on integer matrices.
- **Cons:** you compute the upper hull too (wasted work, potentially a lot when the points
  are "one-sided"); non-simplicial facets must be handled (merge or perturb); facet/ridge
  adjacency bookkeeping is no easier than (A)'s.

### 6.3 (C) Preprocess the affine hull with exact integer linear algebra

Independent of A/B, and strongly recommended: instead of CGAL's incremental
dimension growth and `Flat_orientation` machinery (§3.6), handle non-full-rank input up
front.

```
1. Translate by p_0 so the configuration spans a linear subspace.
2. Compute the Hermite Normal Form (or just a row-echelon form over Q, then clear
   denominators) of the (n−1) × d matrix of differences. Rank r ≤ d.
3. If r < d: express every point in the resulting lattice basis of the affine hull,
   giving exact integer coordinates in Z^r.
4. Triangulate in dimension r with the full-dimensional algorithm.
5. Report simplices as index tuples — the indices are unchanged.
```

Benefits: deletes `Coaffine.h` (~300 lines of traits), `insert_increase_dimension`
(~130 lines of very tricky combinatorics), `Contained_in_affine_hull_d`, and the entire
"`current_dimension() < maximal_dimension()`" branch of `is_in_conflict`,
`compute_conflict_zone` and `locate`. It is also *more correct* than the current wrapper,
which silently emits garbage for rank-deficient input (§7.4).

Cost: an exact integer HNF / row reduction, ~150–250 lines, well-understood, and useful to
the TOPCOM-replacement side too.

For the full-dimensional algorithm itself, seed the triangulation with an explicit initial
simplex: pick `r+1` affinely independent points by greedy rank extension (using the
`orient` predicate you already have), build the one cell plus its `r+1` infinite cells
directly, then insert the rest. That replaces dimension growth with ~60 lines.

### 6.4 (D) Bind to Qhull

The `qhull` C library (via a Rust binding, or via `cc`-built vendored sources) does
arbitrary-dimensional convex hulls and Delaunay, and the standard way to get a *regular*
triangulation from it is exactly the lifting of §2.2 (this is how `scipy.spatial.Delaunay`
works internally).

- **Pros:** mature, fast, arbitrary `d`, handles the "all points on a sphere" pathologies
  with its merged-facet machinery.
- **Cons, and they are decisive here:**
  - It reintroduces a **C dependency**, which is the thing the port is trying to remove.
    (Qhull is C, not C++, so wheels are *easier* than with CGAL — but it is still a
    non-Rust build dependency.)
  - Qhull is **floating point** with a joggle/merge heuristic. It is not exact. For lattice
    point configurations with many cospherical points — precisely this project's input —
    Qhull's `Qt`/`QJ` options either produce a triangulation that is not the regular one,
    or fail. Joggling (`QJ`) perturbs the *input*, which changes the answer.
  - Qhull's license is its own non-OSI-standard permissive license ("may be used ... free
    of charge" with attribution), which is fine for most purposes but is neither MIT nor
    BSD and is worth a legal glance.
  - No native weighted/regular support — you must lift by hand anyway, so you gain only
    the hull code.

### 6.5 (E) Flip-based: start from a placing triangulation, flip down to regular

Start from any triangulation of the configuration (a **placing/pushing** triangulation is
easy to build: it is the lower hull for a lexicographic/generic height, i.e. literally a
special case of the same lifting), then perform bistellar flips that decrease the lifted
"volume functional" `Σ_σ ∫ ĥ` until no locally-non-regular facet remains. Termination is
guaranteed because the functional strictly decreases (Lawson-style; see De Loera–Rambau–
Santos, *Triangulations*, Ch. 5).

- **Pros:** the project already needs bistellar-flip machinery for the TOPCOM replacement
  (`src/triangulumancer/Flip.cpp`, `include/triangulumancer/Flip.hpp`). If the flip code
  exists and is exact, the regular triangulation is maybe 200 extra lines. Same single
  predicate.
- **Cons:** the number of flips is not well bounded in `d > 2` (it is `Θ(n²)` in 2D, and
  the higher-dimensional bound is weak); for large `n` this will be much slower than the
  incremental hull. Also needs a starting triangulation, which is itself a lower-hull
  computation.

Good as a **cross-check / oracle** in tests, not as the primary implementation.

### 6.6 Recommendation

> **Implement (A)'s data structure with (B)'s single predicate, preceded by (C)'s
> affine-hull reduction, on the numeric stack of §5.5.**

Concretely, in order:

1. `sign_of_det` over exact integers (f64 filter → `i128` Bareiss → BigInt Bareiss).
   One function, heavily tested, the foundation of everything.
2. `orient(d)` and `power_side(d)` as thin wrappers over it, with the lift-based last
   column (§4.3) and index-based SoS perturbation (§4.4).
3. Affine-hull reduction via exact HNF (§6.3); explicit initial simplex.
4. The TDS: arena of cells with `verts`/`neigh`/`mirror`, an infinite vertex.
5. Walk-based location (simplified per §3.4), conflict BFS, hash-map-based hole
   retriangulation (§3.6).
6. Hidden points as a side effect of the hole retriangulation; **report them** to Python
   rather than dropping them silently.
7. BRIO/Hilbert insertion order last, only if profiling says so.

Keep (E) as a test oracle: for small random inputs, check that the flip-based and the
incremental results agree, and check both against the `is_valid` invariants of §8.

Do not use (D).

**None of this can be outsourced to a crate.** §10 surveys the ecosystem: there is no
d-dimensional regular triangulation, no d-dimensional convex hull, no general-`d` weighted
predicate and no exact linear algebra library in Rust. The two crates that come closest —
`la-stack` (general-`D` exact determinant sign via Bareiss) and `delaunay` (general-`D`
orientation/in-sphere and Simulation of Simplicity, exact only through `D = 6`) — are both
BSD-3-Clause and are worth reading or vendoring for steps 1–2 above.

---

## 7. Edge cases

### 7.1 Duplicate points

CGAL's `locate` returns `ON_VERTEX`; `Regular_triangulation::insert`
(`Regular_triangulation.h:805-831`) then:
- equal weights → return the existing vertex, the new point is silently *ignored*
  (not even recorded as hidden);
- `current_dimension() == 0` and bigger weight → replace the point, old one hidden;
- otherwise fall through to `insert_in_conflicting_cell`, which resolves it via the
  conflict test (the lower-lift one ends up hidden).

The wrapper's index scan (`CGAL.cpp:70-83`) maps the surviving vertex to the **first**
input index with those coordinates. So with duplicates at different heights, the reported
index can be the wrong one. The Rust port should carry indices explicitly and define the
tie-break (recommend: smallest index wins among equal `(point, lift)`; among equal points
with different lifts the smaller lift wins, and the other is reported as hidden).

### 7.2 Hidden points

`Regular_triangulation` exposes `number_of_hidden_vertices()`
(`Regular_triangulation.h:380-383`) and stores them in `m_hidden_points`
(`Regular_triangulation.h:461`). **The wrapper uses neither.** The Python user currently
has no way to tell that a point they passed did not make it into the triangulation — the
simplices array simply never mentions that index.

This matters for the project's semantics: TOPCOM distinguishes *fine* (all points used)
from non-fine triangulations, and `triangulate_with_weights` producing a non-fine result
is normal and expected (the README's own example shows 8 simplices with heights vs.
5 with weights). **Recommendation: return the set of used / hidden point indices
alongside the simplices.**

### 7.3 Cospherical / non-general-position input

Extremely common for lattice points (a unit square is already cospherical). Handled by
§4.4. Consequences to document:

- The output is *a* triangulation refining the regular subdivision, not *the* canonical
  one. It is a deterministic function of the perturbation rule.
- **The Rust port will in general produce a different (but equally valid) triangulation
  than CGAL on degenerate input**, because CGAL perturbs by lexicographic coordinate order
  and the recommendation here is to perturb by input index. Tests must compare *invariants*
  (§8), not literal simplex lists, except on inputs verified to be in general position.
- If bit-for-bit agreement with the current CGAL output is a hard requirement, implement
  CGAL's comparator exactly: sort the `d+2` points by lexicographic comparison of bare
  coordinates and replicate the `adjust_sign` bookkeeping of
  `Regular_triangulation.h:1036-1048`. This is possible but brittle (it inherits the
  duplicate-point ordering ambiguity), and is not recommended.

### 7.4 Lower-dimensional (non-full-rank) configurations — a live bug

If the input points span an affine subspace of dimension `r < d`, CGAL leaves
`current_dimension() == r`. Full cells then have `r+1` meaningful vertices and the slots
`r+1 .. d` hold **null vertex handles**. But `CGAL.cpp:95-98` does

```cpp
for (int i = 0; i < dim + 1; i++)
  simplices_buf[...] = index_of_vertex[it->vertex(i)];
```

`index_of_vertex` is a `std::map`, so `operator[]` on a null handle **inserts and returns
0**. The output therefore silently contains index `0` padding, and the array shape
`(n_simplices, dim+1)` is wrong for the actual `r`-simplices. Confirmed by reading
`CGAL.cpp:61`, `CGAL.cpp:87-98` and `Triangulation_ds_full_cell.h:105-109` (vertices above
`current_dimension()` are default-constructed).

The Rust port must either (a) reduce to the affine hull up front (§6.3) and emit
`(n, r+1)` simplices, or (b) detect `r < d` and raise. Option (a) is strictly better and
also happens to be simpler.

### 7.5 Points not in general position generally

Beyond cosphericity: collinear/coplanar subsets are fine (orientation `0` only arises in
`locate` classification, where `COPLANAR` is a legitimate outcome, and in the perturbation
cascade, where the next term is tried). Nothing extra to do beyond exactness +
perturbation.

### 7.6 `std::round(CGAL::to_double(...))` — the index-recovery scan

`CGAL.cpp:68`:

```cpp
vert[i] = std::round(CGAL::to_double(ccd(it->point(), i)));
```

`Epick_d`'s `FT` is `double` and the stored coordinate is exactly the `double` that the
`i64` was converted to at `CGAL.cpp:40`. So:

- **Safe** for `|coord| ≤ 2^53 ≈ 9.0e15`. The `double` round-trips and `std::round` is a
  no-op on an exactly-representable integer.
- **Unsafe** for `|coord| > 2^53`: distinct `i64` coordinates can map to the same `double`,
  so (i) the geometry CGAL computes is not the geometry of the input, and (ii) the index
  recovery matches the wrong point. The wrapper does not check for this. Since the Python
  API accepts arbitrary `int64`, this is reachable.
- Also note `vert` is declared `std::vector<int>` (`CGAL.cpp:66`) — 32 bits — and compared
  against `int64_t` buffer entries, so it truncates for `|coord| > 2^31`.

Aside from correctness, the scan is `O(V · n · d)` — quadratic in the number of points. For
`n = 10^5`, `d = 10` that is `~10^{11}` integer comparisons. Even the existing C++ should
replace it with `insert_and_index` (`Triangulation.h:727-752`, which stores the index in
`vertex->data()`) or a hash map keyed on the coordinate tuple. In Rust, carry the index in
the vertex struct and the problem disappears entirely.

There is also dead code at `CGAL.cpp:70-71`: `unsigned int index = 0;` is immediately
shadowed by the loop's `int index`.

### 7.7 `d > 10`: `Dynamic_dimension_tag`

`CGAL.cpp:129-130`. This changes only compile-time specialisation, not semantics:

- Coordinates move from a fixed `std::array` to a heap `std::vector`
  (`Cartesian_LA_base.h:38-46`, `Mix_vector`).
- `Orientation_of_points` uses the generic matrix version rather than the unrolled
  `Dimension_tag<d>` template (`function_objects_cartesian.h:42` vs `:80`).
- The static-filter layer is inert (it only covers `d ≤ 6`).
- `Functors_without_division<Dynamic_dimension_tag>` is the **empty** typeset
  (`Cartesian_filter_K.h:23`), so the exact fallback uses the rational field type. (As
  noted in §5.1, this is true for `d ≥ 7` anyway.)
- Eigen matrices become fully dynamic, so `LA_eigen::determinant` takes the
  `Eigen::determinant()` (`PartialPivLU`) path for all `n`.

Net: `d > 10` is slower but not different. A Rust implementation with a runtime `d` and
`SmallVec`-backed cells covers the whole range uniformly with no `match` on dimension.

### 7.8 Small / trivial inputs

- `n = 0`: CGAL leaves `current_dimension() == −1`; `finite_full_cells` is empty; the
  wrapper produces a `(0, d+1)` array. Fine.
- `n = 1`: `current_dimension() == 0`, no finite full cells. The wrapper produces
  `(0, d+1)`. Arguably it should be one 0-simplex; match whatever behaviour the tests
  expect.
- `n < d+1`: always rank-deficient → §7.4.
- All points identical: one vertex, everything else silently dropped.

### 7.9 Mismatched parameter length

`CGAL.cpp:32-35` throws if `n_points != params.size()`. Keep this.

### 7.10 `is_valid` is not actually checked

`CGAL_assertion(t.is_valid())` (`CGAL.cpp:57`) compiles away under `NDEBUG`, which is how
release wheels are built. So the project currently has **no** runtime validity checking.
The Rust port should keep the checks behind a `debug_assertions` / feature flag but
actually enable them in CI.

---

## 8. Validation strategy

Invariants to assert (cheap, `O(n·d)` predicate calls each) after a full build, and after
every insertion in debug mode:

1. **Combinatorial.** `c.neigh[i].neigh[c.mirror[i]] == c` for all `c, i`. Every vertex's
   stored incident cell actually contains it. Cell vertex sets have no repeats.
2. **Orientation.** Every finite cell is positively oriented and non-degenerate
   (mirrors `Triangulation.h:1323-1360`). Every infinite cell is negatively oriented after
   substituting the mirror vertex for `∞` (`Triangulation.h:1334-1342`).
3. **Empty power sphere.** For every finite cell and every finite opposite vertex of a
   neighbour, `power_side ≠ ON_POSITIVE_SIDE`
   (mirrors `Regular_triangulation.h:1129-1157`).
4. **Volume.** `Σ_σ |det(σ)| == d! · vol(conv(points))`. Exact with integer arithmetic;
   compute the right-hand side independently (e.g. from the convex hull facets, or just
   compare against a second, independently-built triangulation). This single check catches
   almost every combinatorial corruption.
5. **Coverage.** Every input point is either a vertex of the triangulation or reported as
   hidden; the two sets partition `0..n`.
6. **Lower-hull certificate.** For every cell `σ` and every point `p_j` not in `σ`, the
   lifted `p̂_j` is on or above the affine hyperplane through `σ̂`. This is the definition
   of "regular"; it is `O(n · #cells)` so use it only on small test cases, but it is the
   ground-truth test.

Cross-checks:

- Against the existing CGAL build (keep it around during development) on **general-position**
  random inputs: the triangulations must be identical up to simplex/vertex ordering.
- Against a brute-force `O(n^{d+2})` reference (enumerate all `(d+1)`-subsets, keep those
  passing the empty-power-sphere test) for `n ≤ 12`, `d ≤ 4`. Slow but decisive, and it
  exercises the predicates independently of the incremental machinery.
- Against the flip-based construction of §6.5.
- Delaunay of points on a common sphere (every point cospherical) and on a lattice grid —
  the maximally degenerate cases.

---

## 9. Port complexity assessment

LOC are rough Rust line counts including doc comments but excluding tests.

| # | Component | LOC | Difficulty | Risk | Notes |
|---|---|---|---|---|---|
| 1 | `sign_of_det`: f64 filter + `i128` Bareiss + BigInt Bareiss | 400–700 | High | **High** | Foundation. Error bound derivation is the subtle part. |
| 2 | Lift construction, `f64`→exact scaling, `i128` `\|p\|²` | 120–200 | Low | Low | §5.3. Replaces the buggy `CGAL.cpp:41-48`. |
| 3 | `orient_d` / `power_side_d` wrappers | 100–150 | Low | Low | §4.3 — both are the same call. |
| 4 | Symbolic perturbation (index-based SoS) | 120–200 | Medium | Medium | §4.4. Easy to get the signs wrong; test against brute force. |
| 5 | Exact HNF / affine-hull reduction | 150–250 | Medium | Low | §6.3. Replaces ~450 lines of CGAL. |
| 6 | TDS: arena, cells, mirror indices, infinite vertex | 500–800 | Medium | Medium | Mostly bookkeeping; borrow-checker friction with arena indices. |
| 7 | Point location walk (simplified) | 150–250 | Medium | Medium | §3.4. Only needs 3 outcomes, not 6. |
| 8 | Conflict-zone BFS | 100–150 | Low | Low | §3.5. Straightforward. |
| 9 | Hole retriangulation (`insert_in_hole` equivalent) | 300–500 | **High** | **High** | §3.6. The hash-map ridge-matching variant is much safer than rotors. |
| 10 | Hidden-point handling + reporting | 80–150 | Low | Low | §3.6, §7.2. |
| 11 | Initial simplex construction (replaces dimension growth) | 80–150 | Medium | Low | §6.3. |
| 12 | Spatial sort (Hilbert median + multiscale BRIO) | 200–300 | Low | Low | Optional; do last. |
| 13 | Validation / invariant checks | 200–300 | Low | Low | §8. Write these *first*, they pay for themselves. |
| 14 | PyO3 surface, numpy in/out, error types | 150–250 | Low | Low | |
| | **Total** | **~2650–4350** | | | |

Ranked by risk:

1. **(1) exact predicates.** Everything else is meaningless if this is wrong, and "wrong"
   manifests as a hang three thousand insertions later. Mitigation: exhaustive differential
   testing against a naive `BigInt` cofactor expansion (`O(n!)`, fine for `n ≤ 8`) on
   randomly generated matrices including deliberately near-singular ones.
2. **(9) hole retriangulation.** The one place where a subtle combinatorial error produces
   a structurally invalid mesh that only blows up later. Mitigation: assert invariant (1)
   of §8 after every single insertion in debug builds; fuzz with random point sets.
3. **(4) perturbation.** Sign errors here produce *valid-looking* but non-regular
   triangulations. Mitigation: invariant (6) of §8 on small inputs.
4. **Performance at `d ≥ 10`.** The BigInt path is `O(d³)` bigint ops per predicate, and
   there are `O(d)` predicates per walk step. If this is too slow, the answer is not
   "use floats" — it is a better filter (tier 1) and a tighter `i128` range check (tier 2).
   Note the *current* CGAL configuration is already slow here (§5.1 item 5), so the bar is
   low.
5. **(7) walk termination.** With exact predicates it terminates; add an iteration cap that
   returns a hard error anyway, so a bug is a failure and not a hang.

What the port *saves* relative to porting CGAL wholesale: no `remove` (~320 lines of the
hardest code), no `Coaffine`/flat-orientation traits (~300 lines), no
`insert_increase_dimension` (~130 lines of very tricky orientation parity bookkeeping), no
constructions, no kernel-traits metaprogramming layer (which is the majority of
`NewKernel_d` by volume).

---

## 10. Rust ecosystem survey

Surveyed 2026-09-12 against crates.io / docs.rs / GitHub. **Re-verify versions and licenses
at implementation time** — several of the relevant crates are moving fast and several of the
most useful ones are less than a year old.

### 10.0 The honest summary

> **Nothing in the Rust ecosystem does what this project needs.** There is no crate — pure
> Rust or FFI — that computes a **d-dimensional regular (weighted Delaunay) triangulation**.
> There is no general-`d` **weighted/power predicate** in any Rust crate. There is no
> pure-Rust **convex hull in arbitrary dimension**. There is no Rust **TOPCOM** equivalent
> (no chirotopes, no secondary polytopes, no flip-graph enumeration). There is no Rust
> **exact linear algebra** library (no LinBox/FLINT/IML analogue).
>
> What the ecosystem *does* give you, and gives you well, is **arbitrary-precision integer
> arithmetic** and **2D/3D robust predicates**. Everything between those primitives and the
> finished triangulation is yours to write.

The one piece of good news: the "missing middle" — a general-`D` exact determinant sign and
a general-`D` Simulation of Simplicity — exists in two small, new, **BSD-3-Clause** crates
(`la-stack` and `delaunay`) that can legitimately be read, depended on, or vendored.

### 10.1 (a) d-dimensional Delaunay / regular triangulations

| Crate | Version (2026-09-12) | Last release | Max dimension | Weighted? | Exact? | Pure Rust? | License | Usable here? |
|---|---|---|---|---|---|---|---|---|
| `delaunay` (acgetchell) | 0.8.1 | 2026-08-28 | const-`D` generic; **routine 2D–5D, exact only through D=6**, D≥7 falls back to an FP circumcenter test | **No** | partly (interval + rational + SoS, ≤ D=6) | yes | BSD-3 | **No** as a dependency; **yes** as a reference |
| `dd-delaunay` | 0.1.1 | 2025-03-27 | d-dim (claimed) | No | claimed | yes | BSD-3 | No — fork of an old `delaunay`, ~zero adoption |
| `nd-triangulation` | 0.3.4 | **2021-02-01** | arbitrary (CGAL dD binding) | **No** (plain Delaunay) | yes (CGAL's) | no — needs system CGAL + Eigen + g++ | **GPL-3.0-or-later** | No — GPL, dead, and it is the dependency we are removing |
| `qhull` / `qhull-sys` | 0.4.1 | 2026-09-08 | arbitrary in the API; **Qhull itself is documented for ≤8-D hulls, ≤7-D Delaunay** | not directly (lift by hand) | **No** — merged facets / joggle | no — vendored C + `cc` + `bindgen`/libclang | MIT + Qhull's own permissive license | See §6.4 — not recommended |
| `spade` | 2.15.1 | 2026-03-24 | **2D only** | No | yes (via `robust`) | yes | MIT/Apache-2.0 | No |
| `delaunator` | 1.1.0 | 2026-05-12 | **2D only** | No | yes | yes | ISC | No |
| `voronator` | 0.2.1 | 2023-02-14 | **2D only** | No | yes | yes | non-standard | No |
| `cdt` | 0.1.0 | **2021-06-25** | **2D only**, constrained | No | yes | yes | MIT/Apache-2.0 | No |
| `tritet` | 3.2.0 | 2026-06-08 | **2D + 3D** (Triangle / TetGen) | No | yes | no (bundled C/C++) | **MIT OR AGPL-3.0** (AGPL with TetGen) | No |
| `gemlab` | 2.4.0 | 2026-07-16 | **2D + 3D** (FEM meshing, depends on `tritet`) | No | inherited | no | MIT | No |
| `kika`, `rita`, `shull`, `startin`, `simple_delaunay_lib`, `scirs2-spatial` | various | 2024–2026 | **2D/3D only** | No | varies | yes | mixed | No |
| `hypertri` | 0.4.0 | 2026-06-08 | claims an `nd` feature | No | claimed | yes | MIT/Apache-2.0 | No — ~0 adoption, unproven |
| `poisson` | 0.10.1 | **2019-10-04** | n-D **sampling**, not a triangulator | — | — | yes | MIT | Not applicable |
| `hull` | 0.1.3 | 2018 | — | — | — | — | MIT | **Name collision — it is a Rust build tool, not geometry** |
| `d-delaunay` | — | — | — | — | — | — | — | **Does not exist on crates.io.** The repo of that name was renamed to `delaunay` |
| `cgal-rs`, `cgal-sys`, `cgal` | — | — | — | — | — | — | — | **Do not exist.** The only CGAL binding is `nd-triangulation` |

**`delaunay` 0.8.1 in detail**, since it is the only serious pure-Rust contender. Its own
documentation states: routine construction coverage targets 2D–5D; exact orientation,
relative in-sphere and complete symbolic tie-breaking are available **through D=6**; D≥7
falls back to the floating-point circumcenter/radius predicate "and therefore lacks
exact-sign protection near degeneracy"; the default coordinate type is **`f64`**. So: no
weights, no `i64` coordinates, no exactness above D=6. Unusable as a dependency for this
project, but `delaunay::geometry::predicates::{simplex_orientation, insphere,
insphere_lifted}` are genuine `<const D: usize>` generics, `delaunay::geometry::sos` is a
general-`D` Simulation of Simplicity, and it contains what appears to be the only published
Rust bistellar-flip implementation. BSD-3-Clause, so reading and vendoring are both fine.

### 10.2 (b) d-dimensional convex hulls

**There is no pure-Rust convex hull in arbitrary dimension** — no beneath-beyond, no
randomized incremental, no gift-wrapping.

| Crate | Version | Last release | Max dim | Pure Rust? | Exact? | License | Notes |
|---|---|---|---|---|---|---|---|
| `qhull` / `qhull-sys` | 0.4.1 | 2026-09-08 | arbitrary in API; Qhull practical ≤ 8-D | no | no | MIT + Qhull license | §6.4 |
| `chull` | 0.2.4 | **2020-12-08** | **2D/3D — panics above** | yes | BigInt in the wrapper | MIT/Apache-2.0 | Dead. Its source *looks* dimension-generic, but every orientation goes through `util::det`, which ends in `unimplemented!("matrix size should be less than 4 dim …")`. Not a latent n-D library. |
| `convex_hull` | 0.2.3 | 2018 | 3D | yes | no | MIT | Dead |
| `convex-hull-rs` | 0.1.0 | 2026-06-07 | 2D | yes | no | MIT/Apache-2.0 | Trivial |
| `quickhull` (Jondolf) | 0.1.0 | 2026-09-11 | 2D/3D | yes | no | MIT/Apache-2.0 | One day old at survey time |
| `parry2d` / `parry3d` | 0.30.x | 2026-08 | **2D / 3D, hard-coded** | yes | no (f64 + epsilons) | Apache-2.0 | Very active, but wrong dimension |
| `polytope` | 0.1.2 | **2017** | constructs polytopes, **no hull algorithm** | yes | no | Apache-2.0 | Dead |
| `nalgebra` | 0.35.0 | 2026-05-24 | — | yes | — | Apache-2.0 | **Has no convex hull at all.** Its geometry lives in `parry` (2D/3D) |
| `math-convex-hull`, `convexhull3d` | 0.3.3 / 0.5.0 | 2026 / 2025 | 3D | yes | no | ISC / GPL-3.0+ | Small |
| `lp-modeler` | 0.5.0 | **2021-03-01** | — | no (shells out to CBC) | — | MIT | **Abandoned. Do not use.** |

**Arbitrary-dimension polyhedral tools reachable from Rust — all copyleft:**

| Crate | Backend | License |
|---|---|---|
| `howzat` / `hullabaloo-core` 0.4.0 | double-description + LRS, arbitrary-d H↔V | **AGPL-3.0-only** |
| `cddlib-sys` / `cddlib-rs` | C cddlib + GMP | **GPL-2.0+ / AGPL-3.0** |
| `lrslib-sys` / `lrslib-rs` | C lrslib (exact rational simplex) | **GPL-2.0+ / AGPL-3.0** |
| `ppl-sys` / `ppl-rs` | C++ Parma Polyhedra Library | **GPL-3.0+ / AGPL-3.0** |

> **Every arbitrary-dimension exact polyhedral tool reachable from Rust is GPL or AGPL.**
> For a permissively licensed pip-installable wheel this is a licensing blocker, not merely
> build friction. (`cdd` and `lrs` on crates.io are name collisions — a cookie decryptor and
> a logic solver.)

**LP solvers** (relevant if the TOPCOM-side port needs regularity / secondary-cone tests):
`good_lp` 1.15.3 (MIT, but its *default* feature is `coin_cbc`, i.e. C++ — use
`default-features = false`), `microlp` 0.6.0 (Apache-2.0, pure Rust, the maintained fork of
the abandoned `minilp`), `clarabel` 0.11.1 (Apache-2.0, conic IPM, f64), `highs`/`highs-sys`
(MIT bindings to C++ HiGHS). **Every Rust LP solver is floating point. There is no exact
rational simplex in Rust** — `lrslib` is that, and it is C under GPL.

### 10.3 (c) Exact / robust geometric predicates

| Crate | Version | Last release | Predicates | **Dimensions** | Pure Rust? | License | Verdict |
|---|---|---|---|---|---|---|---|
| `robust` (georust) | 1.2.0 | 2025-05-10 | `orient2d`, `orient3d`, `incircle`, `insphere` | **2D/3D only** | yes, zero deps, `no_std` | MIT/Apache-2.0 | Mature (23.7M downloads). **Use it for the `d = 2` and `d = 3` fast paths** — it will beat anything hand-written there. Useless above 3D. |
| `geometry-predicates` | 0.3.0 | **2021-02-10** | same four, plus raw expansion primitives | **2D/3D only** | yes (c2rust transpile) | MIT | Unmaintained; needs an `exactinit()` call |
| `geogram_predicates` | 0.4.0 | 2026-08-22 | orient2d/3d, in_circle/in_sphere, `det_3d/4d`, **`side1_sos`…`side4_sos` (power-side / weighted!)** | 2D/3D; the `side*` predicates are generic over ambient `D` but **assert `D ∈ {3,4,6,7,8}`** | yes as of 0.4 (pure-Rust port of Geogram PCK) | **LGPL-3.0 OR MIT** | The closest thing to a weighted predicate in Rust — and still only 2D/3D weighted points. **This is the sharpest single gap vs. CGAL.** |
| **`la-stack`** | 0.4.6 | 2026-09-08 | **`Matrix<D>::det_sign_exact()`** (Bareiss over `num-bigint`), `IntervalMatrix<D>`, `RationalMatrix<D>::{det, det_sign, solve}` | **general `D`** (stack ≤ 64; rational ≤ 32; interval ≤ 7) | yes | **BSD-3-Clause** | **The single most relevant crate in this survey.** It is exactly the primitive of §5.5 tier 3. New and small (~3k downloads), so read it before depending on it — but vendoring is permitted and sensible. |
| **`delaunay::geometry::predicates` / `::sos`** | 0.8.1 | 2026-08-28 | `simplex_orientation<D>`, `insphere<D>`, `insphere_lifted<D>`, general-`D` SoS | general const-`D`, complete only **through D=6** | yes | **BSD-3-Clause** | Best available reference implementation of §4.4. Not published standalone. |
| `hyperlimit` | 0.4.0 | 2026-06-08 | `predicates::nd::{orient_d, insphere_d, affine_independent_d}` | arbitrary `d` (runtime `Vec`) | yes | MIT/Apache-2.0 | Unproven (~400 lifetime downloads); hyperreal backend is slow |
| `robust-predicates` | 0.1.4 | 2022-10-20 | orient/incircle/insphere | 2D/3D | **no — `cc` build dep, compiles C** | MIT | Abandoned; disqualified for easy wheels |
| `robust-geo` | 0.1.7 | 2021-02-08 | orient/in_circle/in_sphere, `sign_det_*` | 2D/3D | yes | MIT | Abandoned |
| `simplicity` | 0.4.2 | 2021-02-08 | SoS-perturbed orient / in_circle / in_sphere | **1D/2D/3D only** | yes | MIT | Abandoned. **The only standalone SoS crate**, and it does not cover our dimensions. |
| `apfp` | 0.1.1 | 2026-01-04 | orient2d, incircle, `apfp_signum!` macro for exact sign of arbitrary expressions | **2D only** | yes, zero deps | Unlicense | New; repo 404s |
| `exactpred`, `fast_polygon` | — | — | — | — | — | — | **Do not exist on crates.io** |
| `gaol` | — | — | — | — | — | MPL | **Name collision** — Servo's OS sandbox, not interval arithmetic |
| `spade`'s predicates | — | — | — | — | — | — | **Not exposed.** It just depends on `robust ^1.2` — use `robust` directly |

**Interval arithmetic** (for filter tier 2, if you want intervals rather than error bounds):
`inari` 2.0.0 (MIT, IEEE 1788, x86-64/aarch64 SIMD; ⚠️ **its default features pull in `rug`
→ LGPL + C**, so use `default-features = false`) and `maryada` 0.2.1 (MIT/Apache-2.0,
`no_std`, IEEE 1788.1, brand new).

**Answer to the question that matters:** is there a general-`d` exact orientation predicate
in Rust? Only in `la-stack` (as a determinant sign) and `delaunay` (as a predicate, capped
at D=6). Is there a general-`d` **weighted/power** predicate? **No — nowhere.** Which is
fine, because §4.3 shows you don't need one: it *is* the orientation predicate in `d+1`.

### 10.4 (d) Exact rational and big-integer arithmetic

| Crate | Version | Last release | License | Pure Rust? | `no_std`? | Verdict |
|---|---|---|---|---|---|---|
| `num-bigint` | 0.4.8 **and** 0.5.1 | 2026-07-05 | MIT/Apache-2.0 | yes | yes (`alloc`) | The safe default. 0.4.7 added **single-digit inlining** (no heap alloc for small values) + Burnikel–Ziegler division, which closed most of its historical small-value gap. |
| `num-rational` | 0.4.2 | 2024-05-08 | MIT/Apache-2.0 | yes | yes | ⚠️ pins `num-bigint = "^0.4"` and there is no 0.5. **Pin `num-bigint` 0.4.8, not 0.5.1**, or you compile two copies. |
| `dashu` / `dashu-int` / `dashu-ratio` | 0.6.0 | 2026-08-09 | **MIT OR Apache-2.0** | yes | yes | **Recommended.** Fastest in class on *small* values, which is what dominates a predicate kernel. Signed `IBig` natively; small values inlined on the stack. |
| `ibig` | 0.3.6 | **2022-09-17** | MIT/Apache-2.0 | yes | yes | Effectively superseded — `dashu-int`'s README says the majority of its code derives from `ibig` |
| `bnum` | 0.14.4 | 2026-03-24 | MIT/Apache-2.0 | yes | **yes, and no-alloc** | Interesting fixed-width option: your bit bound is known at compile time (§5.2), so one 3072-bit type covers every `d ≤ 30` with zero allocation. But 0.14 is a redesign with novel `u8` digit storage and the roadmap still lists faster mul/div as future work. **Benchmark before committing.** |
| `crypto-bigint` | 0.7.5 | 2026-06-22 | Apache-2.0/MIT | yes | yes | **Constant-time by default** — pure overhead for geometry. Avoid. |
| `ruint` | 1.20.0 | 2026-07-30 | MIT | yes | yes | Unsigned only — awkward for signed determinants |
| `fixed` | 1.31.0 | 2026-03-20 | MIT/Apache-2.0 | yes | yes | **128-bit ceiling, fixed-*point* not big-int. Not applicable.** |
| `rug` / `gmp-mpfr-sys` | 1.30.0 / 1.7.1 | 2026-04 | **LGPL-3.0+** | **no** — builds GMP/MPFR/MPC from C | no | **Blocked twice over**: LGPL (a maturin wheel statically links, triggering §4 relinking obligations) *and* it needs `gcc`/`m4`/`make`, requires the **Rust GNU toolchain + MSYS2 on Windows (MSVC unsupported)**, and refuses paths containing spaces. |
| `malachite` (+ `-base`, `-nz`, `-q`) | 0.11.0 | 2026-08-28 | **LGPL-3.0-only** | yes | yes | **Blocked.** Verified at HEAD: `malachite-base/Cargo.toml` reads `license = "LGPL-3.0-only"`; there is no permissive subset and no static-linking exception. Fast, but rule it out for permissive wheels. |
| `fraction` | 0.17.0 | 2026-08-22 | MIT/Apache-2.0 | yes | partial | Usable but not needed if you scale lifts to integers (§5.3) |
| `ramp` | 0.7.0 | 2022 | Apache-2.0 | yes (nightly asm) | no | Dead |

**Extended-precision floats** (filter tier candidates): `qd` 0.8.0 (MIT, **double-double
only despite the name**, ~106-bit mantissa, SIMD-aware, by the `faer` author — best choice
here), `twofloat` 0.8.4 (BSD-3, double-double, own docs flag preliminary math functions),
`astro-float` 0.9.6 (MIT, pure-Rust MPFR analogue, correctly rounded), `dashu-float` 0.6.0
(free if you already use dashu). `doubledouble` and `double-double` **do not exist**.
**There is no quad-double (~212-bit) crate in Rust** — between ~106 bits and arbitrary
precision there is nothing.

**Benchmarks** (DRMacIver/rust-bigint-benchmarks, M2 Max, Criterion). Small values (~100
bits) — the regime that dominates predicate work, since most determinants resolve in the
first few words — ns per op:

| Op | dashu | ibig | num-bigint | malachite | rug |
|---|---|---|---|---|---|
| Add | **3.7** | 25.7 | 18.7 | 18.7 | 21.9 |
| Div | **5.7** | 65.2 | 112.0 | 71.2 | 32.4 |

Medium values (~1k–4k bits): `rug` wins on raw mul/div, `dashu` and `malachite` win on GCD.
Two caveats: the benchmark used `num-bigint` **0.4.6**, i.e. before the 0.4.7 small-value
inlining landed; and `rug`/GMP's edge only appears above ~1 kbit — below that its
per-value heap allocation makes it 2–6× *slower* than `dashu`.

**Exact linear algebra is a void.** `nalgebra` **cannot** do this: `Matrix::determinant`
for `n > 3` goes through LU, which requires `ComplexField` (needs `sqrt`, `abs`,
transcendentals), so `BigRational`/`IBig` can never satisfy the bound — see
`dimforge/nalgebra#763` and `#503`. `faer` and `simba` are float-only. The "exact linear
algebra" crates that surface in search (`neco-linear-exact`, `bicmath-linear-algebra`,
`deep_causality_linear`) are all single-author, weeks old, and not shippable.
**`la-stack` is the only credible existing implementation, and it is tiny and new.**

> **Budget for writing fraction-free Bareiss elimination yourself.** It is roughly 200 lines
> and it is the load-bearing primitive of the entire port. This is the single largest
> unpriced item in §9 line 1.

### 10.5 (e) Point configurations / regular subdivisions / TOPCOM-like

**Nothing.** A crates.io full-text search returns **zero** results for `secondary polytope`
and **zero** for `chirotope`. There is no Rust TOPCOM binding or port (TOPCOM is C++ only;
SageMath and polymake shell out to it). There is no polymake, Normaliz, 4ti2 or LattE
binding on crates.io. There is no secondary polytope / secondary fan / secondary cone
implementation, no enumeration of regular subdivisions or of triangulations of a point
configuration, no flip-graph traversal, and no oriented-matroid crate.

The only adjacent Rust code:

| Crate | Version | License | What it has |
|---|---|---|---|
| `delaunay` | 0.8.1 | BSD-3 | **Bistellar flips** — the only published Rust implementation found — plus general-`D` SoS |
| `howzat` / `hullabaloo-*` | 0.4.0 | **AGPL-3.0-only** | Arbitrary-`d` H↔V via double-description + LRS. Would serve for secondary-cone work; AGPL and ~0 adoption |
| `nauty-pet` / `nauty-Traces-sys` | 0.15.0 / 0.11.0 | Apache-2.0 | Mature nauty bindings (C). Relevant if the port needs symmetry groups, as TOPCOM's `SymmetryGroup` / `SymmetricFlipGraph` do |
| `canonaut` | 1.0.0 | — | Pure-Rust nauty port. Unproven |
| `lll-rs` | 0.2.0 | **2020** | LLL lattice reduction. Dead — but note §6.3 needs only HNF, not LLL |

**Prior art worth reading.** [`ndbroadbent/cyrus`](https://github.com/ndbroadbent/cyrus)
is a Rust reimplementation of CYTools' core, advertising "FRSTs, secondary cones, regular
triangulations with exact integer hulls". Its dependency set is instructive and reaches the
same conclusions as this document: `delaunay` (d-dimensional triangulations), `malachite`
(arbitrary precision — a licensing choice this project should **not** copy, §10.4),
`num-rational`, `good_lp` with HiGHS (for Kähler-cone tests), `lll-rs`, `nalgebra`,
`ndarray`, `faer`. The author's stated philosophy — "don't waste time searching for Rust
crates with bindings; find the algorithm in any language, understand it, and port it" — is
the correct conclusion from this survey.

### 10.6 Recommended dependency set

| Need | Pick | Why |
|---|---|---|
| Big integers | **`dashu-int` 0.6** (`IBig`), MIT/Apache-2.0 | Fastest on small values, which dominate predicate work; signed natively; small values inlined on the stack; `no_std`; active |
| Big integers (conservative alternative) | **`num-bigint` 0.4.8** + `num-rational` 0.4.2 | 580M downloads, most-audited. Pin **0.4.8**, not 0.5.1, so `num-rational` resolves to one copy |
| Fixed-width alternative | `bnum` 0.14 — **benchmark first** | The bit bound is known at compile time (§5.2); one 3072-bit type covers `d ≤ 30` with zero allocation |
| `i128` tier | std | §5.5 tier 2, no dependency |
| f64 filter tier | hand-rolled error bounds, optionally `qd` 0.8 (double-double) | Rust has no stable rounding-mode control, so CGAL's `Interval_nt_advanced` is not portable; error-bounded round-to-nearest is the idiom. `f64::mul_add` (FMA) is stable. |
| `d = 2`, `d = 3` fast paths | **`robust` 1.2.0** | 23.7M downloads, MIT/Apache-2.0, zero deps, `no_std`. Will beat anything hand-written at those dimensions. |
| Exact determinant sign, general `D` | **read / vendor `la-stack`** (BSD-3) | The exact primitive of §5.5 tier 3 |
| SoS reference | **read `delaunay::geometry::sos`** (BSD-3) | General-`D` Simulation of Simplicity |
| LP (only if the TOPCOM side needs it) | `good_lp` with `default-features = false` + `microlp` | Only pure-Rust path; **float-only**, so any answer near degeneracy needs exact verification |
| Python bindings | PyO3 + maturin | As already planned |

Everything in that table is MIT and/or Apache-2.0 or BSD-3-Clause, pure Rust with no C
dependency, and builds cleanly under maturin on Linux, macOS and Windows-MSVC — which is
more than can be said for any path through Qhull, CGAL, cddlib, lrslib, GMP, `rug` or
Malachite.

### 10.7 The decision that gates the arithmetic choice

> **The `f64` weights, not the `i64` coordinates, are the real precision hazard.**

Coordinates bound cleanly (§5.2). But arbitrary `f64` weights convert to exact rationals
with denominators up to `2^1074`, and the common-denominator scaling of §5.3 then blows the
bit bound wide open — which silently breaks any fixed-width (`bnum`) plan. **Settle this
before choosing the arithmetic backend:**

- **Option 1 (recommended):** define the API to accept weights/heights on a common `2^k`
  grid, validate `k` (reject subnormals and absurd exponent spreads), and document the bound.
  Then `bnum` with a compile-time width is viable and fast.
- **Option 2:** accept arbitrary `f64`, carry the lifted column as a rational with a shared
  denominator, track the exponent spread explicitly, and use an arbitrary-precision backend
  (`dashu-int` / `num-bigint`).
- **Option 3 (best of all, if the callers can supply it):** add integer/rational height
  inputs to the Python API. For lattice-polytope work these are the natural inputs anyway,
  and they make the whole pipeline exact with no scaling.

---

## 11. References

### CGAL source (paths relative to the sparse checkout described in the preamble)

**Algorithm**

- `Triangulation/include/CGAL/Regular_triangulation.h:27` — class definition.
- `Triangulation/include/CGAL/Regular_triangulation.h:219-238` — range `insert`, spatial sort, hint chaining.
- `Triangulation/include/CGAL/Regular_triangulation.h:307-350` — `Conflict_predicate` (finite + infinite cell cases).
- `Triangulation/include/CGAL/Regular_triangulation.h:406-449` — conflict-zone vertex bookkeeping (the hidden-point mechanism).
- `Triangulation/include/CGAL/Regular_triangulation.h:794-834` — `insert(p, lt, f, ft, s)` dispatch.
- `Triangulation/include/CGAL/Regular_triangulation.h:845-890` — `insert_outside_affine_hull`.
- `Triangulation/include/CGAL/Regular_triangulation.h:962-1008` — `insert_in_conflicting_cell` (the main path).
- `Triangulation/include/CGAL/Regular_triangulation.h:1013-1069` — `perturbed_power_side_of_power_sphere` (SoS).
- `Triangulation/include/CGAL/Regular_triangulation.h:1071-1092` — `is_in_conflict`.
- `Triangulation/include/CGAL/Regular_triangulation.h:1094-1119` — `compute_conflict_zone`.
- `Triangulation/include/CGAL/Regular_triangulation.h:1122-1163` — `is_valid` (empty power sphere).
- `Triangulation/include/CGAL/Triangulation.h:1086-1255` — `do_locate`, the remembering stochastic walk.
- `Triangulation/include/CGAL/Triangulation.h:889-918` — base `insert` dispatch (unused by RT).
- `Triangulation/include/CGAL/Triangulation.h:993-1023` — `insert_outside_convex_hull` (unused by RT).
- `Triangulation/include/CGAL/Triangulation.h:1029-1076` — base `insert_outside_affine_hull`.
- `Triangulation/include/CGAL/Triangulation.h:781-804` — `Outside_convex_hull_traversal_predicate`.
- `Triangulation/include/CGAL/Triangulation.h:249-272` — `Rotor` and `rotate_rotor`.
- `Triangulation/include/CGAL/Triangulation.h:1321-1360` — `Triangulation::is_valid`.
- `Triangulation/include/CGAL/Triangulation.h:727-752` — `insert_and_index` (the fix for the index scan).
- `Triangulation/include/CGAL/Triangulation_data_structure.h:665-704` — `gather_full_cells` (conflict BFS).
- `Triangulation/include/CGAL/Triangulation_data_structure.h:933-1032` — `insert_in_tagged_hole` (hole coning + rotor walk).
- `Triangulation/include/CGAL/Triangulation_data_structure.h:1035-1062` — `insert_in_hole`.
- `Triangulation/include/CGAL/Triangulation_data_structure.h:1090-1160` — `do_insert_increase_dimension`.
- `Triangulation/include/CGAL/Triangulation_data_structure.h:1164-1203` — `insert_increase_dimension`.
- `Triangulation/include/CGAL/Triangulation_ds_full_cell.h:29-310` — full-cell storage and accessors.
- `Triangulation/include/CGAL/Triangulation_ds_vertex.h` — vertex storage (point + one incident cell).
- `Triangulation/include/CGAL/TDS_full_cell_default_storage_policy.h:30-91` — XOR mirror-vertex trick.
- `Triangulation/include/CGAL/TDS_full_cell_mirror_storage_policy.h` — explicit mirror indices.
- `Triangulation/include/CGAL/Regular_triangulation_traits_adapter.h:32-278` — weighted-point glue.
- `Triangulation/include/CGAL/Triangulation/internal/utilities.h:94-112` — `Compare_points_for_perturbation`.

**Predicates and kernel**

- `NewKernel_d/include/CGAL/NewKernel_d/function_objects_cartesian.h:42-77` — generic `Orientation_of_points`.
- `NewKernel_d/include/CGAL/NewKernel_d/function_objects_cartesian.h:80-135` — unrolled `d ≤ 6` and `d = 1`.
- `NewKernel_d/include/CGAL/NewKernel_d/function_objects_cartesian.h:482-530` — `Power_side_of_power_sphere_raw` (**the formula**).
- `NewKernel_d/include/CGAL/NewKernel_d/function_objects_cartesian.h:536-600` — `Side_of_oriented_sphere` (unweighted).
- `NewKernel_d/include/CGAL/NewKernel_d/Types/Weighted_point.h:105-144` — weighted wrappers.
- `NewKernel_d/include/CGAL/NewKernel_d/Coaffine.h:22-26` — `Flat_orientation`.
- `NewKernel_d/include/CGAL/NewKernel_d/Coaffine.h:53-108` — `Construct_flat_orientation`.
- `NewKernel_d/include/CGAL/Epick_d.h:28-80` — the filtered kernel stack.
- `NewKernel_d/include/CGAL/NewKernel_d/Filtered_predicate2.h:48-99` — interval filter + exact fallback.
- `NewKernel_d/include/CGAL/NewKernel_d/Cartesian_filter_K.h:23-41` — `Functors_without_division` (why the power test uses the *field* type).
- `NewKernel_d/include/CGAL/NewKernel_d/Cartesian_static_filters.h:15-24` — static filters only for `Orientation_2..6`, `Side_of_oriented_{circle_2,sphere_3}`.
- `NewKernel_d/include/CGAL/NewKernel_d/Cartesian_LA_base.h:28-32` — `CGAL_EIGEN3_ENABLED` is mandatory.
- `NewKernel_d/include/CGAL/NewKernel_d/LA_eigen/LA.h:104-170` — `determinant` (unrolled to 7, then `PartialPivLU`).
- `Number_types/include/CGAL/Number_types/internal/Exact_type_selector.h:129-180` — exact backend selection.
- `Installation/include/CGAL/Installation/internal/enable_third_party_libraries.h:15-58` — `CGAL_DISABLE_GMP` → Boost.Multiprecision backend.

**Spatial sorting**

- `Spatial_sorting/include/CGAL/spatial_sort.h:81-101` — the `Point_d` overload and its thresholds.
- `Spatial_sorting/include/CGAL/spatial_sort.h:160-172` — default `Hilbert_sort_median_policy`.
- `Spatial_sorting/include/CGAL/Multiscale_sort.h:37-45` — BRIO.
- `Spatial_sorting/include/CGAL/Hilbert_sort_median_d.h:47-160` — `d`-dimensional Hilbert sort.

### This project

- `src/triangulumancer/CGAL.cpp:12-102` — the whole wrapper.
- `src/triangulumancer/CGAL.cpp:41-48` — weights↔heights (**`int` overflow bug**, §2.4).
- `src/triangulumancer/CGAL.cpp:59-84` — `O(n·V·d)` index recovery (§7.6).
- `src/triangulumancer/CGAL.cpp:87-99` — simplex extraction (**breaks for rank-deficient input**, §7.4).
- `include/triangulumancer/CGAL.hpp` — the interface to preserve.
- `include/triangulumancer/PVConfiguration.hpp:76-79` — the Python-facing entry points.
- `CMakeLists.txt:31` — `CGAL_DISABLE_GMP ON`.
- `CMakeLists.txt:47,52` — `CGAL::CGAL`, `Eigen3::Eigen`, `CGAL_EIGEN3_ENABLED`.

### CGAL manual

- dD Triangulations, user manual: https://doc.cgal.org/latest/Triangulation/index.html
  (especially §"Regular Triangulations" and §"Implementation and Complexity").
- `CGAL::Regular_triangulation` reference:
  https://doc.cgal.org/latest/Triangulation/classCGAL_1_1Regular__triangulation.html
- `TriangulationDataStructure` concept:
  https://doc.cgal.org/latest/TDS/classTriangulationDataStructure.html
- `RegularTriangulationTraits_d` concept:
  https://doc.cgal.org/latest/Triangulation/classRegularTriangulationTraits__d.html
- dD Geometry Kernel (`Epick_d`, `Epeck_d`):
  https://doc.cgal.org/latest/Kernel_d/index.html
- Spatial Sorting: https://doc.cgal.org/latest/Spatial_sorting/index.html

### Papers and books

- S. Boissonnat, O. Devillers, S. Hornus, *Incremental construction of the Delaunay
  triangulation and the Delaunay graph in medium dimension*, SoCG 2009. — The algorithm
  CGAL's dD package implements; describes the walk, the conflict zone, and the data
  structure.
- C. Jamin, S. Pion, M. Teillaud, *dD Triangulations* (CGAL package documentation and the
  accompanying design notes). Clément Jamin is the author of
  `Regular_triangulation.h`.
- K. Q. Brown, *Voronoi diagrams from convex hulls*, IPL 9(5), 1979. — The lifting map.
- H. Edelsbrunner, R. Seidel, *Voronoi diagrams and arrangements*, DCG 1, 1986. — Regular
  triangulations as lower hulls; power diagrams.
- G. M. Ziegler, *Lectures on Polytopes*, Springer 1995, §5 ("Regular subdivisions and the
  secondary polytope"). — The `h ↦ S(h)` correspondence used by both this code and TOPCOM.
- J. A. De Loera, J. Rambau, F. Santos, *Triangulations: Structures for Algorithms and
  Applications*, Springer 2010. — Chapters 2 (regular subdivisions, the lifting), 5
  (flips, flip-connectivity), 8 (the TOPCOM algorithms). The reference for the
  TOPCOM-replacement side too.
- H. Edelsbrunner, E. P. Mücke, *Simulation of Simplicity: a technique to cope with
  degenerate cases in geometric algorithms*, ACM TOG 9(1), 1990. — The `ε^{π(i)}`
  perturbation of §4.4.
- I. Emiris, J. Canny, *A general approach to removing degeneracies*, SIAM J. Comput.
  24(3), 1995. — Cheaper deterministic perturbations for determinant predicates.
- J. R. Shewchuk, *Adaptive precision floating-point arithmetic and fast robust geometric
  predicates*, DCG 18, 1997. — The error-bounded filter idiom; the reason the `robust`
  crate exists; also the reason a general-`d` version is not available off the shelf.
- E. H. Bareiss, *Sylvester's identity and multistep integer-preserving Gaussian
  elimination*, Math. Comp. 22, 1968. — The fraction-free determinant algorithm
  recommended in §5.5.
- C. B. Barber, D. P. Dobkin, H. Huhdanpaa, *The Quickhull algorithm for convex hulls*,
  ACM TOMS 22(4), 1996. — Qhull; relevant to §6.4.
- N. Amenta, S. Choi, G. Rote, *Incremental constructions con BRIO*, SoCG 2003. — The
  biased randomized insertion order that `Multiscale_sort` implements.
