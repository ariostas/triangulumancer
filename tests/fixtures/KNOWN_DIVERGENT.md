# Known divergences from the C++ oracle

The fixture corpus in this directory records the output of `triangulumancer`
0.3.0 — the last C++ release — over a corpus of configurations. The Rust port is
expected to reproduce it *except* where the oracle is known to be wrong.

Those cases carry an `untrusted` list naming the affected results. The test
suites must **not assert against** them — neither equality nor inequality.

Asserting inequality would be tempting, but a broken computation can still land
on the right answer by luck, and a test that fails for a *correct* implementation
is worse than no test. So where the correct value is independently known, it is
written down below as a **positive assertion** instead. Those are stronger, and
they are the real acceptance criteria for the phases that fix each bug.

The bugs are catalogued as B1–B5 in [`PLAN.md`](../../PLAN.md) section 3, and are
deliberately not fixed in the C++.

## B1 — `all_connected_triangulations(only_fine=True)` with a non-fine seed

`only_fine` is not an output filter in the flip-graph path. It is forwarded as
`forbid_vertex_removal`, which *deletes edges* from the flip graph, and that is
only sound starting from a fine triangulation. TOPCOM's own driver seeds with
`FineTriang` in this branch; `src/triangulumancer/TOPCOM.cpp:160` always seeds
with `PlacingTriang`, which skips points interior to the hull of the points
placed before them.

**Marking:** computed from the oracle's own output. If the placing triangulation
does not use every point, `all_connected_triangulations_fine` is marked.

**Correct values**, from `all_triangulations(only_fine=True)` — the extension
graph prunes correctly and unconditionally, so it is trustworthy here, and for
every configuration in the corpus whose flip graph is connected the two must
agree:

| configuration | oracle `conn_fine` | correct |
|---|---:|---:|
| `octagon_vectors` | 4 | **1** |
| `interior_point_triangle` | 2 | **1** |
| `many_interior_points` | 18 | **8** |
| `moderate_coords` | 4 | **1** |
| `duplicate_points` | 1 | **0** |
| `square_plus_axes` | *does not terminate* | **64** |

Note `many_interior_points`: the oracle returns 18, which is exactly its
unfiltered count. `only_fine=True` filtered nothing at all.

**Observed severity is worse than "wrong results".** For `square_plus_axes` — the
nine points of the 3×3 grid, ordered with the origin fifth rather than first —
the call does not terminate. The same nine points with the origin first
(`grid_3x3`) complete in well under a second and give the right answer. The bug
is order-sensitive, which is exactly what a placing-order-dependent seed
predicts, and a released version hanging on a nine-point two-dimensional input
is the sharpest single argument in the corpus for the rewrite.

**Phase 4** fixes this by seeding with the fine triangulation, and yielding
nothing when that is not fine (no fine triangulation exists then). All six values
above are asserted in `tests/flipgraph.rs`, each also as a set equal to the
oracle's `all_triangulations(only_fine=True)`.

## B3 — integer overflow in the height→weight conversion

`CGAL.cpp:45-46` truncates `int64_t` coordinates to `int` and squares them in
`int` arithmetic while computing `weight = -height + |p|²`. This overflows for
`|coord| >= 46341`.

**Marking:** every `triangulate_with_heights__*` result on a case tagged
`large_coords`.

**Phase 6** fixes this by computing the lift exactly.

**Correct values** (asserted in `tests/regular.rs`, `known_divergent_b3`). By
hand: with heights, the regular triangulation of a square is the diagonal with
the lower midpoint, and where all four (or five) lifted points are coplanar the
tie goes as for every other cospherical case (`src/regular/predicates.rs`: the
lexicographically largest point counts as raised).

| configuration | method | correct |
|---|---|---|
| `large_coords_b3` | `heights__alt` | `[[0,1,3],[0,2,3]]` |
| `large_coords_b3` | `heights__ones`, `heights__ramp` (both coplanar) | `[[0,1,2],[1,2,3]]` |
| `moderate_coords` | `heights__alt` (centre above the 0–3 diagonal) | `[[0,1,3],[0,2,3]]` |
| `moderate_coords` | `heights__ones` (all coplanar) | `[[0,1,4],[0,2,4],[1,3,4],[2,3,4]]` |
| `moderate_coords` | `heights__ramp` (corners coplanar, centre above) | `[[0,1,2],[1,2,3]]` |

These happen to equal the oracle's output — a broken computation landing on
the right answer, which is why the entries stay untrusted rather than being
asserted unequal. The same test adds cases beyond the corpus — weights
`(0, 0, 0, 2·10¹⁰ + 1)` on the square of side `10⁵` (whose `|p|²` overflows 32 bits), correctly
`[[0,1,3],[0,2,3]]` — and squares with coordinates `2⁶²`.

## B4 — rank-deficient input yields a malformed simplices array

`CGAL.cpp:87-98` looks vertices up in a `std::map` with `operator[]`, so null
vertex handles silently insert index `0`. For a configuration that does not span
its ambient space the result is an array padded with zeros, of the wrong shape.

**Marking:** every `delaunay_triangulation` and `triangulate_with_*` result on a
case tagged `rank_deficient`, or whose `placing_triangulation` the oracle rejected
as not full rank. The second condition catches `single_point`, which is
rank-deficient without being tagged so; it was missing until phase 1, when the
rank test over the corpus (`tests/fixture_rank.rs`) tripped over it.

**Observed:** for four collinear points in the plane the oracle returns
`[[0,0,1],[0,1,2],[0,2,3]]` — each 1-simplex padded with a spurious leading `0`,
in an array of the wrong width. For a single point in the plane it returns
`[[0,0,0]]`. The correct answers are the 1-dimensional triangulation
`[[0,1],[1,2],[2,3]]` and the 0-dimensional `[[0]]`, in the configuration's own
affine hull.

**Phase 1** fixes this structurally, via the exact affine-hull reduction
(`Config::reduce_to_hull`): a rank-deficient configuration is reduced to its own
affine hull rather than producing garbage. Phase 1 checks the hulls themselves —
the collinear case reduces to the points `0, 1, 2, 3` on a line, the single point
to dimension 0 — and **phase 6** asserts the triangulations above.

**Asserted** in `tests/regular.rs` (`known_divergent_b4`): for
`rank_deficient_collinear`, every method except `heights__alt` gives
`[[0,1],[1,2],[2,3]]`, and `heights__alt` (heights `-1, 0, 1, -1`, so the
middle points are above the segment) gives `[[0,3]]` with points 1 and 2
hidden; `single_point` gives `[[0]]`; and `rank_deficient_coplanar_3d` under
Delaunay gives the cocircular unit square of its own plane, `[[0,1,2],[1,2,3]]`
(the oracle's rows with the padding `0` removed).

## B5 — index recovery by coordinate scan

`CGAL.cpp:59-84` maps CGAL vertices back to input indices with an `O(n·V·d)`
scan comparing `std::round(CGAL::to_double(coord))`. With duplicate points it
picks an arbitrary one of them, and it breaks above `2^53`.

**Marking:** every `delaunay_triangulation` and `triangulate_with_*` result on a
case tagged `duplicates`.

Unlike B1, B3 and B4, this one is not a wrong answer — dropping a duplicate point
is correct, and reporting index 1 rather than index 3 as the surviving vertex is
merely arbitrary. It is marked `untrusted` because there is nothing to assert,
not because the oracle is wrong. The Rust port should make the choice
deterministic (lowest index wins) and report the hidden points.

**Phase 6** fixes this by carrying the index through insertion instead of
recovering it afterwards.

**Asserted** in `tests/regular.rs` (`known_divergent_b5`), for
`duplicate_points` (points 1 and 3 coincide at `(1, 0)`): the lower lift wins,
and on equal lifts the lower index. So the survivor is point 3 for
`heights__alt` (heights `0` vs `-1`) and `weights__ramp` (lifts `1 − 2` vs
`1 − 4`), and point 1 for the other five results; the simplex is `{0, 2,
survivor}` and the other copy is the one hidden point. The same test checks
that distinct points `(2⁶⁰, 0)` and `(2⁶⁰ + 1, 1)`, which are one double apart,
are both used.

## Not a bug: timeouts and crashes

A result may also be recorded as `{"timeout": n}` or `{"crash": code}` rather
than `{"ok": ...}` or `{"error": ...}`. These are not assertions about the Rust
implementation — they record that the oracle could not produce an answer within
the budget, or exited the interpreter outright. The Rust port is free to
succeed where the oracle timed out, and is expected to; `{"crash": ...}` entries
should become ordinary exceptions.

Regenerate with a larger budget via:

```bash
python scripts/gen_fixtures.py --timeout 900 --only <name>
```

**`vc_degenerate_flip` does not simply need a larger budget.** Its four
enumeration entries are recorded as timeouts at 30s, and a re-run at 900s was
stopped under system memory pressure while `all_triangulations` was still in
flight, several minutes in. Sixteen vectors of rank four is not a large input,
and the oracle collects every triangulation into a `std::vector` before
returning any of them.

That is the concrete case for the streaming enumeration in
[`PLAN.md`](../../PLAN.md) section 5.4: the Rust core yields triangulations
through a callback or iterator and collects into a `Vec` only when asked, so
counting or filtering a large enumeration costs no memory. It is also why
phase 5's acceptance criteria should include this configuration finishing.

**Phase 5 outcome.** The Rust enumeration streams it in constant memory (about 4 MB,
however long it runs) and yields its first triangulations at once, but it has *more than
8.3 billion triangulations and more than 4.3 billion fine ones* — counts reached in 44
minutes on several threads before the runs were stopped. So the oracle's problem was
never only memory: the complete count takes hours. It is the `#[ignore]`d
`vc_degenerate_flip_count` in `tests/extension.rs`. Never collect it into a list.
