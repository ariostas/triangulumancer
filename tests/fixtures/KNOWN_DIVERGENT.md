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

**Phase 4** fixes this by seeding with the fine triangulation.

## B3 — integer overflow in the height→weight conversion

`CGAL.cpp:45-46` truncates `int64_t` coordinates to `int` and squares them in
`int` arithmetic while computing `weight = -height + |p|²`. This overflows for
`|coord| >= 46341`.

**Marking:** every `triangulate_with_heights__*` result on a case tagged
`large_coords`.

**Phase 6** fixes this by computing the lift exactly.

## B4 — rank-deficient input yields a malformed simplices array

`CGAL.cpp:87-98` looks vertices up in a `std::map` with `operator[]`, so null
vertex handles silently insert index `0`. For a configuration that does not span
its ambient space the result is an array padded with zeros, of the wrong shape.

**Marking:** every `delaunay_triangulation` and `triangulate_with_*` result on a
case tagged `rank_deficient`.

**Observed:** for four collinear points in the plane the oracle returns
`[[0,0,1],[0,1,2],[0,2,3]]` — each 1-simplex padded with a spurious leading `0`,
in an array of the wrong width. For a single point in the plane it returns
`[[0,0,0]]`. The correct answers are the 1-dimensional triangulation
`[[0,1],[1,2],[2,3]]` and the 0-dimensional `[[0]]`, in the configuration's own
affine hull.

**Phase 1** fixes this structurally, via the exact affine-hull reduction: a
rank-deficient configuration is reduced to its own affine hull rather than
producing garbage.

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

## Not a bug: timeouts and crashes

A result may also be recorded as `{"timeout": n}` or `{"crash": code}` rather
than `{"ok": ...}` or `{"error": ...}`. These are not assertions about the Rust
implementation — they record that the oracle could not produce an answer within
the budget, or exited the interpreter outright. The Rust port is free to
succeed where the oracle timed out, and is expected to; `{"crash": ...}` entries
should become ordinary exceptions.

Regenerate with a larger budget via:

```bash
python scripts/gen_fixtures.py --timeout 600 --only <name>
```
