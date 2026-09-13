#!/usr/bin/env python3
"""Generate the differential-testing fixture corpus from the C++ oracle.

The Rust port is validated against ``triangulumancer`` 0.3.0 — the last release
built on CGAL and TOPCOM. This script installs that release into an isolated
virtualenv, runs every public method over a corpus of configurations, and
records the results as compressed JSON for the Rust and Python test suites to
compare against.

Usage::

    python scripts/gen_fixtures.py                 # core tier, the committed one
    python scripts/gen_fixtures.py --tier exhaustive
    python scripts/gen_fixtures.py --list          # show the corpus, install nothing

Two tiers:

``core``
    A curated corpus: every configuration appearing in the test suite, standard
    polytope families, and deliberately degenerate inputs. Small enough to
    commit and to run in CI. This is the default.

``exhaustive``
    Every 0/1 configuration with ``n <= 8`` points in ``d <= 4`` dimensions, up
    to the symmetries of the cube. This is the mitigation for the two riskiest
    parts of the port — the special-position circuit handling of phase 3 and the
    extension-graph enumeration of phase 5 (see PLAN.md section 11). It is large
    and slow, so it is generated on demand and not committed.

Known divergences
-----------------

The oracle has known bugs (PLAN.md section 3), which are deliberately *not*
fixed. Results that those bugs touch are listed per entry under ``untrusted``:
the test suites record them for reference but must not assert against them.
The important one is **B1**: when a configuration has points the placing
triangulation does not use, ``all_connected_triangulations(only_fine=True)``
seeds from a non-fine triangulation and is wrong. This script detects that from
the oracle's own output. Where the *correct* value is independently known it is
written down in ``tests/fixtures/KNOWN_DIVERGENT.md`` as a positive assertion,
which is stronger and safer than asserting mere inequality.

Isolation
---------

Every oracle call runs in its own subprocess. This is not defensiveness for its
own sake: TOPCOM calls ``exit(1)`` on several internal error paths, which would
otherwise take down the whole run, and some enumerations do not terminate in any
useful time. ``signal.alarm`` cannot bound either, because a Python signal
handler only runs between bytecode instructions and these calls sit inside a C
extension holding the GIL. With a subprocess per call, a hang becomes a recorded
timeout and a hard exit becomes a recorded crash — both of which are themselves
findings worth keeping in the corpus.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

ORACLE_VERSION = "0.3.0"
REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"

# Per-method wall-clock budget in the emit subprocess. A configuration whose
# enumeration exceeds this is recorded as a timeout rather than hanging the run.
DEFAULT_TIMEOUT_SECONDS = 120


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

Matrix = list[list[int]]


class Case:
    """One configuration to feed to the oracle."""

    def __init__(
        self,
        name: str,
        kind: str,
        matrix: Matrix,
        tags: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.kind = kind
        self.matrix = matrix
        self.tags = tags

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "matrix": self.matrix,
            "tags": list(self.tags),
        }

    @staticmethod
    def from_json(obj: dict[str, Any]) -> Case:
        return Case(
            name=obj["name"],
            kind=obj["kind"],
            matrix=obj["matrix"],
            tags=tuple(obj["tags"]),
        )


def _cyclic_polytope(dim: int, n: int) -> Matrix:
    """Points on the moment curve, the standard neighbourly example."""
    return [[t**k for k in range(1, dim + 1)] for t in range(1, n + 1)]


def _simplex(dim: int) -> Matrix:
    return [[0] * dim, *[[1 if j == i else 0 for j in range(dim)] for i in range(dim)]]


def _product_of_simplices(a: int, b: int) -> Matrix:
    """Delta_a x Delta_b, whose triangulation counts are tabulated in the
    literature and make a good independent check."""
    return [list(p) + list(q) for p in _simplex(a) for q in _simplex(b)]


def _cube(dim: int) -> Matrix:
    return [list(c) for c in itertools.product([0, 1], repeat=dim)]


def _cross_polytope(dim: int) -> Matrix:
    pts: Matrix = []
    for i in range(dim):
        for s in (1, -1):
            pts.append([s if j == i else 0 for j in range(dim)])
    return pts


def core_corpus() -> list[Case]:
    """The curated, committed corpus."""
    grid_3x3: Matrix = [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [-1, 1],
        [-1, 0],
        [-1, -1],
        [0, -1],
        [1, -1],
    ]
    octagon_vectors: Matrix = [
        [1, 0],
        [1, 1],
        [0, 1],
        [-1, 1],
        [-1, 0],
        [-1, -1],
        [0, -1],
        [1, -1],
    ]
    # From test_vc_triangulation_with_degenerate_flip: fails in TOPCOM 1.1.2.
    degenerate_flip: Matrix = [
        [3, 1, -2, -1],
        [0, 0, 0, 1],
        [-1, -3, 1, 1],
        [0, 2, 1, -1],
        [2, 0, -1, 0],
        [1, -1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, -2, 1, 1],
        [3, 1, -1, -1],
        [4, 3, -1, -2],
        [1, -1, 0, 1],
        [0, -3, 1, 2],
        [2, 2, 0, -1],
    ]

    cases: list[Case] = [
        # --- straight from the 0.3 test suite ---------------------------------
        Case("simplex_2d", "point", _simplex(2), ("regression",)),
        Case("grid_3x3", "point", grid_3x3, ("regression", "interior")),
        Case("unit_square", "point", [[0, 0], [1, 0], [1, 1], [0, 1]], ("regression",)),
        Case("simplex_4d", "point", _simplex(4), ("regression",)),
        Case(
            "square_plus_axes",
            "point",
            [
                [1, 1],
                [-1, 1],
                [1, -1],
                [-1, -1],
                [0, 0],
                [0, 1],
                [0, -1],
                [1, 0],
                [-1, 0],
            ],
            ("regression", "interior"),
        ),
        Case("simplicial_cone_3d", "vector", _simplex(3)[1:], ("regression",)),
        Case("octagon_vectors", "vector", octagon_vectors, ("regression",)),
        Case(
            "vc_square_4d",
            "vector",
            [[0, 0, 1], [1, 0, 3], [1, 1, 1], [0, 1, 2]],
            ("regression",),
        ),
        Case(
            "vc_degenerate_flip",
            "vector",
            degenerate_flip,
            ("regression", "degenerate", "slow"),
        ),
        # --- standard families ------------------------------------------------
        Case("cyclic_2_5", "point", _cyclic_polytope(2, 5), ("family",)),
        Case("cyclic_3_6", "point", _cyclic_polytope(3, 6), ("family",)),
        Case("cyclic_3_7", "point", _cyclic_polytope(3, 7), ("family", "slow")),
        Case("cyclic_4_7", "point", _cyclic_polytope(4, 7), ("family", "slow")),
        Case("prod_simplices_1_1", "point", _product_of_simplices(1, 1), ("family",)),
        Case("prod_simplices_2_1", "point", _product_of_simplices(2, 1), ("family",)),
        Case(
            "prod_simplices_2_2",
            "point",
            _product_of_simplices(2, 2),
            ("family", "slow"),
        ),
        Case("cube_2d", "point", _cube(2), ("family",)),
        Case("cube_3d", "point", _cube(3), ("family",)),
        Case("cross_2d", "point", _cross_polytope(2), ("family",)),
        Case("cross_3d", "point", _cross_polytope(3), ("family",)),
        # --- degenerate and adversarial ---------------------------------------
        Case(
            "duplicate_points",
            "point",
            [[0, 0], [1, 0], [0, 1], [1, 0]],
            ("degenerate", "duplicates"),
        ),
        Case(
            "cospherical_square",
            "point",
            [[1, 0], [0, 1], [-1, 0], [0, -1]],
            ("degenerate", "cospherical"),
        ),
        Case(
            "rank_deficient_collinear",
            "point",
            [[0, 0], [1, 1], [2, 2], [3, 3]],
            ("degenerate", "rank_deficient"),
        ),
        Case(
            "rank_deficient_coplanar_3d",
            "point",
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
            ("degenerate", "rank_deficient"),
        ),
        Case(
            "interior_point_triangle",
            "point",
            [[0, 0], [3, 0], [0, 3], [1, 1]],
            ("degenerate", "interior"),
        ),
        Case(
            "many_interior_points",
            "point",
            [[0, 0], [4, 0], [0, 4], [1, 1], [2, 1], [1, 2]],
            ("degenerate", "interior"),
        ),
        # Exercises B3: the oracle squares coordinates in `int`, which overflows
        # at |coord| >= 46341. Recorded for reference; the Rust port must differ.
        Case(
            "large_coords_b3",
            "point",
            [[0, 0], [50000, 0], [0, 50000], [50000, 50000]],
            ("degenerate", "large_coords"),
        ),
        Case(
            "moderate_coords",
            "point",
            [[0, 0], [1000, 0], [0, 1000], [1000, 1000], [500, 500]],
            ("degenerate", "large_coords", "interior"),
        ),
        # --- trivial / boundary shapes ----------------------------------------
        Case("single_point", "point", [[0, 0]], ("trivial",)),
        Case("segment_1d", "point", [[0], [1]], ("trivial",)),
        Case("segment_1d_three", "point", [[0], [1], [2]], ("trivial", "interior")),
    ]
    return cases


def _canonical_01_configs(dim: int, n: int) -> Iterator[Matrix]:
    """0/1 configurations of `n` points in `dim` dimensions, deduplicated up to
    the symmetries of the cube (coordinate permutations and per-axis flips).

    Without the dedup the sweep is dominated by relabellings that exercise
    exactly the same combinatorics.
    """
    vertices = [tuple(c) for c in itertools.product([0, 1], repeat=dim)]
    index = {v: i for i, v in enumerate(vertices)}

    # The hyperoctahedral group acting on vertex indices.
    symmetries: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(dim)):
        for flip in itertools.product([0, 1], repeat=dim):
            mapping = []
            for v in vertices:
                w = tuple(v[perm[i]] ^ flip[i] for i in range(dim))
                mapping.append(index[w])
            symmetries.append(tuple(mapping))

    seen: set[tuple[int, ...]] = set()
    for subset in itertools.combinations(range(len(vertices)), n):
        if subset in seen:
            continue
        orbit = {tuple(sorted(sym[i] for i in subset)) for sym in symmetries}
        seen |= orbit
        yield [list(vertices[i]) for i in subset]


def exhaustive_corpus() -> list[Case]:
    """Every 0/1 configuration with n <= 8 points in d <= 4 dimensions."""
    cases: list[Case] = []
    for dim in range(1, 5):
        for n in range(dim + 1, 9):
            if n > 2**dim:
                continue
            for i, matrix in enumerate(_canonical_01_configs(dim, n)):
                cases.append(
                    Case(f"01_d{dim}_n{n}_{i:04d}", "point", matrix, ("exhaustive",))
                )
    return cases


CORPORA = {"core": core_corpus, "exhaustive": exhaustive_corpus}


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


# Deterministic per-point parameter vectors for the two weighted entry points.
# `ones` should reproduce the Delaunay triangulation, which is a useful internal
# consistency check on the fixtures themselves.
def _params_for(n: int) -> dict[str, list[float]]:
    return {
        "ramp": [float(i + 1) for i in range(n)],
        "ones": [1.0] * n,
        "alt": [float((i % 3) - 1) for i in range(n)],
    }


COMMON_METHODS = (
    "placing_triangulation",
    "fine_triangulation",
    "all_triangulations",
    "all_triangulations_fine",
    "all_connected_triangulations",
    "all_connected_triangulations_fine",
    "flips_of_placing",
    "neighbors_of_placing",
)

POINT_ONLY_METHODS = (
    "delaunay_triangulation",
    *[
        f"triangulate_with_{k}__{p}"
        for k in ("weights", "heights")
        for p in ("ramp", "ones", "alt")
    ],
)


def methods_for(case: Case) -> tuple[str, ...]:
    if case.kind == "point":
        return COMMON_METHODS + POINT_ONLY_METHODS
    return COMMON_METHODS


# ---------------------------------------------------------------------------
# Emit mode: one method, one configuration, inside the oracle virtualenv
# ---------------------------------------------------------------------------


def _canonical_simplices(simplices: Any) -> list[list[int]]:
    """Sort within each simplex and then between simplices.

    The oracle's output order is an artifact of TOPCOM's hash tables and is not
    reproducible; the Rust port pins its own canonical order (PLAN.md section
    5.6). Comparing canonical forms is what makes the fixtures meaningful.
    """
    return sorted(sorted(int(v) for v in simp) for simp in simplices)


def _canonical_triangulations(triangs: Any) -> list[list[list[int]]]:
    return sorted(_canonical_simplices(t.simplices) for t in triangs)


def run_method(module: Any, case: Case, method: str) -> Any:
    """Run one oracle method. Raises on oracle failure; the caller records it."""
    cls = (
        module.PointConfiguration
        if case.kind == "point"
        else module.VectorConfiguration
    )

    def fresh() -> Any:
        # The oracle locks a configuration once any triangulation is built, so
        # every method gets a fresh instance.
        return cls(case.matrix)

    if method in (
        "placing_triangulation",
        "fine_triangulation",
        "delaunay_triangulation",
    ):
        return _canonical_simplices(getattr(fresh(), method)().simplices)

    if method == "all_triangulations":
        return _canonical_triangulations(fresh().all_triangulations())
    if method == "all_triangulations_fine":
        return _canonical_triangulations(fresh().all_triangulations(True))
    if method == "all_connected_triangulations":
        return _canonical_triangulations(fresh().all_connected_triangulations())
    if method == "all_connected_triangulations_fine":
        return _canonical_triangulations(fresh().all_connected_triangulations(True))

    if method == "flips_of_placing":
        flips = fresh().placing_triangulation().flips()
        return sorted(
            (
                {
                    "pre": _canonical_simplices(f.pre.simplices),
                    "post": _canonical_simplices(f.post.simplices),
                }
                for f in flips
            ),
            key=lambda d: (d["pre"], d["post"]),
        )
    if method == "neighbors_of_placing":
        return _canonical_triangulations(fresh().placing_triangulation().neighbors())

    if method.startswith("triangulate_with_"):
        kind, label = method[len("triangulate_with_") :].split("__")
        params = _params_for(len(case.matrix))[label]
        call = getattr(fresh(), f"triangulate_with_{kind}")
        return _canonical_simplices(call(params).simplices)

    msg = f"unknown method {method!r}"
    raise ValueError(msg)


def emit(case_path: Path, method: str, out_path: Path) -> int:
    # Imported lazily: this is only reachable inside the oracle virtualenv, and
    # the driver half of this script must run without the oracle installed.
    import triangulumancer as module  # noqa: PLC0415

    case = Case.from_json(json.loads(case_path.read_text()))
    try:
        result: dict[str, Any] = {"ok": run_method(module, case, method)}
    except Exception as exc:  # noqa: BLE001 - recording oracle failures is the point
        result = {"error": f"{type(exc).__name__}: {exc}"}
    out_path.write_text(json.dumps(result))
    return 0


# ---------------------------------------------------------------------------
# Driver mode
# ---------------------------------------------------------------------------


def _make_oracle_venv(venv: Path) -> Path:
    """Create a virtualenv holding the pinned oracle release."""
    print(
        f"creating oracle venv with triangulumancer=={ORACLE_VERSION}", file=sys.stderr
    )
    subprocess.run(["uv", "venv", "--quiet", str(venv)], check=True)
    python = venv / "bin" / "python"
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--quiet",
            "--python",
            str(python),
            f"triangulumancer=={ORACLE_VERSION}",
        ],
        check=True,
    )
    return python


def _run_one(
    python: Path, tmpdir: Path, case: Case, method: str, timeout: int
) -> dict[str, Any]:
    """Run a single oracle call in its own process.

    A subprocess is the only reliable way to bound these calls. `signal.alarm`
    cannot interrupt a C extension that holds the GIL, and TOPCOM calls `exit(1)`
    on several internal error paths — both of which would otherwise take down
    the whole run. Here a hang is a timeout and a hard exit is a crash record.
    """
    case_path = tmpdir / "case.json"
    out_path = tmpdir / "result.json"
    case_path.write_text(json.dumps(case.to_json()))
    out_path.unlink(missing_ok=True)

    cmd = [
        str(python),
        str(Path(__file__).resolve()),
        "--emit",
        "--case-file",
        str(case_path),
        "--method",
        method,
        "--out-file",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"timeout": timeout}

    if out_path.exists():
        return json.loads(out_path.read_text())  # type: ignore[no-any-return]
    return {
        "crash": proc.returncode,
        "stderr": proc.stderr.decode(errors="replace")[-500:],
    }


def _untrusted(case: Case, results: dict[str, dict[str, Any]]) -> list[str]:
    """Which results the test suites must *not* assert against.

    Two different situations end up here, and it is worth being clear that this
    is deliberately weaker than "the Rust answer must differ":

    * The oracle is definitely wrong (B1, B3, B4). Asserting inequality would be
      tempting, but a broken computation can still land on the right answer by
      luck, and a test that fails for a *correct* implementation is worse than
      no test. Where the correct value is independently known it is recorded in
      tests/fixtures/KNOWN_DIVERGENT.md instead, as a positive assertion.
    * The oracle's answer is one of several defensible ones (B5: which of a set
      of duplicate points is reported as the vertex). Nothing is wrong here;
      there is simply nothing to assert.

    Computed from the oracle's own output wherever possible, so the marking
    stays correct if the corpus changes. See PLAN.md section 3.
    """
    untrusted: set[str] = set()

    # B1: only_fine in the flip graph is `forbid_vertex_removal`, not a filter,
    # so it needs a fine seed. The oracle always seeds with the placing
    # triangulation; when that does not use every point, the walk is wrong.
    placing = results.get("placing_triangulation", {}).get("ok")
    if placing is not None:
        used = {v for simp in placing for v in simp}
        if len(used) != len(case.matrix):
            untrusted.add("all_connected_triangulations_fine")

    # B3: the height->weight conversion squares coordinates in `int`, which
    # overflows for |coord| >= 46341. Weights are unaffected.
    if "large_coords" in case.tags:
        untrusted.update(k for k in results if k.startswith("triangulate_with_heights"))

    # B4: rank-deficient input hits `std::map::operator[]` on null vertex
    # handles, so the simplices array is padded with index 0 and has the wrong
    # shape. B5: with duplicate points the coordinate scan picks an arbitrary
    # one of them.
    if {"rank_deficient", "duplicates"} & set(case.tags):
        untrusted.update(
            k for k in results if k.startswith(("triangulate_with_", "delaunay"))
        )

    return sorted(untrusted)


def drive(tier: str, timeout: int, only: str | None) -> int:
    cases = CORPORA[tier]()
    if only is not None:
        cases = [c for c in cases if only in c.name]
    total_calls = sum(len(methods_for(c)) for c in cases)
    print(
        f"corpus '{tier}': {len(cases)} configurations, {total_calls} oracle calls",
        file=sys.stderr,
    )

    entries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        python = _make_oracle_venv(tmpdir / "oracle-venv")

        for i, case in enumerate(cases, 1):
            results: dict[str, dict[str, Any]] = {}
            notes: list[str] = []
            for method in methods_for(case):
                res = _run_one(python, tmpdir, case, method, timeout)
                results[method] = res
                if "timeout" in res:
                    notes.append(f"{method}: timeout")
                elif "crash" in res:
                    notes.append(f"{method}: crash({res['crash']})")
                elif "error" in res:
                    notes.append(f"{method}: raised")
            entry = case.to_json()
            entry["results"] = results
            entry["untrusted"] = _untrusted(case, results)
            entry["params"] = (
                _params_for(len(case.matrix)) if case.kind == "point" else {}
            )
            entries.append(entry)

            suffix = ("  [" + "; ".join(notes) + "]") if notes else ""
            print(
                f"[{i}/{len(cases)}] {case.name}{suffix}", file=sys.stderr, flush=True
            )

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIXTURE_DIR / f"{tier}.json.gz"

    # With --only, merge into the existing corpus instead of replacing it, so a
    # single slow configuration can be re-run with a larger budget without
    # regenerating everything.
    if only is not None and out_path.exists():
        with gzip.open(out_path, "rt", encoding="utf-8") as fh:
            existing = json.load(fh)["entries"]
        regenerated = {e["name"] for e in entries}
        merged = [e for e in existing if e["name"] not in regenerated] + entries
        order = {c.name: i for i, c in enumerate(CORPORA[tier]())}
        entries = sorted(merged, key=lambda e: order.get(e["name"], len(order)))
        print(
            f"merged {len(regenerated)} entries into the existing corpus",
            file=sys.stderr,
        )

    payload = {"oracle_version": ORACLE_VERSION, "tier": tier, "entries": entries}
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)

    def count(key: str) -> int:
        return sum(1 for e in entries for r in e["results"].values() if key in r)

    print(
        f"\nwrote {out_path.relative_to(REPO_ROOT)} "
        f"({len(entries)} entries, {out_path.stat().st_size / 1024:.0f} KiB)\n"
        f"  {sum(1 for e in entries if e['untrusted'])} entries with untrusted results\n"
        f"  {count('error')} calls raised, {count('timeout')} timed out, "
        f"{count('crash')} crashed the interpreter",
        file=sys.stderr,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=sorted(CORPORA), default="core")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="per-call wall-clock budget in seconds",
    )
    parser.add_argument(
        "--only", help="restrict to configurations whose name contains this"
    )
    parser.add_argument("--list", action="store_true", help="print the corpus and exit")
    parser.add_argument("--emit", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--method", help=argparse.SUPPRESS)
    parser.add_argument("--out-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.emit:
        if args.case_file is None or args.method is None or args.out_file is None:
            parser.error("--emit requires --case-file, --method and --out-file")
        return emit(args.case_file, args.method, args.out_file)

    if args.list:
        for case in CORPORA[args.tier]():
            tags = ",".join(case.tags) or "-"
            print(f"{case.name:32s} {case.kind:6s} n={len(case.matrix):<4d} {tags}")
        return 0

    return drive(args.tier, args.timeout, args.only)


if __name__ == "__main__":
    raise SystemExit(main())
