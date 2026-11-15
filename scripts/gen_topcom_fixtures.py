#!/usr/bin/env python3
"""Generate ``tests/fixtures/topcom.json.gz`` from TOPCOM's own command-line tools.

The main corpus (``gen_fixtures.py``) records the 0.3.0 Python package, and so
only reaches TOPCOM through the six methods 0.3 exposed. This script calls
TOPCOM directly, which buys two things the main corpus cannot:

* **the chirotope itself**, from ``points2chiro`` — the differential test for
  the oriented-matroid layer (``PLAN.md`` section 6.1);
* **many more placing and fine triangulations**, from ``points2placingtriang``
  and ``points2finetriang``, over a few hundred seeded random configurations,
  most of them degenerate (small coordinates, duplicates, collinearities,
  vector configurations that are not pointed);
* **circuits**, from ``chiro2circuits`` fed ``points2chiro``'s output (phase 3).
  Not ``points2circuits``: given points, it enumerates circuits by a subset
  graph search that orients and labels them differently, while
  ``chiro2circuits`` runs TOPCOM's ``Circuit(chiro, dependent_set)`` over every
  ``(rank + 1)``-subset, which is what the flip code uses;
* **flip sets**, from ``points2flips``, of the placing and fine triangulations
  and of up to three more triangulations per configuration, fed to it as
  seeds (phase 3);
* **flip-graph counts and sets**, from ``points2ntriangs`` /
  ``points2nfinetriangs`` and ``points2triangs`` (phase 4).

Besides the named and random configurations, every full-rank 0/1
configuration of ``gen_fixtures.py``'s exhaustive tier (``n <= 8`` points in
``d <= 4`` dimensions, up to the symmetries of the cube) is included, under
names starting ``01_``: the exhaustive small-case sweep of ``PLAN.md`` section
7, phase 3, against TOPCOM itself rather than the 0.3 bindings.

Usage::

    python scripts/gen_topcom_fixtures.py --topcom-bin DIR

where ``DIR`` holds the binaries, built from the TOPCOM revision 0.3 was
built against (``ariostas/TOPCOM`` at ``a4333293e9c3cbcf5b8420b89ee30f51078d47fc``,
the one ``TOPCOM.md`` cites). Only GMP is needed; the ``cddlib`` parts of
TOPCOM are not. From a checkout at ``$T``, a ``Makefile`` like this in an empty
directory builds them::

    CXXFLAGS=-std=c++17 -O1 -w -DSTL_SYMMETRIES -DSTL_CONTAINERS -DSTL_FLIPS \\
      -DSTL_CHIROTOPE -DGMPRATIONAL -DPERMLIB_DOMAIN_INT -DUSE_LEXORDER \\
      '-DPACKAGE="TOPCOM"' '-DVERSION="1.1.2"' \\
      -I$(T)/lib-src -I$(T)/wrap-gmp-gmpxx -I$(T)/lib-src-reg
    SRCS=$(wildcard $(T)/lib-src/*.cc)
    OBJS=$(notdir $(SRCS:.cc=.o))
    vpath %.cc $(T)/lib-src $(T)/src
    all: points2chiro points2placingtriang points2finetriang chiro2circuits \\
      points2flips points2ntriangs points2nfinetriangs points2triangs
    %.o: %.cc
    	g++ $(CXXFLAGS) -c $< -o $@
    points2%: points2%.o $(OBJS)
    	g++ $^ -lgmpxx -lgmp -lpthread -o $@
    chiro2%: chiro2%.o $(OBJS)
    	g++ $^ -lgmpxx -lgmp -lpthread -o $@

The enumeration tools link ``ComputeTriangs.cc``, which includes
``RegularityCheck.hh``, which needs cddlib's headers. Regularity checking is
never requested here, so instead of cddlib put a stub ``RegularityCheck.hh``
in ``./stubreg`` and add ``-I./stubreg`` to the front of ``CXXFLAGS``: a class
``RegularityCheck`` in namespace ``topcom`` with a constructor taking
``(const PointConfiguration&, const Chirotope&, const Incidences&, const
SimplicialComplex&)``, static no-op ``init()`` and ``term()``, a ``heights()``
returning ``Vector()``, and an ``is_regular()`` that aborts.

The command-line tools differ from the 0.3 bindings in ways the generator
avoids rather than records: they reject zero vectors and rank below two, and
they silently reduce rank-deficient input instead of rejecting it. Every
configuration here is full rank with no zero column. Simplices are stored in
canonical order (sorted within, then between), as in the main corpus; flips
as ``[removed_side, added_side]`` pairs of TOPCOM's ``FlipRep``, sorted.

The flip-graph enumerations run with a timeout (``--enum-timeout``) and every
TOPCOM process under a ``MEMORY_LIMIT`` address-space cap; a count that did
not finish is recorded as ``null``. The full set of triangulations is
recorded only when there are at most ``MAX_TRIANG_SET`` of them.

Note that ``points2nfinetriangs`` seeds with TOPCOM's fine triangulation,
which is not fine when there are duplicate points (``PLAN.md`` section 13,
phase 2 decisions); its count is then that of a walk from a non-fine seed
with vertex removal forbidden, which is not the number of fine
triangulations. Such entries carry ``"fine_seed_not_fine": true``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import random
import re
import resource
import subprocess
import sys
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any

from gen_fixtures import exhaustive_corpus

FIXTURE = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "topcom.json.gz"
)
TOPCOM_REV = "a4333293e9c3cbcf5b8420b89ee30f51078d47fc"
# Chirotopes are recorded only up to this many bases, to keep the file small.
MAX_CHIRO_BASES = 800
# Circuits are recorded only up to this many (rank + 1)-subsets.
MAX_CIRCUIT_SETS = 3000
# Whole flip-graph components are recorded only up to this many triangulations.
MAX_TRIANG_SET = 40
# Flips of this many triangulations besides placing and fine, from the component.
EXTRA_FLIP_SEEDS = 3
# Address-space cap per TOPCOM process; the flip-graph tools keep every
# triangulation in memory and can otherwise exhaust the machine.
MEMORY_LIMIT = 1 << 30


def _limit_memory() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))


Matrix = list[list[int]]


def rank(rows: Matrix) -> int:
    """Exact rank, by Gaussian elimination over the rationals."""
    m = [[Fraction(x) for x in r] for r in rows]
    r = 0
    cols = len(m[0]) if m else 0
    for c in range(cols):
        pivot = next((i for i in range(r, len(m)) if m[i][c] != 0), None)
        if pivot is None:
            continue
        m[r], m[pivot] = m[pivot], m[r]
        for i in range(len(m)):
            if i != r and m[i][c] != 0:
                f = m[i][c] / m[r][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[r], strict=True)]
        r += 1
    return r


def columns(kind: str, matrix: Matrix) -> Matrix:
    """The homogenised columns TOPCOM sees."""
    return [[*p, 1] for p in matrix] if kind == "point" else [list(p) for p in matrix]


def random_case(rng: random.Random, index: int) -> dict[str, Any] | None:
    kind = rng.choice(["point", "point", "vector"])
    dim = (
        rng.choice([1, 2, 2, 3, 3, 4]) if kind == "point" else rng.choice([2, 3, 3, 4])
    )
    r = dim + 1 if kind == "point" else dim
    n = rng.randint(r, min(r + 9, 13))
    bound = rng.choice([1, 1, 2, 2, 3, 10])
    matrix = [[rng.randint(-bound, bound) for _ in range(dim)] for _ in range(n)]
    if kind == "point" and rng.random() < 0.3:
        matrix = [[abs(x) for x in p] for p in matrix]  # 0/1-ish, more interior points
    if rng.random() < 0.2:  # a duplicate
        matrix.append(list(rng.choice(matrix)))
    cols = columns(kind, matrix)
    if any(all(x == 0 for x in c) for c in cols) or rank(cols) < r or r < 2:
        return None
    return {"name": f"random_{index:04d}", "kind": kind, "matrix": matrix}


def named_cases() -> list[dict[str, Any]]:
    """A few hand-picked configurations, mostly for their chirotopes."""
    grid = [[x, y] for x in (-1, 0, 1) for y in (-1, 0, 1)]
    octagon = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    cyclic = [[t, t * t, t**3] for t in range(7)]
    cube = [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    degenerate = [
        [3, 1, -2, -1], [0, 0, 0, 1], [-1, -3, 1, 1], [0, 2, 1, -1],
        [2, 0, -1, 0], [1, -1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0],
        [-1, 0, 0, 0], [1, 0, 0, 0], [0, -2, 1, 1], [3, 1, -1, -1],
        [4, 3, -1, -2], [1, -1, 0, 1], [0, -3, 1, 2], [2, 2, 0, -1],
    ]  # fmt: skip
    return [
        {"name": "grid_3x3", "kind": "point", "matrix": grid},
        {"name": "octagon_vectors", "kind": "vector", "matrix": octagon},
        {"name": "cyclic_3_7", "kind": "point", "matrix": cyclic},
        {"name": "cube_3d", "kind": "point", "matrix": cube},
        {"name": "vc_degenerate_flip", "kind": "vector", "matrix": degenerate},
        {
            "name": "large_coords",
            "kind": "point",
            "matrix": [[0, 0], [2**40, 3], [5, 2**41], [-(2**39), 7], [1, 1]],
        },
    ]


def zero_one_cases() -> list[dict[str, Any]]:
    """The full-rank configurations of the exhaustive 0/1 tier."""
    cases = []
    for case in exhaustive_corpus():
        if rank(columns("point", case.matrix)) == len(case.matrix[0]) + 1:
            cases.append({"name": case.name, "kind": "point", "matrix": case.matrix})
    return cases


def run_text(binary: Path, text: str) -> str:
    out = subprocess.run(
        [str(binary)],
        input=text,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        preexec_fn=_limit_memory,
    )
    return out.stdout


def run(
    binary: Path, cols: Matrix, seed: list[list[int]] | None = None, timeout: int = 120
) -> str:
    text = "[" + ",".join("[" + ",".join(map(str, c)) + "]" for c in cols) + "]"
    if seed is not None:
        # An empty symmetry group, then the seed triangulation.
        text += (
            "\n[]\n{"
            + ",".join("{" + ",".join(map(str, s)) + "}" for s in seed)
            + "}\n"
        )
    out = subprocess.run(
        [str(binary)],
        input=text,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
        preexec_fn=_limit_memory,
    )
    return out.stdout


def run_or_none(binary: Path, cols: Matrix, timeout: int) -> str | None:
    try:
        return run(binary, cols, timeout=timeout)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        # Out of time, or out of memory under MEMORY_LIMIT.
        return None


def parse_triangulation(text: str) -> list[list[int]]:
    simplices = [
        sorted(int(x) for x in s.split(",") if x)
        for s in re.findall(r"\{([\d,]*)\}", text)
    ]
    return sorted(simplices)


def parse_pairs(text: str) -> list[list[list[int]]]:
    """``[{a,b},{c}]`` pairs, as printed for circuits and flips, sorted."""
    pairs = [
        [[int(x) for x in a.split(",") if x], [int(x) for x in b.split(",") if x]]
        for a, b in re.findall(r"\[\{([\d,]*)\},\{([\d,]*)\}\]", text)
    ]
    return sorted(pairs)


def parse_triangulations(text: str) -> list[list[list[int]]]:
    """``T[i] := {...};`` lines, as printed by ``points2triangs``, sorted."""
    return sorted(
        parse_triangulation(line.partition(":=")[2])
        for line in text.splitlines()
        if ":=" in line
    )


def parse_chirotope(text: str) -> str:
    header, _, body = text.strip().partition(":")
    return header.strip() + ":" + "".join(body.split())


def evaluate(case: dict[str, Any], bin_dir: Path, enum_timeout: int) -> None:
    """Run every tool on one configuration, recording the results in `case`."""
    cols = columns(case["kind"], case["matrix"])
    n, r = len(cols), len(cols[0])
    if comb(n, r) <= MAX_CHIRO_BASES:
        case["chirotope"] = parse_chirotope(run(bin_dir / "points2chiro", cols))
    case["placing"] = parse_triangulation(run(bin_dir / "points2placingtriang", cols))
    case["fine"] = parse_triangulation(run(bin_dir / "points2finetriang", cols))
    case["fine_seed_not_fine"] = len({v for s in case["fine"] for v in s}) < n
    if n == r:
        # A single simplex. The enumeration tools fail on it outright
        # ("allocation of 2048 blocks failed"), and there is nothing to flip.
        case["n_connected"] = case["n_connected_fine"] = None
        case["flips"] = []
        return
    if comb(n, r + 1) <= MAX_CIRCUIT_SETS:
        chiro = run(bin_dir / "points2chiro", cols)
        case["circuits"] = parse_pairs(run_text(bin_dir / "chiro2circuits", chiro))

    count = run_or_none(bin_dir / "points2ntriangs", cols, enum_timeout)
    case["n_connected"] = None if count is None else int(count.split()[-1])
    count = run_or_none(bin_dir / "points2nfinetriangs", cols, enum_timeout)
    case["n_connected_fine"] = None if count is None else int(count.split()[-1])

    seeds = [case["placing"], case["fine"]]
    if case["n_connected"] is not None and case["n_connected"] <= MAX_TRIANG_SET:
        component = parse_triangulations(run(bin_dir / "points2triangs", cols))
        case["connected"] = component
        others = [t for t in component if t not in seeds]
        step = max(1, len(others) // EXTRA_FLIP_SEEDS)
        seeds += others[::step][:EXTRA_FLIP_SEEDS]
    case["flips"] = [
        {
            "triangulation": t,
            "flips": parse_pairs(run(bin_dir / "points2flips", cols, seed=t)),
        }
        for t in seeds
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--topcom-bin", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300, help="random configurations")
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument(
        "--enum-timeout",
        type=int,
        default=20,
        help="seconds per flip-graph enumeration",
    )
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cases = named_cases()
    index = 0
    while len(cases) < args.count + len(named_cases()):
        case = random_case(rng, index)
        index += 1
        if case is not None:
            cases.append(case)

    cases += zero_one_cases()

    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(lambda c: evaluate(c, args.topcom_bin, args.enum_timeout), cases))

    payload = {"topcom_rev": TOPCOM_REV, "seed": args.seed, "entries": cases}
    with gzip.open(FIXTURE, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"), sort_keys=True)
    n_chiro = sum("chirotope" in c for c in cases)
    n_circ = sum("circuits" in c for c in cases)
    n_count = sum(c["n_connected"] is not None for c in cases)
    print(
        f"{len(cases)} configurations, {n_chiro} chirotopes, {n_circ} circuit sets, "
        f"{n_count} flip-graph counts -> {FIXTURE}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
