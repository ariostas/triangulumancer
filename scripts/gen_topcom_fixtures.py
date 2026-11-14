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
  vector configurations that are not pointed).

Usage::

    python scripts/gen_topcom_fixtures.py --topcom-bin DIR

where ``DIR`` holds the three binaries, built from the TOPCOM revision 0.3 was
built against (``ariostas/TOPCOM`` at ``a4333293e9c3cbcf5b8420b89ee30f51078d47fc``,
the one ``TOPCOM.md`` cites). Only GMP is needed; the ``cddlib`` parts of
TOPCOM are not. From a checkout at ``$T``, a ``Makefile`` like this in an empty
directory builds them::

    CXXFLAGS=-std=c++17 -O1 -w -DSTL_SYMMETRIES -DSTL_CONTAINERS -DSTL_FLIPS \\
      -DSTL_CHIROTOPE -DGMPRATIONAL -DPERMLIB_DOMAIN_INT -DUSE_LEXORDER \\
      '-DPACKAGE="TOPCOM"' '-DVERSION="1.1.2"' \\
      -I$(T)/lib-src -I$(T)/wrap-gmp-gmpxx -I$(T)/lib-src-reg
    SRCS=$(filter-out %SymmetricExtensionGraph.cc %SymmetricExtensionGraphMaster.cc \\
      %SymmetricFlipGraph.cc %ComputeTriangs.cc,$(wildcard $(T)/lib-src/*.cc))
    OBJS=$(notdir $(SRCS:.cc=.o))
    vpath %.cc $(T)/lib-src $(T)/src
    all: points2chiro points2placingtriang points2finetriang
    %.o: %.cc
    	g++ $(CXXFLAGS) -c $< -o $@
    points2%: points2%.o $(OBJS)
    	g++ $^ -lgmpxx -lgmp -lpthread -o $@

The command-line tools differ from the 0.3 bindings in ways the generator
avoids rather than records: they reject zero vectors and rank below two, and
they silently reduce rank-deficient input instead of rejecting it. Every
configuration here is full rank with no zero column. Simplices are stored in
canonical order (sorted within, then between), as in the main corpus.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import subprocess
import sys
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "topcom.json.gz"
TOPCOM_REV = "a4333293e9c3cbcf5b8420b89ee30f51078d47fc"
# Chirotopes are recorded only up to this many bases, to keep the file small.
MAX_CHIRO_BASES = 800

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
    dim = rng.choice([1, 2, 2, 3, 3, 4]) if kind == "point" else rng.choice([2, 3, 3, 4])
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


def run(binary: Path, cols: Matrix) -> str:
    text = "[" + ",".join("[" + ",".join(map(str, c)) + "]" for c in cols) + "]"
    out = subprocess.run(
        [str(binary)], input=text, capture_output=True, text=True, check=True, timeout=120
    )
    return out.stdout


def parse_triangulation(text: str) -> list[list[int]]:
    simplices = [
        sorted(int(x) for x in s.split(",") if x) for s in re.findall(r"\{([\d,]*)\}", text)
    ]
    return sorted(simplices)


def parse_chirotope(text: str) -> str:
    header, _, body = text.strip().partition(":")
    return header.strip() + ":" + "".join(body.split())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--topcom-bin", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300, help="random configurations")
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cases = named_cases()
    index = 0
    while len(cases) < args.count + len(named_cases()):
        case = random_case(rng, index)
        index += 1
        if case is not None:
            cases.append(case)

    for case in cases:
        cols = columns(case["kind"], case["matrix"])
        r = len(cols[0])
        if comb(len(cols), r) <= MAX_CHIRO_BASES:
            case["chirotope"] = parse_chirotope(run(args.topcom_bin / "points2chiro", cols))
        case["placing"] = parse_triangulation(
            run(args.topcom_bin / "points2placingtriang", cols)
        )
        case["fine"] = parse_triangulation(run(args.topcom_bin / "points2finetriang", cols))

    payload = {"topcom_rev": TOPCOM_REV, "seed": args.seed, "entries": cases}
    with gzip.open(FIXTURE, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"), sort_keys=True)
    n_chiro = sum("chirotope" in c for c in cases)
    print(f"{len(cases)} configurations, {n_chiro} chirotopes -> {FIXTURE}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
