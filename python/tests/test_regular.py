"""Regular and Delaunay triangulations (phase 6 of the port).

Behaviour that is new in, or pinned down by, the Rust port: exact integer and
``Fraction`` input, ``hidden_points``, rank-deficient input, duplicates, and
errors instead of garbage.
"""

from __future__ import annotations

import math
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

from triangulumancer import PointConfiguration


def square() -> PointConfiguration:
    return PointConfiguration([[0, 0], [1, 0], [1, 1], [0, 1]])


def test_delaunay_tie_breaking_matches_0_3() -> None:
    # Cocircular: the diagonal is chosen as CGAL chose it.
    assert square().delaunay_triangulation().simplices.tolist() == [
        [0, 1, 3],
        [1, 2, 3],
    ]


def test_exact_heights() -> None:
    p = PointConfiguration([[0, 0], [3, 0], [0, 3], [1, 1]])
    t = p.triangulate_with_heights([0, 0, 0, 1])
    assert t.simplices.tolist() == [[0, 1, 2]]
    assert t.hidden_points == [3]
    assert not t.is_fine

    # The plane through the corners at heights 0, 1, 0 passes through the
    # interior point at height exactly 1/3, which is not a double. A hair above
    # it the point is hidden, a hair below it is not.
    t = p.triangulate_with_heights([0, 1, 0, Fraction(1, 3) + Fraction(1, 10**30)])
    assert t.hidden_points == [3]
    t = p.triangulate_with_heights([0, 1, 0, Fraction(1, 3) - Fraction(1, 10**30)])
    assert t.hidden_points == []
    assert t.n_simplices == 3
    # Exactly on it is a tie, broken as CGAL breaks it: the lexicographically
    # largest point, (3, 0), counts as raised, so (1, 1) is a vertex.
    t = p.triangulate_with_heights([0, 1, 0, Fraction(1, 3)])
    assert t.hidden_points == []

    # Arbitrarily large integers, and numpy arrays of either kind.
    t = p.triangulate_with_heights([10**40, 10**40, 10**40, 10**40 + 1])
    assert t.hidden_points == [3]
    t = p.triangulate_with_heights(np.array([0, 0, 0, -1], dtype=np.int64))
    assert t.hidden_points == []
    t = p.triangulate_with_weights(np.array([0.0, 0.0, 0.0, 0.5]))
    assert t.n_simplices == 3


def test_floats_are_exact_doubles() -> None:
    # The same answers from floats and from the exact rationals they denote.
    p = PointConfiguration([[0, 0], [3, 0], [0, 3], [1, 1], [2, 0]])
    heights = [0.1, 0.7, -0.3, 0.2, 0.05]
    exact = [Fraction(h) for h in heights]
    a = p.triangulate_with_heights(heights).simplices.tolist()
    p = PointConfiguration([[0, 0], [3, 0], [0, 3], [1, 1], [2, 0]])
    assert p.triangulate_with_heights(exact).simplices.tolist() == a


def test_large_coordinates() -> None:
    # B3: 0.3 squared coordinates in 32 bits.
    s = 100_000
    p = PointConfiguration([[0, 0], [s, 0], [0, s], [s, s]])
    t = p.triangulate_with_weights([0, 0, 0, 2 * s * s + 1])
    assert t.simplices.tolist() == [[0, 1, 3], [0, 2, 3]]


def test_rank_deficient() -> None:
    # B4: triangulated in the affine hull, with simplices of the right size.
    p = PointConfiguration([[0, 0], [1, 1], [2, 2], [3, 3]])
    t = p.delaunay_triangulation()
    assert t.simplices.tolist() == [[0, 1], [1, 2], [2, 3]]
    assert t.simplices.shape == (3, 2)
    t = PointConfiguration([[5, 5]]).delaunay_triangulation()
    assert t.simplices.tolist() == [[0]]


def test_duplicates() -> None:
    # B5: the lowest index wins a tie, and the loser is reported.
    p = PointConfiguration([[0, 0], [1, 0], [0, 1], [1, 0]])
    t = p.delaunay_triangulation()
    assert t.simplices.tolist() == [[0, 1, 2]]
    assert t.hidden_points == [3]
    p = PointConfiguration([[0, 0], [1, 0], [0, 1], [1, 0]])
    t = p.triangulate_with_heights([0, 1, 0, 0])
    assert t.hidden_points == [1]


def test_errors() -> None:
    p = square()
    with pytest.raises(ValueError, match="number of parameters"):
        p.triangulate_with_weights([1, 2, 3])
    with pytest.raises(ValueError, match="not finite"):
        p.triangulate_with_heights([0, math.nan, 0, 0])
    with pytest.raises(ValueError, match="subnormal"):
        p.triangulate_with_heights([0, 5e-324, 0, 0])
    with pytest.raises(TypeError):
        p.triangulate_with_heights([0, "x", 0, 0])
    with pytest.raises(ValueError, match="at least one"):
        PointConfiguration().delaunay_triangulation()
    # A failed triangulation does not lock the configuration.
    p.add_point([2, 2])
    assert p.n_points == 5


def test_locks_on_success() -> None:
    p = square()
    p.triangulate_with_heights([0, 0, 0, 0])
    with pytest.raises(RuntimeError):
        p.add_point([2, 2])


def test_many_points() -> None:
    rng = np.random.default_rng(6)
    pts = rng.integers(-1000, 1000, size=(2000, 2))
    t = PointConfiguration(pts).delaunay_triangulation()
    distinct = len({tuple(r) for r in pts.tolist()})
    assert len(t.hidden_points) == 2000 - distinct
    pts = rng.integers(-5, 6, size=(300, 3))
    t = PointConfiguration(pts).triangulate_with_heights(rng.integers(0, 50, size=300))
    assert t.n_simplices > 0


def test_large_delaunay_memory_is_linear() -> None:
    if sys.platform == "win32":
        pytest.skip("needs the resource module")
    # Simplices with labels of 128 or more must cost O(rank) each, not O(n):
    # with an n-bit bitset per simplex this run peaked at ~300 MiB, and the cost
    # grew quadratically (40 000 points took over 1 GiB). Run in a subprocess so
    # that the peak is this computation's alone.
    code = """
import resource, sys
import numpy as np
from triangulumancer import PointConfiguration
pts = np.random.default_rng(7).integers(-10**6, 10**6, size=(20000, 3))
t = PointConfiguration(pts).delaunay_triangulation()
assert t.n_simplices > 100000, t.n_simplices
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(peak // (1 if sys.platform == "darwin" else 1024) // 1024)
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    peak_mib = int(out.stdout.strip())
    assert peak_mib < 200, f"peak RSS {peak_mib} MiB"
