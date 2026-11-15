from __future__ import annotations

import subprocess
import sys

import pytest

from triangulumancer import PointConfiguration, Triangulation, VectorConfiguration


def test_cgal_simplex() -> None:
    p = PointConfiguration([[0, 0], [1, 0], [0, 1]])
    t = p.delaunay_triangulation()
    assert len(t.simplices) == 1
    assert set(t.simplices[0]) == {0, 1, 2}

    t = p.triangulate_with_heights([1, 2, 3])
    assert len(t.simplices) == 1
    assert set(t.simplices[0]) == {0, 1, 2}

    t = p.triangulate_with_weights([1, 2, 3])
    assert len(t.simplices) == 1
    assert set(t.simplices[0]) == {0, 1, 2}


def test_cgal_square() -> None:
    p = PointConfiguration(
        [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    )
    t = p.delaunay_triangulation()
    assert len(t.simplices) == 8

    t = p.triangulate_with_heights([1, 1, 0, 1, 0, 1, 0, 1, 0])
    assert len(t.simplices) == 2

    t = p.triangulate_with_heights([1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert len(t.simplices) == 6

    t = p.triangulate_with_weights([1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert len(t.simplices) == 8


def test_locking_pc() -> None:
    p = PointConfiguration([[0, 0], [1, 0], [1, 1]])
    p.delaunay_triangulation()
    with pytest.raises(RuntimeError):
        p.add_point([2, 2])


def test_topcom_simplex() -> None:
    p = PointConfiguration([[0, 0], [1, 0], [0, 1]])
    t = p.placing_triangulation()
    assert len(t.simplices) == 1
    assert set(t.simplices[0]) == {0, 1, 2}

    assert len(t.neighbors()) == 0

    t = p.fine_triangulation()
    assert len(t.simplices) == 1
    assert set(t.simplices[0]) == {0, 1, 2}

    assert len(p.all_connected_triangulations()) == 1
    assert len(p.all_connected_triangulations(True)) == 1

    assert len(p.all_triangulations()) == 1
    assert len(p.all_triangulations(True)) == 1


def test_topcom_square() -> None:
    p = PointConfiguration(
        [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    )
    t = p.placing_triangulation()
    assert len(t.simplices) == 8

    assert len(t.neighbors()) == 6

    t = p.fine_triangulation()
    assert len(t.simplices) == 8

    assert len(p.all_connected_triangulations()) == 387
    assert len(p.all_connected_triangulations(True)) == 64

    assert len(p.all_triangulations()) == 387
    assert len(p.all_triangulations(True)) == 64


def test_topcom_simplicial_cone() -> None:
    vc = VectorConfiguration([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t = vc.placing_triangulation()
    assert len(t.simplices) == 1
    assert set(t.simplices[0]) == {0, 1, 2}

    assert len(vc.all_triangulations()) == 1
    assert len(vc.all_triangulations(True)) == 1


def test_topcom_square_vc() -> None:
    vc = VectorConfiguration(
        [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    )
    t = vc.placing_triangulation()
    assert len(t.simplices) == 6  # simps: 01, 12, 23, 34, 45, 50

    assert len(t.neighbors()) == 6

    assert len(vc.all_triangulations()) == 131
    assert (
        len(vc.all_triangulations(True)) == 1
    )  # simps: 01, 12, 23, 34, 45, 56, 67, 70


def test_direct_construction() -> None:
    p = PointConfiguration([[0, 0], [1, 0], [0, 1]])
    simplices = [[0, 1, 2]]
    t = Triangulation(p, simplices)

    assert t.simplices.tolist() == simplices


def test_flips() -> None:
    p = PointConfiguration([[0, 0], [1, 0], [0, 1]])
    t = p.placing_triangulation()
    assert len(t.flips()) == 0
    assert len(t.bistellar_flips()) == 0

    p = PointConfiguration(
        [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    )
    t = p.placing_triangulation()
    flips = t.flips()
    assert len(flips) == 6

    for flip in flips:
        assert flip.pre.n_simplices == 2
        assert flip.post.n_simplices in (1, 2)

    assert len(t.bistellar_flips()) == 6

    p = PointConfiguration([[0, 0], [1, 0], [1, 1], [0, 1]])
    t = p.placing_triangulation()
    flips = t.flips()
    assert len(flips) == 1
    assert len(flips[0].pre.simplices) == len(flips[0].post.simplices) == 2

    vc = VectorConfiguration([[0, 0, 1], [1, 0, 3], [1, 1, 1], [0, 1, 2]])
    t = vc.placing_triangulation()
    flips = t.flips()
    assert len(flips) == 1
    assert len(flips[0].pre.simplices) == len(flips[0].post.simplices) == 2


def test_vc_triangulation_with_degenerate_flip() -> None:
    # This fails in TOPCOM 1.1.2
    vecs = [
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

    vc = VectorConfiguration(vecs)
    t = vc.fine_triangulation()

    assert len(t.simplices) == 46


def test_large_fine_memory_is_bounded() -> None:
    if sys.platform == "win32":
        pytest.skip("needs the resource module")
    # The fine triangulation and the validity check ask for tens of millions
    # of distinct chirotope signs on a grid this size. With every sign
    # memoised, this run peaked at ~200 MiB and a 60 x 60 grid at 1.4 GiB; the
    # cache is bounded now. Run in a subprocess so that the peak is this
    # computation's alone.
    code = """
import random, resource, sys
from triangulumancer import PointConfiguration, Triangulation
pts = [[x, y] for x in range(40) for y in range(40)]
random.Random(1).shuffle(pts)
p = PointConfiguration(pts)
t = p.fine_triangulation()
assert t.n_simplices == 2 * 39 * 39, t.n_simplices
Triangulation(p, t.simplices)
try:
    # This process's own peak. On Linux ru_maxrss would not do: it carries
    # over the peak of the process that spawned this one (pytest, which may be
    # bigger than the limit), across fork and exec.
    with open("/proc/self/status") as f:
        kib = next(int(line.split()[1]) for line in f if line.startswith("VmHWM:"))
except OSError:
    # ru_maxrss is in bytes on macOS (which has no /proc) and in KiB on Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kib = peak // (1024 if sys.platform == "darwin" else 1)
print(kib // 1024)
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    peak_mib = int(out.stdout.strip())
    assert peak_mib < 100, f"peak RSS {peak_mib} MiB"
