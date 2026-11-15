"""Bad and extreme input: every case raises a specific Python exception or
gives a sensible answer, and none panics (a ``PanicException`` is a
``BaseException``, so ``pytest.raises(Exception)`` would not catch it)."""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
import pytest

from triangulumancer import PointConfiguration, Triangulation, VectorConfiguration

BIG = 2**63 - 1


def square() -> PointConfiguration:
    return PointConfiguration([[0, 0], [1, 0], [1, 1], [0, 1]])


def corners() -> PointConfiguration:
    return PointConfiguration(
        [[-BIG - 1, -BIG - 1], [BIG, -BIG - 1], [BIG, BIG], [-BIG - 1, BIG], [0, 0]]
    )


RAISES: list[tuple[str, Callable[[], object], type[Exception], str | None]] = [
    ("nan", lambda: square().triangulate_with_weights([0, 0, math.nan, 0]), ValueError, "finite"),
    ("inf", lambda: square().triangulate_with_heights([0, 0, math.inf, 0]), ValueError, "finite"),
    ("subnormal", lambda: square().triangulate_with_heights([0, 0, 5e-324, 0]), ValueError, "subnormal"),
    ("5000 digits", lambda: square().triangulate_with_heights([0, 1, 10**5000, 0]), ValueError, "orders of magnitude"),
    ("string", lambda: square().triangulate_with_heights("abcd"), TypeError, None),
    ("none", lambda: square().triangulate_with_heights(None), TypeError, None),
    ("2d array", lambda: square().triangulate_with_heights(np.zeros((4, 1))), TypeError, None),
    ("complex", lambda: square().triangulate_with_heights([0, 1j, 0, 0]), TypeError, None),
    ("empty weights", lambda: PointConfiguration().triangulate_with_weights([]), ValueError, "at least one"),
    ("empty placing", lambda: PointConfiguration().placing_triangulation(), ValueError, "at least one"),
    ("empty fine", lambda: PointConfiguration().fine_triangulation(), ValueError, "at least one"),
    ("empty connected", lambda: PointConfiguration().all_connected_triangulations(), ValueError, "at least one"),
    ("empty iter", lambda: PointConfiguration().iter_all_connected_triangulations(), ValueError, "at least one"),
    ("empty all", lambda: PointConfiguration().all_triangulations(), ValueError, "at least one"),
    ("empty vectors", lambda: VectorConfiguration().placing_triangulation(), ValueError, "at least one"),
    ("one point placing", lambda: PointConfiguration([[5, 5]]).placing_triangulation(), ValueError, "full rank"),
    ("collinear flips", lambda: PointConfiguration([[0, 0], [1, 1], [2, 2]]).delaunay_triangulation().flips(), ValueError, "full rank"),
    ("coords too big", lambda: PointConfiguration([[2**64, 0], [1, 0], [0, 1]]), OverflowError, None),
    ("nan coords", lambda: PointConfiguration([[math.nan, 0], [1, 0], [0, 1]]), ValueError, None),
    ("negative label", lambda: Triangulation(square(), [[0, 1, -1]]), ValueError, "negative"),
    ("label too big", lambda: Triangulation(square(), [[0, 1, 2**40]]), IndexError, "out of range"),
    ("3d simplices", lambda: Triangulation(square(), np.zeros((1, 3, 1), dtype=int)), ValueError, "two-dimensional"),
    ("no simplices", lambda: Triangulation(square(), []), ValueError, "no simplices"),
    ("listed twice", lambda: Triangulation(square(), [[0, 1, 2], [0, 1, 2]], validate=False), ValueError, "twice"),
    ("repeated label", lambda: Triangulation(square(), [[0, 0, 1]], validate=False), ValueError, "repeated"),
    ("config type", lambda: Triangulation([[0, 0]], [[0]]), TypeError, "Configuration"),
]  # fmt: skip


def test_raises() -> None:
    # One test rather than `parametrize`, which mypy sees as untyped; on a
    # failure `--showlocals` names the case.
    for name, call, exc, match in RAISES:
        assert name
        with pytest.raises(exc, match=match):
            call()


def test_exact_and_odd_scalars() -> None:
    flat = [[0, 1, 3], [1, 2, 3]]
    other = [[0, 1, 2], [0, 2, 3]]
    t = square().triangulate_with_heights
    assert t([0, 0, 10**400, 0]).simplices.tolist() == flat
    assert t([10**5000] * 4).simplices.tolist() == flat
    # One non-zero height is the same as height one, however large.
    assert t([0, 0, 10**5000, 0]).simplices.tolist() == flat
    assert t([-(10**5000)] * 4).simplices.tolist() == flat
    assert t([Fraction(1, 3), 0, Fraction(1, 7), 0]).simplices.tolist() == flat
    assert t([True, False, True, False]).simplices.tolist() == flat
    assert t(np.array([0, 1, 0, 0], dtype=np.float32)).simplices.tolist() == other
    assert (
        t(np.array([0, 2**64 - 1, 0, 0], dtype=np.uint64)).simplices.tolist() == other
    )
    assert (
        t(np.array([0, Fraction(1, 2), 0, 0], dtype=object)).simplices.tolist() == other
    )


def test_degenerate_and_extreme_configurations() -> None:
    assert PointConfiguration([[]]).delaunay_triangulation().simplices.tolist() == [[0]]
    t = PointConfiguration(np.zeros((3, 0), dtype=int)).placing_triangulation()
    assert t.simplices.tolist() == [[0]]
    assert t.hidden_points == [1, 2]
    t = PointConfiguration([[5, 5]] * 3).delaunay_triangulation()
    assert t.simplices.tolist() == [[0]]
    assert t.hidden_points == [1, 2]
    t = corners().delaunay_triangulation()
    assert t.simplices.tolist() == [[0, 1, 4], [0, 3, 4], [1, 2, 4], [2, 3, 4]]
    t = corners().placing_triangulation()
    assert t.simplices.tolist() == [[0, 1, 2], [0, 2, 3]]
    assert len(t.flips()) == 2
    v = VectorConfiguration(
        [[-BIG - 1, -BIG - 1], [BIG, -BIG - 1], [BIG, BIG], [-BIG - 1, BIG]]
    )
    assert sorted(t.n_simplices for t in v.all_connected_triangulations()) == [3, 4]
    # Exact coordinates near the i64 limit can overflow the affine-hull
    # reduction; that is a RuntimeError, not a wrong answer.
    with pytest.raises(RuntimeError, match="overflow"):
        PointConfiguration(
            [[-BIG - 1, -BIG - 1], [BIG, BIG], [0, 0]]
        ).delaunay_triangulation()


def test_failure_does_not_lock() -> None:
    p = PointConfiguration([[0, 0], [1, 0], [0, 1]])
    with pytest.raises(ValueError, match="finite"):
        p.triangulate_with_weights([math.nan, 0, 0])
    p.add_point([1, 1])
    assert p.n_points == 4
