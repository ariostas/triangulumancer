"""The flip-graph enumeration (phase 4 of the port).

Covers what is new in, or fixed by, the Rust port: the streaming iterator,
canonical order, ``Triangulation`` equality and hashing, and bug B1 (the
``only_fine`` enumeration seeded with a non-fine triangulation).
"""

from __future__ import annotations

import pytest

from triangulumancer import PointConfiguration, VectorConfiguration


def grid() -> PointConfiguration:
    return PointConfiguration([[x, y] for x in range(3) for y in range(3)])


def test_list_is_canonical() -> None:
    ts = grid().all_connected_triangulations()
    assert len(ts) == 387
    keys = [t.simplices.tolist() for t in ts]
    assert keys == sorted(keys)


def test_iterator_streams_the_same_set() -> None:
    p = grid()
    it = p.iter_all_connected_triangulations()
    first = next(it)
    assert first == p.placing_triangulation()
    rest = list(it)
    assert len(rest) == 386
    assert set(rest) | {first} == set(p.all_connected_triangulations())
    assert len(list(p.iter_all_connected_triangulations(only_fine=True))) == 64


def test_iterator_locks_the_configuration() -> None:
    p = grid()
    p.iter_all_connected_triangulations()
    with pytest.raises(RuntimeError):
        p.add_point([5, 5])


def test_equality_and_hashing() -> None:
    p = grid()
    a = p.fine_triangulation()
    b = p.fine_triangulation()
    assert a == b
    assert hash(a) == hash(b)
    assert a != a.neighbors()[0]
    assert a != grid().fine_triangulation()  # a different configuration object
    assert a != "a triangulation"
    assert len({a, b}) == 1


def test_b1_square_plus_axes() -> None:
    # The 3x3 grid with the origin fifth: 0.3 did not terminate here.
    p = PointConfiguration(
        [[1, 1], [-1, 1], [1, -1], [-1, -1], [0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]
    )
    fine = p.all_connected_triangulations(only_fine=True)
    assert len(fine) == 64
    assert all(t.is_fine for t in fine)


def test_b1_interior_points() -> None:
    p = PointConfiguration([[0, 0], [3, 0], [0, 3], [1, 1]])
    assert len(p.all_connected_triangulations()) == 2
    (t,) = p.all_connected_triangulations(only_fine=True)
    assert t.is_fine
    vc = VectorConfiguration(
        [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    )
    assert len(vc.all_connected_triangulations()) == 131
    assert len(vc.all_connected_triangulations(only_fine=True)) == 1


def test_no_fine_triangulation_with_duplicates() -> None:
    p = PointConfiguration([[0, 0], [1, 0], [0, 1], [1, 0]])
    assert p.all_connected_triangulations(only_fine=True) == []
    assert len(p.all_connected_triangulations()) == 2


def test_rank_deficient_is_an_error() -> None:
    p = PointConfiguration([[0, 0], [1, 1], [2, 2]])
    with pytest.raises(ValueError, match="not full rank"):
        p.all_connected_triangulations()
    with pytest.raises(ValueError, match="not full rank"):
        p.iter_all_connected_triangulations()
    p.add_point([3, 3])  # still unlocked
