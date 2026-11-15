"""The extension-graph enumeration (phase 5 of the port).

Covers what is new in the Rust port: the streaming iterator, canonical order,
agreement with the flip graph, and the ``only_fine`` semantics shared with it.
"""

from __future__ import annotations

import itertools

import pytest

from triangulumancer import PointConfiguration, VectorConfiguration


def grid() -> PointConfiguration:
    return PointConfiguration([[x, y] for x in range(3) for y in range(3)])


def test_list_is_canonical() -> None:
    ts = grid().all_triangulations()
    assert len(ts) == 387
    keys = [t.simplices.tolist() for t in ts]
    assert keys == sorted(keys)


def test_same_as_the_flip_graph_in_the_plane() -> None:
    p = grid()
    assert p.all_triangulations() == p.all_connected_triangulations()
    assert p.all_triangulations(only_fine=True) == p.all_connected_triangulations(
        only_fine=True
    )


def test_iterator_streams_the_same_set() -> None:
    p = grid()
    found = list(p.iter_all_triangulations())
    assert len(found) == 387
    assert set(found) == set(p.all_triangulations())
    fine = list(p.iter_all_triangulations(only_fine=True))
    assert len(fine) == 64
    assert all(t.is_fine for t in fine)


def test_iterator_is_lazy() -> None:
    # 16 vectors of rank 4 with billions of triangulations: 0.3 ran out of
    # memory collecting them, while the first few stream out at once.
    vc = VectorConfiguration(
        [
            [3, 1, -2, -1], [0, 0, 0, 1], [-1, -3, 1, 1], [0, 2, 1, -1],
            [2, 0, -1, 0], [1, -1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0],
            [-1, 0, 0, 0], [1, 0, 0, 0], [0, -2, 1, 1], [3, 1, -1, -1],
            [4, 3, -1, -2], [1, -1, 0, 1], [0, -3, 1, 2], [2, 2, 0, -1],
        ]
    )  # fmt: skip
    first = list(itertools.islice(vc.iter_all_triangulations(), 100))
    assert len(set(first)) == 100


def test_iterator_locks_the_configuration() -> None:
    p = grid()
    p.iter_all_triangulations()
    with pytest.raises(RuntimeError):
        p.add_point([5, 5])


def test_only_fine() -> None:
    p = PointConfiguration([[0, 0], [3, 0], [0, 3], [1, 1]])
    assert len(p.all_triangulations()) == 2
    (t,) = p.all_triangulations(only_fine=True)
    assert t.is_fine
    vc = VectorConfiguration(
        [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    )
    assert len(vc.all_triangulations()) == 131
    assert len(vc.all_triangulations(only_fine=True)) == 1


def test_no_fine_triangulation_with_duplicates() -> None:
    p = PointConfiguration([[0, 0], [1, 0], [0, 1], [1, 0]])
    assert p.all_triangulations(only_fine=True) == []
    assert list(p.iter_all_triangulations(only_fine=True)) == []
    assert len(p.all_triangulations()) == 2


def test_rank_deficient_is_an_error() -> None:
    p = PointConfiguration([[0, 0], [1, 1], [2, 2]])
    with pytest.raises(ValueError, match="not full rank"):
        p.all_triangulations()
    with pytest.raises(ValueError, match="not full rank"):
        p.iter_all_triangulations()
    p.add_point([3, 3])  # still unlocked
