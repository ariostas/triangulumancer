"""The ``Triangulation`` class and the placing/fine methods (phase 2 of the port).

These cover behaviour that is new in, or pinned down by, the Rust port:
canonical simplex order, ``is_fine``, validation in the constructor, and when
the configuration is locked.
"""

from __future__ import annotations

import numpy as np
import pytest

from triangulumancer import PointConfiguration, Triangulation, VectorConfiguration


def interior_point_triangle() -> PointConfiguration:
    return PointConfiguration([[0, 0], [3, 0], [0, 3], [1, 1]])


def test_placing_and_fine() -> None:
    p = interior_point_triangle()
    t = p.placing_triangulation()
    assert t.simplices.tolist() == [[0, 1, 2]]
    assert not t.is_fine
    t = p.fine_triangulation()
    assert t.simplices.tolist() == [[0, 1, 3], [0, 2, 3], [1, 2, 3]]
    assert t.is_fine
    assert t.n_simplices == 3
    assert t.dim == 2


def test_simplices_array() -> None:
    vc = VectorConfiguration([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1]])
    t = vc.fine_triangulation()
    s = t.simplices
    assert s.dtype == np.int64
    assert s.shape == (t.n_simplices, 2)
    rows = s.tolist()
    assert rows == sorted(rows)
    assert all(r == sorted(r) for r in rows)


def test_repr() -> None:
    t = interior_point_triangle().fine_triangulation()
    assert repr(t) == "A triangulation with 3 simplices of a point configuration with 4 points"
    vc = VectorConfiguration([[1, 0], [0, 1], [-1, -1]])
    t = vc.placing_triangulation()
    assert repr(t) == "A triangulation with 3 simplices of a vector configuration with 3 vectors"


def test_locking() -> None:
    p = interior_point_triangle()
    p.fine_triangulation()
    with pytest.raises(RuntimeError):
        p.add_point([1, 2])

    vc = VectorConfiguration([[1, 0], [0, 1]])
    vc.placing_triangulation()
    with pytest.raises(RuntimeError):
        vc.add_vector([1, 1])


def test_a_failed_triangulation_does_not_lock() -> None:
    p = PointConfiguration([[0, 0], [1, 1], [2, 2]])
    with pytest.raises(ValueError, match="not full rank"):
        p.placing_triangulation()
    p.add_point([0, 1])
    assert p.placing_triangulation().n_simplices == 2


def test_direct_construction_validates() -> None:
    p = PointConfiguration([[0, 0], [1, 0], [1, 1], [0, 1]])
    t = Triangulation(p, [[3, 2, 0], [0, 1, 2]])
    assert t.simplices.tolist() == [[0, 1, 2], [0, 2, 3]]

    with pytest.raises(ValueError, match="invalid triangulation"):
        Triangulation(p, [[0, 1, 2]])
    with pytest.raises(ValueError, match="invalid triangulation"):
        Triangulation(p, [[0, 1, 2], [0, 1, 3]])
    with pytest.raises(ValueError):
        Triangulation(p, [[0, 1]])
    with pytest.raises(IndexError):
        Triangulation(p, [[0, 1, 4]])
    with pytest.raises(TypeError):
        Triangulation([[0, 0]], [[0, 1, 2]])

    t = Triangulation(p, [[0, 1, 2]], validate=False)
    assert t.n_simplices == 1
    assert not t.is_fine


def test_direct_construction_locks_only_on_success() -> None:
    p = PointConfiguration([[0, 0], [1, 0], [0, 1]])
    with pytest.raises(ValueError):
        Triangulation(p, [[0, 1, 1]])
    p.add_point([1, 1])
    Triangulation(p, [[0, 1, 2], [1, 2, 3]])
    with pytest.raises(RuntimeError):
        p.add_point([2, 2])


def test_direct_construction_of_vector_triangulations() -> None:
    vc = VectorConfiguration([[1, 0], [0, 1], [-1, 0], [0, -1]])
    t = Triangulation(vc, [[0, 1], [1, 2], [2, 3], [0, 3]])
    assert t.is_fine
    assert t.dim == 2
    with pytest.raises(ValueError):
        Triangulation(vc, [[0, 1], [1, 2], [2, 3]])
