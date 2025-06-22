from triangulumancer import PointConfiguration

import pytest


def test_cgal_simplex():
    p = PointConfiguration([[0, 0], [1, 0], [0, 1]])
    t = p.delaunay_triangulation()
    assert len(t.simplices()) == 1
    assert set(t.simplices()[0]) == {0, 1, 2}

    t = p.triangulate_with_heights([1, 2, 3])
    assert len(t.simplices()) == 1
    assert set(t.simplices()[0]) == {0, 1, 2}

    t = p.triangulate_with_weights([1, 2, 3])
    assert len(t.simplices()) == 1
    assert set(t.simplices()[0]) == {0, 1, 2}


def test_cgal_square():
    p = PointConfiguration(
        [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    )
    t = p.delaunay_triangulation()
    assert len(t.simplices()) == 8

    t = p.triangulate_with_heights([1, 1, 0, 1, 0, 1, 0, 1, 0])
    assert len(t.simplices()) == 2

    t = p.triangulate_with_heights([1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert len(t.simplices()) == 6

    t = p.triangulate_with_weights([1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert len(t.simplices()) == 8


def test_locking_pc():
    p = PointConfiguration([[0, 0], [1, 0], [1, 1]])
    p.delaunay_triangulation()
    with pytest.raises(RuntimeError):
        p.add_point([2, 2])


def test_topcom_simplex():
    p = PointConfiguration([[0, 0], [1, 0], [0, 1]])
    t = p.placing_triangulation()
    assert len(t.simplices()) == 1
    assert set(t.simplices()[0]) == {0, 1, 2}

    assert len(t.neighbors()) == 0

    assert len(p.all_connected_triangulations()) == 1
    assert len(p.all_connected_triangulations(True)) == 1

    assert len(p.all_triangulations()) == 1
    assert len(p.all_triangulations(True)) == 1


def test_topcom_square():
    p = PointConfiguration(
        [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    )
    t = p.placing_triangulation()
    assert len(t.simplices()) == 8

    assert len(t.neighbors()) == 6

    assert len(p.all_connected_triangulations()) == 387
    assert len(p.all_connected_triangulations(True)) == 64

    assert len(p.all_triangulations()) == 387
    assert len(p.all_triangulations(True)) == 64
