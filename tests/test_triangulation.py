from triangulumancer import PointConfiguration, VectorConfiguration, Triangulation

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

    t = p.fine_triangulation()
    assert len(t.simplices()) == 1
    assert set(t.simplices()[0]) == {0, 1, 2}

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

    t = p.fine_triangulation()
    assert len(t.simplices()) == 8

    assert len(p.all_connected_triangulations()) == 387
    assert len(p.all_connected_triangulations(True)) == 64

    assert len(p.all_triangulations()) == 387
    assert len(p.all_triangulations(True)) == 64


def test_topcom_simplicial_cone():
    vc = VectorConfiguration([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t = vc.placing_triangulation()
    assert len(t.simplices()) == 1
    assert set(t.simplices()[0]) == {0, 1, 2}

    assert len(vc.all_triangulations()) == 1
    assert len(vc.all_triangulations(True)) == 1


def test_topcom_square_vc():
    vc = VectorConfiguration(
        [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    )
    t = vc.placing_triangulation()
    assert len(t.simplices()) == 6  # simps: 01, 12, 23, 34, 45, 50

    assert len(t.neighbors()) == 6

    assert len(vc.all_triangulations()) == 131
    assert (
        len(vc.all_triangulations(True)) == 1
    )  # simps: 01, 12, 23, 34, 45, 56, 67, 70


def test_direct_construction():
    p = PointConfiguration([[0, 0], [1, 0], [0, 1]])
    simplices = [[0, 1, 2]]
    t = Triangulation(p, simplices)

    assert t.simplices().tolist() == simplices
