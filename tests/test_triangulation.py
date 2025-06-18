from triangulumancer import PointConfiguration


def test_delaunay_simplex():
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


def test_delaunay_square():
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
