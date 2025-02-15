from __future__ import annotations

import numpy as np

from triangulumancer import PointConfiguration


def test_dim():
    p = PointConfiguration([[1, 1], [-1, 1], [1, -1], [-1, -1]])
    assert p.dim() == 2

    p = PointConfiguration(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-1, -1, -1, -1],
        ]
    )
    assert p.dim() == 4


def test_points():
    points = np.array(
        [[1, 1], [-1, 1], [1, -1], [-1, -1], [0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]],
        dtype=np.int64,
    )
    sorted_points = sorted(tuple(pt) for pt in points)

    p = PointConfiguration(sorted_points)
    returned_points = p.points()
    sorted_computed_points = sorted(tuple(pt) for pt in returned_points)

    assert len(returned_points) == 9
    assert sorted_computed_points == sorted_points


def test_add_points():
    points = np.array(
        [[1, 1], [-1, 1], [1, -1], [-1, -1], [0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]],
        dtype=np.int64,
    )
    p = PointConfiguration(points)
    p.add_point([2, 2])

    returned_points = p.points()
    assert returned_points[-1].tolist() == [2, 2]
