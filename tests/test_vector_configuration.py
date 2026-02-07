from __future__ import annotations

import numpy as np

from triangulumancer import VectorConfiguration


def test_dim():
    vc = VectorConfiguration([[1, 1], [-1, 1], [1, -1], [-1, -1]])
    assert vc.dim() == 2

    vc = VectorConfiguration(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-1, -1, -1, -1],
        ]
    )
    assert vc.dim() == 4


def test_vectors():
    vectors = np.array(
        [[1, 1], [-1, 1], [1, -1], [-1, -1], [0, 1], [0, -1], [1, 0], [-1, 0]],
        dtype=np.int64,
    )
    sorted_vectors = sorted(tuple(vec) for vec in vectors)

    vc = VectorConfiguration(sorted_vectors)
    returned_vectors = vc.vectors()
    sorted_computed_vectors = sorted(tuple(vec) for vec in returned_vectors)

    assert len(returned_vectors) == 8
    assert sorted_computed_vectors == sorted_vectors


def test_add_vectors():
    vectors = np.array(
        [[1, 1], [-1, 1], [1, -1], [-1, -1], [0, 1], [0, -1], [1, 0], [-1, 0]],
        dtype=np.int64,
    )
    vc = VectorConfiguration(vectors)
    vc.add_vectors([2, 2])

    returned_vectors = vc.vectors()
    assert returned_vectors[-1].tolist() == [2, 2]
