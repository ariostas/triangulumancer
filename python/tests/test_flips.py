"""Flips and neighbours (phase 3 of the port).

Behaviour pinned down by the Rust port beyond the 0.3 test suite: canonical
order of flips and neighbours, ``Flip.__repr__``, and that ``pre``/``post``
are the removed and added simplices.
"""

from __future__ import annotations

from triangulumancer import PointConfiguration, Triangulation, VectorConfiguration


def test_flip_of_the_square() -> None:
    p = PointConfiguration([[0, 0], [1, 0], [1, 1], [0, 1]])
    t = p.placing_triangulation()
    (flip,) = t.flips()
    assert flip.pre.simplices.tolist() == [[0, 1, 2], [0, 2, 3]]
    assert flip.post.simplices.tolist() == [[0, 1, 3], [1, 2, 3]]
    (n,) = t.neighbors()
    assert n.simplices.tolist() == [[0, 1, 3], [1, 2, 3]]
    assert [s.simplices.tolist() for s in n.neighbors()] == [t.simplices.tolist()]


def test_flip_repr() -> None:
    p = PointConfiguration([[0, 0], [1, 0], [1, 1], [0, 1]])
    (flip,) = p.placing_triangulation().flips()
    assert repr(flip) == (
        "A bistellar flip of a triangulation of a 2-dimensional point configuration with 4 points"
    )
    vc = VectorConfiguration([[0, 0, 1], [1, 0, 3], [1, 1, 1], [0, 1, 2]])
    (flip,) = vc.placing_triangulation().flips()
    assert repr(flip) == (
        "A bistellar flip of a triangulation of a 3-dimensional vector configuration with 4 vectors"
    )


def test_order_is_canonical() -> None:
    p = PointConfiguration([[x, y] for x in range(3) for y in range(3)])
    t = p.fine_triangulation()
    flips = t.flips()
    keys = [(f.pre.simplices.tolist(), f.post.simplices.tolist()) for f in flips]
    assert keys == sorted(keys)
    neighbors = [n.simplices.tolist() for n in t.neighbors()]
    assert neighbors == sorted(neighbors)
    assert len(neighbors) == len(flips)
    # Every neighbour is a valid triangulation.
    for n in t.neighbors():
        Triangulation(p, n.simplices)
