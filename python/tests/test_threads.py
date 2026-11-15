"""Using one configuration, or one iterator, from several threads.

Computations run on a snapshot of the configuration with the GIL released, and
lock it only when they finish, so other threads can keep using it meanwhile.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from triangulumancer import PointConfiguration


def big_config(n: int = 20000) -> PointConfiguration:
    pts = np.random.default_rng(3).integers(-(10**6), 10**6, size=(n, 3))
    return PointConfiguration(pts)


def test_reads_while_computing() -> None:
    # 0.3 held the configuration borrowed for the whole computation, so any
    # other thread touching it got "Already mutably borrowed".
    pc = big_config()
    done = threading.Event()
    errors: list[BaseException] = []
    reads = 0

    def reader() -> None:
        nonlocal reads
        while not done.is_set():
            try:
                assert pc.n_points == 20000
                assert pc.dim == 3
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
                return
            reads += 1

    th = threading.Thread(target=reader)
    th.start()
    try:
        t = pc.delaunay_triangulation()
    finally:
        done.set()
        th.join()
    assert not errors, errors[0]
    assert reads > 0
    assert t.n_simplices > 0


def test_concurrent_computations() -> None:
    pc = big_config(5000)
    results: list[int] = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            results.append(pc.delaunay_triangulation().n_simplices)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(3)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors, errors[0]
    assert len(results) == 3
    assert len(set(results)) == 1


def test_modification_during_computation() -> None:
    # Adding points while a triangulation is computed either happens before
    # the lock, and the computation then fails rather than lock a
    # configuration its result does not describe, or after it, and is refused.
    pc = big_config()
    outcome: list[object] = []

    def compute() -> None:
        try:
            outcome.append(pc.delaunay_triangulation())
        except RuntimeError as e:
            outcome.append(e)

    th = threading.Thread(target=compute)
    th.start()
    added = 0
    refusal = ""
    while th.is_alive():
        try:
            pc.add_point([0, 0, 0])
            added += 1
        except RuntimeError as e:
            refusal = str(e)
            break
    th.join()
    assert not refusal or "locked" in refusal
    (result,) = outcome
    if isinstance(result, RuntimeError):
        assert "modified" in str(result)
        assert added > 0
        pc.add_point([1, 1, 1])  # still unlocked
    else:
        # Any addition before the lock would have failed the computation, so
        # the triangulation describes the configuration it locked.
        assert added == 0
        assert pc.n_points == 20000
        with pytest.raises(RuntimeError, match="locked"):
            pc.add_point([1, 1, 1])


def test_shared_iterator() -> None:
    pc = PointConfiguration([[x, y] for x in range(3) for y in range(3)])
    expected = set(pc.all_connected_triangulations())
    it = pc.iter_all_connected_triangulations()
    out = []
    lock = threading.Lock()

    def work() -> None:
        for t in it:
            with lock:
                out.append(t)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(out) == len(set(out)) == len(expected)
    assert set(out) == expected
    with pytest.raises(StopIteration):
        next(it)
