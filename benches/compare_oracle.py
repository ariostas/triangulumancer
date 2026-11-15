"""Time the same workloads through Python on two builds of triangulumancer.

Run it once with the 0.3.0 oracle installed and once with this tree's build,
and compare::

    uv venv /tmp/oracle && uv pip install --python /tmp/oracle/bin/python \\
        triangulumancer==0.3.0
    /tmp/oracle/bin/python benches/compare_oracle.py > oracle.json
    .venv-dev/bin/python benches/compare_oracle.py > new.json
    python3 benches/compare_oracle.py --table oracle.json new.json

The configurations are those of `benches/common/mod.rs` (the same SplitMix64
stream), so these numbers line up with the Criterion benchmarks. Each workload
runs in its own subprocess under a timeout, so a workload the oracle cannot
finish (or crashes on) is recorded as such instead of stopping the run. Cap
memory around the whole run (see PLAN.md section 14).
"""

# A command line tool: it reports on stdout and stderr.
# ruff: noqa: T201

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

MASK = (1 << 64) - 1


class Rng:
    """SplitMix64, as `benches/common/mod.rs`."""

    def __init__(self, seed: int) -> None:
        self.state = seed

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def below(self, bound: int) -> int:
        return self.next_u64() % bound

    def unit(self) -> float:
        return (self.next_u64() >> 11) / float(1 << 53)


def random_points(seed: int, n: int, dim: int, rng_range: int) -> list[list[int]]:
    rng = Rng(seed)
    flat = [rng.below(rng_range) for _ in range(n * dim)]
    return [flat[i * dim : (i + 1) * dim] for i in range(n)]


def random_weights(seed: int, n: int, scale: float) -> list[float]:
    rng = Rng(seed ^ 0x5EED)
    return [rng.unit() * scale for _ in range(n)]


def grid(a: int, b: int) -> list[list[int]]:
    return [[x, y] for x in range(a) for y in range(b)]


def workloads() -> dict[str, Callable[[], Callable[[], Any]]]:
    """Name -> setup, returning the timed call. Setup is not timed."""
    # Imported here, so that the driver process never loads either build.
    from triangulumancer import PointConfiguration as P  # noqa: PLC0415

    def construction(method: str, pts: list[list[int]]) -> Callable[[], Any]:
        # A fresh configuration per call: neither build caches anything on
        # it, but this keeps both honest.
        def call() -> Any:
            return getattr(P(pts), method)()

        return call

    def neighbors() -> Callable[[], Any]:
        t = P(random_points(3, 40, 3, 100)).fine_triangulation()
        return t.neighbors

    def connected(only_fine: bool) -> Callable[[], Any]:
        return lambda: len(P(grid(3, 4)).all_connected_triangulations(only_fine))

    def everything() -> Callable[[], Any]:
        return lambda: len(P(grid(3, 4)).all_triangulations())

    def delaunay(seed: int, n: int, dim: int, rng_range: int) -> Callable[[], Any]:
        pts = random_points(seed, n, dim, rng_range)
        return lambda: P(pts).delaunay_triangulation()

    def weights() -> Callable[[], Any]:
        pts = random_points(11, 2000, 3, 1 << 20)
        w = random_weights(11, 2000, 1e10)
        return lambda: P(pts).triangulate_with_weights(w)

    r100 = random_points(1, 100, 3, 1000)
    r60 = random_points(2, 60, 4, 1000)
    return {
        "placing/random_100x3": lambda: construction("placing_triangulation", r100),
        "fine/random_100x3": lambda: construction("fine_triangulation", r100),
        "placing/random_60x4": lambda: construction("placing_triangulation", r60),
        "fine/random_60x4": lambda: construction("fine_triangulation", r60),
        "neighbors/random_40x3": neighbors,
        "all_connected/grid_3x4": lambda: connected(False),
        "all_connected_fine/grid_3x4": lambda: connected(True),
        "all_triangulations/grid_3x4": everything,
        "delaunay/random_10000x2": lambda: delaunay(10, 10_000, 2, 1 << 20),
        "delaunay/random_2000x3": lambda: delaunay(11, 2_000, 3, 1 << 20),
        "delaunay/random_300x5": lambda: delaunay(12, 300, 5, 1 << 20),
        "delaunay/random_40x10": lambda: delaunay(13, 40, 10, 1 << 20),
        "delaunay/lattice_2000x3_range8": lambda: delaunay(14, 2_000, 3, 8),
        "weights/float_2000x3": weights,
    }


def run_one(name: str, budget: float) -> dict[str, Any]:
    call = workloads()[name]()
    times: list[float] = []
    start = time.perf_counter()
    while len(times) < 3 or (time.perf_counter() - start < budget and len(times) < 50):
        t0 = time.perf_counter()
        call()
        times.append(time.perf_counter() - t0)
    return {"median": statistics.median(times), "min": min(times), "runs": len(times)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--one", help="run a single workload (internal)")
    ap.add_argument("--only", help="run only workloads containing this string")
    ap.add_argument("--timeout", type=int, default=300, help="seconds per workload")
    ap.add_argument("--budget", type=float, default=2.0, help="seconds of repeats")
    ap.add_argument("--table", nargs=2, metavar=("OLD", "NEW"))
    args = ap.parse_args()

    if args.table:
        old = json.loads(Path(args.table[0]).read_text())
        new = json.loads(Path(args.table[1]).read_text())
        print("| workload | 0.3.0 | this build | speed-up |")
        print("|---|---:|---:|---:|")
        for name, n in new.items():
            o = old.get(name, {})
            fmt = lambda r: (
                f"{r['median'] * 1e3:.3g} ms" if "median" in r else r.get("error", "-")
            )
            ratio = (
                f"{o['median'] / n['median']:.1f}x"
                if "median" in o and "median" in n
                else "-"
            )
            print(f"| `{name}` | {fmt(o)} | {fmt(n)} | {ratio} |")
        return

    if args.one:
        print(json.dumps(run_one(args.one, args.budget)))
        return

    results: dict[str, Any] = {}
    for name in workloads():
        if args.only and args.only not in name:
            continue
        cmd = [
            sys.executable,
            "-I",
            __file__,
            "--one",
            name,
            "--budget",
            str(args.budget),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=args.timeout, check=False
            )
        except subprocess.TimeoutExpired:
            results[name] = {"error": f"timeout ({args.timeout} s)"}
        else:
            if proc.returncode == 0:
                results[name] = json.loads(proc.stdout.strip().splitlines()[-1])
            else:
                results[name] = {"error": f"exit {proc.returncode}"}
        print(name, results[name], file=sys.stderr, flush=True)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
