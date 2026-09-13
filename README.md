# Triangulumancer

[![Rust CI](https://github.com/ariostas/triangulumancer/actions/workflows/rust.yml/badge.svg)](https://github.com/ariostas/triangulumancer/actions/workflows/rust.yml)
[![Python CI](https://github.com/ariostas/triangulumancer/actions/workflows/python.yml/badge.svg)](https://github.com/ariostas/triangulumancer/actions/workflows/python.yml)

This project provides tools to construct $n$-dimensional triangulations of point
and vector configurations: Delaunay and regular triangulations from given
heights or weights, placing (pushing) and fine triangulations, the bistellar
flips available from a triangulation, and the complete list of triangulations of
a configuration.

## Status

**The implementation is being rewritten in Rust.** Versions up to 0.3.0 were C++,
using [CGAL](https://www.cgal.org/) for regular and Delaunay triangulations and
[TOPCOM](https://www.wm.uni-bayreuth.de/de/team/rambau_joerg/TOPCOM/) for
everything else. The rewrite removes both dependencies, which means no CMake, no
system libraries to install, and Windows support.

The plan lives in [`PLAN.md`](PLAN.md), with the algorithm details in
[`CGAL.md`](CGAL.md) and [`TOPCOM.md`](TOPCOM.md). Work in progress lives on the
`rust-migration` branch; `0.3.0` remains what `pip install triangulumancer`
gives you until the port reaches parity. The last C++ commit is tagged
`legacy-cpp`.

## Installation

```bash
pip install triangulumancer
```

Or from a checkout:

```bash
git clone https://github.com/ariostas/triangulumancer.git
cd triangulumancer
pip install .
```

Building from source needs a Rust toolchain ([rustup](https://rustup.rs/)) and
nothing else.

## Usage

```python
>>> from triangulumancer import PointConfiguration
>>>
>>> p = PointConfiguration(
...     [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
... )
>>> p
A 2-dimensional point configuration with 9 points
>>> t_delaunay = p.delaunay_triangulation()
>>> t_delaunay
A triangulation with 8 simplices of a point configuration with 9 points
>>> t_heights = p.triangulate_with_heights([1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> t_heights
A triangulation with 8 simplices of a point configuration with 9 points
>>> t_weights = p.triangulate_with_weights([1, 2, 3, 4, 5, 6, 7, 8, 9])
A triangulation with 5 simplices of a point configuration with 9 points
>>> t_placing = p.placing_triangulation()
A triangulation with 8 simplices of a point configuration with 9 points
>>> t_placing_neighbors = t_placing.neighbors()
>>> len(t_placing_neighbors)
6
>>> list_t_connected = p.all_connected_triangulations()
>>> len(list_t_connected)
387
>>> list_t_connected_fine = p.all_connected_triangulations(only_fine=True)
>>> len(list_t_connected_fine)
64
>>> list_t_all = p.all_triangulations()
>>> len(list_t_all)
387
>>> list_t_all_fine = p.all_triangulations(only_fine=True)
>>> len(list_t_all_fine)
64
```

There is equivalent functionality for vector configurations, but it is not fully
complete and has not been tested as much, so we have not included it in the
example above.

Please open an issue if you have any problems or want to see more functionality.
Pull requests are also welcome!

## License

`triangulumancer` is distributed under the
[GNU General Public License version 3](https://www.gnu.org/licenses/gpl-3.0.txt)
or later.

Releases up to 0.3.0 bundled CGAL and TOPCOM, whose licenses also applied to
those artifacts. The Rust implementation has no such dependencies, though it is
a reimplementation informed by both, and remains under the GPL.
