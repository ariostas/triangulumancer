# Triangulumancer

[![Rust CI](https://github.com/ariostas/triangulumancer/actions/workflows/rust.yml/badge.svg)](https://github.com/ariostas/triangulumancer/actions/workflows/rust.yml)
[![Python CI](https://github.com/ariostas/triangulumancer/actions/workflows/python.yml/badge.svg)](https://github.com/ariostas/triangulumancer/actions/workflows/python.yml)
[![PyPI](https://img.shields.io/pypi/v/triangulumancer)](https://pypi.org/project/triangulumancer/)
[![crates.io](https://img.shields.io/crates/v/triangulumancer)](https://crates.io/crates/triangulumancer)
[![docs.rs](https://img.shields.io/docsrs/triangulumancer)](https://docs.rs/triangulumancer)

Triangulumancer constructs triangulations of point and vector configurations in
any dimension: Delaunay triangulations and regular triangulations from given
heights or weights, placing (pushing) and fine triangulations, the bistellar
flips of a triangulation and its neighbours, and the complete list of
triangulations of a configuration, either the flip-connected ones or all of
them.

It is written in Rust and usable from both Python and Rust. All arithmetic is
exact.

Up to version 0.3, triangulumancer wrapped [CGAL](https://www.cgal.org/) and
[TOPCOM](https://www.wm.uni-bayreuth.de/de/team/rambau_joerg/TOPCOM/). Version
0.4 reimplements what it used from both in pure Rust, so it has no system
dependencies, builds anywhere Rust does, and runs on **Windows** too. The Python
API is unchanged apart from some additions and bug fixes; see
[`CHANGELOG.md`](CHANGELOG.md) for the full list of differences.

## Installation

```bash
pip install triangulumancer
```

There are wheels for Linux, macOS and Windows, for CPython 3.10 and later
including free-threaded 3.14. To build from a checkout instead, you need a Rust
toolchain ([rustup](https://rustup.rs/)) and nothing else:

```bash
git clone https://github.com/ariostas/triangulumancer.git
cd triangulumancer
pip install .
```

For the Rust library, add it to your `Cargo.toml`:

```bash
cargo add triangulumancer
```

## Usage from Python

Configurations are built from integer coordinates, one row per point (or vector).
Every triangulation method returns a `Triangulation`, whose `simplices` is a
NumPy array of point indices with one row per simplex.

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
>>> t_weights
A triangulation with 5 simplices of a point configuration with 9 points
>>> t_placing = p.placing_triangulation()
>>> t_placing
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

### Triangulations

Simplices are listed with their vertices in increasing order, and sorted. The
output of every method is deterministic, including the order of the lists the
enumerations return.

```python
>>> t_placing.simplices.tolist()
[[0, 1, 2], [0, 1, 6], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 6], [1, 6, 7], [1, 7, 8]]
>>> t_placing.dim, t_placing.n_simplices
(2, 8)
>>> t_placing.is_fine
True
>>> p.fine_triangulation() == t_placing
True
```

A regular triangulation need not use every point. The ones it leaves out are
listed by `hidden_points`:

```python
>>> t_weights.is_fine
False
>>> t_weights.hidden_points
[0, 1]
```

Heights and weights may be floats, which are used as their exact binary values,
or integers of any size and `fractions.Fraction`s, which keep the whole
computation exact:

```python
>>> from fractions import Fraction
>>> t = p.triangulate_with_heights([Fraction(1, 3), 0, 0, 0, 0, 0, 0, 0, 10**30])
>>> t.simplices.tolist()
[[1, 2, 3], [1, 3, 7], [1, 7, 8], [3, 4, 7], [4, 5, 7], [5, 6, 7]]
>>> t.hidden_points
[0]
```

A triangulation can also be given explicitly. It is checked to be one, unless
`validate=False` is passed:

```python
>>> from triangulumancer import Triangulation
>>> square = PointConfiguration([[0, 0], [1, 0], [0, 1], [1, 1]])
>>> t = Triangulation(square, [[1, 2, 3], [0, 1, 2]])
>>> t.simplices.tolist()
[[0, 1, 2], [1, 2, 3]]
>>> Triangulation(square, [[0, 1, 2]])
Traceback (most recent call last):
    ...
ValueError: invalid triangulation: ...
```

Triangulations compare equal when they have the same simplices and belong to the
same configuration object, and they can be used in sets and as dictionary keys.

Once a configuration has been triangulated it can no longer be extended, since
triangulations refer to its points by index:

```python
>>> square.add_point([2, 2])
Traceback (most recent call last):
    ...
RuntimeError: the configuration is locked because a triangulation of it exists, so more points or vectors cannot be added
```

### Flips

`flips()` (or its alias `bistellar_flips()`) lists the bistellar flips of a
triangulation. Each `Flip` has a `pre` and a `post`: the simplices it removes
and the ones it adds. `neighbors()` returns the triangulations the flips lead
to.

```python
>>> f = t_placing.flips()[0]
>>> f
A bistellar flip of a triangulation of a 2-dimensional point configuration with 9 points
>>> f.pre.simplices.tolist(), f.post.simplices.tolist()
([[0, 1, 2], [0, 2, 3]], [[0, 1, 3], [1, 2, 3]])
```

### Enumerating triangulations

`all_triangulations()` finds every triangulation of the configuration, and
`all_connected_triangulations()` the ones connected to the placing triangulation
by flips. With `only_fine=True`, the first returns every fine triangulation, and
the second the fine triangulations connected to the fine triangulation through
flips between fine triangulations.

Both build the whole list in memory. The `iter_` variants produce the
triangulations one at a time instead, in the order the search finds them, so
configurations with more triangulations than fit in memory can still be
counted or searched. `iter_all_triangulations` runs in constant memory.

```python
>>> it = p.iter_all_triangulations(only_fine=True)
>>> next(it)
A triangulation with 8 simplices of a point configuration with 9 points
>>> sum(1 for _ in p.iter_all_connected_triangulations())
387
```

Computations release the GIL, so several can run in parallel threads.

### Vector configurations

`VectorConfiguration` has the same methods except the three regular ones
(`delaunay_triangulation`, `triangulate_with_heights` and
`triangulate_with_weights`), with `vectors`, `n_vectors` and
`add_vector`/`add_vectors` in place of `points`, `n_points` and
`add_point`/`add_points`. Its simplices have as many vertices as its dimension.

```python
>>> from triangulumancer import VectorConfiguration
>>> v = VectorConfiguration([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1]])
>>> v
A 2-dimensional vector configuration with 5 vectors
>>> v.placing_triangulation()
A triangulation with 4 simplices of a vector configuration with 5 vectors
>>> len(v.all_triangulations())
5
```

### Errors

Invalid input raises `ValueError` (for example a configuration that is empty or
does not span its space, where the method needs it to), `TypeError` or
`IndexError`. Misuse, such as extending a locked configuration, raises
`RuntimeError`.

## Usage from Rust

The [crate documentation](https://docs.rs/triangulumancer) describes the
library in full. The combinatorial algorithms work on an oriented matroid
(a `Chirotope`), and the regular ones on the configuration itself:

```rust
use triangulumancer::config::PointConfig;
use triangulumancer::regular::{self, Scalar};
use triangulumancer::triang::{self, ExtensionGraph};

fn main() -> triangulumancer::Result<()> {
    // The 3×3 grid, row-major: point i is coords[2 * i..2 * i + 2].
    let coords = [0, 0, 1, 0, 1, 1, 0, 1, -1, 1, -1, 0, -1, -1, 0, -1, 1, -1];
    let config = PointConfig::from_row_major(&coords, 9, 2)?;

    // Regular triangulations need the coordinates.
    let delaunay = regular::delaunay(&config)?;
    assert_eq!(delaunay.len(), 8);
    let heights: Vec<Scalar> = (1..=9).map(Scalar::from).collect();
    let t = regular::with_heights(&config, &heights)?;
    println!("{:?}", t.to_vecs());

    // Everything else works on the oriented matroid.
    let chiro = config.chirotope()?;
    let placing = triang::placing(&chiro)?;
    assert_eq!(triang::neighbors(&chiro, &placing)?.len(), 6);
    assert_eq!(triang::all_connected(&chiro, false)?.len(), 387);
    assert_eq!(triang::all_triangulations(&chiro, true)?.len(), 64);

    // Or stream them, here counting without building any.
    assert_eq!(ExtensionGraph::new(&chiro, false)?.count(), 387);
    Ok(())
}
```

The Python bindings are behind the `python` feature, which is off by default, so
the Rust library does not depend on PyO3.

## Contributing

Please open an issue if you have any problems or want to see more functionality.
Pull requests are welcome too; see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

`triangulumancer` is distributed under the
[GNU General Public License version 3](https://www.gnu.org/licenses/gpl-3.0.txt)
or later. Releases up to 0.3.0 bundled CGAL and TOPCOM, whose licenses also
applied to those packages; from 0.4.0 on it has no such dependencies. It remains
under the GPL as a reimplementation of algorithms from both, written by closely
following their sources.
