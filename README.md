# Triangulumancer

[![Build and test](https://github.com/ariostas/triangulumancer/actions/workflows/test.yml/badge.svg)](https://github.com/ariostas/triangulumancer/actions/workflows/test.yml)

This project provides tools to construct $n$-dimensional triangulations of point configurations. It uses [CGAL](https://www.cgal.org/)'s `dD Triangulations` package to construct Delaunay and regular triangulations with given heights/weights, and it uses [TOPCOM](https://www.wm.uni-bayreuth.de/de/team/rambau_joerg/TOPCOM/) to find pulling/pushing triangulations, to find the list of available bistellar flips, and to find the complete list of triangulations of a point configuration.

The functionality and documentation of this project are currently fairly limited. If there is enough interest, I will continue to expand on both functionality and documentation. Please open an issue to ask for more functionality, or even better, make a pull request!

## Installation

Triangulumancer can be installed in most cases simply by running
```bash
pip install triangulumancer
```

If you want to tweak compilation parameters or anything else, you can clone the repository and build the wheel yourself.
```bash
git clone --recurse-submodules https://github.com/ariostas/triangulumancer.git
cd triangulumancer
pip install .
# You might need to set your LD_LIBRARY_PATH to include extern/topcom/external/lib.
```

## Usage

Here is a basic example that shows the available functionality.

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
>>> 387
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

## License

All original `triangulumancer` code is distributed under the [GNU General Public License version 3](https://www.gnu.org/licenses/gpl-3.0.txt). The libraries it depends on, which can be found in the `extern` directory, are redistributed under their corresponding licenses.
