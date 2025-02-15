# Triangulumancer

> [!WARNING]
> This project is still in early stages and offers very limited functionality. The code and documentation are under active development and may change significantly.

This project provides tools to construct $n$-dimensional triangulations of point configurations. It uses [CGAL](https://www.cgal.org/)'s `dD Triangulations` package to construct Delaunay and regular triangulations with given heights/weights, and it uses [TOPCOM](https://www.wm.uni-bayreuth.de/de/team/rambau_joerg/TOPCOM/) to find pulling/pushing triangulations, to find the list of available bistellar flips, and to find the complete list of triangulations of a point configuration.


## License

All original `triangulumancer` code is distributed under the [GNU General Public License version 3](https://www.gnu.org/licenses/gpl-3.0.txt). The libraries it depends on, which can be found in the `extern` directory, are redistributed under their corresponding licenses.
