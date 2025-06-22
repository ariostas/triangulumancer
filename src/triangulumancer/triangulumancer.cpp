#include <pybind11/pybind11.h>

#include "triangulumancer/PointConfiguration.hpp"
#include "triangulumancer/Triangulation.hpp"

using namespace triangulumancer;

PYBIND11_MODULE(triangulumancer, m) {
  pybind11::class_<PointConfiguration>(m, "PointConfiguration")
      .def(pybind11::init<>())
      .def(pybind11::init<pybind11::array_t<int> const &>())
      .def("n_points", &PointConfiguration::n_points)
      .def("dim", &PointConfiguration::dim)
      .def("__repr__", &PointConfiguration::repr)
      .def("points", &PointConfiguration::points)
      .def("add_point", &PointConfiguration::add_points)
      .def("add_points", &PointConfiguration::add_points)
      .def("placing_triangulation", &PointConfiguration::placing_triangulation)
      .def("all_connected_triangulations",
           &PointConfiguration::all_connected_triangulations,
           pybind11::arg("only_fine") = false)
      .def("all_triangulations", &PointConfiguration::all_triangulations,
           pybind11::arg("only_fine") = false)
      .def("triangulate_with_heights",
           &PointConfiguration::triangulate_with_heights)
      .def("triangulate_with_weights",
           &PointConfiguration::triangulate_with_weights)
      .def("delaunay_triangulation",
           &PointConfiguration::delaunay_triangulation);
  pybind11::class_<Triangulation>(m, "Triangulation")
      .def("n_simplices", &Triangulation::n_simplices)
      .def("dim", &Triangulation::dim)
      .def("__repr__", &Triangulation::repr)
      .def("simplices", &Triangulation::simplices)
      .def("neighbors", &Triangulation::neighbors);
}
