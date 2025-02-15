#include <pybind11/pybind11.h>

#include "triangulumancer/PointConfiguration.hpp"

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
      .def("add_points", &PointConfiguration::add_points);
}
