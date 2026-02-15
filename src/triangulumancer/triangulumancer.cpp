#include <pybind11/pybind11.h>

#include "triangulumancer/Flip.hpp"
#include "triangulumancer/PVConfiguration.hpp"
#include "triangulumancer/Triangulation.hpp"

using namespace triangulumancer;

PYBIND11_MODULE(triangulumancer, m, pybind11::mod_gil_not_used(),
                pybind11::multiple_interpreters::per_interpreter_gil()) {
  pybind11::class_<PointConfiguration>(m, "PointConfiguration")
      .def(pybind11::init<>())
      .def(pybind11::init<pybind11::array_t<int> const &>())
      .def("n_points", &PointConfiguration::n_points)
      .def("dim", &PointConfiguration::dim)
      .def("__repr__", &PointConfiguration::repr)
      .def("points", &PointConfiguration::points)
      .def("add_point", &PointConfiguration::add_point)
      .def("add_points", &PointConfiguration::add_point)
      .def("placing_triangulation", &PointConfiguration::placing_triangulation)
      .def("fine_triangulation", &PointConfiguration::fine_triangulation)
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
  pybind11::class_<VectorConfiguration>(m, "VectorConfiguration")
      .def(pybind11::init<>())
      .def(pybind11::init<pybind11::array_t<int> const &>())
      .def("n_vectors", &VectorConfiguration::n_vectors)
      .def("dim", &VectorConfiguration::dim)
      .def("__repr__", &VectorConfiguration::repr)
      .def("vectors", &VectorConfiguration::vectors)
      .def("add_vector", &VectorConfiguration::add_vector)
      .def("add_vectors", &VectorConfiguration::add_vector)
      .def("placing_triangulation", &VectorConfiguration::placing_triangulation)
      .def("fine_triangulation", &VectorConfiguration::fine_triangulation)
      .def("all_connected_triangulations",
           &VectorConfiguration::all_connected_triangulations,
           pybind11::arg("only_fine") = false)
      .def("all_triangulations", &VectorConfiguration::all_triangulations,
           pybind11::arg("only_fine") = false);
  pybind11::class_<Triangulation>(m, "Triangulation")
      .def(pybind11::init<PointConfiguration const &,
                          pybind11::array_t<int> const &>())
      .def("n_simplices", &Triangulation::n_simplices)
      .def("dim", &Triangulation::dim)
      .def("__repr__", &Triangulation::repr)
      .def("simplices", &Triangulation::simplices)
      .def("flips", &Triangulation::flips)
      .def("bistellar_flips", &Triangulation::flips)
      .def("neighbors", &Triangulation::neighbors);
  pybind11::class_<Flip>(m, "Flip")
      .def("__repr__", &Flip::repr)
      .def_readonly("parent_triangulation", &Flip::parent_triangulation)
      .def_readonly("pre", &Flip::pre)
      .def_readonly("post", &Flip::post);
}
