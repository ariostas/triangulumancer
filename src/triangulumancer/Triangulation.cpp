#include "triangulumancer/Triangulation.hpp"
#include "triangulumancer/PointConfiguration.hpp"

using namespace triangulumancer;

size_t Triangulation::n_simplices() const {
  pybind11::buffer_info buf = simplices_.request();
  return buf.shape[0];
}

size_t Triangulation::dim() const { return pc->dim(); }

std::string Triangulation::repr() const {
  return "A triangulation with " + std::to_string(n_simplices()) + " of a " +
         "point configuration with " + std::to_string(pc->n_points()) +
         " points";
}

pybind11::array_t<int64_t> Triangulation::simplices() { return simplices_; }
