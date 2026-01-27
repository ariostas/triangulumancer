#include "triangulumancer/Triangulation.hpp"
#include "triangulumancer/PointConfiguration.hpp"
#include "triangulumancer/TOPCOM.hpp"
#include "triangulumancer/VectorConfiguration.hpp"

using namespace triangulumancer;

Triangulation::Triangulation(std::shared_ptr<PointConfigurationData> pc_data_in,
                             pybind11::array_t<int64_t> simplices_in)
    : isPC(true), pc(pc_data_in), m_simplices(simplices_in) {
  if (!pc.pc_data->is_locked) {
    pc.pc_data->is_locked = true;
  }
}

Triangulation::Triangulation(PointConfiguration const &pc_in,
                             pybind11::array_t<int64_t> simplices_in)
    : isPC(true), pc(pc_in), m_simplices(simplices_in) {
  if (!pc.pc_data->is_locked) {
    pc.pc_data->is_locked = true;
  }
}

Triangulation::Triangulation(
    std::shared_ptr<VectorConfigurationData> vc_data_in,
    pybind11::array_t<int64_t> simplices_in)
    : isPC(false), vc(vc_data_in), m_simplices(simplices_in) {
  if (!vc.vc_data->is_locked) {
    vc.vc_data->is_locked = true;
  }
}

Triangulation::Triangulation(VectorConfiguration const &vc_in,
                             pybind11::array_t<int64_t> simplices_in)
    : isPC(false), vc(vc_in), m_simplices(simplices_in) {
  if (!vc.vc_data->is_locked) {
    vc.vc_data->is_locked = true;
  }
}

size_t Triangulation::n_simplices() const {
  pybind11::buffer_info buf = m_simplices.request();
  return buf.shape[0];
}

size_t Triangulation::dim() const { return pc.dim(); }

std::string Triangulation::repr() const {
  std::string msg;
  msg = "A triangulation with " + std::to_string(n_simplices()) +
        " simplices of a ";
  if (isPC) {
    msg = msg + "point configuration with " + std::to_string(pc.n_points()) +
          " points";
  } else {
    msg = msg + "vector configuration with " + std::to_string(vc.n_vectors()) +
          " vectors";
  }
  return msg;
}

pybind11::array_t<int64_t> Triangulation::simplices() const {
  return m_simplices;
}

std::vector<Triangulation> Triangulation::neighbors() const {
  return top::find_neighbors(*this);
}
