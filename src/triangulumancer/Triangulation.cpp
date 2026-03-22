#include "triangulumancer/Triangulation.hpp"
#include "triangulumancer/PVConfiguration.hpp"
#include "triangulumancer/TOPCOM.hpp"

using namespace triangulumancer;

Triangulation::Triangulation(std::shared_ptr<PVConfigurationData> pvc_data_in,
                             pybind11::array_t<int64_t> simplices_in)
    : pvc(pvc_data_in), m_simplices(simplices_in) {
  if (!pvc.pvc_data->is_locked) {
    pvc.pvc_data->is_locked = true;
  }
}

Triangulation::Triangulation(PVConfiguration const &pvc_in,
                             pybind11::array_t<int64_t> simplices_in)
    : pvc(pvc_in), m_simplices(simplices_in) {
  if (!pvc.pvc_data->is_locked) {
    pvc.pvc_data->is_locked = true;
  }
}

size_t Triangulation::n_simplices() const {
  pybind11::buffer_info buf = m_simplices.request();
  return buf.shape[0];
}

size_t Triangulation::dim() const { return pvc.dim(); }

std::string Triangulation::repr() const {
  std::string msg;
  msg = "A triangulation with " + std::to_string(n_simplices()) +
        " simplices of a ";
  bool is_pc = pvc.pvc_data->config_type == ConfigurationType::Point;
  if (is_pc) {
    msg = msg + "point configuration with " + std::to_string(pvc.n_pv()) +
          " points";
  } else {
    msg = msg + "vector configuration with " + std::to_string(pvc.n_pv()) +
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
