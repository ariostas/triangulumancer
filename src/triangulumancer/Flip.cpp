#include "triangulumancer/Flip.hpp"

using namespace triangulumancer;

Flip::Flip(Triangulation const &pre, Triangulation const &post)
    : pre(pre), post(post) {}

std::string Flip::repr() const {
  std::string msg;
  msg = "A bistellar flip of a triangulation of a " +
        std::to_string(pre.dim()) + "-dimensional ";
  bool is_pc = pre.pvc.pvc_data->config_type == ConfigurationType::Point;
  if (is_pc) {
    msg = msg + "point configuration with " + std::to_string(pre.pvc.n_pv()) +
          " points";
  } else {
    msg = msg + "vector configuration with " + std::to_string(pre.pvc.n_pv()) +
          " vectors";
  }
  return msg;
}
