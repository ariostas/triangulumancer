#include "triangulumancer/Flip.hpp"

using namespace triangulumancer;

Flip::Flip(Triangulation const &parent_triangulation_in,
           Triangulation const &pre_in, Triangulation const &post_in)
    : parent_triangulation(parent_triangulation_in), pre(pre_in),
      post(post_in) {}

std::string Flip::repr() const {
  std::string msg;
  msg = "A bistellar flip between a triangulation with " +
        std::to_string(pre.n_simplices()) +
        " simplices and a triangulation with " +
        std::to_string(post.n_simplices()) + " simplices, both of a " +
        std::to_string(parent_triangulation.dim()) + "-dimensional ";
  bool is_pc = parent_triangulation.pvc.pvc_data->config_type ==
               ConfigurationType::Point;
  if (is_pc) {
    msg = msg + "point configuration with " +
          std::to_string(parent_triangulation.pvc.n_pv()) + " points";
  } else {
    msg = msg + "vector configuration with " +
          std::to_string(parent_triangulation.pvc.n_pv()) + " vectors";
  }
  return msg;
}
