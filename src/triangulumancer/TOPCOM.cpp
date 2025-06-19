#include "triangulumancer/TOPCOM.hpp"

#include <iterator>

// TOPCOM includes
#include "PlacingTriang.hh"

namespace triangulumancer::top {

Triangulation
simplicial_complex_to_triangulation(PointConfiguration const &pc,
                                    ::topcom::SimplicialComplex &sc) {
  size_t dim = pc.dim();
  // std::difference doesn't work with these iterators
  size_t n_simplices = 0;
  for (auto it = sc.begin(); it != sc.end(); it++, n_simplices++) {
  }

  auto simplices = pybind11::array_t<int64_t>({n_simplices, dim + 1});
  auto simplices_buf = simplices.mutable_data();

  unsigned int simplex_idx = 0;
  for (auto sc_it = sc.begin(); sc_it != sc.end(); sc_it++, simplex_idx++) {
    unsigned int dim_idx = 0;
    for (auto simp_it = sc_it->begin(); simp_it != sc_it->end();
         simp_it++, dim_idx++) {
      simplices_buf[simplex_idx * (dim + 1) + dim_idx] = *simp_it;
    }
  }

  return Triangulation(pc, simplices);
}

Triangulation triangulate_placing(PointConfiguration const &pc) {
  ::topcom::PointConfiguration points = pc.pc_data->topcom_pc;

  if (points.rank() < points.rowdim()) {
    throw std::runtime_error("Points are not full rank");
  }
  if ((points.no() < 1) || (points.rank() < 1)) {
    throw std::runtime_error("Number of points and rank must be at least one");
  }
  if (points.rank() > points.no()) {
    throw std::runtime_error("Rank must not be larger than number of points");
  }

  ::topcom::Chirotope chiro(points, false);
  ::topcom::PlacingTriang t(chiro);

  return simplicial_complex_to_triangulation(pc, t);
}

} // namespace triangulumancer::top
