#include "triangulumancer/TOPCOM.hpp"

// TOPCOM includes
#include "PlacingTriang.hh"
#include "SymmetricExtensionGraphMaster.hh"
#include "SymmetricFlipGraph.hh"
#include "Symmetry.hh"
#include "TriangFlips.hh"
#include "TriangNode.hh"

namespace triangulumancer::top {

Triangulation
simplicial_complex_to_triangulation(PointConfiguration const &pc,
                                    topcom::SimplicialComplex const &sc) {
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

topcom::SimplicialComplex
triangulation_to_simplicial_complex(Triangulation const &t) {
  auto simplices = t.simplices();
  auto simplices_buf = simplices.data();

  size_t n_simplices = t.n_simplices();
  size_t dim = t.dim() + 1;

  topcom::SimplicialComplex sc;
  topcom::Simplex simp;

  for (unsigned int simp_idx = 0; simp_idx < n_simplices; simp_idx++) {
    simp.clear();
    for (unsigned int dim_idx = 0; dim_idx < dim; dim_idx++) {
      simp += simplices_buf[simp_idx * dim + dim_idx];
    }
    sc += simp;
  }
  return sc;
}

void validate_point_configuration(topcom::PointConfiguration const &points) {
  if (points.rank() < points.rowdim()) {
    throw std::runtime_error("Points are not full rank");
  }
  if ((points.no() < 1) || (points.rank() < 1)) {
    throw std::runtime_error("Number of points and rank must be at least one");
  }
  if (points.rank() > points.no()) {
    throw std::runtime_error("Rank must not be larger than number of points");
  }
}

Triangulation triangulate_placing(PointConfiguration const &pc) {
  topcom::PointConfiguration points = pc.pc_data->topcom_pc;

  validate_point_configuration(points);

  topcom::Chirotope chiro(points, false);
  topcom::PlacingTriang t(chiro);

  return simplicial_complex_to_triangulation(pc, t);
}

std::vector<Triangulation> find_neighbors(Triangulation const &t) {
  std::vector<Triangulation> neighbors;

  topcom::PointConfiguration points = t.pc.pc_data->topcom_pc;

  validate_point_configuration(points);

  topcom::Chirotope chiro(points, false);

  size_t no(chiro.no());
  size_t rank(chiro.rank());
  topcom::SymmetryGroup symmetries(no);

  topcom::SimplicialComplex seed = triangulation_to_simplicial_complex(t);

  const topcom::symmetryptr_datapair seed_symmetryptrs(
      symmetries.stabilizer_ptrs(seed));

  const topcom::TriangNode tn(0, no, rank, seed);
  const topcom::TriangFlips tf(chiro, tn, seed_symmetryptrs, false);
  topcom::MarkedFlips mf = tf.flips();

  for (auto t_it = mf.begin(); t_it != mf.end(); t_it++) {
    auto fl = topcom::Flip(tn, t_it->first);
    auto sc =
        static_cast<topcom::SimplicialComplex>(topcom::TriangNode(0, tn, fl));
    neighbors.push_back(simplicial_complex_to_triangulation(t.pc, sc));
  }

  return neighbors;
}

std::vector<Triangulation>
find_all_connected_triangulations(PointConfiguration const &pc,
                                  bool only_fine) {
  std::vector<Triangulation> all_triangs;

  topcom::PointConfiguration points = pc.pc_data->topcom_pc;

  validate_point_configuration(points);

  topcom::Chirotope chiro(points, false);

  size_t no(chiro.no());
  size_t rank(chiro.rank());
  topcom::SymmetryGroup symmetries(no);

  topcom::SimplicialComplex seed = topcom::PlacingTriang(chiro);

  const topcom::symmetryptr_datapair seed_symmetryptrs(
      symmetries.stabilizer_ptrs(seed));

  topcom::Volumes *voltableptr = nullptr;

  auto callback = [&](topcom::SimplicialComplex sc) {
    all_triangs.push_back(simplicial_complex_to_triangulation(pc, sc));
  };

  topcom::SymmetricFlipGraph sfg(no, rank, points, chiro, symmetries, seed,
                                 seed_symmetryptrs, voltableptr, true,
                                 only_fine, callback);

  return all_triangs;
}

std::vector<Triangulation> find_all_triangulations(PointConfiguration const &pc,
                                                   bool only_fine) {
  std::vector<Triangulation> all_triangs;

  topcom::PointConfiguration points = pc.pc_data->topcom_pc;

  validate_point_configuration(points);

  topcom::Chirotope chiro(points, false);

  size_t no(chiro.no());
  size_t rank(chiro.rank());
  topcom::SymmetryGroup symmetries(no);

  auto callback = [&](topcom::SimplicialComplex sc) {
    all_triangs.push_back(simplicial_complex_to_triangulation(pc, sc));
  };

  topcom::SymmetricExtensionGraphMaster segm(
      no, rank, points, chiro, symmetries, true, only_fine, false, callback);

  return all_triangs;
}

} // namespace triangulumancer::top
