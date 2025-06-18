#include "triangulumancer/CGAL.hpp"

#include <CGAL/Epick_d.h>
#include <CGAL/Regular_triangulation.h>
#include <CGAL/assertions.h>

#include <cmath>
#include <vector>

namespace triangulumancer::cgal {

template <typename TDim>
Triangulation triangulate_cgal(PointConfiguration const &pc,
                               std::vector<double> const &params,
                               bool use_heights) {
  typedef CGAL::Epick_d<TDim> K;
  typedef CGAL::Regular_triangulation<K> T;
  typedef typename K::Point_d Bare_point;
  typedef typename K::Weighted_point_d Weighted_point;
  typedef CGAL::Regular_triangulation_traits_adapter<K> RK;
  typedef typename RK::Compute_coordinate_d Ccd;
  // Define some objects for later
  RK traits = RK();
  const Ccd ccd = traits.compute_coordinate_d_object();

  auto points = pc.points();
  auto points_buf = points.data();

  auto n_points = pc.n_points();
  auto dim = pc.dim();

  if (n_points != params.size()) {
    throw std::runtime_error(
        "Number of parameters must match number of points");
  }

  std::vector<Weighted_point> w_points;
  w_points.reserve(n_points);
  for (int i = 0; i < n_points; i++) {
    Bare_point p(&points_buf[dim * i], &points_buf[dim * (i + 1)]);
    double weight = params[i];
    if (use_heights) {
      weight *= -1;
      for (int j = 0; j < dim; j++) {
        int sqrt_h0 = points_buf[i * dim + j];
        weight += sqrt_h0 * sqrt_h0;
      }
    }

    Weighted_point wp(p, weight);
    w_points.push_back(wp);
  }

  // Construct triangulation
  T t(dim);
  t.insert(w_points.begin(), w_points.end());
  CGAL_assertion(t.is_valid());

  // Match vertices to indices in the order they were given
  // TODO: This whole part should be refactored
  std::map<typename T::Vertex_handle, int> index_of_vertex;
  for (typename T::Vertex_iterator it = t.vertices_begin();
       it != t.vertices_end(); ++it) {
    if (t.is_infinite(it))
      continue;
    std::vector<int> vert(dim, 0);
    for (int i = 0; i < dim; i++) {
      vert[i] = std::round(CGAL::to_double(ccd(it->point(), i)));
    }
    unsigned int index = 0;
    for (int index = 0; index < n_points; index++) {
      bool matches = true;
      for (int i = 0; i < dim; i++) {
        if (vert[i] != points_buf[index * dim + i]) {
          matches = false;
          break;
        }
      }
      if (matches) {
        index_of_vertex[it] = index;
        break;
      }
    }
  }

  // Construct simplices array
  size_t n_simplices =
      std::distance(t.finite_full_cells_begin(), t.finite_full_cells_end());
  auto simplices = pybind11::array_t<int64_t>({n_simplices, dim + 1});
  auto simplices_buf = simplices.mutable_data();

  unsigned int simplex_idx = 0;
  for (typename T::Finite_full_cell_iterator it = t.finite_full_cells_begin();
       it != t.finite_full_cells_end(); it++, simplex_idx++) {
    for (int i = 0; i < dim + 1; i++) {
      simplices_buf[simplex_idx * (dim + 1) + i] =
          index_of_vertex[it->vertex(i)];
    }
  }

  return Triangulation(pc, simplices);
}

Triangulation triangulate_cgal_infer_dim(PointConfiguration const &pc,
                                         std::vector<double> const &params,
                                         bool use_heights) {
  switch (pc.dim()) {
  case 1:
    return triangulate_cgal<CGAL::Dimension_tag<1>>(pc, params, use_heights);
  case 2:
    return triangulate_cgal<CGAL::Dimension_tag<2>>(pc, params, use_heights);
  case 3:
    return triangulate_cgal<CGAL::Dimension_tag<3>>(pc, params, use_heights);
  case 4:
    return triangulate_cgal<CGAL::Dimension_tag<4>>(pc, params, use_heights);
  case 5:
    return triangulate_cgal<CGAL::Dimension_tag<5>>(pc, params, use_heights);
  case 6:
    return triangulate_cgal<CGAL::Dimension_tag<6>>(pc, params, use_heights);
  case 7:
    return triangulate_cgal<CGAL::Dimension_tag<7>>(pc, params, use_heights);
  case 8:
    return triangulate_cgal<CGAL::Dimension_tag<8>>(pc, params, use_heights);
  case 9:
    return triangulate_cgal<CGAL::Dimension_tag<9>>(pc, params, use_heights);
  case 10:
    return triangulate_cgal<CGAL::Dimension_tag<10>>(pc, params, use_heights);
  default:
    return triangulate_cgal<CGAL::Dynamic_dimension_tag>(pc, params,
                                                         use_heights);
  }
}

Triangulation triangulate_delaunay(PointConfiguration const &pc) {
  auto weights = std::vector(pc.n_points(), 0.);
  return triangulate_cgal_infer_dim(pc, weights, false);
}

} // namespace triangulumancer::cgal
