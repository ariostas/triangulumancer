#pragma once

#include <CGAL/Epick_d.h>
#include <CGAL/Regular_triangulation.h>
#include <CGAL/assertions.h>

#include <vector>

#include "triangulumancer/PointConfiguration.hpp"
#include "triangulumancer/Triangulation.hpp"

namespace triangulumancer {

enum TriangParamType {
  Heights,
  Weights,
};

template <typename TDim>
Triangulation
triangulate_cgal(PointConfiguration &pc, std::vector<double> &params,
                 TriangParamType param_type = TriangParamType::Heights) {
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
    Bare_point p(&points_buf[i + dim], &points[i * (dim + 1)]);
    double weight = params[i];
    if (paramType == TriangParamType::Height) {
      weight *= -1;
      for (int j = 0; j < dim; j++) {
        int sqrt_h0 = points[i * dim + j];
        weight += sqrt_h0 * sqrt_h0;
      }
    }

    Weighted_point wp(p, (weights.size() == 0 ? 0 : weights[i]));
    weight = p w_points.push_back(wp);
  }

  // // Read point list
  // std::vector<std::vector<int> > points = readPoints(std::cin);
  // int nPoints = points.size();
  // if(nPoints == 0){
  //   std::cerr << "Error: points list was empty" << std::endl;
  //   return 1;
  // }
  // int D = points[0].size();
  // // Read weights or heights
  // std::vector<double> weights = readWeights(std::cin);
  // if(nPoints != weights.size()){
  //   std::cerr << (useHeights ? "Heights" : "Weights") << " not specified or
  //   size mismatch. Computing Delaunay triangulation..." << std::endl;
  //   weights.clear();
  // }
  // // If using heights, convert them into weights
  // if(param_type == TriangParamType::Heights){
  //   for(int i = 0; i < nPoints; i++){
  //     double h0 = 0;
  //     for(int j = 0; j < D; j++){
  //       h0 += points[i][j]*points[i][j];
  //     }
  //     weights[i] = h0 - weights[i];
  //   }
  // }

  // // Produce triangulation
  // std::vector<Weighted_point> w_points;
  // w_points.reserve(nPoints);
  // for(int i = 0; i < nPoints; i++) {
  //   Bare_point p(points[i].begin(), points[i].end());
  //   Weighted_point wp(p,(weights.size() == 0 ? 0 : weights[i]));
  //   w_points.push_back(wp);
  // }

  // // Define triangulation object
  // T t(D);
  // CGAL_assertion(t.empty());

  // // Insert the points in the triangulation
  // t.insert(w_points.begin(), w_points.end());

  // // Check if the triangulation is ok
  // CGAL_assertion( t.is_valid() );

  // // Match vertices to indices in the order they were given
  // std::map<T::Vertex_handle, int> index_of_vertex;
  // for(T::Vertex_iterator it = t.vertices_begin(); it != t.vertices_end();
  // ++it){
  //   if(t.is_infinite(it))
  //     continue;
  //   std::vector<int> vert(D,0);
  //   for(int i = 0; i < D; i++){
  //     vert[i] = CGAL::to_double(ccd(it->point(), i));
  //   }
  //   index_of_vertex[it] = std::distance(points.begin(),
  //   std::find(points.begin(), points.end(), vert));
  // }

  // // Print the simplices (and weights)
  // printf("[");
  // for(T::Finite_full_cell_iterator it = t.finite_full_cells_begin(); it !=
  // t.finite_full_cells_end(); ++it){
  //   for(int i = 0; i < D+1; i++){
  //     if(i == 0) printf("[%i,",index_of_vertex[it->vertex(i)]);
  //     else if(i == D) printf("%i]",index_of_vertex[it->vertex(i)]);
  //     else printf("%i,",index_of_vertex[it->vertex(i)]);
  //   }
  //   if(it != t.finite_full_cells_end() && std::next(it) !=
  //   t.finite_full_cells_end()) printf(",");
  // }
  // printf("]\n");

  return Triangulation{};
}

Triangulation triangulate_delaunay(PointConfiguration pc) {
  auto weights = std::vector(pc.n_points(), 0.);
  switch (pc.dim()) {
  case 1:
    return triangulate_cgal<CGAL::Dimension_tag<1>>(pc, weights,
                                                    TriangParamType::Weights);
  case 2:
    return triangulate_cgal<CGAL::Dimension_tag<2>>(pc, weights,
                                                    TriangParamType::Weights);
  case 3:
    return triangulate_cgal<CGAL::Dimension_tag<3>>(pc, weights,
                                                    TriangParamType::Weights);
  case 4:
    return triangulate_cgal<CGAL::Dimension_tag<4>>(pc, weights,
                                                    TriangParamType::Weights);
  case 5:
    return triangulate_cgal<CGAL::Dimension_tag<5>>(pc, weights,
                                                    TriangParamType::Weights);
  case 6:
    return triangulate_cgal<CGAL::Dimension_tag<6>>(pc, weights,
                                                    TriangParamType::Weights);
  case 7:
    return triangulate_cgal<CGAL::Dimension_tag<7>>(pc, weights,
                                                    TriangParamType::Weights);
  case 8:
    return triangulate_cgal<CGAL::Dimension_tag<8>>(pc, weights,
                                                    TriangParamType::Weights);
  case 9:
    return triangulate_cgal<CGAL::Dimension_tag<9>>(pc, weights,
                                                    TriangParamType::Weights);
  case 10:
    return triangulate_cgal<CGAL::Dimension_tag<10>>(pc, weights,
                                                     TriangParamType::Weights);
  default:
    return triangulate_cgal<CGAL::Dynamic_dimension_tag>(
        pc, weights, TriangParamType::Weights);
  }
}

} // namespace triangulumancer
