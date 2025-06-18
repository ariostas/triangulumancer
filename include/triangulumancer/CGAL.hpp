#pragma once

#include "triangulumancer/PointConfiguration.hpp"
#include "triangulumancer/Triangulation.hpp"

namespace triangulumancer::cgal {

template <typename TDim>
Triangulation triangulate_cgal(PointConfiguration const &pc,
                               std::vector<double> const &params,
                               bool use_heights);

Triangulation triangulate_cgal_infer_dim(PointConfiguration const &pc,
                                         std::vector<double> const &params,
                                         bool use_heights);

Triangulation
triangulate_delaunay(triangulumancer::PointConfiguration const &pc);

} // namespace triangulumancer::cgal
