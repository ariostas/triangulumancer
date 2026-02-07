#pragma once

#include "triangulumancer/PointConfiguration.hpp"
#include "triangulumancer/Triangulation.hpp"
#include "triangulumancer/VectorConfiguration.hpp"

namespace triangulumancer::top {

Triangulation
triangulate_placing(triangulumancer::PointConfiguration const &pc);

Triangulation
triangulate_placing(triangulumancer::VectorConfiguration const &vc);

Triangulation triangulate_fine(triangulumancer::PointConfiguration const &pc);
// Triangulation triangulate_fine(triangulumancer::VectorConfiguration const
// &vc);

std::vector<Triangulation> find_neighbors(Triangulation const &t);

std::vector<Triangulation>
find_all_connected_triangulations(PointConfiguration const &pc,
                                  bool only_fine = false);

std::vector<Triangulation> find_all_triangulations(PointConfiguration const &pc,
                                                   bool only_fine = false);
std::vector<Triangulation>
find_all_triangulations(VectorConfiguration const &pc, bool only_fine = false);

} // namespace triangulumancer::top
