#pragma once

#include "triangulumancer/PointConfiguration.hpp"
#include "triangulumancer/Triangulation.hpp"

namespace triangulumancer::top {

Triangulation
triangulate_placing(triangulumancer::PointConfiguration const &pc);

std::vector<Triangulation> find_neighbors(Triangulation const &t);

std::vector<Triangulation>
find_all_connected_triangulations(PointConfiguration const &pc,
                                  bool only_fine = false);

std::vector<Triangulation> find_all_triangulations(PointConfiguration const &pc,
                                                   bool only_fine = false);

} // namespace triangulumancer::top
