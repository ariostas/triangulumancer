#pragma once

#include "triangulumancer/PVConfiguration.hpp"
#include "triangulumancer/Triangulation.hpp"

namespace triangulumancer::top {

Triangulation triangulate_placing(triangulumancer::PVConfiguration const &pvc);

Triangulation triangulate_fine(triangulumancer::PVConfiguration const &pvc);

std::vector<Triangulation> find_neighbors(Triangulation const &t);

std::vector<Triangulation>
find_all_connected_triangulations(PVConfiguration const &pc,
                                  bool only_fine = false);

std::vector<Triangulation> find_all_triangulations(PVConfiguration const &pvc,
                                                   bool only_fine = false);

} // namespace triangulumancer::top
