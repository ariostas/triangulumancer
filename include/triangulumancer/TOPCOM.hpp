#pragma once

#include "triangulumancer/Flip.hpp"
#include "triangulumancer/PVConfiguration.hpp"
#include "triangulumancer/Triangulation.hpp"

namespace triangulumancer::top {

Triangulation triangulate_placing(PVConfiguration const &pvc);

Triangulation triangulate_fine(PVConfiguration const &pvc);

std::vector<Flip> find_flips(Triangulation const &t);

std::vector<Triangulation> find_neighbors(Triangulation const &t);

std::vector<Triangulation>
find_all_connected_triangulations(PVConfiguration const &pc,
                                  bool only_fine = false);

std::vector<Triangulation> find_all_triangulations(PVConfiguration const &pvc,
                                                   bool only_fine = false);

} // namespace triangulumancer::top
