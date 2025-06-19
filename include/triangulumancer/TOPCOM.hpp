#pragma once

#include "triangulumancer/PointConfiguration.hpp"
#include "triangulumancer/Triangulation.hpp"

namespace triangulumancer::top {

Triangulation
triangulate_placing(triangulumancer::PointConfiguration const &pc);

std::vector<Triangulation> find_neighbors(Triangulation const &t);

} // namespace triangulumancer::top
