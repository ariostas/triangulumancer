#pragma once

#include "triangulumancer/PointConfiguration.hpp"
#include "triangulumancer/Triangulation.hpp"

namespace triangulumancer::top {

Triangulation
triangulate_placing(triangulumancer::PointConfiguration const &pc);

} // namespace triangulumancer::top
