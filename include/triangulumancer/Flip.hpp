#pragma once

#include <string>

#include "triangulumancer/Triangulation.hpp"

namespace triangulumancer {

struct Flip {
  Triangulation pre;
  Triangulation post;

  // Constructors
  Flip() = delete;
  Flip(Triangulation const &pre, Triangulation const &post);

  // Basic info
  std::string repr() const;
};

} // namespace triangulumancer
