#pragma once

#include <string>

#include "triangulumancer/Triangulation.hpp"

namespace triangulumancer {

struct Flip {
  Triangulation parent_triangulation;

  Triangulation pre;
  Triangulation post;

  // Constructors
  Flip() = delete;
  Flip(Triangulation const &parent_triangulation_in,
       Triangulation const &pre_in, Triangulation const &post_in);

  // Basic info
  std::string repr() const;
};

} // namespace triangulumancer
