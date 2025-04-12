#pragma once

#include <memory>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace triangulumancer {

class PointConfiguration;

struct Triangulation {
  std::shared_ptr<PointConfiguration> pc;

  pybind11::array_t<int64_t> simplices_;

  // Constructors
  Triangulation() = delete;
  Triangulation(std::shared_ptr<PointConfiguration> pc) : pc(pc) {}
  Triangulation(std::shared_ptr<PointConfiguration> pc,
                pybind11::array_t<int64_t> simplices_)
      : pc(pc), simplices_(simplices_) {}

  // Basic info
  size_t n_simplices() const;
  size_t dim() const;
  std::string repr() const;
  pybind11::array_t<int64_t> simplices();

  // TOPCOM functionality
  // neighbor_triangulations
};

} // namespace triangulumancer
