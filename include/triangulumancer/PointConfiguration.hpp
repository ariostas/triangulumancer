#pragma once

#include <memory>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

// TOPCOM imports
#include "PointConfiguration.hh"

namespace triangulumancer {

class Triangulation;

struct PointConfiguration {
  // This is what stores the main data
  // Points are stored with an extra 1 at the end
  // Topcom uses column-major order
  topcom::PointConfiguration pc;

  // We keep a copy of the points in a more standard data type
  std::optional<pybind11::array_t<int64_t>> points_;
  bool has_new_points;

  // We keep a copy that Triangulations can use
  std::shared_ptr<PointConfiguration> shared_copy;

  // Constructors
  PointConfiguration() : pc(), points_(std::nullopt), has_new_points(false) {}
  PointConfiguration(pybind11::array_t<int64_t> const &matrix);

  // Basic info
  size_t n_points() const;
  size_t dim() const;
  std::string repr() const;
  pybind11::array_t<int64_t> points();

  // Modifications
  void add_points(pybind11::array_t<int64_t> const &matrix);
  // Removing points complicates things a lot, so it's not supported

  // TOPCOM functionality
  // all_triangulations
  // pushing_triangulation

  // CGAL functionality
  // Triangulation triangulate(std::vector<int64_t> &parameters);
};

} // namespace triangulumancer
