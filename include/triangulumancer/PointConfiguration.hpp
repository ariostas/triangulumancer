#pragma once

#include <memory>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

// TOPCOM imports
#include "PointConfiguration.hh"

namespace triangulumancer {

class Triangulation;

// We keep the actual data in a separate struct so that it's easier
// to share between triangulations
struct PointConfigurationData {
  // This is what stores the main data
  // Points are stored with an extra 1 at the end
  // Topcom uses column-major order
  topcom::PointConfiguration topcom_pc;

  // We keep a copy of the points in a more standard data type
  std::optional<pybind11::array_t<int64_t>> points;
  bool has_new_points;

  // After constructing triangulations with it, we lock it
  // so that it is no longer possible to add more points.
  bool is_locked;

  PointConfigurationData();
  PointConfigurationData(PointConfigurationData &pc_data) = default;
};

class PointConfiguration {
public:
  std::shared_ptr<PointConfigurationData> pc_data;

  // Constructors
  PointConfiguration();
  PointConfiguration(std::shared_ptr<PointConfigurationData> pc_data_in);
  PointConfiguration(pybind11::array_t<int64_t> const &matrix);

  // Basic info
  size_t n_points() const;
  size_t dim() const;
  std::string repr() const;
  pybind11::array_t<int64_t> points() const;

  // Modifications
  void add_points(pybind11::array_t<int64_t> const &matrix);
  // Removing points complicates things a lot, so it's not supported

  // TOPCOM functionality
  std::vector<Triangulation>
  all_connected_triangulations(bool only_fine = false) const;
  std::vector<Triangulation> all_triangulations(bool only_fine = false) const;
  Triangulation placing_triangulation() const;

  // CGAL functionality
  Triangulation triangulate_with_heights(std::vector<double> const &heights);
  Triangulation triangulate_with_weights(std::vector<double> const &weights);
  Triangulation delaunay_triangulation() const;
};

} // namespace triangulumancer
