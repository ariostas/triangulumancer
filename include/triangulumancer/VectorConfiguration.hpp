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
struct VectorConfigurationData {
  // This is what stores the main data
  // Vectors are stored ***WITHOUT*** an extra 1 at the end
  // Topcom uses column-major order
  topcom::PointConfiguration topcom_vc;

  // We keep a copy of the vectors in a more standard data type
  std::optional<pybind11::array_t<int64_t>> vectors;
  bool has_new_vectors;

  // After constructing triangulations with it, we lock it
  // so that it is no longer possible to add more vectors.
  bool is_locked;

  VectorConfigurationData();
  VectorConfigurationData(VectorConfigurationData &vc_data) = default;
};

class VectorConfiguration {
public:
  std::shared_ptr<VectorConfigurationData> vc_data;

  // Constructors
  VectorConfiguration();
  VectorConfiguration(std::shared_ptr<VectorConfigurationData> vc_data_in);
  VectorConfiguration(pybind11::array_t<int64_t> const &matrix);

  // Basic info
  size_t n_vectors() const;
  size_t dim() const;
  std::string repr() const;
  pybind11::array_t<int64_t> vectors() const;

  // Modifications
  void add_vectors(pybind11::array_t<int64_t> const &matrix);
  // Removing vectors complicates things a lot, so it's not supported

  // TOPCOM functionality
  std::vector<Triangulation>
  all_connected_triangulations(bool only_fine = false) const;
  std::vector<Triangulation> all_triangulations(bool only_fine = false) const;
  Triangulation placing_triangulation() const;
  Triangulation fine_triangulation() const;

  // CGAL functionality
  Triangulation triangulate_with_heights(std::vector<double> const &heights);
  Triangulation triangulate_with_weights(std::vector<double> const &weights);
  Triangulation delaunay_triangulation() const;
};

} // namespace triangulumancer
