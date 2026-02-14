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

enum ConfigurationType { Point, Vector };

// We keep the actual data in a separate struct so that it's easier
// to share between triangulations
struct PVConfigurationData {
  // This is what stores the main data
  // Points from a point configuration are stored with an extra 1 at the end
  // Topcom uses column-major order
  topcom::PointConfiguration topcom_pc;

  // Whether this is a point configuration or a vector configuration
  ConfigurationType config_type;

  // We keep a copy of the points/vectors in a more standard data type
  std::optional<pybind11::array_t<int64_t>> pv;
  bool has_new_pv;

  // After constructing triangulations with it, we lock it
  // so that it is no longer possible to add more points/vectors.
  bool is_locked;

  PVConfigurationData(ConfigurationType config_type_in);
  PVConfigurationData(PVConfigurationData &pvc_data) = default;
};

class PVConfiguration {
public:
  std::shared_ptr<PVConfigurationData> pvc_data;

  // Constructors
  PVConfiguration(std::shared_ptr<PVConfigurationData> pvc_data_in);
  PVConfiguration(pybind11::array_t<int64_t> const &matrix,
                  ConfigurationType config_type);

  // Basic info
  size_t n_pv() const;
  size_t dim() const;
  std::string repr() const;
  pybind11::array_t<int64_t> pv() const;

  // Modifications
  void add_pv(pybind11::array_t<int64_t> const &matrix);
  // Removing points/vectors complicates things a lot, so it's not supported

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

class PointConfiguration : public PVConfiguration {
public:
  // Constructors
  PointConfiguration();
  PointConfiguration(std::shared_ptr<PVConfigurationData> pvc_data_in);
  PointConfiguration(pybind11::array_t<int64_t> const &matrix);

  // Aliases
  size_t n_points() const;
  pybind11::array_t<int64_t> points() const;
  void add_point(pybind11::array_t<int64_t> const &matrix);
};

class VectorConfiguration : public PVConfiguration {
public:
  // Constructors
  VectorConfiguration();
  VectorConfiguration(std::shared_ptr<PVConfigurationData> pvc_data_in);
  VectorConfiguration(pybind11::array_t<int64_t> const &matrix);

  // Aliases
  size_t n_vectors() const;
  pybind11::array_t<int64_t> vectors() const;
  void add_vector(pybind11::array_t<int64_t> const &matrix);
};

} // namespace triangulumancer
