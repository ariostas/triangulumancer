#pragma once

#include <memory>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "triangulumancer/PointConfiguration.hpp"
#include "triangulumancer/VectorConfiguration.hpp"

namespace triangulumancer {

struct Triangulation {
  // This is only a light wrapper
  PointConfiguration pc;
  VectorConfiguration vc;
  bool isPC;

  pybind11::array_t<int64_t> m_simplices;

  // Constructors
  Triangulation() = delete;
  Triangulation(std::shared_ptr<PointConfigurationData> pc_data_in,
                pybind11::array_t<int64_t> simplices_in);
  Triangulation(PointConfiguration const &pc,
                pybind11::array_t<int64_t> simplices_in);
  Triangulation(std::shared_ptr<VectorConfigurationData> vc_data_in,
                pybind11::array_t<int64_t> simplices_in);
  Triangulation(VectorConfiguration const &vc,
                pybind11::array_t<int64_t> simplices_in);

  // Basic info
  size_t n_simplices() const;
  size_t dim() const;
  std::string repr() const;
  pybind11::array_t<int64_t> simplices() const;

  // TOPCOM functionality
  std::vector<Triangulation> neighbors() const;
};

} // namespace triangulumancer
