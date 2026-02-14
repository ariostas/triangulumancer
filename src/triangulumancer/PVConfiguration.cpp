#include "triangulumancer/PVConfiguration.hpp"

#include "triangulumancer/CGAL.hpp"
#include "triangulumancer/TOPCOM.hpp"

using namespace triangulumancer;

// PVConfigurationData

PVConfigurationData::PVConfigurationData(ConfigurationType config_type_in)
    : config_type(config_type_in), is_locked(false) {};

// PVConfiguration

PVConfiguration::PVConfiguration(
    std::shared_ptr<PVConfigurationData> pvc_data_in)
    : pvc_data(pvc_data_in) {}

PVConfiguration::PVConfiguration(pybind11::array_t<int64_t> const &matrix,
                                 ConfigurationType config_type)
    : pvc_data(std::make_shared<PVConfigurationData>(config_type)) {

  pybind11::buffer_info buf = matrix.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("Number of dimensions must be two");
  }

  ssize_t n_pv = buf.shape[0];
  ssize_t d = buf.shape[1];
  bool is_pc = config_type == ConfigurationType::Point;

  pvc_data->topcom_pc = topcom::Matrix(d + 1 * is_pc, n_pv);
  int64_t *ptr = static_cast<int64_t *>(buf.ptr);
  for (ssize_t i = 0; i < n_pv; i++) {
    for (ssize_t j = 0; j < d + 1 * is_pc; j++) {
      if (j == d) {
        pvc_data->topcom_pc(j, i) = 1;
      } else {
        pvc_data->topcom_pc(j, i) = (signed long)ptr[i * d + j];
      }
    }
  }

  pvc_data->has_new_pv = true;
}

size_t PVConfiguration::n_pv() const { return pvc_data->topcom_pc.coldim(); }

size_t PVConfiguration::dim() const {
  bool is_pc = pvc_data->config_type == ConfigurationType::Point;
  return (pvc_data->topcom_pc.rowdim() > 0)
             ? pvc_data->topcom_pc.rowdim() - 1 * is_pc
             : 0;
}

std::string PVConfiguration::repr() const {
  switch (pvc_data->config_type) {
  case ConfigurationType::Point:
    return "A " + std::to_string(dim()) + "-dimensional point configuration" +
           " with " + std::to_string(n_pv()) + " points";
  case ConfigurationType::Vector:
    return "A " + std::to_string(dim()) + "-dimensional vector configuration" +
           " with " + std::to_string(n_pv()) + " vectors";
  default:
    return "Unknown configuration type";
  }
}

pybind11::array_t<int64_t> PVConfiguration::pv() const {

  if (!pvc_data->has_new_pv && pvc_data->pv.has_value()) {
    return pvc_data->pv.value();
  }

  size_t n_pv_ = n_pv();
  size_t d = dim();

  auto result = pybind11::array_t<int64_t>({n_pv_, d});

  auto buf = result.mutable_data();

  for (size_t i = 0; i < n_pv_; i++) {
    for (size_t j = 0; j < d; j++) {
      buf[i * d + j] = pvc_data->topcom_pc(j, i).get_num().get_si();
    }
  }

  pvc_data->pv = result;
  pvc_data->has_new_pv = false;
  return pvc_data->pv.value();
}

void PVConfiguration::add_pv(pybind11::array_t<int64_t> const &matrix) {
  bool is_pc = pvc_data->config_type == ConfigurationType::Point;
  if (pvc_data->is_locked) {
    std::string msg = (is_pc ? "Point" : "Vector") +
                      std::string(" configuration is locked, so more ") +
                      (is_pc ? "points" : "vectors") + " can be added";
    throw std::runtime_error(msg);
  }
  pybind11::buffer_info buf = matrix.request();
  size_t d = dim();
  int64_t *ptr = static_cast<int64_t *>(buf.ptr);
  if (buf.ndim == 1) {
    if (buf.shape[0] != d) {
      throw std::runtime_error("Dimension mismatch");
    }
    auto v = topcom::Vector(d + 1 * is_pc);
    for (size_t i = 0; i < d; i++) {
      v(i) = (signed long)ptr[i];
    }
    if (is_pc) {
      v(d) = 1;
    }
    pvc_data->topcom_pc.push_back(std::move(v));
    pvc_data->has_new_pv = true;
  } else if (buf.ndim == 2) {
    size_t n_pv = buf.shape[0];
    if (buf.shape[1] != d) {
      throw std::runtime_error("Dimension mismatch");
    }
    for (size_t i = 0; i < n_pv; i++) {
      auto v = topcom::Vector(d + 1 * is_pc);
      for (size_t j = 0; j < d + 1 * is_pc; j++) {
        if (j == d) {
          v(j) = 1;
        } else {
          v(j) = (signed long)ptr[i * n_pv + j];
        }
      }
      pvc_data->topcom_pc.push_back(std::move(v));
    }
    pvc_data->has_new_pv = true;
  } else {
    throw std::runtime_error("Input must be a vector or a matrix");
  }
}

Triangulation PVConfiguration::placing_triangulation() const {
  return top::triangulate_placing(*this);
}

Triangulation PVConfiguration::fine_triangulation() const {
  return top::triangulate_fine(*this);
}

std::vector<Triangulation>
PVConfiguration::all_connected_triangulations(bool only_fine) const {
  return top::find_all_connected_triangulations(*this, only_fine);
}

std::vector<Triangulation>
PVConfiguration::all_triangulations(bool only_fine) const {
  return top::find_all_triangulations(*this, only_fine);
}

Triangulation
PVConfiguration::triangulate_with_heights(std::vector<double> const &heights) {
  if (pvc_data->config_type != ConfigurationType::Point) {
    throw std::runtime_error("Triangulation with heights is only implemented "
                             "for point configurations");
  }
  return cgal::triangulate_cgal_infer_dim(PointConfiguration(this->pvc_data),
                                          heights, true);
}

Triangulation
PVConfiguration::triangulate_with_weights(std::vector<double> const &weights) {
  if (pvc_data->config_type != ConfigurationType::Point) {
    throw std::runtime_error("Triangulation with weights is only implemented "
                             "for point configurations");
  }
  return cgal::triangulate_cgal_infer_dim(PointConfiguration(this->pvc_data),
                                          weights, false);
}

Triangulation PVConfiguration::delaunay_triangulation() const {
  if (pvc_data->config_type != ConfigurationType::Point) {
    throw std::runtime_error(
        "Delaunay triangulation is only implemented for point configurations");
  }
  return cgal::triangulate_delaunay(PointConfiguration(this->pvc_data));
}

// PointConfiguration

PointConfiguration::PointConfiguration()
    : PVConfiguration(
          std::make_shared<PVConfigurationData>(ConfigurationType::Point)) {}

PointConfiguration::PointConfiguration(
    std::shared_ptr<PVConfigurationData> pvc_data_in)
    : PVConfiguration(pvc_data_in) {
  if (pvc_data_in->config_type != ConfigurationType::Point) {
    throw std::runtime_error("Expected a point configuration data");
  }
}

PointConfiguration::PointConfiguration(pybind11::array_t<int64_t> const &matrix)
    : PVConfiguration(matrix, ConfigurationType::Point) {}

size_t PointConfiguration::n_points() const { return n_pv(); }

void PointConfiguration::add_point(pybind11::array_t<int64_t> const &matrix) {
  add_pv(matrix);
}

// VectorConfiguration

VectorConfiguration::VectorConfiguration()
    : PVConfiguration(
          std::make_shared<PVConfigurationData>(ConfigurationType::Vector)) {}

VectorConfiguration::VectorConfiguration(
    std::shared_ptr<PVConfigurationData> pvc_data_in)
    : PVConfiguration(pvc_data_in) {
  if (pvc_data_in->config_type != ConfigurationType::Vector) {
    throw std::runtime_error("Expected a vector configuration data");
  }
}

VectorConfiguration::VectorConfiguration(
    pybind11::array_t<int64_t> const &matrix)
    : PVConfiguration(matrix, ConfigurationType::Vector) {}

size_t VectorConfiguration::n_vectors() const { return n_pv(); }

void VectorConfiguration::add_vector(pybind11::array_t<int64_t> const &matrix) {
  add_pv(matrix);
}
