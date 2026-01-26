#include "triangulumancer/VectorConfiguration.hpp"

#include "triangulumancer/VectorConfiguration.hpp"
#include "triangulumancer/TOPCOM.hpp"

using namespace triangulumancer;

VectorConfigurationData::VectorConfigurationData() : is_locked(false) {};

VectorConfiguration::VectorConfiguration()
    : vc_data(std::make_shared<VectorConfigurationData>()) {}

VectorConfiguration::VectorConfiguration(
    std::shared_ptr<VectorConfigurationData> vc_data_in)
    : vc_data(vc_data_in) {}

VectorConfiguration::VectorConfiguration(pybind11::array_t<int64_t> const &matrix)
    : vc_data(std::make_shared<VectorConfigurationData>()) {

  pybind11::buffer_info buf = matrix.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("Number of dimensions must be two");
  }

  ssize_t n_pts = buf.shape[0];
  ssize_t d = buf.shape[1];

  vc_data->topcom_vc = topcom::Matrix(d, n_pts);
  int64_t *ptr = static_cast<int64_t *>(buf.ptr);
  for (ssize_t i = 0; i < n_pts; i++) {
    for (ssize_t j = 0; j < d; j++) {
      vc_data->topcom_vc(j, i) = (signed long)ptr[i * d + j];
    }
  }

  vc_data->has_new_vectors = true;
}

size_t VectorConfiguration::n_vectors() const {
  return vc_data->topcom_vc.coldim();
}

size_t VectorConfiguration::dim() const {
  return vc_data->topcom_vc.rowdim();
}

std::string VectorConfiguration::repr() const {
  return "A " + std::to_string(dim()) + "-dimensional vector configuration" +
         " with " + std::to_string(n_vectors()) + " vectors";
}

pybind11::array_t<int64_t> VectorConfiguration::vectors() const {

  if (!vc_data->has_new_vectors && vc_data->vectors.has_value()) {
    return vc_data->vectors.value();
  }

  size_t n_pts = n_vectors();
  size_t d = dim();

  auto result = pybind11::array_t<int64_t>({n_pts, d});

  auto buf = result.mutable_data();

  for (size_t i = 0; i < n_pts; i++) {
    for (size_t j = 0; j < d; j++) {
      buf[i * d + j] = vc_data->topcom_vc(j, i).get_num().get_si();
    }
  }

  vc_data->vectors = result;
  vc_data->has_new_vectors = false;
  return vc_data->vectors.value();
}

void VectorConfiguration::add_vectors(pybind11::array_t<int64_t> const &matrix) {
  if (vc_data->is_locked) {
    throw std::runtime_error(
        "Vector configuration is locked, so more vectors can be added");
  }
  pybind11::buffer_info buf = matrix.request();
  size_t d = dim();
  int64_t *ptr = static_cast<int64_t *>(buf.ptr);
  if (buf.ndim == 1) {
    if (buf.shape[0] != d) {
      throw std::runtime_error("Dimension mismatch");
    }
    auto v = topcom::Vector(d);
    for (size_t i = 0; i < d; i++) {
      v(i) = (signed long)ptr[i];
    }
    vc_data->topcom_vc.push_back(std::move(v));
    vc_data->has_new_vectors = true;
  } else if (buf.ndim == 2) {
    size_t n_pts = buf.shape[0];
    if (buf.shape[1] != d) {
      throw std::runtime_error("Dimension mismatch");
    }
    for (size_t i = 0; i < n_pts; i++) {
      auto v = topcom::Vector(d);
      for (size_t j = 0; j < d; j++) {
        v(j) = (signed long)ptr[i * n_pts + j];
      }
      vc_data->topcom_vc.push_back(std::move(v));
    }
    vc_data->has_new_vectors = true;
  } else {
    throw std::runtime_error("Input must be a vector or a matrix");
  }
}

Triangulation VectorConfiguration::placing_triangulation() const {
  return top::triangulate_placing(*this);
}

/*
Triangulation VectorConfiguration::fine_triangulation() const {
  return top::triangulate_fine(*this);
}

std::vector<Triangulation>
VectorConfiguration::all_connected_triangulations(bool only_fine) const {
  return top::find_all_connected_triangulations(*this, only_fine);
}
*/

std::vector<Triangulation>
VectorConfiguration::all_triangulations(bool only_fine) const {
  return top::find_all_triangulations(*this, only_fine);
}
