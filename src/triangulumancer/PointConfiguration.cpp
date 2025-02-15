#include "triangulumancer/PointConfiguration.hpp"

using namespace triangulumancer;

PointConfiguration::PointConfiguration(
    pybind11::array_t<int64_t> const &matrix) {

  pybind11::buffer_info buf = matrix.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("Number of dimensions must be two");
  }

  ssize_t n_pts = buf.shape[0];
  ssize_t d = buf.shape[1];

  pc = topcom::Matrix(d + 1, n_pts);
  int64_t *ptr = static_cast<int64_t *>(buf.ptr);
  for (ssize_t i = 0; i < n_pts; i++) {
    for (ssize_t j = 0; j <= d; j++) {
      if (j == d) {
        pc(j, i) = 1;
      } else {
        pc(j, i) = (signed long)ptr[i * d + j];
      }
    }
  }

  has_new_points = true;
}

size_t PointConfiguration::n_points() const { return pc.coldim(); }

size_t PointConfiguration::dim() const {
  return (pc.rowdim() > 0) ? pc.rowdim() - 1 : 0;
}

std::string PointConfiguration::repr() const {
  return "A " + std::to_string(dim()) + "-dimensional point configuration" +
         " with " + std::to_string(n_points()) + " points";
}

pybind11::array_t<int64_t> PointConfiguration::points() {

  if (!has_new_points && points_.has_value()) {
    return points_.value();
  }

  size_t n_pts = n_points();
  size_t d = dim();

  auto result = pybind11::array_t<int64_t>({n_pts, d});

  auto buf = result.mutable_data();

  for (size_t i = 0; i < n_pts; i++) {
    for (size_t j = 0; j < d; j++) {
      buf[i * d + j] = pc(j, i).get_num().get_si();
    }
  }

  points_ = result;
  has_new_points = false;
  return points_.value();
}

void PointConfiguration::add_points(pybind11::array_t<int64_t> const &matrix) {
  pybind11::buffer_info buf = matrix.request();
  size_t d = dim();
  int64_t *ptr = static_cast<int64_t *>(buf.ptr);
  if (buf.ndim == 1) {
    if (buf.shape[0] != d) {
      throw std::runtime_error("Dimension mismatch");
    }
    auto v = topcom::Vector(d + 1);
    for (size_t i = 0; i < d; i++) {
      v(i) = (signed long)ptr[i];
    }
    v(d) = 1;
    pc.push_back(std::move(v));
    has_new_points = true;
  } else if (buf.ndim == 2) {
    size_t n_pts = buf.shape[0];
    if (buf.shape[1] != d) {
      throw std::runtime_error("Dimension mismatch");
    }
    for (size_t i = 0; i < n_pts; i++) {
      auto v = topcom::Vector(d + 1);
      for (size_t j = 0; j <= d; j++) {
        if (j == d) {
          v(j) = 1;
        } else {
          v(j) = (signed long)ptr[i * n_pts + j];
        }
      }
      pc.push_back(std::move(v));
    }
    has_new_points = true;
  } else {
    throw std::runtime_error("Input must be a vector or a matrix");
  }
}
