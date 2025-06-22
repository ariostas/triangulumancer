#include "triangulumancer/PointConfiguration.hpp"

#include "triangulumancer/CGAL.hpp"
#include "triangulumancer/PointConfiguration.hpp"
#include "triangulumancer/TOPCOM.hpp"

using namespace triangulumancer;

PointConfigurationData::PointConfigurationData() : is_locked(false) {};

PointConfiguration::PointConfiguration()
    : pc_data(std::make_shared<PointConfigurationData>()) {}

PointConfiguration::PointConfiguration(
    std::shared_ptr<PointConfigurationData> pc_data_in)
    : pc_data(pc_data_in) {}

PointConfiguration::PointConfiguration(pybind11::array_t<int64_t> const &matrix)
    : pc_data(std::make_shared<PointConfigurationData>()) {

  pybind11::buffer_info buf = matrix.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("Number of dimensions must be two");
  }

  ssize_t n_pts = buf.shape[0];
  ssize_t d = buf.shape[1];

  pc_data->topcom_pc = topcom::Matrix(d + 1, n_pts);
  int64_t *ptr = static_cast<int64_t *>(buf.ptr);
  for (ssize_t i = 0; i < n_pts; i++) {
    for (ssize_t j = 0; j <= d; j++) {
      if (j == d) {
        pc_data->topcom_pc(j, i) = 1;
      } else {
        pc_data->topcom_pc(j, i) = (signed long)ptr[i * d + j];
      }
    }
  }

  pc_data->has_new_points = true;
}

size_t PointConfiguration::n_points() const {
  return pc_data->topcom_pc.coldim();
}

size_t PointConfiguration::dim() const {
  return (pc_data->topcom_pc.rowdim() > 0) ? pc_data->topcom_pc.rowdim() - 1
                                           : 0;
}

std::string PointConfiguration::repr() const {
  return "A " + std::to_string(dim()) + "-dimensional point configuration" +
         " with " + std::to_string(n_points()) + " points";
}

pybind11::array_t<int64_t> PointConfiguration::points() const {

  if (!pc_data->has_new_points && pc_data->points.has_value()) {
    return pc_data->points.value();
  }

  size_t n_pts = n_points();
  size_t d = dim();

  auto result = pybind11::array_t<int64_t>({n_pts, d});

  auto buf = result.mutable_data();

  for (size_t i = 0; i < n_pts; i++) {
    for (size_t j = 0; j < d; j++) {
      buf[i * d + j] = pc_data->topcom_pc(j, i).get_num().get_si();
    }
  }

  pc_data->points = result;
  pc_data->has_new_points = false;
  return pc_data->points.value();
}

void PointConfiguration::add_points(pybind11::array_t<int64_t> const &matrix) {
  if (pc_data->is_locked) {
    throw std::runtime_error(
        "Point configuration is locked, so more points can be added");
  }
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
    pc_data->topcom_pc.push_back(std::move(v));
    pc_data->has_new_points = true;
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
      pc_data->topcom_pc.push_back(std::move(v));
    }
    pc_data->has_new_points = true;
  } else {
    throw std::runtime_error("Input must be a vector or a matrix");
  }
}

Triangulation PointConfiguration::placing_triangulation() const {
  return top::triangulate_placing(*this);
}

std::vector<Triangulation>
PointConfiguration::all_connected_triangulations(bool only_fine) const {
  return top::find_all_connected_triangulations(*this, only_fine);
}

std::vector<Triangulation>
PointConfiguration::all_triangulations(bool only_fine) const {
  return top::find_all_triangulations(*this, only_fine);
}

Triangulation PointConfiguration::triangulate_with_heights(
    std::vector<double> const &heights) {
  return cgal::triangulate_cgal_infer_dim(*this, heights, true);
}

Triangulation PointConfiguration::triangulate_with_weights(
    std::vector<double> const &weights) {
  return cgal::triangulate_cgal_infer_dim(*this, weights, false);
}

Triangulation PointConfiguration::delaunay_triangulation() const {
  return cgal::triangulate_delaunay(*this);
}
