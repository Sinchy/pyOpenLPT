#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "CircleIdentifier.h"

namespace py = pybind11;

void bind_CircleIdentifier(py::module_ &m) {
  py::class_<CircleIdentifier>(m, "CircleIdentifier")
      .def(py::init<const Image &>(), py::arg("img_input"))

      .def(
          "BubbleCenterAndSizeByCircle",
          [](CircleIdentifier &self, double rmin, double rmax, double sense) {
            std::vector<Pt2D> centers;
            std::vector<double> radii;
            std::vector<double> metrics = self.BubbleCenterAndSizeByCircle(
                centers, radii, rmin, rmax, sense);

            return py::make_tuple(centers, radii, metrics);
          },
          py::arg("rmin"), py::arg("rmax"), py::arg("sense"));
}