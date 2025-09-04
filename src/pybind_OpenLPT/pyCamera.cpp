#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Camera.h"

namespace py = pybind11;

void bind_Camera(py::module_& m) {
    py::enum_<CameraType>(m, "CameraType")
        .value("PINHOLE", CameraType::PINHOLE)
        .value("POLYNOMIAL", CameraType::POLYNOMIAL)
        .value("PINPLATE", CameraType::PINPLATE)
        .export_values();

    py::class_<Camera>(m, "Camera")
        .def(py::init<>())
        .def(py::init<std::string>(), py::arg("file_name"))
        .def_readwrite("_type", &Camera::_type)
        .def_readwrite("_max_intensity", &Camera::_max_intensity)
        .def_readwrite("_is_active", &Camera::_is_active);
}
