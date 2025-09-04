#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Shake.h"

namespace py = pybind11;

void bind_Shake(py::module_& m) {
    py::enum_<ObjFlag>(m, "ObjFlag")
        .value("None", ObjFlag::None)
        .value("Ghost", ObjFlag::Ghost)
        .value("Repeated", ObjFlag::Repeated)
        .export_values();

    py::class_<Shake>(m, "Shake")
        .def(py::init<const std::vector<Camera>&, const ObjectConfig&>(),
             py::arg("cams"), py::arg("obj_cfg"))
        .def("run_shake", &Shake::runShake, py::arg("objs"), py::arg("img_orig"))
        .def("cal_residual_image", &Shake::calResidualImage,
             py::arg("objs"), py::arg("img_orig"), py::arg("flags") = nullptr);
}
