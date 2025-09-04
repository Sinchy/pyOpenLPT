#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "IPR.h"
#include "Config.h"

namespace py = pybind11;

void bind_IPR(py::module_& m) {
    py::class_<IPR>(m, "IPR")
        // keep_alive<1,2>: IPR keeps refs to cams (argument 2) — make cams live as long as self
        .def(py::init<std::vector<Camera>&>(), py::arg("cams"), py::keep_alive<1,2>())
        .def("run_ipr", &IPR::runIPR, py::arg("cfg"), py::arg("images"))
        .def("save_obj_info", &IPR::saveObjInfo, py::arg("filename"), py::arg("objs"), py::arg("cfg"));
}
