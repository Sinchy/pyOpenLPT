// pyObjectFinder.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ObjectFinder.h"
#include "ObjectInfo.h"
#include "Config.h"
#include "Matrix.h"

namespace py = pybind11;

void bind_ObjectFinder(py::module_& m) {
    py::class_<ObjectFinder2D>(m, "ObjectFinder2D")
        .def(py::init<>())

        // 直接返回 std::vector<std::unique_ptr<Object2D>>
        .def("findObject2D",
             [](ObjectFinder2D& self, const Image& image, const ObjectConfig& obj_cfg) {
                 return self.findObject2D(image, obj_cfg);
             },
             py::arg("image"), py::arg("obj_cfg"));
}
