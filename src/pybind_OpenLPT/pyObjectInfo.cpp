#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ObjectInfo.h"

namespace py = pybind11;

// Helper: expose vector<unique_ptr<Object2D>> as (size, get(i)->ref)
static size_t obj2d_size(const Object3D& o) { return o._obj2d_list.size(); }
static Object2D* obj2d_at(Object3D& o, size_t i) { return o._obj2d_list.at(i).get(); }

void bind_ObjectInfo(py::module_& m) {
    py::class_<Object2D>(m, "Object2D")
        .def_property("_pt_center",
            [](Object2D& o){ return &o._pt_center; },
            [](Object2D& o, const Pt2D& v){ o._pt_center = v; },
            py::return_value_policy::reference_internal);

    py::class_<Tracer2D, Object2D>(m, "Tracer2D")
        .def(py::init<>())
        .def(py::init<const Pt2D&>())
        .def_readwrite("_r_px", &Tracer2D::_r_px);

    py::class_<Bubble2D, Object2D>(m, "Bubble2D")
        .def(py::init<>())
        .def(py::init<const Pt2D&, double>())
        .def_readwrite("_r_px", &Bubble2D::_r_px);

    py::class_<Object3D>(m, "Object3D")
        .def_readwrite("_pt_center", &Object3D::_pt_center)
        .def_readwrite("_is_tracked", &Object3D::_is_tracked)
        .def("project_object2d", &Object3D::projectObject2D, py::arg("cam_list"))
        .def("is_reconstructable", &Object3D::isReconstructable, py::arg("cam_list"))
        .def("obj2d_size", &obj2d_size)
        .def("obj2d_at", &obj2d_at, py::return_value_policy::reference_internal);
}
