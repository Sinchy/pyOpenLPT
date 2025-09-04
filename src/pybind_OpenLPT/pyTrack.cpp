#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Track.h"

namespace py = pybind11;

static size_t len_obj3d_list(const Track& t) { return t._obj3d_list.size(); }
static const Object3D* obj3d_at(const Track& t, size_t i) { return t._obj3d_list.at(i).get(); }

void bind_Track(py::module_& m) {
    py::class_<Track>(m, "Track")
        .def(py::init<>())
        .def_readwrite("_t_list", &Track::_t_list)
        .def_readwrite("_active", &Track::_active)
        .def("add_next", &Track::addNext, py::arg("obj3d"), py::arg("t"))
        .def("save_track", &Track::saveTrack, py::arg("ostream"), py::arg("track_id"))
        .def("load_track", &Track::loadTrack, py::arg("ifstream"), py::arg("cfg"), py::arg("cams"))
        .def("size", &len_obj3d_list)
        .def("obj3d_at", &obj3d_at, py::return_value_policy::reference);
}
