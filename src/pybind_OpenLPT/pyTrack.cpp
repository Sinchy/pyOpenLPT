#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Track.h"
#include "ObjectInfo.h"     // Object3D / Bubble3D / Tracer3D
#include "pybind_utils.h"   // make_unique_obj3d_list

namespace py = pybind11;

void bind_Track(py::module_& m) {
    py::class_<Track>(m, "Track")
        .def(py::init<>())

        // 仅绑定公开方法；不要暴露/触碰私有成员
        .def("save_track", &Track::saveTrack, py::arg("ostream"), py::arg("track_id"))
        .def("load_track", &Track::loadTrack, py::arg("ifstream"), py::arg("cfg"), py::arg("cams"))

        // 关键：add_next 适配器（Python 传 Object3D，绑定层重建 unique_ptr 后调库）
        .def("add_next",
             [](Track& self, Object3D& obj3d, int t) {
                 std::vector<Object3D*> tmp{ &obj3d };
                 auto vec = make_unique_obj3d_list(tmp);   // vector<unique_ptr<Object3D>>，size=1
                 self.addNext(std::move(vec[0]), t);       // 假设签名为 addNext(std::unique_ptr<Object3D>, int)
             },
             py::arg("obj3d"), py::arg("t"));
}
