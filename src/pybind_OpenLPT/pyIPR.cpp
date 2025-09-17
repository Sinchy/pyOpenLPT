// pyIPR.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "IPR.h"
#include "Config.h"
#include "Camera.h"
#include "ImageIO.h"
#include "ObjectInfo.h"     // Object3D / Bubble3D / ...
#include "pybind_utils.h"   // make_unique_obj3d_list (我们刚建的工具头)

namespace py = pybind11;

void bind_IPR(py::module_& m) {
    py::class_<IPR>(m, "IPR")
        // 若 IPR 内部保存对 cams 的引用，keep_alive 能避免悬垂
        .def(py::init<std::vector<Camera>&>(),
             py::arg("cams"),
             py::keep_alive<1,2>())

        // 直接转发（假设签名不含 unique_ptr 入参）
        .def("run_ipr",
             &IPR::runIPR,
             py::arg("cfg"), py::arg("images"))

        // 适配器：Python 传 list[Object3D] -> 绑定层重建 vector<unique_ptr<Object3D>>
        .def("save_obj_info",
             [](IPR& self,
                const std::string& filename,
                const std::vector<Object3D*>& objs,   // Python: list[Bubble3D/...]
                const ObjectConfig& cfg) {
                 auto out = make_unique_obj3d_list(objs);  // 重建 unique_ptr 容器
                 py::gil_scoped_release nogil;
                 self.saveObjInfo(filename, out, cfg);
             },
             py::arg("filename"), py::arg("objs"), py::arg("cfg"));
}
