#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "STB.h"
#include "Config.h"
#include "ImageIO.h"
#include "Track.h"

namespace py = pybind11;

void bind_STB(py::module_& m)
{
    py::class_<STB>(m, "STB")
        // STB(BasicSetting const& setting, std::string const& type, std::string const& obj_cfg_path)
        .def(py::init<const BasicSetting&, const std::string&, const std::string&>(),
             py::arg("setting"), py::arg("obj_type"), py::arg("obj_cfg_path"))
        // 处理一帧（把图像作为 Image = Matrix<double> 的列表传进来）
        .def("process_frame",
         [](STB& self, int frame_id, std::vector<Image> img_list) {
             self.processFrame(frame_id, img_list);  
             return img_list;                         
         },
         py::arg("frame_id"), py::arg("img_list"),
         "Run STB on a frame; returns the modified residual images.")
        // 批量保存/加载（便于 Python 快速验证）
        .def("saveTracksAll", &STB::saveTracksAll, py::arg("folder"), py::arg("t"))
        .def("loadTracksAll", &STB::loadTracksAll, py::arg("folder"), py::arg("t"));
}
