#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ImageIO.h"

namespace py = pybind11;

void bind_ImageIO(py::module_& m) {
    py::class_<ImageParam>(m, "ImageParam")
        .def(py::init<>())
        .def_readwrite("n_row", &ImageParam::n_row)
        .def_readwrite("n_col", &ImageParam::n_col)
        .def_readwrite("bits_per_sample", &ImageParam::bits_per_sample)
        .def_readwrite("n_channel", &ImageParam::n_channel)
        .def("__repr__", [](const ImageParam& p){
            return "<ImageParam n_row=" + std::to_string(p.n_row) +
                   " n_col=" + std::to_string(p.n_col) +
                   " bits_per_sample=" + std::to_string(p.bits_per_sample) +
                   " n_channel=" + std::to_string(p.n_channel) + ">";
        });

    py::class_<ImageIO>(m, "ImageIO")
        .def(py::init<>())
        .def(py::init<std::string, std::string>(),
             py::arg("folder_path"), py::arg("file_img_path"))
        .def(py::init<const ImageIO&>())
        .def("loadImgPath", &ImageIO::loadImgPath,
             py::arg("folder_path"), py::arg("file_img_path"))
        .def("loadImg", &ImageIO::loadImg,
             py::arg("img_id"),
             py::return_value_policy::move)
        .def("saveImg", &ImageIO::saveImg,
             py::arg("save_path"), py::arg("image"))
        .def("setImgParam", &ImageIO::setImgParam,
             py::arg("img_param"))
        .def("getImgParam", &ImageIO::getImgParam)
        .def("getCurrImgID", &ImageIO::getCurrImgID)
        .def("getImgPath", &ImageIO::getImgPath)
        .def("__repr__", [](const ImageIO& io){
            return "<ImageIO at " + std::to_string(reinterpret_cast<std::uintptr_t>(&io)) + ">";
        });
}
