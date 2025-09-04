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
        .def_readwrite("n_channel", &ImageParam::n_channel);

    py::class_<ImageIO>(m, "ImageIO")
        .def(py::init<>())
        .def(py::init<std::string,std::string>(), py::arg("folder_path"), py::arg("file_img_path"))
        .def("load_img_path", &ImageIO::loadImgPath, py::arg("folder_path"), py::arg("file_img_path"))
        .def("load_img", &ImageIO::loadImg, py::arg("img_id"))
        .def("save_img", &ImageIO::saveImg, py::arg("save_path"), py::arg("image"))
        .def("set_img_param", &ImageIO::setImgParam, py::arg("param"))
        .def("get_img_param", &ImageIO::getImgParam)
        .def("get_curr_img_id", &ImageIO::getCurrImgID)
        .def("get_img_path", &ImageIO::getImgPath);
}
