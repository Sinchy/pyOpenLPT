#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Matrix.h"

namespace py = pybind11;

static double get_m(const Matrix<double>& M, int i, int j) { return M(i, j); }
static void   set_m(Matrix<double>& M, int i, int j, double v) { M(i, j) = v; }

void bind_Matrix(py::module_& m) {
    // Pt2D / Pt3D are small wrappers over Matrix<double> (we expose a simple API)
    py::class_<Pt2D>(m, "Pt2D")
        .def(py::init<>())
        .def(py::init<double,double>(), py::arg("x"), py::arg("y"))
        .def_property("x", [](const Pt2D& p){ return p[0]; }, [](Pt2D& p, double v){ p[0]=v; })
        .def_property("y", [](const Pt2D& p){ return p[1]; }, [](Pt2D& p, double v){ p[1]=v; });

    py::class_<Pt3D>(m, "Pt3D")
        .def(py::init<>())
        .def(py::init<double,double,double>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_property("x", [](const Pt3D& p){ return p[0]; }, [](Pt3D& p, double v){ p[0]=v; })
        .def_property("y", [](const Pt3D& p){ return p[1]; }, [](Pt3D& p, double v){ p[1]=v; })
        .def_property("z", [](const Pt3D& p){ return p[2]; }, [](Pt3D& p, double v){ p[2]=v; });

    // Image is Matrix<double> with image semantics; 给出最小读写接口
    py::class_<Image>(m, "Image")
        .def(py::init<>())
        .def(py::init<int,int,double>(), py::arg("rows"), py::arg("cols"), py::arg("val") = 0.0)
        .def("rows", [](const Image& I){ return I.getDimRow(); })
        .def("cols", [](const Image& I){ return I.getDimCol(); })
        .def("get", &get_m, py::arg("i"), py::arg("j"))
        .def("set", &set_m, py::arg("i"), py::arg("j"), py::arg("value"));
}
