#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "STBCommons.h"

namespace py = pybind11;

// real impl
static void _bind_STBCommons_impl(py::module_& m) {
    // ----- AxisLimit -----
    py::class_<AxisLimit>(m, "AxisLimit")
        .def(py::init<>())
        .def(py::init<double,double,double,double,double,double>(),
             py::arg("x_min"), py::arg("x_max"),
             py::arg("y_min"), py::arg("y_max"),
             py::arg("z_min"), py::arg("z_max"))
        .def_readwrite("x_min", &AxisLimit::x_min)
        .def_readwrite("x_max", &AxisLimit::x_max)
        .def_readwrite("y_min", &AxisLimit::y_min)
        .def_readwrite("y_max", &AxisLimit::y_max)
        .def_readwrite("z_min", &AxisLimit::z_min)
        .def_readwrite("z_max", &AxisLimit::z_max)
        .def("check", &AxisLimit::check, py::arg("x"), py::arg("y"), py::arg("z"));

    // ----- PixelRange -----
    py::class_<PixelRange>(m, "PixelRange")
        .def(py::init<>())
        .def_readwrite("row_min", &PixelRange::row_min)
        .def_readwrite("row_max", &PixelRange::row_max)
        .def_readwrite("col_min", &PixelRange::col_min)
        .def_readwrite("col_max", &PixelRange::col_max)
        .def("set_row_range", &PixelRange::setRowRange, py::arg("row"))
        .def("set_col_range", &PixelRange::setColRange, py::arg("col"))
        .def("set_range",     &PixelRange::setRange,    py::arg("row"), py::arg("col"))
        .def("num_rows",      &PixelRange::getNumOfRow)
        .def("num_cols",      &PixelRange::getNumOfCol);

    // ----- Common numeric constants (macro -> module attributes) -----
    m.attr("SAVEPRECISION")     = py::int_(SAVEPRECISION);
    m.attr("SMALLNUMBER")       = py::float_(SMALLNUMBER);
    m.attr("SQRTSMALLNUMBER")   = py::float_(SQRTSMALLNUMBER);
    m.attr("MAGSMALLNUMBER")    = py::float_(MAGSMALLNUMBER);
    m.attr("LOGSMALLNUMBER")    = py::float_(LOGSMALLNUMBER);
    m.attr("UNDISTORT_MAX_ITER")= py::int_(UNDISTORT_MAX_ITER);
    m.attr("UNDISTORT_EPS")     = py::float_(UNDISTORT_EPS);
    m.attr("IMGPTINIT")         = py::int_(IMGPTINIT);
    m.attr("WIENER_MAX_ITER")   = py::int_(WIENER_MAX_ITER);
    m.attr("UNLINKED")          = py::int_(UNLINKED);
    m.attr("MAX_ERR_LINEARFIT") = py::float_(MAX_ERR_LINEARFIT);
    m.attr("LEN_LONG_TRACK")    = py::int_(LEN_LONG_TRACK);
}

// two exported names, to match either style in your pyOpenLPT.cpp
void bind_STBCommons(py::module_& m) { _bind_STBCommons_impl(m); }
void bind_stbcommons(py::module_& m) { _bind_STBCommons_impl(m); }
