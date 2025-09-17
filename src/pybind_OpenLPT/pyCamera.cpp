#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Camera.h"
#include "STBCommons.h"
#include "myMATH.h"

namespace py = pybind11;

void bind_Camera(py::module_& m) {
    // ========== Enums ==========
    py::enum_<CameraType>(m, "CameraType")
        .value("PINHOLE",    CameraType::PINHOLE)
        .value("POLYNOMIAL", CameraType::POLYNOMIAL)
        .value("PINPLATE",   CameraType::PINPLATE)
        .export_values();

    py::enum_<RefPlane>(m, "RefPlane")
        .value("REF_X", RefPlane::REF_X)
        .value("REF_Y", RefPlane::REF_Y)
        .value("REF_Z", RefPlane::REF_Z)
        .export_values();

    // ========== Structs ==========
    py::class_<PinholeParam>(m, "PinholeParam")
        .def(py::init<>())
        .def_readwrite("n_row",        &PinholeParam::n_row)
        .def_readwrite("n_col",        &PinholeParam::n_col)
        .def_readwrite("cam_mtx",      &PinholeParam::cam_mtx)
        .def_readwrite("is_distorted", &PinholeParam::is_distorted)
        .def_readwrite("n_dist_coeff", &PinholeParam::n_dist_coeff)
        .def_readwrite("dist_coeff",   &PinholeParam::dist_coeff)
        .def_readwrite("r_mtx",        &PinholeParam::r_mtx)
        .def_readwrite("t_vec",        &PinholeParam::t_vec)
        .def_readwrite("r_mtx_inv",    &PinholeParam::r_mtx_inv)
        .def_readwrite("t_vec_inv",    &PinholeParam::t_vec_inv)
        .def("__repr__", [](const PinholeParam& p){
            return "<PinholeParam " + std::to_string(p.n_col) + "x" + std::to_string(p.n_row) +
                   (p.is_distorted ? " distorted" : " undistorted") + ">";
        });

    py::class_<PolyParam>(m, "PolyParam")
        .def(py::init<>())
        .def_readwrite("n_row",     &PolyParam::n_row)
        .def_readwrite("n_col",     &PolyParam::n_col)
        .def_readwrite("ref_plane", &PolyParam::ref_plane)
        .def_readwrite("plane",     &PolyParam::plane)
        .def_readwrite("n_coeff",   &PolyParam::n_coeff)
        .def_readwrite("u_coeffs",  &PolyParam::u_coeffs)
        .def_readwrite("du_coeffs", &PolyParam::du_coeffs)
        .def_readwrite("v_coeffs",  &PolyParam::v_coeffs)
        .def_readwrite("dv_coeffs", &PolyParam::dv_coeffs)
        .def("__repr__", [](const PolyParam& p){
            return "<PolyParam " + std::to_string(p.n_col) + "x" + std::to_string(p.n_row) +
                   ", n_coeff=" + std::to_string(p.n_coeff) + ">";
        });

    py::class_<PinPlateParam, PinholeParam>(m, "PinPlateParam")
        .def(py::init<>())
        .def_readwrite("plane",             &PinPlateParam::plane)
        .def_readwrite("pt3d_closest",      &PinPlateParam::pt3d_closest)
        .def_readwrite("refract_array",     &PinPlateParam::refract_array)
        .def_readwrite("w_array",           &PinPlateParam::w_array)
        .def_readwrite("n_plate",           &PinPlateParam::n_plate)
        .def_readwrite("proj_tol",          &PinPlateParam::proj_tol)
        .def_readwrite("proj_nmax",         &PinPlateParam::proj_nmax)
        .def_readwrite("lr",                &PinPlateParam::lr)
        .def_readwrite("refract_ratio_max", &PinPlateParam::refract_ratio_max)
        .def("__repr__", [](const PinPlateParam& p){
            return "<PinPlateParam n_plate=" + std::to_string(p.n_plate) + ">";
        });

    // ========== Camera ==========
    py::class_<Camera>(m, "Camera")
        // 构造器
        .def(py::init<>())
        .def(py::init<std::string>(), py::arg("file_name"))
        .def(py::init<const Camera&>(), py::arg("other"))

        // 加载/保存
        .def("loadParameters", [](Camera& c, const std::string& path){ c.loadParameters(path); },
             py::arg("file_name"))
        .def("saveParameters", &Camera::saveParameters, py::arg("file_name"))

        // 成员变量
        .def_readwrite("_type",           &Camera::_type)
        .def_readwrite("_pinhole_param",  &Camera::_pinhole_param)
        .def_readwrite("_poly_param",     &Camera::_poly_param)
        .def_readwrite("_pinplate_param", &Camera::_pinplate_param)
        .def_readwrite("_max_intensity",  &Camera::_max_intensity)
        .def_readwrite("_is_active",      &Camera::_is_active)

        // 方法
        .def("updatePolyDuDv",    &Camera::updatePolyDuDv)
        .def("updatePt3dClosest", &Camera::updatePt3dClosest)
        .def("rmtxTorvec",        &Camera::rmtxTorvec, py::arg("r_mtx"))

        .def("getNRow", &Camera::getNRow)
        .def("getNCol", &Camera::getNCol)

        .def("project", &Camera::project,
             py::arg("pt_world"), py::arg("is_print_detail") = false)

        .def("worldToUndistImg", &Camera::worldToUndistImg,
             py::arg("pt_world"), py::arg("param"))

        .def("distort", &Camera::distort,
             py::arg("pt_img_undist"), py::arg("param"))

        .def("undistort", &Camera::undistort,
             py::arg("pt_img_dist"), py::arg("param"))

        .def("polyProject", &Camera::polyProject, py::arg("pt_world"))

        .def("polyImgToWorld", &Camera::polyImgToWorld,
             py::arg("pt_img_dist"), py::arg("plane_world"))

        .def("refractPlate", &Camera::refractPlate, py::arg("pt_world"))

        .def("lineOfSight",     &Camera::lineOfSight,     py::arg("pt_img_undist"))
        .def("pinholeLine",     &Camera::pinholeLine,     py::arg("pt_img_undist"))
        .def("pinplateLine",    &Camera::pinplateLine,    py::arg("pt_img_undist"))
        .def("polyLineOfSight", &Camera::polyLineOfSight, py::arg("pt_img_dist"))

        .def("__repr__", [](const Camera& c){
            return "<Camera type=" + std::to_string(static_cast<int>(c._type)) +
                   " active=" + (c._is_active ? "True" : "False") +
                   " size=" + std::to_string(c.getNCol()) + "x" + std::to_string(c.getNRow()) +
                   ">";
        });
}
