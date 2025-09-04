#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BubbleRefImg.h"
#include "ObjectInfo.h"
#include "Camera.h"
#include "Matrix.h"

namespace py = pybind11;

// A Python-friendly wrapper that accepts plain coordinates
// and constructs native unique_ptr<Object2D/3D> containers internally.
static bool _cal_ref_from_points(
    BubbleRefImg& self,
    const std::vector<Pt3D>& obj3d_centers,
    const std::vector<std::vector<Pt2D>>& bb2d_centers_all,
    const std::vector<Camera>& cams,
    const std::vector<Image>& images,
    double r_thres, int n_bb_thres)
{
    // build objs_out
    std::vector<std::unique_ptr<Object3D>> objs_out;
    objs_out.reserve(obj3d_centers.size());
    for (const auto& p : obj3d_centers) {
        // minimal: use Tracer3D to carry a center (works for the algorithm which reads pt only)
        auto up = std::make_unique<Tracer3D>();
        up->_pt_center = p;
        objs_out.emplace_back(std::move(up));
    }

    // build bb2d_list_all (use Tracer2D with center-only)
    std::vector<std::vector<std::unique_ptr<Object2D>>> bb2d_all(bb2d_centers_all.size());
    for (size_t c=0; c<bb2d_centers_all.size(); ++c) {
        bb2d_all[c].reserve(bb2d_centers_all[c].size());
        for (const auto& p : bb2d_centers_all[c]) {
            auto up = std::make_unique<Tracer2D>(p);
            bb2d_all[c].emplace_back(std::move(up));
        }
    }

    return self.calBubbleRefImg(objs_out, bb2d_all, cams, images, r_thres, n_bb_thres);
}

static void _bind_BubbleRefImg_impl(py::module_& m) {
    py::class_<BubbleRefImg>(m, "BubbleRefImg")
        .def(py::init<>())

        // Original signature (advanced): you can pass lists-of-unique_ptrs
        .def("cal_bubble_ref_img",
             &BubbleRefImg::calBubbleRefImg,
             py::arg("objs_out"),
             py::arg("bb2d_list_all"),
             py::arg("cams"),
             py::arg("images"),
             py::arg("r_thres") = 6.0,
             py::arg("n_bb_thres") = 5)

        // Friendly wrapper: accept coordinates
        //   obj3d_centers      -> [ (X,Y,Z), ... ]
        //   bb2d_centers_all   -> [ [ (x,y), ... ] for each cam ]
        .def("cal_bubble_ref_from_points",
             &_cal_ref_from_points,
             py::arg("obj3d_centers"),
             py::arg("bb2d_centers_all"),
             py::arg("cams"),
             py::arg("images"),
             py::arg("r_thres") = 6.0,
             py::arg("n_bb_thres") = 5)

        // Accessors
        .def("__getitem__",
             [](const BubbleRefImg& self, int camID) -> const Image& {
                 return self[camID];
             }, py::return_value_policy::reference_internal, py::arg("cam_id"))
        .def("get_int_ref", &BubbleRefImg::getIntRef, py::arg("cam_id"))
        .def_readwrite("_is_valid", &BubbleRefImg::_is_valid);
}

void bind_BubbleRefImg(py::module_& m) { _bind_BubbleRefImg_impl(m); }
void bind_bubblerefimg(py::module_& m) { _bind_BubbleRefImg_impl(m); }
