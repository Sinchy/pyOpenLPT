#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "StereoMatch.h"
#include "ObjectInfo.h"   // Tracer2D / Bubble2D / Object2D / Object3D / Pt2D, Pt3D
#include "Camera.h"
#include "Config.h"       // ObjectConfig, ObjectKind
#include "pybind_utils.h" // make_unique_obj2d_grid

namespace py = pybind11;

/**
 * pyStereoMatch
 * Wrapper class that manages ownership of cameras, config and 2D objects,
 * and internally holds a StereoMatch instance that references them.
 */
/** pyStereoMatch: wrapper that owns cams + 2D objects and references obj_cfg */
class pyStereoMatch {
public:
    pyStereoMatch(const std::vector<Camera>& cams,
                  const std::vector<std::vector<Object2D*>>& obj2d_by_cam,
                  const ObjectConfig& obj_cfg)
        : _cams_copy(cams),
          _obj2d_owned(make_unique_obj2d_grid(obj2d_by_cam)),
          _obj_cfg_ptr(&obj_cfg),
          _matcher(std::make_unique<StereoMatch>(_cams_copy, _obj2d_owned, *_obj_cfg_ptr)) {}

    std::vector<std::unique_ptr<Object3D>> match() const {
        return _matcher->match();
    }

private:
    std::vector<Camera> _cams_copy;  // own a copy
    std::vector<std::vector<std::unique_ptr<Object2D>>> _obj2d_owned; // own deep copy
    const ObjectConfig* _obj_cfg_ptr = nullptr; // just a pointer; do not own
    std::unique_ptr<StereoMatch> _matcher;
};

void bind_StereoMatch(py::module_& m) {
    py::class_<pyStereoMatch>(m, "StereoMatch")
        .def(py::init<const std::vector<Camera>&,
                      const std::vector<std::vector<Object2D*>>&,
                      const ObjectConfig&>(),
             py::arg("cams"),
             py::arg("obj2d_by_cam"),
             py::arg("obj_cfg"),
             // ensure Python obj_cfg outlives self (extra safety though we only store a pointer)
             py::keep_alive<1, 4>())
        .def("match", &pyStereoMatch::match);
}
