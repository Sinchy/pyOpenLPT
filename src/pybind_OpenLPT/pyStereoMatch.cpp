#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "StereoMatch.h"
#include "ObjectInfo.h"   // Tracer2D / Bubble2D / Object2D / Object3D / Pt2D, Pt3D
#include "Camera.h"
#include "Config.h"       // ObjectConfig, ObjectKind

namespace py = pybind11;

/**
 * A small holder that owns inputs so StereoMatch can safely keep const-pointers.
 *  - cams_     : copy of cameras
 *  - owned_    : vector<vector<unique_ptr<Object2D>>> built from Python inputs
 *  - cfg_      : non-owning pointer to a concrete ObjectConfig passed from Python
 *  - core_     : the real StereoMatch engine constructed from the three above
 */
class PyStereoMatch {
public:
    // ctor #1: Friendly — pass 2D centers per camera; kind is deduced from cfg.kind()
    PyStereoMatch(const std::vector<Camera>& cams,
                  const std::vector<std::vector<Pt2D>>& bb2d_per_cam,
                  const ObjectConfig& cfg)
        : cams_(cams), cfg_(&cfg)
    {
        const bool as_tracer = (cfg.kind() == ObjectKind::Tracer);
        owned_.resize(bb2d_per_cam.size());
        for (size_t c = 0; c < bb2d_per_cam.size(); ++c) {
            owned_[c].reserve(bb2d_per_cam[c].size());
            for (const auto& p : bb2d_per_cam[c]) {
                if (as_tracer) {
                    owned_[c].emplace_back(std::make_unique<Tracer2D>(p));
                } else {
                    // Minimal radius placeholder for Bubble2D; tune if needed
                    owned_[c].emplace_back(std::make_unique<Bubble2D>(p, /*r_px=*/2.0));
                }
            }
        }
        core_ = std::make_unique<StereoMatch>(cams_, owned_, *cfg_);
    }

    // ctor #2: Advanced — pass existing Object2D objects (Python-owned pointers). We clone them.
    PyStereoMatch(const std::vector<Camera>& cams,
                  const std::vector<std::vector<Object2D*>>& obj2d_per_cam,
                  const ObjectConfig& cfg)
        : cams_(cams), cfg_(&cfg)
    {
        owned_.resize(obj2d_per_cam.size());
        for (size_t c = 0; c < obj2d_per_cam.size(); ++c) {
            owned_[c].reserve(obj2d_per_cam[c].size());
            for (auto* p : obj2d_per_cam[c]) {
                if (!p) continue;
                owned_[c].emplace_back(p->clone()); // virtual clone()
            }
        }
        core_ = std::make_unique<StereoMatch>(cams_, owned_, *cfg_);
    }

    // Run and return only 3D centers (simple Python-friendly output).
    std::vector<Pt3D> match_centers() const {
        auto objs = core_->match();  // vector<unique_ptr<Object3D>>
        std::vector<Pt3D> out;
        out.reserve(objs.size());
        for (auto& up : objs) out.push_back(up->_pt_center);
        return out;
    }

private:
    std::vector<Camera> cams_;   // keep a copy so references in StereoMatch stay valid
    const ObjectConfig* cfg_;    // non-owning; lifetime managed by Python object
    std::vector<std::vector<std::unique_ptr<Object2D>>> owned_;
    std::unique_ptr<StereoMatch> core_;
};

void bind_StereoMatch(py::module_& m)
{
    py::class_<PyStereoMatch>(m, "StereoMatch")
        // Friendly ctor: cams + [[(x,y),...], ...] + cfg  (kind auto-detected)
        .def(py::init<
                 const std::vector<Camera>&,
                 const std::vector<std::vector<Pt2D>>&,
                 const ObjectConfig&
             >(),
             py::arg("cams"),
             py::arg("bb2d_per_cam"),
             py::arg("config"),
             // keep_alive<1,4>: keep 'config' alive as long as self lives
             py::keep_alive<1, 4>(),
             "Construct from 2D centers per camera. Kind is deduced from config.kind()."
        )
        // Advanced ctor: cams + [[Object2D*, ...], ...] + cfg  (clone into owned_)
        .def(py::init<
                 const std::vector<Camera>&,
                 const std::vector<std::vector<Object2D*>>&,
                 const ObjectConfig&
             >(),
             py::arg("cams"),
             py::arg("obj2d_per_cam"),
             py::arg("config"),
             py::keep_alive<1, 4>(),
             "Construct from existing Object2D instances; they are cloned internally."
        )
        // Run
        .def("match", &PyStereoMatch::match_centers,
             "Run stereo matching and return a list of 3D centers [(X,Y,Z), ...].");
}
