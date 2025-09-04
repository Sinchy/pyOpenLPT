#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ObjectFinder.h"
#include "ObjectInfo.h"
#include "Matrix.h"
#include "Config.h"

namespace py = pybind11;

// helpers: convert returned vector<unique_ptr<Object2D>> into light-weight results
static std::vector<Pt2D> _to_centers(std::vector<std::unique_ptr<Object2D>> v) {
    std::vector<Pt2D> out;
    out.reserve(v.size());
    for (auto& u : v) out.push_back(u->_pt_center);
    return out;
}

static std::vector<std::tuple<double,double,double>>
_to_circles(std::vector<std::unique_ptr<Object2D>> v) {
    std::vector<std::tuple<double,double,double>> out;
    out.reserve(v.size());
    for (auto& u : v) {
        double r = 0.0;
        if (auto* t = dynamic_cast<Tracer2D*>(u.get()))  r = t->_r_px;
        else if (auto* b = dynamic_cast<Bubble2D*>(u.get())) r = b->_r_px;
        out.emplace_back(u->_pt_center[0], u->_pt_center[1], r);
    }
    return out;
}

static void _bind_ObjectFinder_impl(py::module_& m) {
    py::class_<ObjectFinder2D>(m, "ObjectFinder2D")
        .def(py::init<>())

        // Raw API (advanced): return Python list of polymorphic Object2D.
        // Ownership moves to Python; you'll get a list of Object2D/Tracer2D/Bubble2D instances.
        .def("find_object2d_raw",
             [](ObjectFinder2D& self, const Image& img, const ObjectConfig& cfg) {
                 return self.findObject2D(img, cfg);
             }, py::arg("image"), py::arg("config"))

        // Convenience: just centers [(x,y), ...]
        .def("find_centers2d",
             [](ObjectFinder2D& self, const Image& img, const ObjectConfig& cfg) {
                 return _to_centers(self.findObject2D(img, cfg));
             }, py::arg("image"), py::arg("config"))

        // Convenience: circles [(x,y,r), ...]
        .def("find_circles2d",
             [](ObjectFinder2D& self, const Image& img, const ObjectConfig& cfg) {
                 return _to_circles(self.findObject2D(img, cfg));
             }, py::arg("image"), py::arg("config"));
}

void bind_ObjectFinder(py::module_& m) { _bind_ObjectFinder_impl(m); }
void bind_objectfinder(py::module_& m) { _bind_ObjectFinder_impl(m); }
