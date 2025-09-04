#include <pybind11/pybind11.h>
namespace py = pybind11;

void bind_Matrix(py::module_&);
void bind_ImageIO(py::module_&);
void bind_Camera(py::module_&);
void bind_ObjectInfo(py::module_&);
void bind_Config(py::module_&);
void bind_IPR(py::module_&);
void bind_Shake(py::module_&);
void bind_Track(py::module_&);
void bind_MyMath(py::module_&); 
void bind_STBCommons(py::module_&);
void bind_OTF(py::module_&);
void bind_PredField(py::module_&);
void bind_STB(py::module_&);
void bind_StereoMatch(py::module_&);
void bind_ObjectFinder(py::module_&);
void bind_BubbleRefImg(py::module_&);

PYBIND11_MODULE(openlpt, m) {
    m.doc() = "OpenLPT Python bindings";

    // Core
    bind_Matrix(m);
    bind_ImageIO(m);
    bind_Camera(m);
    bind_ObjectInfo(m);
    bind_Config(m);
    bind_IPR(m);
    bind_Shake(m);
    bind_Track(m);
    bind_MyMath(m);
    bind_STBCommons(m);
    bind_OTF(m);
    bind_PredField(m);
    bind_STB(m);
    bind_StereoMatch(m);
    bind_ObjectFinder(m);
    bind_BubbleRefImg(m);
}
