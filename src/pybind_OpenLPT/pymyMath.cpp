#include <pybind11/pybind11.h>

namespace py = pybind11;

// 先占位：等你需要把 myMATH 里的具体函数暴露出来时，再补充绑定
void bind_MyMath(py::module_& m)
{
    auto sub = m.def_submodule("mymath", "Lightweight math helpers");
    // 示例：sub.def("linspace", ...);
}
