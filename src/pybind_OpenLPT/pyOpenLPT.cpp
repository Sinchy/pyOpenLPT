// pyOpenLPT.cpp
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <string>
#include <memory>
#include <iostream>

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

int run_openlpt(const std::string& config_path);

struct PythonStreamRedirector {
    std::unique_ptr<py::scoped_ostream_redirect> out;
    std::unique_ptr<py::scoped_estream_redirect> err;

    PythonStreamRedirector() {
        py::object sys = py::module_::import("sys");
        out = std::make_unique<py::scoped_ostream_redirect>(std::cout, sys.attr("stdout"));
        err = std::make_unique<py::scoped_estream_redirect>(std::cerr, sys.attr("stderr"));
    }
    PythonStreamRedirector(py::object py_stdout, py::object py_stderr) {
        out = std::make_unique<py::scoped_ostream_redirect>(std::cout, py_stdout);
        err = std::make_unique<py::scoped_estream_redirect>(std::cerr, py_stderr);
    }
    void close() { err.reset(); out.reset(); }
};

void bind_PythonStreamRedirector(py::module_& m) {
    py::class_<PythonStreamRedirector>(m, "PythonStreamRedirector")
        .def(py::init<>())
        .def(py::init<py::object, py::object>(), py::arg("stdout"), py::arg("stderr"))
        .def("close", &PythonStreamRedirector::close)
        .def("__enter__", [](PythonStreamRedirector &self) { return &self; },
             py::return_value_policy::reference)
        .def("__exit__", [](PythonStreamRedirector &self, py::object, py::object, py::object) {
            self.close(); return false;
        });
}

PYBIND11_MODULE(pyopenlpt, m) {
    m.doc() = "OpenLPT Python bindings";

    bind_PythonStreamRedirector(m);

    bind_Config(m);
    bind_Matrix(m);
    bind_ImageIO(m);
    bind_Camera(m);
    bind_ObjectInfo(m);
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

    m.def("run",
          [](const std::string& config_file_path) {
              py::object sys = py::module_::import("sys");
              py::scoped_ostream_redirect  out(std::cout, sys.attr("stdout"));
              py::scoped_estream_redirect  err(std::cerr, sys.attr("stderr"));
              py::gil_scoped_release nogil;
              int rc = run_openlpt(config_file_path);
              if (rc != 0)
                  throw py::value_error("OpenLPT failed with return code = " + std::to_string(rc));
          },
          py::arg("config_file_path"));
}
